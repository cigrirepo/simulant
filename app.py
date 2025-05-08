import os
import re
import json
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.linalg import cholesky
import plotly.express as px
import openai
from fpdf import FPDF

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="RiskSim360", layout="wide")


def init_openai():
    """
    Initialize OpenAI API key from env or Streamlit secrets.
    """
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.error(
            "OpenAI API key not found. "
            "Set OPENAI_API_KEY in environment or in Streamlit secrets."
        )
    else:
        openai.api_key = key


init_openai()

# ── Session State Initialization ──────────────────────────────────────────────
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {}
if "parsed_df" not in st.session_state:
    st.session_state["parsed_df"] = None


# ── Helper Functions ──────────────────────────────────────────────────────────
def parse_assumptions_df(df: pd.DataFrame) -> list[dict]:
    """
    Convert a structured DataFrame into a list of assumption dicts.
    Expects columns: ['Driver','Distribution','Param1','Param2','Param3'].
    """
    assumptions = []
    for _, row in df.iterrows():
        assumptions.append({
            "driver": str(row["Driver"]),
            "dist": str(row["Distribution"]).lower(),
            "params": [
                float(row["Param1"]),
                float(row["Param2"]),
                float(row["Param3"]),
            ],
        })
    return assumptions


def apply_correlation(n_sims: int, assumptions: list, corr_matrix: np.ndarray) -> np.ndarray:
    """
    Generate correlated standard normals via Cholesky decomposition.
    """
    L = cholesky(corr_matrix, lower=True)
    z = np.random.standard_normal((n_sims, len(assumptions)))
    return z @ L.T


def sample_from_copula(corr_normals: np.ndarray, assumptions: list) -> np.ndarray:
    """
    Transform correlated normals to target distributions via inverse CDF.
    """
    samples = np.zeros_like(corr_normals)
    for i, a in enumerate(assumptions):
        q = stats.norm.cdf(corr_normals[:, i])
        mn, p2, p3 = a["params"]
        dist = a["dist"]

        if dist == "triangular":
            mode, mx = p2, p3
            c = (mode - mn) / (mx - mn)
            samples[:, i] = stats.triang(c, loc=mn, scale=(mx - mn)).ppf(q)
        elif dist == "normal":
            samples[:, i] = stats.norm(loc=mn, scale=p2).ppf(q)
        elif dist == "lognormal":
            samples[:, i] = stats.lognorm(s=p2, scale=np.exp(mn)).ppf(q)
        elif dist == "uniform":
            samples[:, i] = stats.uniform(loc=mn, scale=(p2 - mn)).ppf(q)
        else:
            samples[:, i] = np.nan

    return samples


def run_monte_carlo(
    assumptions: list[dict],
    n_sims: int,
    corr_matrix: np.ndarray | None = None
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation, optionally applying input correlations.
    Returns a DataFrame of shape (n_sims, num_drivers).
    """
    if corr_matrix is not None:
        corr_normals = apply_correlation(n_sims, assumptions, corr_matrix)
        sims = sample_from_copula(corr_normals, assumptions)
    else:
        sims = np.zeros((n_sims, len(assumptions)))
        for i, a in enumerate(assumptions):
            mn, p2, p3 = a["params"]
            dist = a["dist"]

            if dist == "triangular":
                mode, mx = p2, p3
                c = (mode - mn) / (mx - mn)
                sims[:, i] = stats.triang(c, loc=mn, scale=(mx - mn)).rvs(n_sims)
            elif dist == "normal":
                sims[:, i] = stats.norm(loc=mn, scale=p2).rvs(n_sims)
            elif dist == "lognormal":
                sims[:, i] = stats.lognorm(s=p2, scale=np.exp(mn)).rvs(n_sims)
            elif dist == "uniform":
                sims[:, i] = stats.uniform(loc=mn, scale=(p2 - mn)).rvs(n_sims)
            else:
                sims[:, i] = np.nan

    cols = [a["driver"] for a in assumptions]
    return pd.DataFrame(sims, columns=cols)


def calculate_npv(
    sim_df: pd.DataFrame,
    cashflow_cols: list[str],
    discount_rate: float
) -> np.ndarray:
    """
    Calculate NPV for each simulation. Assumes cashflow_cols correspond to t=0,1,... in order.
    """
    periods = np.arange(len(cashflow_cols))
    pv = sim_df[cashflow_cols].values / ((1 + discount_rate) ** periods)
    return pv.sum(axis=1)


def calculate_var_cvar(npv_array: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """
    Compute Value-at-Risk and Conditional VaR at the alpha level.
    """
    var = np.percentile(npv_array, alpha * 100)
    cvar = npv_array[npv_array <= var].mean()
    return var, cvar


def tornado_chart(impact_df: pd.DataFrame) -> px.bar:
    """
    Generate a horizontal bar chart of driver impacts on NPV.
    """
    return px.bar(
        impact_df.sort_values("Impact"),
        x="Impact",
        y="Driver",
        orientation="h",
        title="Tornado Chart: Driver Impacts on NPV",
    )


def generate_risk_mermaid(drivers: list[str]) -> str:
    """
    Build a Mermaid flowchart snippet connecting each driver to a single outcome node.
    """
    nodes = [f"    node{i}[{d}]" for i, d in enumerate(drivers, 1)]
    arrows = [f"    node{i} --> Outcome" for i in range(1, len(drivers) + 1)]
    return (
        "```mermaid\n"
        "flowchart LR\n"
        + "\n".join(nodes)
        + "\n    Outcome((Net Present Value))\n"
        + "\n".join(arrows)
        + "\n```"
    )


def generate_narrative(findings: dict) -> str:
    """
    Use OpenAI to produce a concise executive summary from findings, with error handling.
    """
    prompt = (
        "Write a concise executive summary of the following risk analysis results:\n"
        + json.dumps(findings, indent=2)
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.5,
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Narrative generation failed: {e}")
        return ""
    m = re.search(r"```(?:json)?(.*?)```", text, re.S)
    return m.group(1).strip() if m else text


def export_pdf(
    hist_fig: px.bar,
    tor_fig: px.bar,
    narrative: str,
    summary_dict: dict
) -> None:
    """
    Generate and stream a PDF containing narrative, summary stats, and charts.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        hpath = os.path.join(tmpdir, "hist.png")
        tpath = os.path.join(tmpdir, "tornado.png")
        hist_fig.write_image(hpath)
        tor_fig.write_image(tpath)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "RiskSim360 Report", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, narrative or "No narrative available.")
        pdf.ln(5)

        for k, v in summary_dict.items():
            pdf.cell(0, 8, f"{k}: {v}", ln=True)
        pdf.ln(5)

        pdf.image(hpath, w=180)
        pdf.ln(5)
        pdf.image(tpath, w=180)

        out_path = os.path.join(tmpdir, "RiskSim360_Report.pdf")
        pdf.output(out_path)

        with open(out_path, "rb") as f:
            st.download_button(
                "📄 Download PDF Report", f,
                file_name="RiskSim360_Report.pdf", mime="application/pdf"
            )


def export_excel(
    sim_df: pd.DataFrame,
    npv_array: np.ndarray,
    assumptions_df: pd.DataFrame
) -> None:
    """
    Generate and stream an Excel workbook with simulation data, NPVs, and assumptions.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        writer = pd.ExcelWriter(tmp.name, engine="xlsxwriter")
        sim_df.to_excel(writer, sheet_name="Simulations", index=False)
        pd.DataFrame({"NPV": npv_array}).to_excel(
            writer, sheet_name="NPV Summary", index=False
        )
        assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)
        writer.close()

        with open(tmp.name, "rb") as f:
            st.download_button(
                "📊 Download Excel Workbook", f,
                file_name="RiskSim360_Output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ── Main App ───────────────────────────────────────────────────────────────────
def main():
    st.title("RiskSim360: Monte Carlo Risk Simulator")
    st.sidebar.header("Inputs & Scenario Manager")

    # -- Structured Assumptions Upload --
    uploaded = st.sidebar.file_uploader("Upload Assumptions (CSV/XLSX)", type=["csv","xlsx"])
    df_upload = None
    if uploaded:
        try:
            df_upload = (pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv")
                         else pd.read_excel(uploaded))
            st.sidebar.success("Assumptions file loaded.")
        except Exception as e:
            st.sidebar.error(f"Failed to load file: {e}")

    # -- Free-Text Parser --
    free_text = st.sidebar.text_area("Or paste assumption text...", height=100)
    if st.sidebar.button("Parse Free-Text") and free_text:
        prompt_text = "Parse the following financial assumptions into JSON...\n" + free_text
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=300,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"```(?:json)?(.*?)```", raw, re.S)
        jstr = m.group(1).strip() if m else raw
        try:
            parsed = json.loads(jstr)
            df_parsed = pd.json_normalize(parsed)
            df_parsed.columns = ["Driver", "Distribution", "Param1", "Param2", "Param3"]
            st.sidebar.success("Parsed assumptions:")
            st.sidebar.dataframe(df_parsed)
            st.session_state["parsed_df"] = df_parsed
        except Exception as e:
            st.sidebar.error(f"Parsing failed: {e}")

    # -- Save Scenario --
    scenario_name = st.sidebar.text_input("Scenario Name")
    if st.sidebar.button("Save Scenario") and scenario_name:
        if df_upload is not None:
            st.session_state["scenarios"][scenario_name] = df_upload
        elif st.session_state["parsed_df"] is not None:
            st.session_state["scenarios"][scenario_name] = st.session_state["parsed_df"]
        st.sidebar.success(f"Saved scenario '{scenario_name}'")

    scenarios = list(st.session_state["scenarios"].keys())
    selected = st.sidebar.selectbox("Select Scenario", scenarios) if scenarios else None

    # -- Simulation Settings --
    n_sims = st.sidebar.number_input("Number of Simulations", min_value=1000,
                                     max_value=100000, value=20000, step=1000)
    discount_rate = st.sidebar.number_input("Discount Rate", min_value=0.0,
                                            max_value=1.0, value=0.1, step=0.01)

    # -- Correlation Matrix --
    corr_matrix = None
    uploaded_corr = st.sidebar.file_uploader("Upload Correlation Matrix (CSV)", type="csv")
    if uploaded_corr:
        try:
            corr_df = pd.read_csv(uploaded_corr, index_col=0)
            if corr_df.shape[0] != corr_df.shape[1] or list(corr_df.columns) != list(corr_df.index):
                raise ValueError("Matrix must be square with matching labels.")
            if np.any(np.linalg.eigvals(corr_df) < 0):
                raise ValueError("Matrix is not positive semidef.")
            corr_matrix = corr_df.values
            st.sidebar.success("Correlation matrix loaded.")
        except Exception as e:
            st.sidebar.error(f"Invalid correlation matrix: {e}")

    # ── Run Simulation ────────────────────────────────────────────────────────────
    if selected and st.sidebar.button("Run Simulation"):
        df_assump = st.session_state["scenarios"][selected]
        assumptions = parse_assumptions_df(df_assump)
        sim_df = run_monte_carlo(assumptions, int(n_sims), corr_matrix)

        drivers = [a["driver"] for a in assumptions]
        npv_arr = calculate_npv(sim_df, drivers, discount_rate)
        var, cvar = calculate_var_cvar(npv_arr)
        base_npv = npv_arr.mean()

        st.subheader(f"Scenario: {selected}")
        histfig = px.histogram(npv_arr, nbins=50, title="NPV Distribution")
        st.plotly_chart(histfig, use_container_width=True)

        impacts = []
        for d in drivers:
            pert = sim_df.copy()
            pert[d] += sim_df[d].std()
            impacts.append({
                "Driver": d,
                "Impact": calculate_npv(pert, drivers, discount_rate).mean() - base_npv,
            })
        tor_df = pd.DataFrame(impacts)
        tor_fig = tornado_chart(tor_df)
        st.plotly_chart(tor_fig, use_container_width=True)

        st.markdown(
            f"**VaR (5%):** ${var:,.2f}   "
            f"**CVaR:** ${cvar:,.2f}   "
            f"**P(NPV<0):** {(npv_arr < 0).mean() * 100:.2f}%"
        )

        st.markdown("### Risk Workflow Diagram")
        st.code(generate_risk_mermaid(drivers), language="markdown")

        # ── AI Narrative ────────────────────────────────────────────────────────
        if st.sidebar.button("Generate AI Narrative"):
            findings = {
                "Scenario": selected,
                "Mean NPV": round(base_npv, 2),
                "Std Dev": round(npv_arr.std(), 2),
                "VaR(5%)": round(var, 2),
                "CVaR": round(cvar, 2),
                "P(NPV<0)": f"{(npv_arr < 0).mean() * 100:.2f}%"
            }
            with st.spinner("Generating AI narrative..."):
                narrative = generate_narrative(findings)
            if narrative:
                st.subheader("Executive Summary")
                st.write(narrative)

        # ── What-If Sliders ─────────────────────────────────────────────────────
        st.markdown("### 🔁 Re-run With Adjusted Inputs")
        adj_vals = {}
        for d in drivers:
            m, s = sim_df[d].mean(), sim_df[d].std()
            if s > 0:
                adj_vals[d] = st.slider(d, float(m - 2*s), float(m + 2*s), float(m), step=float(s/10))
            else:
                adj_vals[d] = st.number_input(f"{d} (constant)", value=float(m))
        adj_df = pd.DataFrame([adj_vals])
        adj_npv = calculate_npv(adj_df, drivers, discount_rate)[0]
        st.markdown(f"**Adjusted NPV:** ${adj_npv:,.2f}   **Δ:** ${adj_npv - base_npv:,.2f}")

        # ── Export Results ──────────────────────────────────────────────────────
        st.markdown("### Export Results")
        summary_dict = {
            "Mean NPV": f"${base_npv:,.2f}",
            "Std Dev": f"${npv_arr.std():,.2f}",
            "VaR(5%)": f"${var:,.2f}",
            "CVaR": f"${cvar:,.2f}",
            "P(NPV<0)": f"{(npv_arr < 0).mean() * 100:.2f}%"
        }
        export_pdf(histfig, tor_fig, narrative if 'narrative' in locals() else "", summary_dict)
        export_excel(sim_df, npv_arr, df_assump)


if __name__ == "__main__":
    main()
