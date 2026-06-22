"""
Step 5: Autonomous BI Dashboard — Dataset-Agnostic
====================================================
Upload ANY CSV dataset. The AI agent:
  1. Reads and profiles the schema
  2. Runs exploratory analysis (statistics, distributions, correlations)
  3. Generates KPI cards and chart specs adapted to your data
  4. Produces an executive summary with key insights

Everything in the dashboard — KPIs, charts, filters, table columns —
is derived dynamically from what the agent discovers in the data.
No hardcoded column names. Works with any dataset.

Usage:
    streamlit run dashboard.py
"""

import io
import json
import os
import re
import sys
import textwrap
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from agent_loop import run_agent_loop
from agent_tools import read_dataset_schema, run_python_analysis
from dashboard_config import generate_dashboard_config, profile_dataset

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst · Autonomous BI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS — premium dark theme, fully generic
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0f0c29 40%, #1a1040 100%);
    color: #e8e0f0;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #0a0619 100%);
    border-right: 1px solid rgba(139,92,246,0.2);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label { color: #c4b5fd !important; }

/* ---- Upload / Landing ---- */
.upload-box {
    border: 2px dashed rgba(139,92,246,0.5);
    border-radius: 20px;
    padding: 60px 40px;
    text-align: center;
    background: linear-gradient(135deg, rgba(139,92,246,0.06) 0%, rgba(59,7,100,0.04) 100%);
    transition: border-color 0.2s;
}
.upload-box:hover { border-color: rgba(167,139,250,0.8); }

/* ---- KPI card ---- */
.kpi-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.18) 0%, rgba(109,40,217,0.08) 100%);
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 16px;
    padding: 20px 22px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
    height: 130px;
    display: flex; flex-direction: column; justify-content: center;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(139,92,246,0.28); }
.kpi-label { font-size:0.72rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#a78bfa; margin-bottom:8px; }
.kpi-value { font-size:2rem; font-weight:800; color:#f5f3ff; line-height:1.1; }
.kpi-sub   { font-size:0.68rem; color:#b8a9d4; margin-top:5px; font-style:italic; }

/* ---- Section title ---- */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; font-weight: 700;
    color: #e9d5ff; margin: 20px 0 10px;
    padding-bottom: 6px;
    border-bottom: 2px solid rgba(139,92,246,0.25);
}

/* ---- Insight block ---- */
.insight-block {
    background: rgba(139,92,246,0.08);
    border-left: 3px solid #7c3aed;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin: 8px 0;
    font-size: 0.88rem; line-height: 1.7;
    color: #e9d5ff; white-space: pre-wrap;
}
.agent-thinking {
    background: rgba(16,185,129,0.06);
    border-left: 3px solid #10b981;
    border-radius: 0 10px 10px 0;
    padding: 10px 16px; margin: 6px 0;
    font-size: 0.82rem; color: #a7f3d0;
    font-family: 'Courier New', monospace;
}
.user-msg {
    background: rgba(99,102,241,0.08);
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 10px 14px; margin: 6px 0;
    font-size: 0.87rem; color: #c7d2fe;
}

/* ---- Stat pill ---- */
.stat-pill {
    display: inline-block;
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.78rem; color: #d4c8f5;
    margin: 2px 3px;
}

/* ---- Override Streamlit muted text (caption, small) ---- */
.stCaption, [data-testid="stCaption"] p,
.element-container .stMarkdown p small,
small { color: #b8a9d4 !important; }
[data-testid="stMetric"] label { color: #c4b5fd !important; }
[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: #c4b5fd !important; }

/* ---- Inputs ---- */
.stTextInput input, .stTextArea textarea {
    background: rgba(20,10,50,0.8) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    color: #e8e0f0 !important; border-radius: 10px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4c1d95);
    color: white; border: none; border-radius: 10px;
    padding: 10px 24px; font-weight: 600; transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #5b21b6);
    transform: translateY(-1px); box-shadow: 0 4px 16px rgba(124,58,237,0.4);
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10,6,25,0.7); border-radius: 12px; padding: 4px;
}
.stTabs [data-baseweb="tab"] { color: #a78bfa; border-radius: 8px; }
.stTabs [aria-selected="true"] { background: rgba(124,58,237,0.3) !important; color: #e9d5ff !important; }
.stFileUploader { border: 2px dashed rgba(139,92,246,0.4) !important; border-radius: 12px; }
.stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
/* ---- Selectbox / radio labels ---- */
.stSelectbox label, .stRadio label, .stMultiSelect label,
.stSlider label, .stDateInput label, .stTextInput label,
.stTextArea label { color: #c4b5fd !important; }
/* ---- Expander header ---- */
.stExpander summary { color: #c4b5fd !important; }
/* ---- dataframe header ---- */
.stDataFrame thead th { color: #e9d5ff !important; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────
PALETTE = ["#a78bfa", "#f472b6", "#34d399", "#60a5fa", "#fbbf24",
           "#f87171", "#c084fc", "#4ade80", "#38bdf8", "#fb923c"]

SCALE_PURPLE = ["#2e1065", "#4c1d95", "#6d28d9", "#7c3aed", "#a78bfa", "#ddd6fe"]
SCALE_GREEN  = ["#14532d", "#166534", "#15803d", "#16a34a", "#4ade80"]
SCALE_HEAT   = ["#dc2626", "#f59e0b", "#fbbf24", "#4ade80", "#22c55e"]


def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8e0f0", family="Inter"),
        legend=dict(bgcolor="rgba(10,6,25,0.8)", bordercolor="rgba(139,92,246,0.3)", borderwidth=1),
        margin=dict(t=40, b=20, l=10, r=10),
    )
    fig.update_xaxes(gridcolor="rgba(139,92,246,0.07)", linecolor="rgba(139,92,246,0.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(139,92,246,0.07)", linecolor="rgba(139,92,246,0.18)", zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────
# SESSION STATE KEYS
# ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "df": None,              # The loaded DataFrame
        "filename": None,        # Original file name
        "agent_analysis": None,  # Full agent analysis result dict
        "dash_config": None,     # KPI + chart config from agent
        "chat_history": [],      # Conversation history
        "analysis_done": False,  # Whether auto-analysis ran
        "tmp_csv_path": None,    # Path to temp CSV file for agent tools
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────
# DATA UTILITIES
# ─────────────────────────────────────────────────────────────
def infer_column_roles(df: pd.DataFrame) -> dict:
    """
    Auto-detect column roles from dtypes:
        numeric   : numeric measures (SUM / AVG)
        datetime  : time axis
        category  : grouping dimensions (low cardinality text)
        text      : free text (high cardinality, skip)
        id        : identifier columns
    """
    roles = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        nuniq = df[col].nunique()
        n = len(df)

        if "datetime" in dtype or "date" in dtype:
            roles[col] = "datetime"
        elif "int" in dtype or "float" in dtype:
            # heuristic: very low nunique int → probably a category/id
            if nuniq <= 2:
                roles[col] = "category"
            elif col.lower().endswith(("_id", "id", "num", "number", "code")) and nuniq == n:
                roles[col] = "id"
            else:
                roles[col] = "numeric"
        elif "object" in dtype or "string" in dtype or "category" in dtype:
            if nuniq <= 30:
                roles[col] = "category"
            else:
                roles[col] = "text"
        elif "bool" in dtype:
            roles[col] = "category"
        else:
            roles[col] = "other"
    return roles


def smart_fmt(val, col_name: str = "", series: pd.Series = None) -> str:
    """Format a numeric value intelligently based on context."""
    if pd.isna(val):
        return "—"
    if isinstance(val, float):
        # percentage heuristic
        if series is not None and series.max() <= 100 and series.min() >= -10:
            return f"{val:.1f}%"
        return f"{val:,.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def save_df_to_tmp(df: pd.DataFrame) -> str:
    """Save DataFrame to a temp CSV so agent tools can read it by path."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w",
        encoding="utf-8", newline=""
    )
    df.to_csv(tmp, index=False)
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────
# AGENT ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────
ANALYSIS_SYSTEM_PROMPT = """
<Persona>
You are an expert Data Analyst AI capable of writing accurate Python code.
</Persona>

<Goal>
Explore the provided dataset and produce a comprehensive BI report.
Your report will be rendered directly inside a Streamlit dashboard.
</Goal>

<Tools>
1. read_dataset_schema(file_path)  - Returns column names, dtypes, shape, sample rows.
2. run_python_analysis(code_string) - Executes Python; save final answer in 'result'.
</Tools>

<Process>
Work in a ReAct loop:
  Thought:      What will you do and why?
  Action:       Tool name
  Action Input: Argument
  Observation:  [Tool result — do NOT generate this yourself]
Repeat until ready for the final answer.
</Process>

<Constraints>
- Do NOT load or print the entire file — use read_dataset_schema first.
- Self-correct on Python errors: use the error to fix the code and retry.
- Max 10 iterations.
</Constraints>

<Output Format>
When ready, output a JSON object (and NOTHING else outside the JSON) with this exact schema:

{
  "dataset_name": "<inferred descriptive name>",
  "summary": "<2-3 sentence plain-English summary of what the dataset contains>",
  "shape": {"rows": <int>, "columns": <int>},
  "column_types": {
    "<col_name>": "<numeric|datetime|category|text|id>",
    ...
  },
  "kpis": [
    {
      "name": "<short human label>",
      "value": <number or string>,
      "calculation": "<e.g. SUM(col) or COUNT(col)>",
      "business_logic": "<why this matters>",
      "column": "<source column name>"
    }
  ],
  "charts": [
    {
      "title": "<chart title>",
      "type": "<bar|line|pie|scatter|histogram|heatmap>",
      "x": "<column name>",
      "y": "<column name or aggregation expression>",
      "color": "<optional column name for color grouping or null>",
      "description": "<1-sentence chart insight>"
    }
  ],
  "key_findings": [
    "<finding 1>",
    "<finding 2>",
    "<finding 3>"
  ],
  "recommendations": [
    "<recommendation 1>",
    "<recommendation 2>"
  ]
}

Produce exactly 4 KPIs and 3-5 charts. Choose charts that are most meaningful for this specific dataset.
</Output Format>
"""


@st.cache_data(show_spinner=False)
def run_full_analysis(file_path: str, file_hash: int) -> dict:
    """
    Run the full agent analysis pipeline on a dataset file.
    Returns the parsed JSON result dict.
    Cached by file_hash so it only runs once per uploaded file.
    """
    captured = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured

    try:
        raw_answer = run_agent_loop(
            user_input=(
                f"Analyse the dataset at '{file_path}'. "
                "Explore its structure, compute key statistics, identify trends, "
                "and return the result as a JSON object following the exact schema in your instructions."
            ),
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            max_iterations=10,
        )
    finally:
        sys.stdout = original_stdout

    trace = captured.getvalue()

    # Extract JSON from the agent's response
    json_match = re.search(r'\{.*\}', raw_answer, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            result["__trace__"] = trace
            return result
        except json.JSONDecodeError:
            pass

    # Fallback: return minimal structure with the raw text
    return {
        "dataset_name": "Loaded Dataset",
        "summary": raw_answer[:500] if raw_answer else "Analysis complete.",
        "shape": {"rows": 0, "columns": 0},
        "column_types": {},
        "kpis": [],
        "charts": [],
        "key_findings": [raw_answer],
        "recommendations": [],
        "__trace__": trace,
        "__raw__": raw_answer,
    }


def compute_kpi_value(kpi: dict, df: pd.DataFrame) -> str:
    """
    Compute a KPI value from the agent spec against the live DataFrame.
    Falls back to the pre-computed agent value if the column isn't found.
    """
    col = kpi.get("column", "")
    calc = kpi.get("calculation", "")
    pre_val = kpi.get("value", "—")

    try:
        if col not in df.columns:
            return str(pre_val) if pre_val is not None else "—"

        m = re.match(r"(\w+)\(", calc.strip())
        agg = m.group(1).upper() if m else "SUM"

        if agg == "SUM":
            v = df[col].sum()
            return f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"
        elif agg == "AVG" or agg == "MEAN":
            v = df[col].mean()
            return f"{v:,.2f}"
        elif agg in ("COUNT", "COUNT_DISTINCT"):
            v = df[col].nunique() if "DISTINCT" in agg else len(df[col].dropna())
            return f"{v:,}"
        elif agg == "MIN":
            return str(df[col].min())
        elif agg == "MAX":
            return str(df[col].max())
        elif agg == "COUNTIF":
            # e.g. COUNTIF(status='Active')
            inner = re.search(r"\((.+)\)", calc)
            if inner:
                parts = inner.group(1).split("=", 1)
                if len(parts) == 2:
                    c2 = parts[0].strip()
                    v2 = parts[1].strip().strip("'\"")
                    if c2 in df.columns:
                        return f"{(df[c2] == v2).sum():,}"

        return str(pre_val) if pre_val is not None else "—"
    except Exception:
        return str(pre_val) if pre_val is not None else "—"


def render_chart(chart_spec: dict, df: pd.DataFrame):
    """Render a Plotly chart from an agent-generated spec."""
    ctype = chart_spec.get("type", "bar").lower()
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    color_col = chart_spec.get("color")
    title = chart_spec.get("title", "Chart")

    # Validate columns exist
    if x_col and x_col not in df.columns:
        st.caption(f"Chart skipped: column '{x_col}' not found in dataset.")
        return
    if y_col and y_col not in df.columns and not re.match(r"\w+\(", str(y_col)):
        st.caption(f"Chart skipped: column '{y_col}' not found in dataset.")
        return
    if color_col and color_col not in df.columns:
        color_col = None

    try:
        # Resolve y aggregation if needed (e.g. "SUM(revenue)")
        y_agg_match = re.match(r"(\w+)\((\w+)\)", str(y_col)) if y_col else None
        if y_agg_match:
            agg_fn, agg_col = y_agg_match.group(1).upper(), y_agg_match.group(2)
            if agg_col in df.columns and x_col in df.columns:
                if agg_fn == "SUM":
                    plot_df = df.groupby(x_col, observed=True)[agg_col].sum().reset_index()
                elif agg_fn in ("AVG", "MEAN"):
                    plot_df = df.groupby(x_col, observed=True)[agg_col].mean().reset_index()
                elif agg_fn == "COUNT":
                    plot_df = df.groupby(x_col, observed=True)[agg_col].count().reset_index()
                else:
                    plot_df = df.groupby(x_col, observed=True)[agg_col].sum().reset_index()
                y_col = agg_col
                color_col = None  # remove color after groupby
            else:
                plot_df = df
        else:
            plot_df = df

        fig = None

        if ctype in ("bar", "bar_chart"):
            if len(plot_df) > 30:
                plot_df = plot_df.sort_values(y_col, ascending=False).head(20)
            fig = px.bar(
                plot_df, x=x_col, y=y_col, title=title,
                color=color_col if color_col else y_col,
                color_continuous_scale=SCALE_PURPLE,
                color_discrete_sequence=PALETTE,
            )
            if not color_col:
                fig.update_coloraxes(showscale=False)

        elif ctype in ("line", "line_chart"):
            if pd.api.types.is_datetime64_any_dtype(df[x_col]) or "date" in x_col.lower():
                plot_df = plot_df.sort_values(x_col)
            fig = px.line(
                plot_df, x=x_col, y=y_col, title=title,
                color=color_col, color_discrete_sequence=PALETTE,
                markers=True,
            )
            fig.update_traces(line_width=2.5)

        elif ctype in ("pie", "pie_chart", "donut"):
            if x_col in plot_df.columns and y_col in plot_df.columns:
                fig = px.pie(
                    plot_df, names=x_col, values=y_col, title=title,
                    hole=0.4, color_discrete_sequence=PALETTE,
                )
                fig.update_traces(textinfo="label+percent", pull=[0.03] * len(plot_df))

        elif ctype in ("scatter", "scatter_chart"):
            y_actual = y_col if y_col in df.columns else None
            if y_actual:
                fig = px.scatter(
                    df, x=x_col, y=y_actual, title=title,
                    color=color_col, color_discrete_sequence=PALETTE,
                    opacity=0.75,
                )

        elif ctype in ("histogram",):
            fig = px.histogram(
                df, x=x_col, title=title,
                color=color_col, color_discrete_sequence=PALETTE,
                nbins=30,
            )

        elif ctype in ("heatmap",):
            num_df = df.select_dtypes(include="number")
            if len(num_df.columns) >= 2:
                corr = num_df.corr()
                fig = px.imshow(
                    corr, title=title,
                    color_continuous_scale=["#2e1065", "#7c3aed", "#e9d5ff"],
                    text_auto=".2f",
                )

        if fig:
            apply_theme(fig)
            st.plotly_chart(fig, width='stretch')
            if chart_spec.get("description"):
                st.caption(f"💡 {chart_spec['description']}")
        else:
            st.caption(f"Could not render chart: {title}")

    except Exception as e:
        st.caption(f"Chart render error ({title}): {e}")


def render_auto_charts(df: pd.DataFrame, roles: dict):
    """
    Render automatic exploratory charts based purely on column roles.
    Used as supplementary charts alongside agent-specified ones.
    """
    num_cols = [c for c, r in roles.items() if r == "numeric"]
    cat_cols = [c for c, r in roles.items() if r == "category"]
    dt_cols  = [c for c, r in roles.items() if r == "datetime"]

    rendered = 0

    # Distribution of first numeric by first category
    if num_cols and cat_cols and rendered < 2:
        nc, cc = num_cols[0], cat_cols[0]
        agg = df.groupby(cc, observed=True)[nc].sum().reset_index().sort_values(nc, ascending=True)
        if 1 < len(agg) <= 30:
            fig = px.bar(
                agg, x=nc, y=cc, orientation="h",
                title=f"{nc} by {cc}",
                color=nc, color_continuous_scale=SCALE_PURPLE,
            )
            fig.update_coloraxes(showscale=False)
            apply_theme(fig)
            st.plotly_chart(fig, width='stretch')
            rendered += 1

    # Time series of first numeric over first date
    if dt_cols and num_cols and rendered < 2:
        dc, nc = dt_cols[0], num_cols[0]
        ts = df.groupby(dc)[nc].sum().reset_index().sort_values(dc)
        if len(ts) > 1:
            fig = px.area(
                ts, x=dc, y=nc,
                title=f"{nc} over time",
                color_discrete_sequence=["#a78bfa"],
            )
            fig.update_traces(fill="tozeroy", fillcolor="rgba(167,139,250,0.12)")
            apply_theme(fig)
            st.plotly_chart(fig, width='stretch')
            rendered += 1

    # Histogram of first numeric
    if num_cols:
        fig = px.histogram(
            df, x=num_cols[0],
            title=f"Distribution: {num_cols[0]}",
            nbins=25, color_discrete_sequence=["#7c3aed"],
        )
        apply_theme(fig)
        st.plotly_chart(fig, width='stretch')


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:12px 0 20px;">'
        '<div style="font-size:2.8rem;">🧠</div>'
        '<div style="font-family:\'Playfair Display\',serif;font-size:1.1rem;'
        'color:#e9d5ff;font-weight:700;">AI Data Analyst</div>'
        '<div style="font-size:0.72rem;color:#b8a9d4;">Autonomous BI · AI Course</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # --- Upload ---
    st.markdown("### 📂 Dataset")
    uploaded = st.file_uploader(
        "Upload any CSV file",
        help="The AI agent will automatically analyse your data and build the dashboard.",
        label_visibility="collapsed",
    )

    # Demo data button
    if st.button("🍰 Load demo (bakery orders)", width='stretch'):
        demo_path = os.path.join(os.path.dirname(__file__), "data", "orders.csv")
        if os.path.exists(demo_path):
            st.session_state["df"] = pd.read_csv(demo_path)
            st.session_state["filename"] = "orders.csv"
            st.session_state["analysis_done"] = False
            st.session_state["agent_analysis"] = None
            st.session_state["dash_config"] = None
            st.session_state["tmp_csv_path"] = demo_path
            st.rerun()

    st.markdown("---")

    # --- Filters (only when data loaded) ---
    if st.session_state["df"] is not None:
        df_all = st.session_state["df"]
        roles = infer_column_roles(df_all)

        st.markdown("### 🔍 Filters")

        # Date filter
        dt_cols = [c for c, r in roles.items() if r == "datetime"]
        date_filter = None
        if dt_cols:
            dc = dt_cols[0]
            if dc in df_all.columns:
                min_d = df_all[dc].min()
                max_d = df_all[dc].max()
                if pd.notna(min_d) and pd.notna(max_d):
                    min_d = pd.Timestamp(min_d).date()
                    max_d = pd.Timestamp(max_d).date()
                    date_filter = st.date_input(
                        f"Date range ({dc})", value=(min_d, max_d),
                        min_value=min_d, max_value=max_d,
                    )
                    date_filter = (dc, date_filter)

        # Category filters (up to 2)
        cat_cols_all = [c for c, r in roles.items() if r == "category"]
        cat_filters = {}
        for cc in cat_cols_all[:2]:
            opts = sorted(df_all[cc].dropna().unique().tolist(), key=str)
            if 1 < len(opts) <= 50:
                sel = st.multiselect(cc.replace("_", " ").title(), opts, default=opts)
                cat_filters[cc] = sel

        st.markdown("---")

    # Agent settings
    st.markdown("### ⚙️ Agent Settings")
    api_key_input = st.text_input(
        "Gemini API Key", 
        type="password", 
        help="Required for dynamic analysis of custom datasets. If left empty, the app uses a simulated Bakery dataset."
    )
    if api_key_input:
        import os
        os.environ["GOOGLE_API_KEY"] = api_key_input
        
    max_iter = st.slider("Max iterations", 3, 10, 8)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.7rem;color:#5c4f7a;text-align:center;">'
        "ReAct Agent · Data Analyst AI Course</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# HANDLE FILE UPLOAD
# ─────────────────────────────────────────────────────────────
def load_uploaded_file(file_obj):
    name = file_obj.name.lower()
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_obj)
    
    encodings = ['utf-8', 'cp1255', 'iso-8859-8', 'windows-1252', 'latin1']
    for enc in encodings:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except UnicodeDecodeError:
            continue
    # Fallback to string replacement
    file_obj.seek(0)
    return pd.read_csv(file_obj, encoding='utf-8', errors='replace')

if uploaded is not None:
    file_key = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("_upload_key") != file_key:
        # New file uploaded — reset state
        df_loaded = load_uploaded_file(uploaded)
        
        # Try to parse datetime columns
        for col in df_loaded.columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    df_loaded[col] = pd.to_datetime(df_loaded[col], errors="coerce")
                except Exception:
                    pass

        st.session_state["df"] = df_loaded
        st.session_state["filename"] = uploaded.name
        st.session_state["analysis_done"] = False
        st.session_state["agent_analysis"] = None
        st.session_state["dash_config"] = None
        st.session_state["_upload_key"] = file_key

        # Save to tmp file so agent tools can read it by path
        tmp_path = save_df_to_tmp(df_loaded)
        st.session_state["tmp_csv_path"] = tmp_path
        st.rerun()


# ─────────────────────────────────────────────────────────────
# LANDING PAGE — No data loaded
# ─────────────────────────────────────────────────────────────
if st.session_state["df"] is None:
    st.markdown("""
    <div style="text-align:center;padding:30px 0 10px;">
        <div style="font-size:4rem;">🧠</div>
        <h1 style="font-family:'Playfair Display',serif;font-size:2.4rem;
                   background:linear-gradient(90deg,#e9d5ff,#a78bfa,#f0abfc);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin:0;">AI Data Analyst</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#d1c4e9;font-size:1rem;margin:8px 0 30px;text-align:center;">'
        'Upload any CSV dataset. The AI agent will analyse it and build your BI dashboard.'
        '</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        drop = st.file_uploader(
            "Drop your CSV here or click to browse",
            label_visibility="collapsed",
        )
        st.markdown(
            "<p style='color:#9d8ec4;font-size:0.82rem;margin-top:12px;'>"
            "Supports any CSV dataset · Sales, HR, Finance, Operations, Healthcare…"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🍰 Try with demo bakery dataset", width='stretch', type="secondary"):
            demo_path = os.path.join(os.path.dirname(__file__), "data", "orders.csv")
            if os.path.exists(demo_path):
                df_demo = pd.read_csv(demo_path)
                df_demo["order_date"] = pd.to_datetime(df_demo["order_date"], errors="coerce")
                st.session_state["df"] = df_demo
                st.session_state["filename"] = "orders.csv"
                st.session_state["analysis_done"] = False
                st.session_state["agent_analysis"] = None
                st.session_state["dash_config"] = None
                st.session_state["tmp_csv_path"] = demo_path
                st.rerun()

        if drop:
            df_loaded = load_uploaded_file(drop)
                
            for col in df_loaded.columns:
                if "date" in col.lower() or "time" in col.lower():
                    try:
                        df_loaded[col] = pd.to_datetime(df_loaded[col], errors="coerce")
                    except Exception:
                        pass
            st.session_state["df"] = df_loaded
            st.session_state["filename"] = drop.name
            st.session_state["analysis_done"] = False
            st.session_state["agent_analysis"] = None
            st.session_state["dash_config"] = None
            tmp_path = save_df_to_tmp(df_loaded)
            st.session_state["tmp_csv_path"] = tmp_path
            st.rerun()

    # Feature pills
    st.markdown("<br><br>", unsafe_allow_html=True)
    features = [
        "🔍 Auto schema detection", "📊 Smart KPI generation", "📈 Adaptive charts",
        "🤖 ReAct agent loop", "💡 Business insights", "💬 Chat with your data",
    ]
    pills_html = "".join(f'<span class="stat-pill">{f}</span>' for f in features)
    st.markdown(f'<div style="text-align:center;">{pills_html}</div>', unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# DATA IS LOADED — Apply sidebar filters
# ─────────────────────────────────────────────────────────────
df_raw = st.session_state["df"]
roles = infer_column_roles(df_raw)

df = df_raw.copy()
try:
    # Apply date filter
    if "date_filter" in dir() and date_filter and len(date_filter) == 2:
        dc, dr = date_filter
        if len(dr) == 2 and dc in df.columns:
            df = df[
                (df[dc].dt.date >= dr[0]) & (df[dc].dt.date <= dr[1])
            ]
    # Apply category filters
    if "cat_filters" in dir():
        for cc, sel in cat_filters.items():
            if sel and cc in df.columns:
                df = df[df[cc].isin(sel)]
except Exception:
    pass


# ─────────────────────────────────────────────────────────────
# RUN AGENT ANALYSIS (auto, once per file)
# ─────────────────────────────────────────────────────────────
if not st.session_state["analysis_done"]:
    tmp_path = st.session_state.get("tmp_csv_path")
    if not tmp_path or not os.path.exists(tmp_path):
        tmp_path = save_df_to_tmp(df_raw)
        st.session_state["tmp_csv_path"] = tmp_path

    analysis_placeholder = st.empty()
    with analysis_placeholder.container():
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;">
            <div style="font-size:2.5rem;margin-bottom:12px;">🧠</div>
            <h3 style="color:#e9d5ff;font-family:'Playfair Display',serif;">
                AI Agent is analysing your dataset…
            </h3>
            <p style="color:#7c6fa0;">
                The ReAct agent is reading schema, computing statistics, and building your dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)

        prog_bar = st.progress(0, text="Starting agent…")

        # Step 1: Schema
        prog_bar.progress(15, text="Step 1/4 — Reading schema…")
        schema_txt = read_dataset_schema(tmp_path)

        # Step 2: Metadata profiling
        prog_bar.progress(35, text="Step 2/4 — Profiling dataset…")
        meta = profile_dataset(df_raw)

        # Step 3: Full agent analysis
        prog_bar.progress(50, text="Step 3/4 — Running AI analysis (this may take a moment)…")
        api_key = os.environ.get('GOOGLE_API_KEY', '')
        file_hash = hash(f"{st.session_state['filename']}_{len(df_raw)}_{list(df_raw.columns)}_{api_key}_v3")
        analysis = run_full_analysis(tmp_path, file_hash)

        # Step 4: Store results
        prog_bar.progress(90, text="Step 4/4 — Building dashboard…")
        st.session_state["agent_analysis"] = analysis
        st.session_state["dash_config"] = {
            "kpis": analysis.get("kpis", []),
            "charts": analysis.get("charts", []),
        }
        st.session_state["analysis_done"] = True
        prog_bar.progress(100, text="Done!")

    analysis_placeholder.empty()
    st.rerun()

# ─────────────────────────────────────────────────────────────
# DASHBOARD — Data is loaded & analysed
# ─────────────────────────────────────────────────────────────
analysis = st.session_state["agent_analysis"] or {}
dash_config = st.session_state["dash_config"] or {"kpis": [], "charts": []}
filename = st.session_state.get("filename", "Dataset")

# ── Header ──────────────────────────────────────────────────
h_col1, h_col2, h_col3 = st.columns([1, 6, 3])
with h_col1:
    st.markdown('<div style="font-size:3rem;padding-top:6px;">🧠</div>', unsafe_allow_html=True)
with h_col2:
    dataset_name = analysis.get("dataset_name", filename)
    st.markdown(
        f'<h1 style="font-family:Playfair Display,serif;font-size:1.8rem;'
        f'background:linear-gradient(90deg,#e9d5ff,#a78bfa,#f0abfc);'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;">'
        f'{dataset_name}</h1>'
        f'<p style="color:#b8a9d4;font-size:0.8rem;margin:0 0 0 2px;">'
        f'{filename} · {len(df_raw):,} rows × {len(df_raw.columns)} columns · '
        f'AI-powered BI Dashboard</p>',
        unsafe_allow_html=True,
    )
with h_col3:
    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button("🔄 Re-analyse", help="Re-run the agent analysis"):
            st.session_state["analysis_done"] = False
            st.session_state["agent_analysis"] = None
            run_full_analysis.clear()
            st.rerun()
    with btn_cols[1]:
        csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export", csv_bytes, f"{filename}_export.csv", "text/csv")

st.markdown("---")

# ── Summary banner ──────────────────────────────────────────
summary_text = analysis.get("summary", "")
if summary_text:
    st.markdown(
        f'<div class="insight-block" style="border-left-color:#a78bfa;">'
        f'<strong style="color:#e9d5ff;">Dataset Summary</strong><br>{summary_text}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab_overview, tab_charts, tab_data, tab_chat, tab_debug = st.tabs([
    "📊 Overview", "📈 Charts", "📋 Data", "💬 Chat", "🔧 Debug"
])


# ═══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════
with tab_overview:

    # ── KPI Cards ────────────────────────────────────────
    kpis = dash_config.get("kpis", [])
    if kpis:
        st.markdown('<p class="section-title">📌 Key Performance Indicators</p>', unsafe_allow_html=True)
        kpi_cols = st.columns(len(kpis))
        for col, kpi in zip(kpi_cols, kpis):
            val = compute_kpi_value(kpi, df)
            with col:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-label">{kpi.get("name", "")}</div>'
                    f'<div class="kpi-value">{val}</div>'
                    f'<div class="kpi-sub">{kpi.get("business_logic", "")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Auto data profile stats ──────────────────────────
    st.markdown('<p class="section-title">📐 Dataset Profile</p>', unsafe_allow_html=True)

    num_cols_list = [c for c, r in roles.items() if r == "numeric"]
    cat_cols_list = [c for c, r in roles.items() if r == "category"]
    dt_cols_list  = [c for c, r in roles.items() if r == "datetime"]

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Rows", f"{len(df):,}")
    p2.metric("Columns", f"{len(df.columns):,}")
    p3.metric("Numeric", len(num_cols_list))
    p4.metric("Category", len(cat_cols_list))
    p5.metric("Datetime", len(dt_cols_list))
    missing_pct = df.isnull().sum().sum() / max(df.size, 1) * 100
    p6.metric("Missing %", f"{missing_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Findings ──────────────────────────────────────
    findings = analysis.get("key_findings", [])
    recommendations = analysis.get("recommendations", [])

    f_col, r_col = st.columns(2)
    with f_col:
        if findings:
            st.markdown('<p class="section-title">💡 Key Findings</p>', unsafe_allow_html=True)
            for i, finding in enumerate(findings, 1):
                st.markdown(
                    f'<div class="insight-block">📍 <strong>Finding {i}:</strong> {finding}</div>',
                    unsafe_allow_html=True,
                )

    with r_col:
        if recommendations:
            st.markdown('<p class="section-title">🎯 Recommendations</p>', unsafe_allow_html=True)
            for i, rec in enumerate(recommendations, 1):
                st.markdown(
                    f'<div class="insight-block" style="border-left-color:#34d399;">'
                    f'✅ <strong>{i}.</strong> {rec}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick stats for numeric columns ───────────────────
    if num_cols_list:
        st.markdown('<p class="section-title">📊 Numeric Summary</p>', unsafe_allow_html=True)
        summary_df = df[num_cols_list].describe().round(2).T
        summary_df.index.name = "Column"
        summary_df = summary_df.reset_index()
        st.dataframe(summary_df, width='stretch', hide_index=True, height=250)

    # ── Category distributions ─────────────────────────────
    if cat_cols_list:
        st.markdown('<p class="section-title">🏷️ Category Distributions</p>', unsafe_allow_html=True)
        cat_display_cols = st.columns(min(len(cat_cols_list), 3))
        for col_widget, cc in zip(cat_display_cols, cat_cols_list[:3]):
            with col_widget:
                vc = df[cc].value_counts().reset_index()
                vc.columns = [cc, "count"]
                fig = px.pie(
                    vc, names=cc, values="count", title=cc.replace("_", " ").title(),
                    hole=0.4, color_discrete_sequence=PALETTE,
                )
                fig.update_traces(textinfo="label+percent")
                apply_theme(fig)
                st.plotly_chart(fig, width='stretch')


# ═══════════════════════════════════════════════════════════
# TAB 2 — CHARTS (agent-generated + automatic)
# ═══════════════════════════════════════════════════════════
with tab_charts:

    agent_charts = dash_config.get("charts", [])

    if agent_charts:
        st.markdown('<p class="section-title">AI-Generated Charts</p>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#b8a9d4;font-size:0.83rem;margin-top:-6px;">'
            'Charts selected and configured automatically by the AI agent based on your data.'
            '</p>',
            unsafe_allow_html=True,
        )

        # Lay out charts in pairs
        for i in range(0, len(agent_charts), 2):
            row_charts = agent_charts[i:i+2]
            cols = st.columns(len(row_charts))
            for col_widget, chart_spec in zip(cols, row_charts):
                with col_widget:
                    render_chart(chart_spec, df)
    else:
        st.info("No agent charts found. Showing automatic charts.")

    # ── Automatic exploratory charts ──────────────────────
    st.markdown('<p class="section-title">🔭 Exploratory Analysis</p>', unsafe_allow_html=True)

    num_cols_list = [c for c, r in roles.items() if r == "numeric"]
    cat_cols_list = [c for c, r in roles.items() if r == "category"]
    dt_cols_list  = [c for c, r in roles.items() if r == "datetime"]

    # Correlation heatmap (if 3+ numeric cols)
    if len(num_cols_list) >= 3:
        st.markdown("**Correlation Matrix**")
        corr = df[num_cols_list].corr().round(3)
        fig_corr = px.imshow(
            corr, title="Numeric Column Correlations",
            color_continuous_scale=["#2e1065", "#7c3aed", "#a78bfa", "#f5f3ff"],
            text_auto=".2f", aspect="auto",
        )
        apply_theme(fig_corr)
        st.plotly_chart(fig_corr, width='stretch')

    # Time series per numeric column (if dates exist)
    if dt_cols_list and num_cols_list:
        dc = dt_cols_list[0]
        chosen_num = st.selectbox(
            "Select metric for time trend",
            num_cols_list,
            key="trend_select",
        )
        ts_df = df.groupby(dc)[chosen_num].sum().reset_index().sort_values(dc)
        if len(ts_df) > 1:
            fig_ts = px.area(
                ts_df, x=dc, y=chosen_num,
                title=f"{chosen_num} over time ({dc})",
                color_discrete_sequence=["#a78bfa"],
            )
            fig_ts.update_traces(fill="tozeroy", fillcolor="rgba(167,139,250,0.12)")
            apply_theme(fig_ts)
            st.plotly_chart(fig_ts, width='stretch')

    # Bar chart: numeric by category
    if cat_cols_list and num_cols_list:
        ec1, ec2 = st.columns(2)
        with ec1:
            chosen_cat = st.selectbox("Group by", cat_cols_list, key="cat_select")
        with ec2:
            chosen_num2 = st.selectbox("Measure", num_cols_list, key="num_select")

        agg_fn = st.radio("Aggregation", ["Sum", "Mean", "Count"], horizontal=True, key="agg_radio")
        agg_map = {"Sum": "sum", "Mean": "mean", "Count": "count"}
        grp = df.groupby(chosen_cat, observed=True)[chosen_num2].agg(agg_map[agg_fn]).reset_index()
        grp.columns = [chosen_cat, chosen_num2]
        grp = grp.sort_values(chosen_num2, ascending=False).head(25)

        fig_bar = px.bar(
            grp, x=chosen_cat, y=chosen_num2,
            title=f"{agg_fn} of {chosen_num2} by {chosen_cat}",
            color=chosen_num2, color_continuous_scale=SCALE_PURPLE,
        )
        fig_bar.update_coloraxes(showscale=False)
        apply_theme(fig_bar)
        st.plotly_chart(fig_bar, width='stretch')

    # Scatter / histogram of any two columns
    if len(num_cols_list) >= 2:
        st.markdown("**Scatter / Histogram Explorer**")
        s1, s2, s3 = st.columns(3)
        with s1: sx = st.selectbox("X axis", num_cols_list, key="sx")
        with s2: sy = st.selectbox("Y axis", num_cols_list, index=min(1, len(num_cols_list)-1), key="sy")
        with s3: sc = st.selectbox("Color by", ["—"] + cat_cols_list, key="sc")

        sc_col = None if sc == "—" else sc
        fig_sc = px.scatter(
            df, x=sx, y=sy, color=sc_col,
            title=f"{sx} vs {sy}",
            color_discrete_sequence=PALETTE,
            opacity=0.7,
        )
        apply_theme(fig_sc)
        st.plotly_chart(fig_sc, width='stretch')


# ═══════════════════════════════════════════════════════════
# TAB 3 — DATA TABLE
# ═══════════════════════════════════════════════════════════
with tab_data:

    st.markdown('<p class="section-title">📋 Dataset</p>', unsafe_allow_html=True)

    # Quick stats per column type
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Showing rows", f"{len(df):,} / {len(df_raw):,}")
    sc2.metric("Missing cells", f"{df.isnull().sum().sum():,}")
    sc3.metric("Duplicate rows", f"{df.duplicated().sum():,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Search
    search = st.text_input(
        "🔍 Search across all columns",
        placeholder="Type any value to filter…",
        key="table_search",
    )

    display_df = df.copy()

    # Format datetime for display
    for col in display_df.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df[col]):
            display_df[col] = display_df[col].dt.strftime("%Y-%m-%d")

    if search.strip():
        mask = pd.Series([False] * len(display_df), index=display_df.index)
        for cn in display_df.columns:
            mask |= display_df[cn].astype(str).str.contains(search.strip(), case=False, na=False)
        display_df = display_df[mask]

    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

    st.dataframe(display_df, width='stretch', height=440, hide_index=True)

    dl_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered data as CSV",
        dl_data,
        file_name=f"{filename}_filtered.csv",
        mime="text/csv",
    )

    # Column info table
    with st.expander("📐 Column details"):
        col_info = []
        for c in df_raw.columns:
            col_info.append({
                "Column": c,
                "Type": str(df_raw[c].dtype),
                "Role": roles.get(c, "?"),
                "Unique": df_raw[c].nunique(),
                "Missing": f"{df_raw[c].isnull().sum()} ({df_raw[c].isnull().mean()*100:.1f}%)",
                "Sample": str(df_raw[c].dropna().iloc[0]) if len(df_raw[c].dropna()) > 0 else "—",
            })
        st.dataframe(pd.DataFrame(col_info), width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 4 — AGENT CHAT
# ═══════════════════════════════════════════════════════════
with tab_chat:

    st.markdown('<p class="section-title">💬 Chat with your Data</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#b8a9d4;font-size:0.83rem;">'
        'The agent uses a ReAct loop with <code>read_dataset_schema</code> and '
        '<code>run_python_analysis</code> tools to answer questions about your data.'
        '</p>',
        unsafe_allow_html=True,
    )

    tmp_path = st.session_state.get("tmp_csv_path", "data/orders.csv")

    # Example prompts derived from dataset structure
    num_cols_list = [c for c, r in roles.items() if r == "numeric"]
    cat_cols_list = [c for c, r in roles.items() if r == "category"]

    example_prompts = [
        f"What are the top 5 rows by {num_cols_list[0]}?" if num_cols_list else "Summarize the dataset.",
        f"How many unique values does {cat_cols_list[0]} have?" if cat_cols_list else "What columns have missing data?",
        f"What is the average {num_cols_list[0]} grouped by {cat_cols_list[0]}?" if num_cols_list and cat_cols_list else "What trends do you see?",
    ]

    st.markdown("**Try these questions:**")
    ex_cols = st.columns(3)
    for ec, ep in zip(ex_cols, example_prompts):
        if ec.button(ep[:45] + ("..." if len(ep) > 45 else ""), key=f"ep_{ep[:15]}"):
            # Set query AND trigger auto-submit in the same rerun
            st.session_state["pending_chat"] = ep
            st.session_state["auto_submit_chat"] = True
            st.rerun()

    # Check if an example was clicked and should be auto-submitted
    auto_submit = st.session_state.pop("auto_submit_chat", False)
    pending_val = st.session_state.pop("pending_chat", "")

    user_query = st.text_area(
        "Your question:",
        value=pending_val,
        placeholder=f"e.g. 'What is the distribution of {num_cols_list[0] if num_cols_list else 'values'}?'",
        height=80,
        key="chat_input",
    )

    run_chat = st.button("Ask Agent", type="primary", key="run_chat")

    # Trigger agent if button pressed OR auto_submit from example click
    should_run = (run_chat or auto_submit) and (user_query.strip() or pending_val.strip())
    effective_query = user_query.strip() or pending_val.strip()

    if should_run:
        chat_system = f"""You are an expert Data Analyst AI.
The user has loaded a CSV dataset at: {tmp_path}
Dataset info: {analysis.get('summary', '')}
Column types: {json.dumps(analysis.get('column_types', {}), ensure_ascii=False)}

Use read_dataset_schema and run_python_analysis tools to answer questions accurately.
Do NOT guess — always use the tools to verify data.
Save your final answer in a Python variable named 'result' when using run_python_analysis.
"""
        cap = io.StringIO()
        old_out = sys.stdout; sys.stdout = cap
        try:
            answer = run_agent_loop(
                user_input=effective_query,
                system_prompt=chat_system,
                max_iterations=max_iter,
            )
        except Exception as e:
            answer = f"Agent error: {e}"
        finally:
            sys.stdout = old_out
            trace_txt = cap.getvalue()

        st.session_state["chat_history"].append({
            "role": "user", "content": effective_query,
        })
        st.session_state["chat_history"].append({
            "role": "agent", "content": answer, "trace": trace_txt,
        })
        st.rerun()

    # Display conversation
    for msg in reversed(st.session_state["chat_history"]):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg">👤 <strong>You:</strong> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="insight-block">🤖 <strong>Agent:</strong><br>'
                f'{msg["content"].replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
            if msg.get("trace"):
                with st.expander("🔍 View agent trace"):
                    st.code(msg["trace"], language="text")

    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear chat"):
            st.session_state["chat_history"] = []
            st.rerun()


# ═══════════════════════════════════════════════════════════
# TAB 5 — DEBUG / RAW OUTPUT
# ═══════════════════════════════════════════════════════════
with tab_debug:

    st.markdown('<p class="section-title">🔧 Debug & Raw Analysis</p>', unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Agent Analysis JSON**")
        display_analysis = {k: v for k, v in analysis.items() if k not in ("__trace__", "__raw__")}
        st.json(display_analysis)

    with d2:
        st.markdown("**Column Roles (auto-detected)**")
        roles_df = pd.DataFrame(
            [{"Column": c, "Role": r} for c, r in roles.items()]
        )
        st.dataframe(roles_df, width='stretch', hide_index=True)

    if analysis.get("__trace__"):
        st.markdown("**Agent Reasoning Trace**")
        with st.expander("View full trace"):
            st.code(analysis["__trace__"], language="text")

    if analysis.get("__raw__"):
        st.markdown("**Raw Agent Output**")
        with st.expander("View raw output"):
            st.text(analysis["__raw__"])

    st.markdown("**Schema (from agent tool)**")
    tmp_path = st.session_state.get("tmp_csv_path", "")
    if tmp_path and os.path.exists(tmp_path):
        st.code(read_dataset_schema(tmp_path), language="text")

    # Custom code runner
    st.markdown('<p class="section-title">🐍 Custom Analysis Code</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#b8a9d4;font-size:0.82rem;">'
        'Run custom Python against your dataset. Save result to <code>result</code>.</p>',
        unsafe_allow_html=True,
    )
    default_code = f"""import pandas as pd
df = pd.read_csv(r'{tmp_path}')
result = df.describe().round(2)
"""
    code_input = st.text_area("Python code:", value=default_code, height=200, key="code_runner")
    if st.button("▶️ Run", key="run_code"):
        output = run_python_analysis(code_input)
        st.code(output, language="text")
