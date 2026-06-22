"""
Step 4 (Part 2): Dashboard Config Generator — The Semantic Prompt
==================================================================
Sends dataset metadata to the LLM with a structured system prompt
that instructs it to identify KPIs and chart specs automatically.

The output is a JSON config dict that the Streamlit dashboard
(Step 5) reads to render the correct widgets and charts.

This is the "semantic" step that bridges raw data → BI design.
"""

import json
import os
import re
from datetime import datetime

import pandas as pd

from data_loader import load_and_prepare_data


# ---------------------------------------------------------------------------
# Step 1: Profile the dataset → extract metadata
# ---------------------------------------------------------------------------

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Analyse the DataFrame and build a metadata dict that describes
    each column's type, cardinality, and sample values.

    The metadata is intentionally compact (no raw data rows) so it
    fits comfortably in the LLM's context window.
    """
    metadata: dict = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": [],
    }

    for col_name in df.columns:
        dtype_str = str(df[col_name].dtype)

        # Convert Timestamps to ISO strings for JSON compatibility
        sample_values = [
            str(val) if isinstance(val, (pd.Timestamp, datetime)) else val
            for val in df[col_name].dropna().head(2).tolist()
        ]

        col_info = {
            "name": col_name,
            "type": dtype_str,
            "unique_values_count": int(df[col_name].nunique()),
            "missing_percentage": round(
                (df[col_name].isnull().sum() / len(df)) * 100, 2
            ),
            "sample_values": sample_values,
        }
        metadata["columns"].append(col_info)

    return metadata


# ---------------------------------------------------------------------------
# Step 2: Send metadata to LLM → receive JSON dashboard config
# ---------------------------------------------------------------------------

def generate_dashboard_config(metadata: dict) -> dict:
    """
    Build the semantic (meta-prompt) that instructs the LLM to:
      - Identify 4 KPI metrics (measures / dimensions / timestamps)
      - Propose 2 chart specs (line, bar, or pie)

    The LLM response is validated with a regex JSON extractor.
    Falls back to a deterministic config if parsing fails.

    Returns
    -------
    dict with keys:
        "kpis"   : list of {name, calculation, business_logic}
        "charts" : list of {title, type, x_axis, y_axis}
    """
    # --- System Prompt: instructs the LLM how to classify columns ---
    system_prompt = """
    You are a BI System Analyst specialising in dashboard design.
    You receive a dataset metadata object (column names, types, sample values).

    Classify each column as one of:
    1. Measure  : numeric (Float/Int) columns like Revenue, Price, Cost
                  → aggregated with SUM or AVG.
    2. Dimension: categorical (String/Text) columns like Status, Type
                  → used for COUNT or Group By.
    3. Time axis: date/datetime columns
                  → used as the X-axis in time-series charts.

    Return *only* valid JSON with no extra text or markdown fences.
    Format:
    {
      "kpis": [
        {"name": "<human label>", "calculation": "<SUM|AVG|COUNT>(column)", "business_logic": "<why this matters>"}
      ],
      "charts": [
        {"title": "<chart title>", "type": "bar_chart|line_chart|pie_chart", "x_axis": "<column>", "y_axis": "<calculation>"}
      ]
    }
    Produce exactly 4 KPIs and 2 charts.
    """

    user_prompt = (
        f"Here is the dataset metadata:\n"
        f"{json.dumps(metadata, indent=2, ensure_ascii=False)}"
    )

    # --- Try real LLM if API key is available ---
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        response_text = _call_gemini_for_config(system_prompt, user_prompt, api_key)
    else:
        # Offline fallback: return a deterministic config based on metadata
        response_text = _mock_llm_response(metadata)

    # --- Parse and validate the JSON response ---
    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
            config = json.loads(clean_json)
            # Validate required keys
            if "kpis" in config and "charts" in config:
                return config
        raise ValueError("No valid JSON found in LLM response.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing error: {e}\nFalling back to deterministic config.")
        return _deterministic_config(metadata)


def _call_gemini_for_config(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """Call the Gemini API and return the raw text response."""
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        combined = f"{system_prompt}\n\n{user_prompt}"
        response = model.generate_content(combined)
        return response.text
    except Exception as e:
        return f"Error: {e}"


def _mock_llm_response(metadata: dict) -> str:
    """
    Deterministic JSON response derived from the metadata columns.
    Used when no API key is available.
    """
    # Auto-detect column categories
    numeric_cols = [
        c["name"] for c in metadata["columns"] if "float" in c["type"] or "int" in c["type"]
    ]
    categorical_cols = [
        c["name"] for c in metadata["columns"] if "object" in c["type"]
    ]
    datetime_cols = [
        c["name"] for c in metadata["columns"] if "datetime" in c["type"]
    ]

    price_col = next((c for c in numeric_cols if "price" in c.lower()), numeric_cols[0] if numeric_cols else "final_price")
    cost_col = next((c for c in numeric_cols if "cost" in c.lower()), None)
    status_col = next((c for c in categorical_cols if "status" in c.lower()), categorical_cols[0] if categorical_cols else "status")
    type_col = next((c for c in categorical_cols if "type" in c.lower()), categorical_cols[-1] if categorical_cols else "product_type")
    date_col = datetime_cols[0] if datetime_cols else "order_date"
    id_col = next((c["name"] for c in metadata["columns"] if "id" in c["name"].lower()), "order_id")

    config = {
        "kpis": [
            {"name": "Total Revenue", "calculation": f"SUM({price_col})", "business_logic": "Measures overall sales income for the period"},
            {"name": "Total Ingredient Cost", "calculation": f"SUM({cost_col})" if cost_col else f"SUM({price_col})", "business_logic": "Tracks raw material expenses"},
            {"name": "Total Orders", "calculation": f"COUNT({id_col})", "business_logic": "Indicates overall order volume"},
            {"name": "Active Statuses", "calculation": f"COUNT_DISTINCT({status_col})", "business_logic": "Tracks how many order stages are currently active"},
        ],
        "charts": [
            {"title": "Revenue Over Time", "type": "line_chart", "x_axis": date_col, "y_axis": f"SUM({price_col})"},
            {"title": f"Orders by {status_col.replace('_', ' ').title()}", "type": "pie_chart", "x_axis": status_col, "y_axis": f"COUNT({id_col})"},
        ],
    }
    return json.dumps(config)


def _deterministic_config(metadata: dict) -> dict:
    """Last-resort fallback — returns a hardcoded config for the bakery dataset."""
    return {
        "kpis": [
            {"name": "Total Revenue", "calculation": "SUM(final_price)", "business_logic": "Measures total sales income"},
            {"name": "Total Cost", "calculation": "SUM(ingredients_cost)", "business_logic": "Tracks ingredient expenditure"},
            {"name": "Total Orders", "calculation": "COUNT(order_id)", "business_logic": "Counts all orders"},
            {"name": "Pending Orders", "calculation": "COUNTIF(status='Pending')", "business_logic": "Orders awaiting action"},
        ],
        "charts": [
            {"title": "Revenue Over Time", "type": "line_chart", "x_axis": "order_date", "y_axis": "SUM(final_price)"},
            {"title": "Orders by Status", "type": "pie_chart", "x_axis": "status", "y_axis": "COUNT(order_id)"},
        ],
    }


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_bi_pipeline(df: pd.DataFrame | None = None) -> dict:
    """
    Full pipeline:
        1. Load data (or accept a DataFrame)
        2. Profile the dataset → metadata
        3. Send metadata to LLM → dashboard config JSON
        4. Return the config
    """
    if df is None:
        df = load_and_prepare_data()

    print("✓ [1/3] Data loaded — profiling dataset...")
    metadata = profile_dataset(df)

    print("✓ [2/3] Metadata extracted — generating dashboard config with LLM...")
    config = generate_dashboard_config(metadata)

    print("✓ [3/3] Dashboard config ready:")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    return config


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_bi_pipeline()
