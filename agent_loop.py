"""
Step 4: Agent Loop (ReAct Pattern)
====================================
Implements the core agent loop that:
  1. Takes user input
  2. Calls the LLM (simulated here with a rule-based parser)
  3. Parses Thought / Action / Action Input from the response
  4. Executes the matching tool
  5. Feeds the Observation back into the conversation
  6. Repeats until the agent outputs a Final Answer or hits max_iterations

Architecture note
-----------------
In a real deployment, step 2 would call an LLM API (OpenAI, Gemini, etc.).
Here we provide:
  a) A real LLM adapter (call_llm_api) that calls the Google Gemini API
     via the google-generativeai SDK if GOOGLE_API_KEY is set.
  b) A rule-based fallback simulator for offline/demo use.

The order-specific get_order_details tool is also defined here so the loop
can look up individual orders by ID.
"""

import json
import os
import re
from typing import Any

import pandas as pd

from agent_prompts import SYSTEM_PROMPT, ORDERS_SYSTEM_PROMPT
from agent_tools import read_dataset_schema, run_python_analysis
from data_loader import orders_df

# ---------------------------------------------------------------------------
# Order-specific tool: get_order_details
# ---------------------------------------------------------------------------

def get_order_details(order_id: Any) -> str:
    """
    Returns full details for a single order as a JSON string.
    Operates on the shared orders_df loaded in data_loader.py.
    """
    order = orders_df[orders_df["order_id"] == int(order_id)]
    if order.empty:
        return json.dumps({"error": f"Order {order_id} not found."})
    # Convert Timestamps to strings so JSON can serialize them
    record = order.copy()
    record["order_date"] = record["order_date"].dt.strftime("%Y-%m-%d")
    return record.to_json(orient="records", force_ascii=False)


# ---------------------------------------------------------------------------
# Tool registry — maps tool names to callables
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "orders.csv")

AVAILABLE_TOOLS: dict = {
    "read_dataset_schema": read_dataset_schema,
    "run_python_analysis": run_python_analysis,
    "get_order_details": get_order_details,
}

# ---------------------------------------------------------------------------
# LLM adapter — Google Generative AI (Gemini)
# ---------------------------------------------------------------------------
import time

_GEMINI_MODEL_CHAIN = ["gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-2.0-flash"]
_last_llm_error = ""
_used_simulator = False

def get_last_llm_status() -> dict:
    return {"used_simulator": _used_simulator, "error": _last_llm_error}

def call_llm_api(messages: list[dict], model_name: str = "") -> dict:
    global _last_llm_error, _used_simulator
    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

    if not api_key:
        _last_llm_error = "No API key configured. Enter your Gemini API key in the sidebar."
        _used_simulator = True
        return _simulate_response(messages)

    models_to_try = [model_name] if model_name else _GEMINI_MODEL_CHAIN

    last_error = ""
    for model in models_to_try:
        result, last_error = _call_universal_llm(messages, model, api_key)
        if result is not None:
            return result
        if _is_key_error(last_error):
            break

    _last_llm_error = last_error
    _used_simulator = True
    print(f"[WARN] All LLM models failed. Last error: {last_error} -- using simulator.")
    return _simulate_response(messages)

def _is_key_error(error_str: str) -> bool:
    s = error_str.lower()
    return any(k in s for k in ("401", "403", "invalid api key", "api_key_invalid", "permission denied", "unauthorized", "authentication"))

def _categorize_llm_error(error_str: str) -> str:
    s = error_str.lower()
    if _is_key_error(error_str):
        return "❌ Invalid API Key — your Gemini API key was rejected (HTTP 401/403)."
    if "404" in s or "model not found" in s or "not found" in s:
        return "❌ Model Not Found (HTTP 404) — the selected Gemini model doesn't exist or your key doesn't have access to it."
    if "429" in s or "rate limit" in s or "resource_exhausted" in s or "quota" in s:
        if "per minute" in s or "rate_limit" in s or "requests per minute" in s:
            return "⏱️ Rate Limited (per-minute) — you've hit the 5 req/min free-tier limit. The agent will automatically retry. Wait ~60 seconds and click 🔄 Re-analyse."
        return "📵 Quota Exhausted — your daily free-tier quota is used up. Wait until midnight (PT) for it to reset, or upgrade your plan."
    return f"⚠️ API Error: {error_str}"

def _call_universal_llm(messages: list[dict], model_name: str, api_key: str):
    try:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        genai.configure(api_key=api_key)
        system_instruction = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [msg["content"]]})
            elif msg["role"] == "tool":
                contents.append({"role": "user", "parts": [f"Observation from {msg.get('name', 'tool')}:\n{msg['content']}"]})
        
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(contents)
                return _parse_react_response(response.text), ""
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate limit" in err_str or "resource_exhausted" in err_str:
                    if "per minute" in err_str or "rate_limit" in err_str:
                        if attempt < max_retries - 1:
                            print(f"[WARN] Rate limited. Retrying in 15 seconds... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(15)
                            continue
                raise e
    except Exception as e:
        return None, str(e)


def _parse_react_response(text: str) -> dict:
    """
    Parse the agent's ReAct-format response.
    Looks for Action: / Action Input: / Final Answer: markers.
    """
    # Check for Final Answer
    final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        return {
            "wants_tool": False,
            "tool_name": None,
            "tool_args": None,
            "text": final_match.group(1).strip(),
        }

    # Check for Action / Action Input
    action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
    input_match = re.search(r"Action Input:\s*(.*?)(?=\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)

    if action_match:
        tool_name = action_match.group(1).strip()
        raw_input = input_match.group(1).strip() if input_match else ""

        # Try to parse JSON args
        try:
            tool_args = json.loads(raw_input)
        except (json.JSONDecodeError, ValueError):
            tool_args = raw_input  # plain string arg (e.g. a file path)

        return {
            "wants_tool": True,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "text": text,
        }

    # No action found — treat as final answer
    return {
        "wants_tool": False,
        "tool_name": None,
        "tool_args": None,
        "text": text,
    }


def _simulate_response(messages: list[dict]) -> dict:
    import re
    import json
    import pandas as pd

    observations = [m for m in messages if m.get("role") == "tool"]
    step = len(observations)

    # Extract file path from user message
    file_path = ""
    for msg in messages:
        m = re.search(r"Analyse the dataset at '(.*?)'", msg.get("content", ""))
        if m:
            file_path = m.group(1)
            break

    if not file_path:
        return {
            "wants_tool": False, "tool_name": None, "tool_args": None,
            "text": "Error: No dataset path found. Please upload a dataset first."
        }

    if step == 0:
        return {
            "wants_tool": True,
            "tool_name": "read_dataset_schema",
            "tool_args": file_path,
            "text": f"Thought: Read the schema first.\nAction: read_dataset_schema\nAction Input: {file_path}"
        }

    if step == 1:
        # Run a comprehensive, BI-aware analysis in one shot
        code = f"""
import pandas as pd, json, re

try:
    df = pd.read_csv(r'{file_path}')
except Exception:
    df = pd.read_csv(r'{file_path}', encoding='cp1255')

rows, cols_n = len(df), len(df.columns)

# ── Classify columns ─────────────────────────────────────────────────
ID_PATTERNS = re.compile(r'\\b(id|_id|code|key|num|number|index|row|no|#)\\b', re.I)

def is_identifier(col, series):
    if ID_PATTERNS.search(col):
        return True
    if series.dtype in ('int64', 'Int64') and series.nunique() == len(series):
        return True  # all-unique int -> likely a key
    return False

all_num  = df.select_dtypes(include='number').columns.tolist()
all_cat  = df.select_dtypes(exclude='number').columns.tolist()
all_dt   = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]

# Filter out identifiers from measures
measures = [c for c in all_num if not is_identifier(c, df[c])]
dims     = [c for c in all_cat if not is_identifier(c, df[c]) and df[c].nunique() <= 50]
dates    = [c for c in all_dt if c in df.columns]

res = {{'rows': rows, 'cols': cols_n, 'measures': measures, 'dims': dims, 'dates': dates}}

# ── Basic stats for each measure ─────────────────────────────────────
stats = {{}}
for m in measures[:6]:
    s = df[m].dropna()
    stats[m] = {{'sum': float(s.sum()), 'mean': float(s.mean()), 'min': float(s.min()), 'max': float(s.max()), 'count': int(len(s))}}
res['stats'] = stats

# ── Cross-tabulations (measure × dimension) ──────────────────────────
xtabs = {{}}
if measures and dims:
    for dm in dims[:2]:
        for ms in measures[:2]:
            key = f'{{dm}}__{{ms}}'
            grp = df.groupby(dm, observed=True)[ms].sum().sort_values(ascending=False)
            xtabs[key] = {{'top': grp.head(5).to_dict(), 'bottom': grp.tail(3).to_dict()}}
res['xtabs'] = xtabs

# ── Top/bottom values per measure ────────────────────────────────────
if dims:
    dim0 = dims[0]
    for ms in measures[:2]:
        grp = df.groupby(dim0, observed=True)[ms].sum().sort_values(ascending=False)
        res[f'top_{{ms}}_by_{{dim0}}'] = grp.head(1).to_dict()
        res[f'pct_top_{{ms}}'] = round(grp.iloc[0] / grp.sum() * 100, 1) if len(grp) > 0 else 0

# ── Category distribution ─────────────────────────────────────────────
if dims:
    dim0 = dims[0]
    vc = df[dim0].value_counts(normalize=True).head(5)
    res['dim0_dist'] = {{dim0: vc.round(3).to_dict()}}

result = json.dumps(res, default=str)
"""
        return {
            "wants_tool": True,
            "tool_name": "run_python_analysis",
            "tool_args": code,
            "text": f"Thought: Run BI analysis.\nAction: run_python_analysis\nAction Input:\n{code}"
        }

    # ── Step 2+: Build the final BI output from computed results ─────────
    last_obs = observations[-1].get("content", "")
    m = re.search(r"\{.*\}", last_obs, re.DOTALL)
    try:
        res = json.loads(m.group(0)) if m else {}
    except Exception:
        res = {}

    rows       = res.get("rows", 0)
    cols_n     = res.get("cols", 0)
    measures   = res.get("measures", [])
    dims       = res.get("dims", [])
    dates      = res.get("dates", [])
    stats      = res.get("stats", {})
    xtabs      = res.get("xtabs", {})

    kpis    = []
    charts  = []
    findings = []
    recs    = []

    # ── KPIs (only from meaningful measures) ─────────────────────────────
    kpis.append({
        "name": "Total Records",
        "value": rows,
        "calculation": "COUNT()",
        "business_logic": "Total volume of records in the dataset.",
        "column": ""
    })

    kpi_count = 1
    for col in measures[:3]:
        if col not in stats:
            continue
        s = stats[col]
        col_label = col.replace("_", " ").title()
        kpis.append({
            "name": f"Total {col_label}",
            "value": round(s["sum"], 2),
            "calculation": f"SUM({col})",
            "business_logic": f"Total cumulative {col_label} — key volume indicator.",
            "column": col
        })
        kpi_count += 1
        if kpi_count >= 3:
            break

    # Add an average KPI for the primary measure
    if measures and measures[0] in stats:
        col = measures[0]
        s = stats[col]
        col_label = col.replace("_", " ").title()
        kpis.append({
            "name": f"Avg {col_label}",
            "value": round(s["mean"], 2),
            "calculation": f"AVG({col})",
            "business_logic": f"Average {col_label} per record — indicates typical magnitude.",
            "column": col
        })

    # ── Charts ────────────────────────────────────────────────────────────
    if dims and measures:
        charts.append({
            "title": f"{measures[0].replace('_', ' ').title()} by {dims[0].replace('_', ' ').title()}",
            "type": "bar",
            "x": dims[0],
            "y": f"SUM({measures[0]})",
            "color": None,
            "description": f"Shows total {measures[0]} broken down by {dims[0]}."
        })

    if len(dims) >= 2 and measures:
        charts.append({
            "title": f"{measures[0].replace('_', ' ').title()} by {dims[1].replace('_', ' ').title()}",
            "type": "bar",
            "x": dims[1],
            "y": f"SUM({measures[0]})",
            "color": None,
            "description": f"Compares {measures[0]} across {dims[1]} segments."
        })

    if dates and measures:
        charts.append({
            "title": f"{measures[0].replace('_', ' ').title()} Over Time",
            "type": "line",
            "x": dates[0],
            "y": measures[0],
            "color": None,
            "description": f"Tracks {measures[0]} trend over time."
        })

    if dims:
        charts.append({
            "title": f"Distribution of {dims[0].replace('_', ' ').title()}",
            "type": "pie",
            "x": dims[0],
            "y": "count",
            "color": None,
            "description": f"Proportion of records by {dims[0]}."
        })

    if len(measures) >= 2:
        charts.append({
            "title": f"{measures[0].replace('_', ' ').title()} vs {measures[1].replace('_', ' ').title()}",
            "type": "scatter",
            "x": measures[0],
            "y": measures[1],
            "color": dims[0] if dims else None,
            "description": f"Reveals correlation between {measures[0]} and {measures[1]}."
        })

    # ── Key Findings (with real numbers) ─────────────────────────────────
    findings.append(
        f"The dataset contains {rows:,} records across {cols_n} columns, "
        f"with {len(measures)} measurable business metrics and {len(dims)} categorical dimensions."
    )

    for col in measures[:2]:
        if col not in stats:
            continue
        s = stats[col]
        col_label = col.replace("_", " ").title()
        findings.append(
            f"Total {col_label} = {s['sum']:,.2f} (avg {s['mean']:,.2f} per record, "
            f"range {s['min']:,.2f} – {s['max']:,.2f})."
        )

    # Cross-tab insight
    if xtabs:
        first_key = next(iter(xtabs))
        dm, ms = first_key.split("__")
        top_data = xtabs[first_key].get("top", {})
        if top_data:
            top_cat = next(iter(top_data))
            top_val = list(top_data.values())[0]
            total = sum(top_data.values())
            pct = round(top_val / total * 100, 1) if total else 0
            dm_label = dm.replace("_", " ").title()
            ms_label = ms.replace("_", " ").title()
            findings.append(
                f"Top {dm_label} by {ms_label}: '{top_cat}' accounts for "
                f"{top_val:,.2f} ({pct}% of the top-5 total)."
            )

    if dims:
        dim0 = dims[0]
        dim0_label = dim0.replace("_", " ").title()
        dim_dist_key = f"dim0_dist"
        dim_dist = res.get(dim_dist_key, {}).get(dim0, {})
        if dim_dist:
            top_dim_val = next(iter(dim_dist))
            top_dim_pct = round(list(dim_dist.values())[0] * 100, 1)
            findings.append(
                f"'{top_dim_val}' is the most frequent {dim0_label}, "
                f"representing {top_dim_pct}% of all records."
            )

    if len(measures) >= 2:
        col1, col2 = measures[0], measures[1]
        if col1 in stats and col2 in stats:
            findings.append(
                f"Secondary metric '{col2.replace('_', ' ').title()}' has a mean of "
                f"{stats[col2]['mean']:,.2f} and total of {stats[col2]['sum']:,.2f}."
            )

    # ── Recommendations ───────────────────────────────────────────────────
    if dims and measures:
        dim_label = dims[0].replace("_", " ").title()
        ms_label = measures[0].replace("_", " ").title()
        recs.append(
            f"Focus resources on the top-performing {dim_label} segments that drive "
            f"the highest {ms_label} — use the Charts tab to identify them visually."
        )
    recs.append(
        "Apply the sidebar filters to drill into specific segments or date ranges "
        "and compare performance across sub-groups."
    )
    if len(measures) >= 2:
        recs.append(
            f"Investigate the relationship between {measures[0].replace('_',' ').title()} "
            f"and {measures[1].replace('_',' ').title()} using the Scatter chart — "
            "strong correlations can inform pricing or resource allocation strategies."
        )

    final_json = {
        "dataset_name": "Business Intelligence Dashboard",
        "summary": (
            f"This dataset contains {rows:,} records and {cols_n} columns. "
            f"It has {len(measures)} numeric business metrics ({', '.join(measures[:3])}) "
            f"and {len(dims)} categorical dimensions ({', '.join(dims[:3])}) "
            f"suitable for BI analysis."
        ),
        "shape": {"rows": rows, "columns": cols_n},
        "column_types": {c: "measure" for c in measures} | {c: "dimension" for c in dims} | {c: "datetime" for c in dates},
        "kpis": kpis,
        "charts": charts,
        "key_findings": [f for f in findings if f],
        "recommendations": recs
    }

    return {
        "wants_tool": False, "tool_name": None, "tool_args": None,
        "text": json.dumps(final_json, ensure_ascii=False)
    }

# ---------------------------------------------------------------------------
# Main Agent Loop
# ---------------------------------------------------------------------------

def run_agent_loop(
    user_input: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_iterations: int = 10,
) -> str:
    """
    Execute the ReAct agent loop.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    print(f"\n[AGENT] Started | query: {user_input}")
    print("-" * 60)

    for iteration in range(max_iterations):
        print(f"\n[Iteration {iteration + 1}/{max_iterations}]")

        # --- Step A: Call the LLM ---
        response = call_llm_api(messages)

        # --- Step B: Does the agent want to use a tool? ---
        if response["wants_tool"]:
            tool_name = response["tool_name"]
            tool_args = response["tool_args"]

            print(f"[TOOL] Calling: {tool_name}")
            print(f"   Args: {str(tool_args)[:120]}...")

            # Validate tool exists
            if tool_name not in AVAILABLE_TOOLS:
                tool_result = f"Error: Tool '{tool_name}' is not available."
            else:
                # Execute the tool
                fn = AVAILABLE_TOOLS[tool_name]
                if isinstance(tool_args, dict):
                    tool_result = fn(**tool_args)
                else:
                    tool_result = fn(tool_args)

            print(f"[OBS] Observation: {str(tool_result)[:200]}...")

            # Append tool result as an Observation so the LLM can see it
            messages.append({"role": "tool", "name": tool_name, "content": tool_result})

        # --- Step C: No tool needed -> Final Answer ---
        else:
            final_answer = response["text"]
            print(f"\n[DONE] Final Answer:\n{final_answer}")
            return final_answer

    # Max iterations reached
    timeout_msg = (
        "[WARN] Agent reached the maximum number of iterations "
        "without producing a final answer."
    )
    print(timeout_msg)
    return timeout_msg


# ---------------------------------------------------------------------------
# Demo execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Order-specific demo: look up an order by ID
    answer = run_agent_loop(
        user_input="What is the status of order 1003? Is there anything special I should know?",
        system_prompt=ORDERS_SYSTEM_PROMPT,
        max_iterations=5,
    )

    print("\n" + "=" * 60)
    print("FULL ANALYSIS (Data Analyst Mode)")
    print("=" * 60)
    answer2 = run_agent_loop(
        user_input=f"Analyse the orders dataset at '{DATA_PATH}'. "
                    "What are the top insights for the manager?",
        system_prompt=SYSTEM_PROMPT,
        max_iterations=10,
    ) 
