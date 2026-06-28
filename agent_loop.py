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

The bakery-specific get_order_details tool is also defined here so the loop
can look up individual orders by ID.
"""

import json
import os
import re
from typing import Any

import pandas as pd

from agent_prompts import BAKERY_SYSTEM_PROMPT, SYSTEM_PROMPT
from agent_tools import read_dataset_schema, run_python_analysis
from data_loader import orders_df

# ---------------------------------------------------------------------------
# Bakery-specific tool: get_order_details
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
# LLM adapter — Universal LLM support (via LiteLLM) or rule-based fallback
# ---------------------------------------------------------------------------

def call_llm_api(messages: list[dict], model_name: str = None) -> dict:
    """
    Call the LLM and return a structured response dict.
    Takes the API key from the Streamlit UI (environment variables) 
    and dynamically routes it via LiteLLM.
    """
    # סטרימליט שומר את המפתח שהזנת בממשק לתוך GOOGLE_API_KEY
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    # אם הזנת מפתח אבל לא הוגדר ממודל ספציפי - נבחר בג'מיני כברירת מחדל (כדי להתאים לממשק שלך)
    if os.environ.get("GOOGLE_API_KEY") and not model_name:
        model_name = "gemini/gemini-1.5-flash"
        
    model = model_name or os.environ.get("LLM_MODEL", "")

    # אם יש גם מודל וגם מפתח, נפעיל את המודל האמיתי. אחרת - נחזור לסימולטור.
    if model and api_key:
        return _call_universal_llm(messages, model, api_key)
    else:
        return _simulate_response(messages)


def _call_universal_llm(messages: list[dict], model_name: str, api_key: str) -> dict:
    """Call ANY LLM API using the litellm library with an explicit API key."""
    try:
        import litellm

        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "tool":
                content = f"Observation from {msg.get('name', 'tool')}:\n{content}"
                role = "user"
                
            formatted_messages.append({"role": role, "content": content})

        # קריאה אוניברסלית באמצעות המפתח שהגיע מהסטרימליט
        response = litellm.completion(
            model=model_name,
            messages=formatted_messages,
            api_key=api_key 
        )
        
        response_text = response.choices[0].message.content
        return _parse_react_response(response_text)

    except ImportError:
        return {
            "wants_tool": False,
            "tool_name": None,
            "tool_args": None,
            "text": "LLM API Error: The 'litellm' package is missing. Please install it using `pip install litellm`.",
        }
    except Exception as e:
        return {
            "wants_tool": False,
            "tool_name": None,
            "tool_args": None,
            "text": f"LLM API Error: {e}",
        }


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
    
    observations = [m for m in messages if m.get("role") == "tool"]
    step = len(observations)
    
    # Extract file path from first message
    file_path = DATA_PATH
    for msg in messages:
        m = re.search(r"Analyse the dataset at '(.*?)'", msg.get("content", ""))
        if m:
            file_path = m.group(1)
            break

    if step == 0:
        return {
            "wants_tool": True,
            "tool_name": "read_dataset_schema",
            "tool_args": file_path,
            "text": "Thought: I need to understand the dataset structure first.\nAction: read_dataset_schema\nAction Input: " + file_path
        }
    elif step == 1:
        code = f"""
import pandas as pd
import json
try:
    df = pd.read_csv(r'{file_path}')
except Exception:
    df = pd.read_csv(r'{file_path}', encoding='cp1255')
num_cols = df.select_dtypes(include=['number']).columns.tolist()
cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
orders = len(df)
res = {{'orders': orders, 'cols': len(df.columns)}}
if num_cols:
    res['num1'] = num_cols[0]
    res['num1_sum'] = float(df[num_cols[0]].sum())
if len(num_cols) > 1:
    res['num2'] = num_cols[1]
    res['num2_avg'] = float(df[num_cols[1]].mean())
if cat_cols:
    res['cat1'] = cat_cols[0]
result = json.dumps(res)
"""
        return {
            "wants_tool": True,
            "tool_name": "run_python_analysis",
            "tool_args": code,
            "text": "Thought: I'll run a dynamic analysis.\nAction: run_python_analysis\nAction Input: \n" + code
        }
    else:
        last_obs = observations[-1].get("content", "")
        m = re.search(r"\{.*\}", last_obs, re.DOTALL)
        
        try:
            res = json.loads(m.group(0)) if m else {'orders': 0, 'cols': 0}
        except:
            res = {'orders': 0, 'cols': 0}
            
        orders = res.get('orders', 0)
        
        kpis = []
        charts = []
        findings = [f"Analyzed {orders} rows and {res.get('cols', 0)} columns successfully."]
        recs = ["Review the auto-generated charts to find interesting patterns.", "Use the filters to drill down into specific dates or categories."]
        
        kpis.append({"name": "Total Rows", "value": orders, "calculation": "COUNT()", "business_logic": "Total volume", "column": ""})
        
        if 'num1' in res:
            kpis.append({"name": f"Total {res['num1']}", "value": res['num1_sum'], "calculation": f"SUM({res['num1']})", "business_logic": f"Sum of {res['num1']}", "column": res['num1']})
            x_col = res.get('cat1', '')
            charts.append({"title": f"{res['num1']} by {x_col}" if x_col else f"{res['num1']} Distribution", "type": "bar", "x": x_col, "y": f"SUM({res['num1']})", "color": None, "description": f"Displays total {res['num1']}."})
            findings.append(f"The total {res['num1']} across the dataset is {res['num1_sum']:,.2f}.")
            
        if 'num2' in res:
            kpis.append({"name": f"Avg {res['num2']}", "value": round(res['num2_avg'], 2), "calculation": f"AVG({res['num2']})", "business_logic": f"Average {res['num2']}", "column": res['num2']})
            findings.append(f"The average {res['num2']} is {res['num2_avg']:,.2f}.")
            
        final_json = {
            "dataset_name": "Dynamic Dashboard",
            "summary": f"Automatically generated BI dashboard for your uploaded dataset containing {orders} rows.",
            "shape": {"rows": orders, "columns": res.get('cols', 0)},
            "column_types": {},
            "kpis": kpis,
            "charts": charts,
            "key_findings": findings,
            "recommendations": recs
        }
        
        return {
            "wants_tool": False,
            "tool_name": None,
            "tool_args": None,
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
    # Bakery-specific demo: look up an order by ID
    answer = run_agent_loop(
        user_input="What is the status of order 1003? Is there anything special I should know?",
        system_prompt=BAKERY_SYSTEM_PROMPT,
        max_iterations=5,
    )

    print("\n" + "=" * 60)
    print("FULL ANALYSIS (Data Analyst Mode)")
    print("=" * 60)
    answer2 = run_agent_loop(
        user_input=f"Analyse the bakery orders dataset at '{DATA_PATH}'. "
                    "What are the top insights for the manager?",
        system_prompt=SYSTEM_PROMPT,
        max_iterations=10,
    ) 
