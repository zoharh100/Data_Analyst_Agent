import json

with open('agent_loop.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = lines[:163]
replacement = '''def _simulate_response(messages: list[dict]) -> dict:
    import re
    import json
    
    observations = [m for m in messages if m.get("role") == "tool"]
    step = len(observations)
    
    # Extract file path from first message
    file_path = DATA_PATH
    if messages:
        m = re.search(r"Analyse the dataset at '(.*?)'", messages[0].get("content", ""))
        if m:
            file_path = m.group(1)

    if step == 0:
        return {
            "wants_tool": True,
            "tool_name": "read_dataset_schema",
            "tool_args": file_path,
            "text": "Thought: I need to understand the dataset structure first.\\nAction: read_dataset_schema\\nAction Input: " + file_path
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
            "text": "Thought: I'll run a dynamic analysis.\\nAction: run_python_analysis\\nAction Input: \\n" + code
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
'''
new_lines.append(replacement)
new_lines.extend(lines[266:])

with open('agent_loop.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Updated agent_loop.py")
