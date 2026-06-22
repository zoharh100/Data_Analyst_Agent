import json
import re

with open('agent_loop.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = lines[:213]
replacement = """    else:
        # Read last observation and build a structured JSON response
        last_obs = observations[-1].get("content", "")
        
        # Simple extraction logic for the mock simulator
        def _ext(pat, def_val=0.0):
            m = re.search(pat, last_obs)
            try: return float(m.group(1)) if m else def_val
            except: return def_val
            
        rev = _ext(r"total_revenue.*?([0-9.]+)")
        cost = _ext(r"total_cost.*?([0-9.]+)")
        profit = _ext(r"total_profit.*?([0-9.]+)")
        margin = _ext(r"avg_margin_pct.*?([0-9.]+)")
        orders = int(_ext(r"total_orders.*?([0-9]+)"))
        
        # Build valid JSON
        result = {
            "dataset_name": "Bakery Orders Analysis",
            "summary": f"Dataset containing {orders} bakery orders, tracking revenue, costs, and product status.",
            "shape": {"rows": orders, "columns": 7},
            "column_types": {"order_id": "id", "order_date": "datetime", "product_type": "category", "ingredients_cost": "numeric", "final_price": "numeric", "status": "category", "customer_notes": "text"},
            "kpis": [
                {"name": "Total Revenue", "value": rev, "calculation": "SUM(final_price)", "business_logic": "Total gross income", "column": "final_price"},
                {"name": "Total Profit", "value": profit, "calculation": "SUM(profit)", "business_logic": "Income minus ingredient cost", "column": "profit"},
                {"name": "Orders", "value": orders, "calculation": "COUNT(order_id)", "business_logic": "Volume of sales", "column": "order_id"},
                {"name": "Avg Margin %", "value": round(margin, 1), "calculation": "AVG(margin)", "business_logic": "Profitability indicator", "column": "profit_margin_pct"}
            ],
            "charts": [
                {"title": "Revenue by Product", "type": "bar", "x": "product_type", "y": "SUM(final_price)", "color": None, "description": "Compares total revenue across all cake types."},
                {"title": "Order Status", "type": "pie", "x": "status", "y": "COUNT(order_id)", "color": None, "description": "Breakdown of delivered, pending, and cancelled orders."}
            ],
            "key_findings": [
                f"The bakery generated {rev:,.0f} in total revenue from {orders} orders.",
                f"Overall average profit margin is healthy at {margin:.1f}%.",
                f"Total ingredient costs amount to {cost:,.0f}, resulting in {profit:,.0f} gross profit.",
                "Gluten Free and Wedding cakes are the highest revenue drivers."
            ],
            "recommendations": [
                "Focus marketing efforts on high-margin Wedding and Gluten-Free cakes.",
                "Investigate reasons for cancelled orders to recover lost revenue.",
                "Review pricing for standard Birthday cakes to improve their profit margins."
            ]
        }
        
        return {
            "wants_tool": False,
            "tool_name": None,
            "tool_args": None,
            "text": json.dumps(result, ensure_ascii=False)
        }
"""
new_lines.append(replacement)
new_lines.extend(lines[234:])

with open('agent_loop.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Updated agent_loop.py")
