"""Verification script to confirm all modules import and run correctly."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Testing data_loader...")
from data_loader import load_and_prepare_data
df = load_and_prepare_data()
print(f"  OK: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")

print("\nTesting agent_tools...")
from agent_tools import read_dataset_schema, run_python_analysis
data_path = os.path.join(os.path.dirname(__file__), "data", "orders.csv")
schema = read_dataset_schema(data_path)
print("  read_dataset_schema OK")

code = (
    "import pandas as pd\n"
    f"df = pd.read_csv(r'{data_path}')\n"
    "df['profit'] = df['final_price'] - df['ingredients_cost']\n"
    "result = df[['product_type','profit']].groupby('product_type')['profit'].sum().to_string()\n"
)
out = run_python_analysis(code)
print(f"  run_python_analysis OK (output length: {len(out)})")

print("\nTesting dashboard_config...")
from dashboard_config import profile_dataset, generate_dashboard_config
meta = profile_dataset(df)
cfg = generate_dashboard_config(meta)
print(f"  OK: {len(cfg.get('kpis', []))} KPIs, {len(cfg.get('charts', []))} charts")
for kpi in cfg.get("kpis", []):
    print(f"    KPI: {kpi['name']} = {kpi['calculation']}")

print("\nTesting agent_loop...")
from agent_loop import run_agent_loop
answer = run_agent_loop("What is the status of order 1003?", max_iterations=3)
print(f"  Agent loop OK (answer length: {len(answer)})")

print("\n=== All modules verified successfully! ===")
print(f"\nTotal revenue: {df['final_price'].sum():,.2f}")
print(f"Total profit:  {df['profit'].sum():,.2f}")
print(f"Avg margin:    {df['profit_margin_pct'].mean():.1f}%")
