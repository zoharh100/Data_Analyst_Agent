"""
Step 3: Mock Data + Data Preparation
=====================================
Loads the orders mock dataset (CSV), converts types,
and computes derived columns (profit, profit_margin_pct).

The 'orders_df' DataFrame is the shared "database" that all
agent tools operate on — analogous to a real BI data warehouse.
"""

import json
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Mock data as JSON (matches the course document exactly, extended to 15 rows)
# ---------------------------------------------------------------------------
MOCK_DATA_JSON = """
[
  {"order_id": 1001, "order_date": "2026-06-15", "product_type": "Laptop Pro 15-inch",
   "ingredients_cost": 1200.50, "final_price": 1800.00, "status": "Delivered",
   "customer_notes": "Great laptop, fast delivery"},
  {"order_id": 1002, "order_date": "2026-06-16", "product_type": "Wireless Headphones",
   "ingredients_cost": 45.00, "final_price": 180.00, "status": "In Progress",
   "customer_notes": "Urgent delivery needed before trip"},
  {"order_id": 1003, "order_date": "2026-06-17", "product_type": "Smartphone 128GB",
   "ingredients_cost": 350.00, "final_price": 620.00, "status": "Pending",
   "customer_notes": "Please include screen protector"},
  {"order_id": 1004, "order_date": "2026-06-18", "product_type": "Mechanical Keyboard",
   "ingredients_cost": 90.00, "final_price": 180.00, "status": "Cancelled",
   "customer_notes": "Client changed mind on switches"},
  {"order_id": 1005, "order_date": "2026-06-19", "product_type": "Laptop Pro 15-inch",
   "ingredients_cost": 1200.00, "final_price": 1800.00, "status": "Delivered",
   "customer_notes": "Perfect for development work"},
  {"order_id": 1006, "order_date": "2026-06-20", "product_type": "Gaming Mouse",
   "ingredients_cost": 50.00, "final_price": 120.00, "status": "Delivered",
   "customer_notes": "Good tracking speed"},
  {"order_id": 1007, "order_date": "2026-06-21", "product_type": "4K Monitor",
   "ingredients_cost": 280.00, "final_price": 450.00, "status": "In Progress",
   "customer_notes": "Rush order, needed for home office setup"},
  {"order_id": 1008, "order_date": "2026-06-22", "product_type": "Smartphone 128GB",
   "ingredients_cost": 360.00, "final_price": 640.00, "status": "Pending",
   "customer_notes": "Verify storage size before shipping"},
  {"order_id": 1009, "order_date": "2026-06-23", "product_type": "Laptop Pro 15-inch",
   "ingredients_cost": 1245.00, "final_price": 1850.00, "status": "Delivered",
   "customer_notes": "Requested extra RAM upgrade last minute"},
  {"order_id": 1010, "order_date": "2026-06-24", "product_type": "Bluetooth Speaker",
   "ingredients_cost": 55.00, "final_price": 120.00, "status": "Cancelled",
   "customer_notes": "Client cancelled, bought locally instead"},
  {"order_id": 1011, "order_date": "2026-06-25", "product_type": "Wireless Headphones",
   "ingredients_cost": 48.00, "final_price": 185.00, "status": "Delivered",
   "customer_notes": "Excellent sound quality"},
  {"order_id": 1012, "order_date": "2026-06-26", "product_type": "Tablet 10-inch",
   "ingredients_cost": 195.00, "final_price": 350.00, "status": "In Progress",
   "customer_notes": "Include stylus pen"},
  {"order_id": 1013, "order_date": "2026-06-27", "product_type": "Smartwatch",
   "ingredients_cost": 85.00, "final_price": 270.00, "status": "Delivered",
   "customer_notes": "Great build quality, repeat customer"},
  {"order_id": 1014, "order_date": "2026-06-28", "product_type": "Smartphone 128GB",
   "ingredients_cost": 355.00, "final_price": 630.00, "status": "Pending",
   "customer_notes": "Client confirmed pickup time 15:00"},
  {"order_id": 1015, "order_date": "2026-06-29", "product_type": "Gaming Mouse",
   "ingredients_cost": 52.00, "final_price": 125.00, "status": "Delivered",
   "customer_notes": "Included RGB lighting is perfect"}
]
"""


def load_and_prepare_data(source: str = "csv") -> pd.DataFrame:
    """
    Load order data and add derived columns.

    Parameters
    ----------
    source : 'json' | 'csv'
        'json'  – load from the in-memory MOCK_DATA_JSON string (default).
        'csv'   – load from data/orders.csv (if it exists).

    Returns
    -------
    pd.DataFrame with additional columns:
        profit              = final_price - ingredients_cost
        profit_margin_pct   = profit / final_price * 100
    """
    if source == "csv":
        csv_path = os.path.join(os.path.dirname(__file__), "data", "orders.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            # Fall back to JSON
            data = json.loads(MOCK_DATA_JSON)
            df = pd.DataFrame(data)
    else:
        data = json.loads(MOCK_DATA_JSON)
        df = pd.DataFrame(data)

    # Convert date column to datetime with mixed format support
    df["order_date"] = pd.to_datetime(df["order_date"], format="mixed")

    # --- Derived columns (used by both agent and dashboard) ---
    df["profit"] = df["final_price"] - df["ingredients_cost"]
    df["profit_margin_pct"] = (df["profit"] / df["final_price"]) * 100

    return df


# ---------------------------------------------------------------------------
# Make orders_df available as a module-level variable so tools can import it
# ---------------------------------------------------------------------------
orders_df: pd.DataFrame = load_and_prepare_data()

# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_prepare_data()
    print("--- Orders (with derived columns) ---")
    print(df[["order_id", "product_type", "final_price", "profit", "profit_margin_pct", "status"]])
    print(f"\nTotal revenue:  ${df['final_price'].sum():,.2f}")
    print(f"Total profit:   ${df['profit'].sum():,.2f}")
    print(f"Avg margin:     {df['profit_margin_pct'].mean():.1f}%")
