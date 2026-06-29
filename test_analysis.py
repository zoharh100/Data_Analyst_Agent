import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyCxAP7Gjglfu-KMiaBoi1sj6RmHTQMWNYI"
import json
from dashboard import run_full_analysis

try:
    res = run_full_analysis("data/orders.csv", 123)
    with open("res.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("Keys in result:", res.keys())
    print("Key findings type:", type(res.get("key_findings")))
    print("Key findings:", res.get("key_findings"))
except Exception as e:
    print("ERROR:", e)
