# AI Data Analyst · Autonomous BI

A Streamlit-based dashboard powered by a ReAct (Reasoning and Acting) AI Agent. This application automatically analyzes uploaded datasets (CSV, Excel), generates Python code on the fly to calculate key metrics, and produces a dynamic BI dashboard complete with KPIs, charts, and key findings.

## Features
- **Dynamic Dataset Parsing:** Upload any tabular dataset (.csv, .xlsx, .xls) and the app will automatically profile it.
- **True Agentic AI Analysis:** Provide a Gemini API Key to enable the ReAct agent. The AI will interpret your data, write custom Python analysis scripts in the background, and dynamically deduce the most important business metrics.
- **Offline Mock Simulator:** If no API Key is provided, the dashboard falls back to a dynamic mock simulator that automatically calculates basic sums, averages, and row counts to simulate AI output.
- **Interactive Dashboard:** View auto-generated KPIs, interactive Plotly charts, data previews, and AI-generated findings.
- **Data Chat:** Ask follow-up questions about your data in the chat interface.

## Quickstart

### 1. Clone the repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd Data_Analyst_Agent
```

### 2. Install dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run dashboard.py
```

## Usage
1. Open the dashboard in your web browser (usually `http://localhost:8501`).
2. Upload your CSV or Excel dataset using the sidebar.
3. (Optional) Paste your **Gemini API Key** in the Agent Settings to unlock the full AI analytical engine.
4. Click **🔄 Re-analyse** to build your dynamic dashboard!