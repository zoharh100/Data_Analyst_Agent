"""
Step 1: Agent Tools
===================
Two core tools for the Data Analyst Agent.
Each tool accepts string arguments and returns a string result,
so the LLM can read and process them easily.
"""

import pandas as pd
import io
import sys


def read_dataset_schema(file_path: str) -> str:
    """
    Reads a CSV file and returns column names, data types,
    and a small sample (first 5 rows) to inform the agent.

    The intent is to keep the context window light — we only
    send a summary/sample rather than the entire dataset.
    """
    try:
        df = pd.read_csv(file_path)
        schema_info = (
            f"Columns and Data types:\n{df.dtypes}\n\n"
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
            f"First 5 rows:\n{df.head()}"
        )
        return str(schema_info)
    except Exception as e:
        return f"Error reading file: {str(e)}"


def run_python_analysis(code_string: str) -> str:
    """
    Executes arbitrary Python code (as a string) that the agent generates,
    and returns the result.

    Important design decisions:
    - The agent must store its final answer in a variable named 'result'.
    - Errors are caught and returned as strings (Observation) so the agent
      can self-correct and try again — this enables the Self-Correction loop.
    - stdout is also captured so print() calls are visible to the agent.
    """
    try:
        # Redirect stdout so the agent can use print() as debug output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        local_vars: dict = {}
        exec(code_string, globals(), local_vars)  # noqa: S102

        sys.stdout = old_stdout
        output_text = captured_output.getvalue()

        # If the agent saved a 'result' variable, return it
        if "result" in local_vars:
            result_str = str(local_vars["result"])
            if output_text:
                return f"{output_text}\nResult: {result_str}"
            return result_str

        # Otherwise return whatever was printed to stdout
        if output_text:
            return output_text

        return (
            "Code executed successfully. "
            "Note: Save your final answer in a variable named 'result'."
        )

    except Exception as e:
        sys.stdout = sys.__stdout__  # restore stdout on error
        # Return the error message as an Observation so the agent can self-correct
        return f"Python Error: {type(e).__name__}: {str(e)}"


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    data_path = os.path.join(os.path.dirname(__file__), "data", "orders.csv")

    print("=== Testing read_dataset_schema ===")
    schema = read_dataset_schema(data_path)
    print(schema)

    print("\n=== Testing run_python_analysis ===")
    code = f"""
import pandas as pd
df = pd.read_csv(r'{data_path}')
df['profit'] = df['final_price'] - df['ingredients_cost']
result = df[['order_id', 'product_type', 'profit', 'status']].to_string(index=False)
"""
    output = run_python_analysis(code)
    print(output)
