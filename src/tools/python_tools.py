import os
from src.utils.ticker import is_valid_ticker
from langchain_experimental.utilities import PythonREPL
from langchain.tools import tool

HOME_DIR = os.environ["HOME"]
DATA_DIR = os.path.join(HOME_DIR, "downloads/ticker_bars")

DOC_STRING = f"""
    A Python shell to execute solely technical analysis code based on the ticker data on path `{DATA_DIR}/{{ticker}}.csv` where ticker is a ticker symbol placeholder.
    Before executing the code, you should call `download_ticker_bars_data` to download the ticker data first to `{DATA_DIR}/{{ticker}}.csv`.
    The format output of code execution is plain text only, 
    you should print the execution output with `print(...)` to show your analysis results.

    Basically you can use pandas, numpy, pandas_ta in this environment to analyze the data, or do the data analysis using native Python.

    GUARDRAILS:
    - You are STRICTLY PROHIBITED from accessing, printing, revealing, deleting, or modifying environment variables (os.environ), system files, or any files outside the `{DATA_DIR}` directory.
    - Never use print(os.environ) or similar commands that dump the entire environment.
    - You may ONLY read the CSV file located at `{DATA_DIR}/{{ticker}}.csv`.
    - You may NOT install new packages or use 'pip'.
    - Network access is restricted; do not attempt to make external API calls.
    - Focus solely on data analysis and technical indicators.
    """


@tool("execute_python_technical_analysis")
def execute_python_technical_analysis(code: str, ticker: str) -> str:
    """
    Execute Python code for technical analysis on the ticker data.

    Args:
        code (str): The Python code to execute.
        ticker (str): The ticker symbol for the data analysis.

    Returns:
        str: The execution output or an error message.
    """

    if not is_valid_ticker(ticker):
        return f"Invalid ticker symbol: `{ticker}`"

    csv_data_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(csv_data_path):
        return (
            f"Data file for ticker `{ticker}` not found."
            "Please call `download_ticker_bars_data` to download the ticker data first."
        )

    python_repl = PythonREPL()

    # Simple static analysis for guardrails
    # Note: This is a basic check and can be bypassed. For production, use a sandbox environment.
    prohibited_keywords = [
        "os.environ",
        "os.getenv",
        "os.putenv",
        "os.unsetenv",
        "write(",
        "delete(",
        "remove(",
        "unlink(",
        "rmdir(",
        "shutil",
        "subprocess",
        "eval",
        "exec",
        "compile",
        "pip",
        "install",
        "wget",
        "curl",
        "requests",
        "urllib",
        "http",
    ]

    # Check for prohibited keywords
    for keyword in prohibited_keywords:
        if keyword in code:
            return f"Security Violation: Prohibited command or keyword `{keyword}` found in code. Please focus on data analysis using pandas/numpy."

    output = python_repl.run(code)

    code = code.strip()
    if not code.startswith("```"):
        code = f"```python\n{code}\n```"

    if not output:
        return f"{code}\nCode executed successfully (no output)."
    return f"{code}\n OUTPUT: \n{output}"


execute_python_technical_analysis.__doc__ = DOC_STRING
