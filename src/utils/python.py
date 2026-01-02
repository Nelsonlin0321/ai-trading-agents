import ast
import os
import subprocess
import uuid


def validate_code(code: str) -> str | None:
    """
    Validates Python code for security vulnerabilities using AST.
    Returns None if valid, or an error message string if invalid.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax Error: {e}"

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [n.name.split(".")[0] for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules = [node.module.split(".")[0]]

            for module in modules:
                if module in [
                    "os",
                    "sys",
                    "subprocess",
                    "shutil",
                    "importlib",
                    "socket",
                    "requests",
                    "urllib",
                ]:
                    return f"Security Violation: Import of restricted module '{module}' is not allowed."

        # Check for dangerous built-ins and functions
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in [
                    "exec",
                    "eval",
                    "compile",
                    "open",
                    "__import__",
                    "exit",
                    "quit",
                    "help",
                ]:
                    return f"Security Violation: Use of restricted function '{node.func.id}' is not allowed."

    return None


def run_python_script(script_path: str, timeout: int = 30):
    """Run a Python script and return the output.

    Args:
        script_path (str): The path to the Python script.
        timeout (int): The maximum execution time in seconds. Defaults to 30.

    Returns:
        str: The output of the script.
    """
    try:
        # Define the command as a list of strings
        # It's generally best practice to use the absolute path to your Python interpreter
        # For this example, we use 'sys.executable' to use the same interpreter as the current script
        # command = [sys.executable, script_path]
        command = ["python3", script_path]

        # Use subprocess.run()
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=timeout
        )

        response = ""
        response += f"STDOUT: {result.stdout}"
        response += f"STDERR: {result.stderr}"
        response += f"Process finished with exit code: {result.returncode}"
        return response
    except subprocess.TimeoutExpired as e:
        return f"Process timed out after {timeout} seconds. STDOUT: {e.stdout} STDERR: {e.stderr}"
    except subprocess.CalledProcessError as e:
        response = ""
        response += f"Process failed with error code {e.returncode}"
        response += f"STDOUT: {e.stdout}"
        response += f"STDERR: {e.stderr}"
        return response
    except FileNotFoundError:
        return "Error: The script file or the python interpreter was not found."


def run_python_code(code: str, timeout: int = 30) -> str:
    """Run Python code and return the output.

    Args:
        code (str): The Python code to run.
        timeout (int): The maximum execution time in seconds. Defaults to 30.

    Returns:
        str: The output of the code.
    """
    # Validate code security
    security_error = validate_code(code)
    if security_error:
        return security_error

    # give it a unique name
    with open(f"{uuid.uuid4()}.py", "w") as f:
        f.write(code)
        temp_path = f.name

    try:
        response = run_python_script(temp_path, timeout=timeout)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return response


if __name__ == "__main__":
    #  python src/utils/python.py
    code = """
import pandas as pd
df = pd.read_csv("/Users/nelsonlin/downloads/ticker_bars/TSLA.csv")
print(df.head())
"""
    print(run_python_code(code))
