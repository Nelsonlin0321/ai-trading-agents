import os
import subprocess
import sys
import uuid


def run_python_script(script_path: str):
    """Run a Python script and return the output.

    Args:
        script_path (str): The path to the Python script.

    Returns:
        str: The output of the script.
    """
    try:
        # Define the command as a list of strings
        # It's generally best practice to use the absolute path to your Python interpreter
        # For this example, we use 'sys.executable' to use the same interpreter as the current script
        command = [sys.executable, script_path]

        # Use subprocess.run()
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        response = ""
        response += f"STDOUT: {result.stdout}"
        response += f"STDERR: {result.stderr}"
        response += f"Process finished with exit code: {result.returncode}"
        return response
    except subprocess.CalledProcessError as e:
        response = ""
        response += f"Process failed with error code {e.returncode}"
        response += f"STDOUT: {e.stdout}"
        response += f"STDERR: {e.stderr}"
        return response
    except FileNotFoundError:
        return "Error: The script file or the python interpreter was not found."


def run_python_code(code: str) -> str:
    """Run Python code and return the output.

    Args:
        code (str): The Python code to run.

    Returns:
        str: The output of the code.
    """
    # give it a unique name
    with open(f"{uuid.uuid4()}.py", "w") as f:
        f.write(code)
        temp_path = f.name

    response = run_python_script(temp_path)
    os.remove(temp_path)
    return response


if __name__ == "__main__":
    #  python src/utils/python.py
    code = """
import pandas as pd
df = pd.read_csv("data/AAPL.csv")
print(df.head())
"""
    print(run_python_code(code))
