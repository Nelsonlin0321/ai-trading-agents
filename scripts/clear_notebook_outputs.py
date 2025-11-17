#!/usr/bin/env python3
"""
Script to clear outputs from Jupyter notebooks.
Used by git pre-commit hook to ensure clean notebooks are committed.
"""

import json
import sys
import os


def clear_notebook_outputs(notebook_path):
    """Clear all outputs from a Jupyter notebook."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Iterate through all cells and clear outputs
    for cell in notebook.get("cells", []):
        if cell["cell_type"] == "code":
            # Clear outputs
            cell["outputs"] = []
            # Reset execution count
            cell["execution_count"] = None

    # Write back to the file
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)  # Use indent=1 for Jupyter's standard
        f.write("\n")  # Ensure a final newline


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python clear_notebook_outputs.py <notebook_path> [notebook_path...]",
            file=sys.stderr,
        )
        sys.exit(1)

    notebook_paths = sys.argv[1:]

    for path in notebook_paths:
        if not os.path.exists(path):
            print(f"File does not exist: {path}", file=sys.stderr)
            continue

        if not path.endswith(".ipynb"):
            print(f"Skipping non-notebook file: {path}", file=sys.stderr)
            continue

        try:
            clear_notebook_outputs(path)
            print(f"Cleared outputs from: {path}")
        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
