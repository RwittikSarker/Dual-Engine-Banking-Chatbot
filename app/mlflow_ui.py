import importlib.abc
import importlib.resources.abc
import os
import sys

# Monkey-patch Python 3.14 importlib.abc.Traversable removal to support MLflow 3.x+
try:
    importlib.abc.Traversable = importlib.resources.abc.Traversable
except AttributeError:
    pass

# Enable MLflow local file store tracking backend for assignment compatibility
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from mlflow.cli import cli

if __name__ == '__main__':
    print("Starting MLflow UI server programmatically with Python 3.14 compatibility patches...")
    
    # Parse and forward command-line arguments to the MLflow CLI
    cli_args = sys.argv[1:]
    
    # Ensure 'ui' is the sub-command
    if not cli_args or cli_args[0] != 'ui':
        cli_args = ['ui'] + cli_args
        
    # Default to host 0.0.0.0 for ease of access if not explicitly overridden
    if '--host' not in cli_args and '-h' not in cli_args:
        cli_args.extend(['--host', '0.0.0.0'])
        
    try:
        # Runs 'mlflow ui' natively in this process, respecting all command-line arguments (like --port)
        cli(args=cli_args)
    except SystemExit:
        pass