import os
import subprocess

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
subprocess.run(["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"])