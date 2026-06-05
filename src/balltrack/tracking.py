"""
tracking.py — one place to point MLflow at the tracking server.

Both `train.py` and `evaluate.py` call `setup_mlflow()` before any logging
happens, so they share the same server URI and experiment. The URI is read from
the `MLFLOW_TRACKING_URI` environment variable (default
`http://127.0.0.1:5000`), keeping it out of the code and out of version control.

Start the server first (see README) with:

    mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

"""

from __future__ import annotations

import os

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "balltrack"


def setup_mlflow(experiment: str = DEFAULT_EXPERIMENT) -> str | None:

    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", experiment)

    try:
        import mlflow
    except ImportError:
        print("(mlflow not installed; skipping experiment tracking)")
        return None

    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
    except Exception as exc:  
        print(f"(mlflow setup failed: {exc}; continuing without tracking)")
        return None

    print(f"MLflow tracking -> {uri}  (experiment: {experiment})")
    return uri
