"""
tracking.py — one place to point MLflow at the tracking server.

Both `train.py` and `evaluate.py` call `setup_mlflow()` before any logging
happens, so they share the same server URI and experiment. The URI is read from
the `MLFLOW_TRACKING_URI` environment variable (default
`http://127.0.0.1:5000`), keeping it out of the code and out of version control.

Start the server first (see README) with:

    mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

Logging degrades gracefully: if MLflow is not installed or the server is
unreachable, `setup_mlflow()` prints a clear warning and returns ``None`` so the
caller can carry on without crashing the run.
"""

from __future__ import annotations

import os

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "balltrack"


def setup_mlflow(experiment: str = DEFAULT_EXPERIMENT) -> str | None:
    """Point MLflow (and the ultralytics callback) at the tracking server.

    Reads the URI from ``MLFLOW_TRACKING_URI`` (default
    :data:`DEFAULT_TRACKING_URI`) and exports both the URI and experiment name to
    the environment so the ultralytics MLflow callback picks them up, *and* calls
    ``set_tracking_uri`` / ``set_experiment`` for our direct logging in
    ``evaluate.py``. Must be called BEFORE any MLflow logging and BEFORE
    ``model.train()``.

    Returns the resolved URI on success, or ``None`` if MLflow is missing or the
    server cannot be reached (caller should then skip logging, not crash).
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    # Export so BOTH our direct calls and the ultralytics callback see the same
    # server. setdefault on the experiment lets an env override win if present.
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
    except Exception as exc:  # server down, bad URI, etc. — never crash the run
        print(f"(mlflow setup failed: {exc}; continuing without tracking)")
        return None

    print(f"MLflow tracking -> {uri}  (experiment: {experiment})")
    return uri
