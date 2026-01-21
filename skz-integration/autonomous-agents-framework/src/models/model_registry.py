from typing import Optional, Any
import os

try:
    import mlflow
except Exception:
    mlflow = None

try:
    import joblib  # type: ignore
except Exception:
    joblib = None


class ModelHandle:
    def __init__(self, model: Any, version: str):
        self.model = model
        self.version = version


def load_model(model_name: str, version: Optional[str] = None) -> Optional[ModelHandle]:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow and uri:
        try:
            mlflow.set_tracking_uri(uri)
            mv = version or "Production"
            model_uri = f"models:/{model_name}/{mv}"
            model = mlflow.pyfunc.load_model(model_uri)
            return ModelHandle(model, mv)
        except Exception:
            pass

    local_path = os.getenv("DECISION_MODEL_PATH")
    local_version = os.getenv("DECISION_MODEL_VERSION", "local")
    if local_path and joblib:
        try:
            model = joblib.load(local_path)
            return ModelHandle(model, local_version)
        except Exception:
            pass

    if os.getenv("ENVIRONMENT", "").lower() == "production":
        raise RuntimeError("Decision model unavailable in production (no MLflow or local model configured)")
    return None
