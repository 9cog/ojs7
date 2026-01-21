import os
from typing import Any, Dict, Optional

try:
    import joblib  # type: ignore
except Exception:
    joblib = None


class MLDecisionEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.getenv("ML_DECISION_MODEL_PATH")
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path or not joblib:
            self._model = None
            return
        try:
            self._model = joblib.load(self.model_path)
        except Exception:
            self._model = None

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if self._model is None:
            return {
                "decision": "review",
                "confidence": 0.5,
                "details": {"provider": "fallback", "reason": "no model available"},
            }
        try:
            y = self._model.predict([self._prepare_features(features)])[0]
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba([self._prepare_features(features)])[0]
                confidence = float(max(proba))
            else:
                confidence = 0.75
            return {
                "decision": str(y),
                "confidence": confidence,
                "details": {"provider": "local_model"},
            }
        except Exception:
            return {
                "decision": "review",
                "confidence": 0.5,
                "details": {"provider": "fallback", "reason": "prediction_failed"},
            }

    def _prepare_features(self, features: Dict[str, Any]) -> Any:
        return [features.get(k) for k in sorted(features.keys())]
