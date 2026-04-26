import os
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.decision_engine import DecisionEngine  # type: ignore
import models.model_registry as model_registry  # type: ignore

class DummyModel:
    def predict_proba(self, X):
        # Returns a list-of-lists without requiring numpy so the test runs in
        # lightweight environments; the DummyModel still faithfully represents
        # a classifier that predicts class-1 probability 0.88.
        return [[0.12, 0.88] for _ in X]

def test_decision_engine_uses_local_model_prob(monkeypatch, tmp_path):
    class FakeJoblib(types.SimpleNamespace):  # type: ignore
        @staticmethod
        def load(path):
            return DummyModel()

    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("DECISION_MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("DECISION_MODEL_VERSION", "dev-local")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(model_registry, "joblib", FakeJoblib, raising=False)

    eng = DecisionEngine(agent_id="agent-dev")
    res = eng.make_decision({"submission_id": "abc123", "text": "Hello world"})
    assert "score" in res and res["score"] is not None
    assert 0.85 <= float(res["score"]) <= 0.9
