import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.decision_engine import DecisionEngine  # type: ignore


def test_decision_engine_smoke_dev(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("DECISION_AB_SPLIT", "control:50,variant:50")
    monkeypatch.setenv("DECISION_AB_STICKY_BY", "submission_id")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("DECISION_MODEL_PATH", "")  # no model available; allowed in dev

    eng = DecisionEngine(agent_id="agent-1")
    res = eng.make_decision({"submission_id": "abc123", "text": "sample"})
    assert isinstance(res, dict)
    assert "variant" in res and "model_version" in res
    assert res["variant"] in ("control", "variant")


def test_decision_engine_prod_requires_model(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DECISION_MODEL_PATH", raising=False)

    eng = DecisionEngine(agent_id="agent-2")
    error = None
    try:
        eng.make_decision({"submission_id": "sub-1"})
    except Exception as e:
        error = e
    assert error is not None
