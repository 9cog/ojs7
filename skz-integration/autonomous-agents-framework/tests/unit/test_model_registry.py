import os
import sys
import types
import importlib
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models import model_registry  # type: ignore


def test_load_model_local_joblib(monkeypatch, tmp_path):
    fake_model_path = tmp_path / "model.joblib"
    fake_model_path.write_bytes(b"FAKE")

    class FakeJoblib(types.SimpleNamespace):  # type: ignore
        @staticmethod
        def load(path):
            assert str(path) == str(fake_model_path)
            return object()

    monkeypatch.setenv("DECISION_MODEL_PATH", str(fake_model_path))
    monkeypatch.setenv("DECISION_MODEL_VERSION", "testlocal")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    monkeypatch.setitem(sys.modules, "joblib", FakeJoblib)
    importlib.reload(model_registry)

    handle = model_registry.load_model("ignored")
    assert handle is not None
    assert handle.version == "testlocal"
