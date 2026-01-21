import os
import importlib
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def reload_factory():
    if "providers.factory" in sys.modules:
        importlib.reload(sys.modules["providers.factory"])
    else:
        import providers.factory  # noqa: F401
    return sys.modules["providers.factory"]


def test_returns_model_when_disabled():
    os.environ["USE_PROVIDER_IMPLEMENTATIONS"] = "false"
    factory = reload_factory()
    ml = factory.get_ml_engine({})
    comm = factory.get_comm_automation({})
    sync = factory.get_data_sync(None)
    assert ml is not None or comm is not None or sync is not None


def test_returns_provider_when_enabled():
    os.environ["USE_PROVIDER_IMPLEMENTATIONS"] = "true"
    factory = reload_factory()
    ml = factory.get_ml_engine({})
    comm = factory.get_comm_automation({})
    sync = factory.get_data_sync(None)
    assert ml is not None
    assert comm is not None
    assert sync is not None
