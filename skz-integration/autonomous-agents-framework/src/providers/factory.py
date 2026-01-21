import os
from typing import Any, Dict, Optional, Tuple

try:
    from .ml_decision_engine import MLDecisionEngine as ProviderMLDecisionEngine
except Exception:
    ProviderMLDecisionEngine = None  # type: ignore

try:
    from .communication_automation import CommunicationAutomation as ProviderCommunicationAutomation
except Exception:
    ProviderCommunicationAutomation = None  # type: ignore

try:
    from .data_sync_manager import DataSyncManager as ProviderDataSyncManager
except Exception:
    ProviderDataSyncManager = None  # type: ignore

try:
    from ..models.ml_decision_engine import DecisionEngine as ModelDecisionEngine
except Exception:
    ModelDecisionEngine = None  # type: ignore

try:
    from ..models.communication_automation import CommunicationAutomation as ModelCommunicationAutomation
except Exception:
    ModelCommunicationAutomation = None  # type: ignore

try:
    from ..data_sync_manager import DataSyncManager as ModelDataSyncManager  # type: ignore
except Exception:
    ModelDataSyncManager = None  # type: ignore


def _flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def use_providers() -> bool:
    return _flag("USE_PROVIDER_IMPLEMENTATIONS", False)


def get_ml_engine(config: Optional[Dict[str, Any]] = None):
    if use_providers() and ProviderMLDecisionEngine:
        return ProviderMLDecisionEngine(model_path=os.getenv("ML_DECISION_MODEL_PATH"))
    if ModelDecisionEngine:
        cfg = config or {}
        return ModelDecisionEngine(cfg)
    return None


def get_comm_automation(config: Optional[Dict[str, Any]] = None):
    if use_providers() and ProviderCommunicationAutomation:
        return ProviderCommunicationAutomation()
    if ModelCommunicationAutomation:
        return ModelCommunicationAutomation(config or {})
    return None


def get_data_sync(ojs_bridge: Optional[Any] = None, dsn: Optional[str] = None, redis_url: Optional[str] = None):
    if use_providers() and ProviderDataSyncManager:
        return ProviderDataSyncManager(dsn=dsn or os.getenv("POSTGRES_DSN"), redis_url=redis_url or os.getenv("REDIS_URL"))
    if ModelDataSyncManager:
        return ModelDataSyncManager(ojs_bridge, db_path=os.getenv("SQLITE_DB_PATH", "data_sync.db"))  # type: ignore
    return None
