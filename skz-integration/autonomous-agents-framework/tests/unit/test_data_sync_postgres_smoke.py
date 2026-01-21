import os
import sys
import pytest
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_sync_manager import DataSyncManager  # type: ignore


@pytest.mark.skipif(not os.getenv("POSTGRESQL_URL"), reason="POSTGRESQL_URL not configured")
def test_datasync_manager_init_with_postgres_env(monkeypatch, tmp_path):
    pg_url = os.getenv("POSTGRESQL_URL")
    assert pg_url is not None

    db_path = tmp_path / "sync_dev.db"
    mgr = DataSyncManager(ojs_bridge=None, db_path=str(db_path))  # type: ignore
    assert mgr is not None
