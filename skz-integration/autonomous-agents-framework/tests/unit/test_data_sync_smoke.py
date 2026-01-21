import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_sync_manager import DataSyncManager  # type: ignore

class DummyOJSBridge:  # minimal methods used in DataSyncManager paths we call
    def get_manuscript(self, manuscript_id): return None
    def get_reviewer(self, reviewer_id): return None
    def get_editorial_decision(self, decision_id): return None
    def update_manuscript(self, manuscript_id, data): return True
    def update_reviewer(self, reviewer_id, data): return True
    def update_editorial_decision(self, decision_id, data): return True


def test_emit_event_and_tables_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"
    monkeypatch.setenv("ENVIRONMENT", "development")
    mgr = DataSyncManager(DummyOJSBridge(), db_path=str(db_path))

    mgr._emit_event("manuscript", "ms-1", "sync_started", {"x": 1})

    con = sqlite3.connect(str(db_path))
    cur = con.execute("SELECT entity_type, entity_id, event_type, payload FROM sync_events")
    rows = cur.fetchall()
    con.close()

    assert rows and rows[0][0] == "manuscript"
    assert rows[0][1] == "ms-1"
    assert rows[0][2] == "sync_started"
