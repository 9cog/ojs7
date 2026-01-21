#!/usr/bin/env python3
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "autonomous-agents-framework" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from providers.factory import get_ml_engine, get_comm_automation, get_data_sync  # type: ignore

def main():
    _ = get_ml_engine({})
    _ = get_comm_automation({})
    _ = get_data_sync(None)
    print("smoke_ok=true")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "autonomous-agents-framework" / "src"
sys.path.insert(0, str(SRC))

from providers.factory import get_ml_engine, get_comm_automation, get_data_sync  # type: ignore

os.environ.setdefault("USE_PROVIDER_IMPLEMENTATIONS", "true")

ml = get_ml_engine({})
pred = ml.predict({"a": 1, "b": 2}) if hasattr(ml, "predict") else {"ok": False}

comm = get_comm_automation({})
email_res = {}
if hasattr(comm, "send_email"):
    email_res = comm.send_email("test@example.com", "Test", "<b>Hi</b>")

sync = get_data_sync(None)
sync_ok = False
if hasattr(sync, "upsert_agent_decision"):
    sync_ok = sync.upsert_agent_decision("agent-1", {"decision": "review", "confidence": 0.5})

print(json.dumps({"ml_pred": pred, "email": email_res, "sync_ok": sync_ok}))
