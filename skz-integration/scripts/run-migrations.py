#!/usr/bin/env python3
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "autonomous-agents-framework", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from providers.migrations import run_guarded_migrations  # type: ignore

if __name__ == "__main__":
    ok = run_guarded_migrations(os.getenv("POSTGRES_DSN"))
    print("migrations_applied=" + ("true" if ok else "false"))
