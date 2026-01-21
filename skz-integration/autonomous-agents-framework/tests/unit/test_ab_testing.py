import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.ab_testing import choose_variant  # type: ignore


def test_choose_variant_sticky():
    os.environ["DECISION_AB_SPLIT"] = "control:50,variant:50"
    os.environ["DECISION_AB_STICKY_BY"] = "submission_id"
    v1, _ = choose_variant({"submission_id": "sub-123"})
    v2, _ = choose_variant({"submission_id": "sub-123"})
    assert v1 == v2


def test_choose_variant_distribution_keys():
    os.environ["DECISION_AB_SPLIT"] = "control:50,variant:50"
    os.environ["DECISION_AB_STICKY_BY"] = ""
    v, mapping = choose_variant({})
    assert set(mapping.keys()) == {"control", "variant"}
