from typing import Dict, Any, Tuple
import os
import hashlib
import random


def _parse_split(split: str) -> Dict[str, int]:
    parts = [p.strip() for p in (split or "").split(",") if p.strip()]
    out: Dict[str, int] = {}
    for p in parts:
        if ":" in p:
            name, pct = p.split(":", 1)
            try:
                out[name.strip()] = int(pct.strip())
            except Exception:
                continue
    if not out:
        out = {"control": 50, "variant": 50}
    total = sum(out.values())
    if total and total != 100:
        factor = 100 / total
        out = {k: int(v * factor) for k, v in out.items()}
        delta = 100 - sum(out.values())
        if delta != 0:
            first_key = next(iter(out.keys()))
            out[first_key] += delta
    return out


def _sticky_hash(key: str) -> int:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100


def choose_variant(context: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
    split_raw = os.getenv("DECISION_AB_SPLIT", "control:50,variant:50")
    mapping = _parse_split(split_raw)
    sticky_by = os.getenv("DECISION_AB_STICKY_BY", "submission_id")
    force = os.getenv("DECISION_AB_FORCE")

    if force and force in mapping:
        return force, mapping

    key = str(context.get(sticky_by, "")) if sticky_by else ""
    if not key:
        variants = list(mapping.keys())
        weights = [mapping[v] for v in variants]
        total = sum(weights)
        r = random.randint(1, total)
        acc = 0
        for v, w in zip(variants, weights):
            acc += w
            if r <= acc:
                return v, mapping
        return variants[0], mapping

    bucket = _sticky_hash(key)
    acc = 0
    for name, pct in mapping.items():
        acc += pct
        if bucket < acc:
            return name, mapping
    return next(iter(mapping.keys())), mapping
