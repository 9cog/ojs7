from typing import Dict, Any, List, Tuple, Optional
import math
import re


def _word_count(text: str) -> int:
    tokens = re.findall(r"\b\w+\b", text or "")
    return len(tokens)


def _sentence_count(text: str) -> int:
    sentences = re.split(r"[.!?]+", text or "")
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)


def _syllable_count(word: str) -> int:
    word = word.lower()
    if len(word) <= 3:
        return 1
    return len(re.findall(r"[aeiouy]+", word)) or 1


def _flesch_kincaid(text: str) -> float:
    words = _word_count(text)
    sentences = max(_sentence_count(text), 1)
    syllables = sum(_syllable_count(w) for w in re.findall(r"\b\w+\b", text or ""))
    if words == 0:
        return 0.0
    asl = words / sentences
    asw = syllables / words
    return 206.835 - (1.015 * asl) - (84.6 * asw)


def _keyword_coverage(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    text_l = (text or "").lower()
    hits = sum(1 for k in keywords if k.lower() in text_l)
    return hits / len(keywords)


def basic_manuscript_features(manuscript: Dict[str, Any], scope_keywords: Optional[List[str]] = None) -> Dict[str, float]:
    title = manuscript.get("title", "") or ""
    abstract = manuscript.get("abstract", "") or ""
    body = manuscript.get("body", "") or ""

    joined = " ".join([title, abstract, body]).strip()
    wc = _word_count(joined)
    sc = _sentence_count(joined)
    fk = _flesch_kincaid(joined)
    kw = _keyword_coverage(joined, scope_keywords or [])

    citations = manuscript.get("citations", [])
    citation_count = len(citations) if isinstance(citations, list) else 0

    review_stats = manuscript.get("review_stats", {}) or {}
    avg_reviewer_score = float(review_stats.get("avg_score", 0.0))
    num_reviews = int(review_stats.get("count", 0))

    return {
        "word_count": float(wc),
        "sentence_count": float(sc),
        "readability_fk": float(fk),
        "keyword_coverage": float(kw),
        "citation_count": float(citation_count),
        "avg_reviewer_score": float(avg_reviewer_score),
        "num_reviews": float(num_reviews),
    }


def to_vector(feature_map: Dict[str, float], ordered_keys: List[str]) -> List[float]:
    return [float(feature_map.get(k, 0.0)) for k in ordered_keys]


FEATURE_KEYS = [
    "word_count",
    "sentence_count",
    "readability_fk",
    "keyword_coverage",
    "citation_count",
    "avg_reviewer_score",
    "num_reviews",
]
