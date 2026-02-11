"""Utility helpers for text normalization and style analysis."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from statistics import mean, pstdev

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
MULTISPACE_RE = re.compile(r"\s+")
CONTRACTION_RE = re.compile(
    r"\b(?:\w+'(?:t|s|re|ve|ll|d|m)|(?:can't|won't|ain't|n't))\b",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [MULTISPACE_RE.sub(" ", chunk).strip() for chunk in text.split("\n\n")]
    return "\n\n".join([p for p in paragraphs if p])


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks."""
    text = normalize_whitespace(text)
    if not text:
        return []
    chunks = SENTENCE_RE.split(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def tokenize_words(text: str) -> list[str]:
    """Extract lowercase words."""
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def char_ngrams(text: str, n: int = 3) -> Counter[str]:
    """Return character n-gram counts."""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    if len(cleaned) < n:
        return Counter([cleaned]) if cleaned else Counter()
    grams = [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]
    return Counter(grams)


def cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    """Cosine similarity between sparse vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0) for k in a)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Set-based lexical overlap."""
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def safe_div(numerator: float, denominator: float) -> float:
    """Division helper that returns 0 for zero denominator."""
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class StyleProfile:
    """Distributional text style features used for verification."""

    avg_sentence_length: float
    sentence_length_std: float
    lexical_diversity: float
    contraction_ratio: float
    comma_rate: float
    exclamation_rate: float
    question_rate: float


def style_profile(text: str) -> StyleProfile:
    """Compute a simple style profile from raw text."""
    sentences = split_sentences(text)
    words = tokenize_words(text)
    sentence_lengths = [len(tokenize_words(sentence)) for sentence in sentences if sentence.strip()]

    if sentence_lengths:
        avg_len = mean(sentence_lengths)
        std_len = pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    else:
        avg_len = 0.0
        std_len = 0.0

    lexical_div = safe_div(len(set(words)), len(words))
    contractions = len(CONTRACTION_RE.findall(text))
    contraction_ratio = safe_div(contractions, len(words))

    sentence_count = len(sentences) or 1
    comma_rate = text.count(",") / sentence_count
    exclamation_rate = text.count("!") / sentence_count
    question_rate = text.count("?") / sentence_count

    return StyleProfile(
        avg_sentence_length=avg_len,
        sentence_length_std=std_len,
        lexical_diversity=lexical_div,
        contraction_ratio=contraction_ratio,
        comma_rate=comma_rate,
        exclamation_rate=exclamation_rate,
        question_rate=question_rate,
    )
