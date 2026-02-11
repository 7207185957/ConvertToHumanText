"""Rule-based text humanization engine."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .text_utils import split_sentences, style_profile, tokenize_words

FORMAL_PHRASE_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\bmoreover\b", "also"),
    (r"\bfurthermore\b", "also"),
    (r"\bin addition\b", "also"),
    (r"\btherefore\b", "so"),
    (r"\bthus\b", "so"),
    (r"\bhowever\b", "still"),
    (r"\butilize\b", "use"),
    (r"\bapproximately\b", "about"),
    (r"\bsufficient\b", "enough"),
    (r"\bassist\b", "help"),
    (r"\bregarding\b", "about"),
    (r"\bendeavor\b", "try"),
    (r"\bcommence\b", "start"),
    (r"\bprior to\b", "before"),
    (r"\bsubsequent to\b", "after"),
]

CONTRACTION_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\bdo not\b", "don't"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bdid not\b", "didn't"),
    (r"\bcannot\b", "can't"),
    (r"\bcan not\b", "can't"),
    (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"),
    (r"\bwas not\b", "wasn't"),
    (r"\bwere not\b", "weren't"),
    (r"\bhave not\b", "haven't"),
    (r"\bhas not\b", "hasn't"),
    (r"\bhad not\b", "hadn't"),
    (r"\bwill not\b", "won't"),
    (r"\bwould not\b", "wouldn't"),
    (r"\bshould not\b", "shouldn't"),
    (r"\bcould not\b", "couldn't"),
    (r"\bit is\b", "it's"),
    (r"\bthat is\b", "that's"),
    (r"\bthere is\b", "there's"),
    (r"\bi am\b", "I'm"),
    (r"\bi have\b", "I've"),
    (r"\byou are\b", "you're"),
    (r"\bthey are\b", "they're"),
    (r"\bwe are\b", "we're"),
]

LEADING_TRANSITION_RE = re.compile(
    r"^(Additionally|Furthermore|Moreover|Consequently|Therefore|Hence|In conclusion)\s*,?\s+",
    re.IGNORECASE,
)

CONNECTORS = [
    "Honestly,",
    "In practice,",
    "That said,",
    "At the same time,",
]


@dataclass(frozen=True)
class HumanizationConfig:
    """Configuration for style tuning."""

    target_sentence_length: float = 16.0
    enable_contractions: bool = True
    add_light_variation: bool = True


def _preserve_case(replacement: str, source: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


class RuleBasedHumanizer:
    """Deterministic rewrites that reduce machine-like wording."""

    def __init__(self, seed: int = 7) -> None:
        self._random = random.Random(seed)

    def humanize(
        self,
        text: str,
        config: HumanizationConfig | None = None,
    ) -> tuple[str, float]:
        """Convert AI-like prose into a more conversational style."""
        cfg = config or HumanizationConfig()
        sentences = split_sentences(text)
        if not sentences:
            return "", 0.0

        rewritten: list[str] = []
        for idx, sentence in enumerate(sentences):
            sentence = self._rewrite_sentence(sentence, idx, cfg)
            rewritten.extend(self._split_if_needed(sentence, cfg.target_sentence_length))

        rewritten = self._merge_short_sentences(rewritten, cfg.target_sentence_length)
        result = " ".join(rewritten).strip()
        confidence = self._human_likeness_confidence(result)
        return result, confidence

    def _rewrite_sentence(self, sentence: str, index: int, config: HumanizationConfig) -> str:
        sentence = sentence.strip()
        sentence = LEADING_TRANSITION_RE.sub("", sentence)

        for pattern, replacement in FORMAL_PHRASE_REPLACEMENTS:
            sentence = re.sub(
                pattern,
                lambda m: _preserve_case(replacement, m.group(0)),
                sentence,
                flags=re.IGNORECASE,
            )

        if config.enable_contractions:
            for pattern, replacement in CONTRACTION_REPLACEMENTS:
                sentence = re.sub(
                    pattern,
                    lambda m: _preserve_case(replacement, m.group(0)),
                    sentence,
                    flags=re.IGNORECASE,
                )

        sentence = re.sub(r"\s+", " ", sentence).strip()
        if config.add_light_variation and index % 4 == 2:
            connector = CONNECTORS[(index // 2) % len(CONNECTORS)]
            if not sentence.lower().startswith(connector.lower()):
                sentence = f"{connector} {sentence[:1].lower()}{sentence[1:]}" if sentence else sentence

        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        if sentence:
            sentence = sentence[:1].upper() + sentence[1:]
        return sentence

    def _split_if_needed(self, sentence: str, target_sentence_length: float) -> list[str]:
        words = tokenize_words(sentence)
        if len(words) <= max(12, int(target_sentence_length * 1.6)):
            return [sentence]

        for separator in [",", ";", " and ", " but "]:
            if separator in sentence:
                if separator in {",", ";"}:
                    left, right = sentence.split(separator, 1)
                else:
                    left, right = sentence.split(separator, 1)
                    right = right[:1].upper() + right[1:]
                left = left.strip()
                right = right.strip()
                if left and left[-1] not in ".!?":
                    left += "."
                if right and right[-1] not in ".!?":
                    right += "."
                return [left, right]
        return [sentence]

    def _merge_short_sentences(self, sentences: list[str], target_sentence_length: float) -> list[str]:
        if len(sentences) < 2:
            return sentences

        merged: list[str] = []
        idx = 0
        min_len = max(7, int(target_sentence_length * 0.55))
        while idx < len(sentences):
            current = sentences[idx]
            cur_len = len(tokenize_words(current))
            if idx < len(sentences) - 1 and cur_len < min_len:
                nxt = sentences[idx + 1]
                merged_sentence = current.rstrip(".!?") + ", and " + nxt[:1].lower() + nxt[1:]
                if merged_sentence and merged_sentence[-1] not in ".!?":
                    merged_sentence += "."
                merged.append(merged_sentence)
                idx += 2
            else:
                merged.append(current)
                idx += 1
        return merged

    def _human_likeness_confidence(self, text: str) -> float:
        profile = style_profile(text)
        sentences = split_sentences(text)
        words = tokenize_words(text)
        if not sentences or not words:
            return 0.0

        diversity_score = self._bounded_score(profile.lexical_diversity, low=0.38, high=0.80)
        sentence_len_score = self._bounded_score(profile.avg_sentence_length, low=9.0, high=23.0)
        burstiness_score = self._bounded_score(profile.sentence_length_std, low=2.2, high=11.0)
        contraction_score = self._bounded_score(profile.contraction_ratio, low=0.01, high=0.09)

        repetition_penalty = self._repetition_penalty(words)
        final_score = (
            0.32 * diversity_score
            + 0.26 * sentence_len_score
            + 0.22 * burstiness_score
            + 0.20 * contraction_score
            - repetition_penalty
        )
        return max(0.0, min(100.0, final_score))

    @staticmethod
    def _bounded_score(value: float, low: float, high: float) -> float:
        if value < low:
            return max(0.0, 100.0 - ((low - value) / max(low, 1e-6)) * 100.0)
        if value > high:
            return max(0.0, 100.0 - ((value - high) / max(high, 1e-6)) * 100.0)
        midpoint = (low + high) / 2
        span = max((high - low) / 2, 1e-6)
        return max(70.0, 100.0 - abs(value - midpoint) / span * 30.0)

    @staticmethod
    def _repetition_penalty(words: list[str]) -> float:
        if len(words) < 8:
            return 0.0
        repeated = 0
        for idx in range(len(words) - 2):
            if words[idx] == words[idx + 1] == words[idx + 2]:
                repeated += 1
        return min(20.0, repeated * 6.0)

    def tune_config_for_reference(
        self,
        reference_avg_sentence_length: float,
        reference_contraction_ratio: float,
    ) -> HumanizationConfig:
        """Build a config tuned to style features from reference text."""
        target_len = max(9.0, min(24.0, reference_avg_sentence_length))
        enable_contractions = reference_contraction_ratio >= 0.01
        return HumanizationConfig(
            target_sentence_length=target_len,
            enable_contractions=enable_contractions,
            add_light_variation=True,
        )
