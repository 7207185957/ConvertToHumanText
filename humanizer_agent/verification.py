"""Verification logic for humanized text outputs."""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass
from statistics import mean

from .text_utils import (
    StyleProfile,
    char_ngrams,
    cosine_similarity,
    jaccard_similarity,
    style_profile,
    tokenize_words,
)


@dataclass(frozen=True)
class VerificationResult:
    """Structured report for reference verification."""

    score: float
    passed: bool
    threshold: float
    reference_count: int
    style_match_score: float
    content_match_score: float
    best_reference_similarity: float
    diagnostics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["score"] = round(self.score, 3)
        payload["style_match_score"] = round(self.style_match_score, 3)
        payload["content_match_score"] = round(self.content_match_score, 3)
        payload["best_reference_similarity"] = round(self.best_reference_similarity, 3)
        payload["diagnostics"] = {
            key: round(value, 4) for key, value in self.diagnostics.items()
        }
        return payload


class TextVerifier:
    """Compare generated text against reference examples."""

    def verify(
        self,
        candidate_text: str,
        reference_examples: list[str],
        min_score: float = 99.9,
    ) -> VerificationResult:
        if not reference_examples:
            raise ValueError("At least one reference example is required for verification.")

        ref_profiles = [style_profile(item) for item in reference_examples]
        reference_profile = self._average_style_profile(ref_profiles)
        candidate_profile = style_profile(candidate_text)

        style_score = self._style_similarity(candidate_profile, reference_profile)
        per_reference_content = [
            self._content_similarity(candidate_text, reference) for reference in reference_examples
        ]
        best_ref_sim = max(per_reference_content) if per_reference_content else 0.0
        content_score = mean(sorted(per_reference_content, reverse=True)[:3]) if per_reference_content else 0.0

        # Heavier weight on style so references can validate writing voice
        # even when content differs.
        total_score = (0.8 * style_score) + (0.2 * content_score)

        diagnostics = {
            "candidate_avg_sentence_length": candidate_profile.avg_sentence_length,
            "reference_avg_sentence_length": reference_profile.avg_sentence_length,
            "candidate_contraction_ratio": candidate_profile.contraction_ratio,
            "reference_contraction_ratio": reference_profile.contraction_ratio,
            "candidate_lexical_diversity": candidate_profile.lexical_diversity,
            "reference_lexical_diversity": reference_profile.lexical_diversity,
        }

        return VerificationResult(
            score=max(0.0, min(100.0, total_score)),
            passed=total_score >= min_score,
            threshold=min_score,
            reference_count=len(reference_examples),
            style_match_score=style_score,
            content_match_score=content_score,
            best_reference_similarity=best_ref_sim,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _average_style_profile(profiles: list[StyleProfile]) -> StyleProfile:
        if not profiles:
            return style_profile("")
        return StyleProfile(
            avg_sentence_length=mean([p.avg_sentence_length for p in profiles]),
            sentence_length_std=mean([p.sentence_length_std for p in profiles]),
            lexical_diversity=mean([p.lexical_diversity for p in profiles]),
            contraction_ratio=mean([p.contraction_ratio for p in profiles]),
            comma_rate=mean([p.comma_rate for p in profiles]),
            exclamation_rate=mean([p.exclamation_rate for p in profiles]),
            question_rate=mean([p.question_rate for p in profiles]),
        )

    def _style_similarity(self, candidate: StyleProfile, reference: StyleProfile) -> float:
        metric_scores = [
            self._tolerance_score(candidate.avg_sentence_length, reference.avg_sentence_length, 8.0),
            self._tolerance_score(candidate.sentence_length_std, reference.sentence_length_std, 6.0),
            self._tolerance_score(candidate.lexical_diversity, reference.lexical_diversity, 0.26),
            self._tolerance_score(candidate.contraction_ratio, reference.contraction_ratio, 0.06),
            self._tolerance_score(candidate.comma_rate, reference.comma_rate, 1.8),
            self._tolerance_score(candidate.exclamation_rate, reference.exclamation_rate, 1.2),
            self._tolerance_score(candidate.question_rate, reference.question_rate, 1.2),
        ]
        return mean(metric_scores)

    @staticmethod
    def _tolerance_score(candidate: float, reference: float, tolerance: float) -> float:
        diff = abs(candidate - reference)
        if tolerance <= 0:
            return 100.0 if diff == 0 else 0.0
        return max(0.0, 100.0 * (1.0 - (diff / tolerance)))

    def _content_similarity(self, candidate_text: str, reference_text: str) -> float:
        candidate_tokens = tokenize_words(candidate_text)
        reference_tokens = tokenize_words(reference_text)
        token_overlap = jaccard_similarity(candidate_tokens, reference_tokens)

        candidate_grams = char_ngrams(candidate_text, n=3)
        reference_grams = char_ngrams(reference_text, n=3)
        trigram_cosine = cosine_similarity(candidate_grams, reference_grams)

        sequence_ratio = difflib.SequenceMatcher(
            None,
            candidate_text.lower(),
            reference_text.lower(),
        ).ratio()

        score = (0.35 * token_overlap) + (0.4 * trigram_cosine) + (0.25 * sequence_ratio)
        return max(0.0, min(100.0, score * 100.0))
