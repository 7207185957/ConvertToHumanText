"""End-to-end agent orchestration for text humanization and verification."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .humanizer import HumanizationConfig, RuleBasedHumanizer
from .references import ReferenceLoader
from .text_utils import style_profile
from .verification import TextVerifier, VerificationResult


@dataclass(frozen=True)
class HumanizationOutput:
    """Full output from a humanization run."""

    original_text: str
    humanized_text: str
    human_likeness_confidence: float
    attempts: int
    verification: VerificationResult | None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "original_text": self.original_text,
            "humanized_text": self.humanized_text,
            "human_likeness_confidence": round(self.human_likeness_confidence, 3),
            "attempts": self.attempts,
        }
        if self.verification is not None:
            payload["verification"] = self.verification.to_dict()
        else:
            payload["verification"] = None
        return payload


class AITextHumanizationAgent:
    """Agent that rewrites AI text and verifies it against reference examples."""

    def __init__(
        self,
        min_verification_score: float = 99.9,
        max_iterations: int = 4,
        seed: int = 7,
    ) -> None:
        self.min_verification_score = min_verification_score
        self.max_iterations = max_iterations
        self.humanizer = RuleBasedHumanizer(seed=seed)
        self.verifier = TextVerifier()
        self.reference_loader = ReferenceLoader()

    def convert(
        self,
        ai_generated_text: str,
        reference_examples: list[str] | None = None,
        reference_source: str | None = None,
        inline_references_text: str | None = None,
    ) -> HumanizationOutput:
        """Humanize text and verify against a reference collection.

        Args:
            ai_generated_text: Input text that should sound more human.
            reference_examples: Optional list of reference examples.
            reference_source: Optional path to references (txt/json/image).
            inline_references_text: Optional multiline references text.
        """
        ai_generated_text = ai_generated_text.strip()
        if not ai_generated_text:
            raise ValueError("Input text is empty.")

        refs = list(reference_examples or [])
        if reference_source:
            refs.extend(self.reference_loader.load_from_source(reference_source))
        if inline_references_text:
            refs.extend(self.reference_loader.load_from_inline_text(inline_references_text))

        # Remove empty duplicates while preserving order.
        unique_refs: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            cleaned = ref.strip()
            if cleaned and cleaned not in seen:
                unique_refs.append(cleaned)
                seen.add(cleaned)

        if unique_refs:
            return self._convert_with_verification(ai_generated_text, unique_refs)

        humanized_text, confidence = self.humanizer.humanize(ai_generated_text)
        return HumanizationOutput(
            original_text=ai_generated_text,
            humanized_text=humanized_text,
            human_likeness_confidence=confidence,
            attempts=1,
            verification=None,
        )

    def _convert_with_verification(
        self, ai_generated_text: str, reference_examples: list[str]
    ) -> HumanizationOutput:
        target_len = mean([style_profile(ref).avg_sentence_length for ref in reference_examples]) or 16.0
        target_contractions = mean(
            [style_profile(ref).contraction_ratio for ref in reference_examples]
        )
        tuned = self.humanizer.tune_config_for_reference(target_len, target_contractions)

        candidate_best = ""
        confidence_best = 0.0
        verification_best: VerificationResult | None = None

        configs = self._iteration_configs(tuned)
        for attempt_idx, config in enumerate(configs, start=1):
            humanized_text, confidence = self.humanizer.humanize(ai_generated_text, config=config)
            verification = self.verifier.verify(
                humanized_text,
                reference_examples=reference_examples,
                min_score=self.min_verification_score,
            )
            if (verification_best is None) or (verification.score > verification_best.score):
                candidate_best = humanized_text
                confidence_best = confidence
                verification_best = verification

            if verification.passed:
                return HumanizationOutput(
                    original_text=ai_generated_text,
                    humanized_text=humanized_text,
                    human_likeness_confidence=confidence,
                    attempts=attempt_idx,
                    verification=verification,
                )

        return HumanizationOutput(
            original_text=ai_generated_text,
            humanized_text=candidate_best,
            human_likeness_confidence=confidence_best,
            attempts=len(configs),
            verification=verification_best,
        )

    def _iteration_configs(self, base: HumanizationConfig) -> list[HumanizationConfig]:
        # Small variations improve chances of matching reference style.
        variations = [
            base,
            HumanizationConfig(
                target_sentence_length=max(9.0, base.target_sentence_length - 2.0),
                enable_contractions=True,
                add_light_variation=True,
            ),
            HumanizationConfig(
                target_sentence_length=min(24.0, base.target_sentence_length + 2.0),
                enable_contractions=base.enable_contractions,
                add_light_variation=True,
            ),
            HumanizationConfig(
                target_sentence_length=base.target_sentence_length,
                enable_contractions=True,
                add_light_variation=False,
            ),
        ]
        return variations[: max(1, self.max_iterations)]
