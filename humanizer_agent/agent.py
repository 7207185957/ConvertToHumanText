"""End-to-end agent orchestration for text humanization and verification."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .external_verification import ExternalDetectorsVerifier, ExternalVerificationSummary
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
    external_verification: ExternalVerificationSummary | None

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
        if self.external_verification is not None:
            payload["external_verification"] = self.external_verification.to_dict()
        else:
            payload["external_verification"] = None
        return payload


class AITextHumanizationAgent:
    """Agent that rewrites AI text and verifies it against reference examples."""

    def __init__(
        self,
        min_verification_score: float = 99.9,
        max_iterations: int = 4,
        seed: int = 7,
        external_providers: list[str] | None = None,
        external_min_human_probability: float = 0.5,
        require_external_verification: bool = False,
        external_timeout_seconds: float = 18.0,
        external_verifier: ExternalDetectorsVerifier | None = None,
    ) -> None:
        self.min_verification_score = min_verification_score
        self.max_iterations = max_iterations
        self.humanizer = RuleBasedHumanizer(seed=seed)
        self.verifier = TextVerifier()
        self.reference_loader = ReferenceLoader()
        self.external_providers = external_providers or []
        self.external_min_human_probability = max(0.0, min(1.0, external_min_human_probability))
        self.require_external_verification = require_external_verification
        self.external_timeout_seconds = max(1.0, external_timeout_seconds)
        self.external_verifier = external_verifier or ExternalDetectorsVerifier()

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

        return self._convert_without_reference(ai_generated_text)

    def _convert_without_reference(self, ai_generated_text: str) -> HumanizationOutput:
        if not self.external_providers:
            humanized_text, confidence = self.humanizer.humanize(ai_generated_text)
            return HumanizationOutput(
                original_text=ai_generated_text,
                humanized_text=humanized_text,
                human_likeness_confidence=confidence,
                attempts=1,
                verification=None,
                external_verification=None,
            )

        configs = self._iteration_configs(HumanizationConfig())
        best_text = ""
        best_confidence = 0.0
        best_external: ExternalVerificationSummary | None = None
        best_rank = float("-inf")

        for attempt_idx, config in enumerate(configs, start=1):
            humanized_text, confidence = self.humanizer.humanize(ai_generated_text, config=config)
            external = self._run_external_verification(humanized_text)
            rank = self._candidate_rank(0.0, external)

            if rank > best_rank:
                best_rank = rank
                best_text = humanized_text
                best_confidence = confidence
                best_external = external

            if (
                self.require_external_verification
                and external is not None
                and external.passed
            ):
                return HumanizationOutput(
                    original_text=ai_generated_text,
                    humanized_text=humanized_text,
                    human_likeness_confidence=confidence,
                    attempts=attempt_idx,
                    verification=None,
                    external_verification=external,
                )

        return HumanizationOutput(
            original_text=ai_generated_text,
            humanized_text=best_text,
            human_likeness_confidence=best_confidence,
            attempts=len(configs),
            verification=None,
            external_verification=best_external,
        )

    def _convert_with_verification(
        self, ai_generated_text: str, reference_examples: list[str]
    ) -> HumanizationOutput:
        reference_profiles = [style_profile(ref) for ref in reference_examples]
        target_len = mean([profile.avg_sentence_length for profile in reference_profiles]) or 16.0
        target_contractions = mean([profile.contraction_ratio for profile in reference_profiles])
        tuned = self.humanizer.tune_config_for_reference(target_len, target_contractions)

        candidate_best = ""
        confidence_best = 0.0
        verification_best: VerificationResult | None = None
        external_best: ExternalVerificationSummary | None = None
        best_rank = float("-inf")

        configs = self._iteration_configs(tuned)
        for attempt_idx, config in enumerate(configs, start=1):
            humanized_text, confidence = self.humanizer.humanize(ai_generated_text, config=config)
            verification = self.verifier.verify(
                humanized_text,
                reference_examples=reference_examples,
                min_score=self.min_verification_score,
            )
            external = self._run_external_verification(humanized_text)
            rank = self._candidate_rank(verification.score, external)

            if rank > best_rank:
                best_rank = rank
                candidate_best = humanized_text
                confidence_best = confidence
                verification_best = verification
                external_best = external

            local_pass = verification.passed
            external_pass = (
                (external is not None and external.passed)
                if self.require_external_verification and self.external_providers
                else True
            )
            if local_pass and external_pass:
                return HumanizationOutput(
                    original_text=ai_generated_text,
                    humanized_text=humanized_text,
                    human_likeness_confidence=confidence,
                    attempts=attempt_idx,
                    verification=verification,
                    external_verification=external,
                )

        return HumanizationOutput(
            original_text=ai_generated_text,
            humanized_text=candidate_best,
            human_likeness_confidence=confidence_best,
            attempts=len(configs),
            verification=verification_best,
            external_verification=external_best,
        )

    def _run_external_verification(
        self, candidate_text: str
    ) -> ExternalVerificationSummary | None:
        if not self.external_providers:
            return None
        return self.external_verifier.verify(
            text=candidate_text,
            providers=self.external_providers,
            min_human_probability=self.external_min_human_probability,
            timeout_seconds=self.external_timeout_seconds,
        )

    @staticmethod
    def _candidate_rank(
        local_verification_score: float,
        external_verification: ExternalVerificationSummary | None,
    ) -> float:
        external_score = 0.0
        if (
            external_verification is not None
            and external_verification.aggregate_human_probability is not None
        ):
            external_score = external_verification.aggregate_human_probability * 100.0
        return (0.7 * local_verification_score) + (0.3 * external_score)

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
