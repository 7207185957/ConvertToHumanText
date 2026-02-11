from __future__ import annotations

import os
import unittest
from unittest import mock

from humanizer_agent.external_verification import ExternalDetectorsVerifier


class ExternalVerificationTests(unittest.TestCase):
    def test_unconfigured_provider_returns_not_configured(self) -> None:
        verifier = ExternalDetectorsVerifier(http_post=self._fake_post)
        with mock.patch.dict(os.environ, {}, clear=True):
            summary = verifier.verify(
                text="Simple text",
                providers=["zerogpt"],
                min_human_probability=0.5,
            )
        self.assertFalse(summary.passed)
        self.assertEqual(summary.configured_count, 0)
        self.assertEqual(summary.available_count, 0)
        self.assertEqual(len(summary.provider_results), 1)
        self.assertFalse(summary.provider_results[0].configured)

    def test_parses_probabilities_and_passes_threshold(self) -> None:
        def post(_url: str, _headers: dict[str, str], _payload: dict[str, object], _timeout: float):
            return 200, {"analysis": {"human_probability": "87%", "ai_probability": "13%"}}

        verifier = ExternalDetectorsVerifier(http_post=post)
        with mock.patch.dict(
            os.environ,
            {"HUMANIZER_ZEROGPT_API_URL": "https://example.test/zerogpt"},
            clear=True,
        ):
            summary = verifier.verify(
                text="Simple text",
                providers=["zerogpt"],
                min_human_probability=0.7,
            )

        self.assertTrue(summary.passed)
        self.assertEqual(summary.available_count, 1)
        self.assertAlmostEqual(summary.aggregate_human_probability or 0.0, 0.87, places=6)
        self.assertAlmostEqual(summary.aggregate_ai_probability or 0.0, 0.13, places=6)
        provider = summary.provider_results[0]
        self.assertTrue(provider.passed)
        self.assertEqual(provider.response_status, 200)

    def test_uses_explicit_dot_path_keys(self) -> None:
        def post(_url: str, _headers: dict[str, str], _payload: dict[str, object], _timeout: float):
            return 200, {"result": {"scores": {"human": 0.64, "ai": 0.36}, "meta": {"pass": True}}}

        verifier = ExternalDetectorsVerifier(http_post=post)
        with mock.patch.dict(
            os.environ,
            {
                "HUMANIZER_GPTZERO_API_URL": "https://example.test/gptzero",
                "HUMANIZER_GPTZERO_HUMAN_SCORE_KEY": "result.scores.human",
                "HUMANIZER_GPTZERO_AI_SCORE_KEY": "result.scores.ai",
                "HUMANIZER_GPTZERO_PASS_KEY": "result.meta.pass",
            },
            clear=True,
        ):
            summary = verifier.verify(
                text="Simple text",
                providers=["gptzero"],
                min_human_probability=0.9,
            )

        self.assertTrue(summary.passed)
        provider = summary.provider_results[0]
        self.assertAlmostEqual(provider.human_probability or 0.0, 0.64, places=6)
        self.assertAlmostEqual(provider.ai_probability or 0.0, 0.36, places=6)
        self.assertTrue(provider.passed)

    @staticmethod
    def _fake_post(
        _url: str, _headers: dict[str, str], _payload: dict[str, object], _timeout: float
    ) -> tuple[int, dict[str, object]]:
        return 200, {}


if __name__ == "__main__":
    unittest.main()
