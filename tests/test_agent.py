from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from humanizer_agent.agent import AITextHumanizationAgent
from humanizer_agent.external_verification import ExternalDetectorsVerifier
from humanizer_agent.references import ReferenceLoader
from humanizer_agent.verification import TextVerifier


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ai_text = (
            "Moreover, it is important that we do not utilize complex terminology. "
            "Therefore, this approach is sufficient for most users."
        )
        self.reference_examples = [
            "I don't use heavy language when simple wording works.",
            "This is usually enough for most people, so I keep it practical.",
        ]

    def test_convert_without_references(self) -> None:
        agent = AITextHumanizationAgent(min_verification_score=99.9)
        output = agent.convert(self.ai_text)
        self.assertTrue(output.humanized_text)
        self.assertNotEqual(output.humanized_text, self.ai_text)
        self.assertIsNone(output.verification)

    def test_convert_with_inline_references(self) -> None:
        agent = AITextHumanizationAgent(min_verification_score=60.0)
        output = agent.convert(self.ai_text, reference_examples=self.reference_examples)
        self.assertIsNotNone(output.verification)
        assert output.verification is not None
        self.assertGreaterEqual(output.verification.score, 0.0)
        self.assertLessEqual(output.verification.score, 100.0)

    def test_reference_loader_from_text_file(self) -> None:
        loader = ReferenceLoader()
        fixture = Path("tests/fixtures/reference_examples.txt")
        refs = loader.load_from_source(fixture)
        self.assertGreaterEqual(len(refs), 2)

    def test_reference_loader_from_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "refs.json"
            path.write_text(json.dumps(self.reference_examples), encoding="utf-8")
            refs = ReferenceLoader().load_from_source(path)
            self.assertEqual(refs, self.reference_examples)

    def test_verifier_scores_identical_reference_high(self) -> None:
        verifier = TextVerifier()
        text = "I don't think this needs heavy wording, so I kept it simple."
        result = verifier.verify(text, [text], min_score=90.0)
        self.assertGreaterEqual(result.score, 90.0)
        self.assertTrue(result.passed)

    def test_convert_with_external_provider_summary(self) -> None:
        def fake_post(
            _url: str, _headers: dict[str, str], _payload: dict[str, object], _timeout: float
        ) -> tuple[int, dict[str, object]]:
            return 200, {"human_probability": 0.88, "ai_probability": 0.12}

        external_verifier = ExternalDetectorsVerifier(http_post=fake_post)
        agent = AITextHumanizationAgent(
            min_verification_score=10.0,
            external_providers=["zerogpt"],
            require_external_verification=True,
            external_verifier=external_verifier,
        )
        with mock.patch.dict(
            os.environ,
            {"HUMANIZER_ZEROGPT_API_URL": "https://example.test/zerogpt"},
            clear=True,
        ):
            output = agent.convert(self.ai_text, reference_examples=self.reference_examples)

        self.assertIsNotNone(output.external_verification)
        assert output.external_verification is not None
        self.assertEqual(output.external_verification.available_count, 1)
        self.assertTrue(output.external_verification.passed)


if __name__ == "__main__":
    unittest.main()
