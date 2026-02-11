from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from humanizer_agent.agent import AITextHumanizationAgent
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


if __name__ == "__main__":
    unittest.main()
