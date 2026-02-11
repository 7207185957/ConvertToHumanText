from __future__ import annotations

import re
import unittest

from humanizer_agent.webapp import create_app


class UITests(unittest.TestCase):
    def setUp(self) -> None:
        app = create_app()
        app.testing = True
        self.client = app.test_client()

    def test_get_homepage_renders(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        payload = response.get_data(as_text=True)
        self.assertIn("AI Text Humanizer + Verification", payload)
        self.assertIn("Input (AI Generated)", payload)
        self.assertIn("Output (Human Generated)", payload)
        self.assertIn("External Detector Sources", payload)
        self.assertIn("ZeroGPT", payload)

    def test_post_humanization_renders_output(self) -> None:
        response = self.client.post(
            "/",
            data={
                "input_text": (
                    "Moreover, it is important that we do not utilize complex language. "
                    "Therefore, this approach is sufficient for users."
                ),
                "references_text": "I keep wording simple.\nThis works for most people.",
                "min_score": "50",
                "max_iterations": "3",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_data(as_text=True)
        self.assertIn("Verification Summary", payload)
        match = re.search(
            r'<textarea\s+id="output_text"[^>]*>(.*?)</textarea>',
            payload,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        assert match is not None
        output_value = match.group(1).strip()
        self.assertTrue(output_value)
        self.assertNotIn("utilize", output_value.lower())

    def test_post_without_input_shows_error(self) -> None:
        response = self.client.post(
            "/",
            data={"input_text": "", "min_score": "99.9", "max_iterations": "4"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_data(as_text=True)
        self.assertIn("Please enter AI-generated text in the left panel.", payload)


if __name__ == "__main__":
    unittest.main()
