"""Reference loading utilities for verification."""

from __future__ import annotations

import json
import re
from pathlib import Path


class ReferenceLoadError(RuntimeError):
    """Raised when reference examples cannot be loaded."""


class ReferenceLoader:
    """Load reference examples from text, JSON, or image sources."""

    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

    def load_from_source(self, source: str | Path) -> list[str]:
        path = Path(source)
        if not path.exists():
            raise ReferenceLoadError(f"Reference source not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return self._extract_examples(path.read_text(encoding="utf-8"))
        if suffix == ".json":
            return self._load_json(path)
        if suffix in self.IMAGE_SUFFIXES:
            raw_text = self._load_from_image(path)
            return self._extract_examples(raw_text)

        raise ReferenceLoadError(
            f"Unsupported reference format '{suffix}'. Use text, JSON, or image files."
        )

    def load_from_inline_text(self, text: str) -> list[str]:
        """Split inline content into usable verification examples."""
        return self._extract_examples(text)

    def _load_json(self, path: Path) -> list[str]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReferenceLoadError(f"Invalid JSON in {path}: {exc}") from exc

        if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
            return [line.strip() for line in payload if line.strip()]

        raise ReferenceLoadError(
            f"JSON references in {path} must be an array of strings."
        )

    def _load_from_image(self, path: Path) -> str:
        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
        except ImportError as exc:
            raise ReferenceLoadError(
                "Image verification requires optional OCR dependencies. "
                "Install with: pip install .[ocr]"
            ) from exc

        image = Image.open(path)
        try:
            text = pytesseract.image_to_string(image)
        finally:
            image.close()

        if not text.strip():
            raise ReferenceLoadError(
                f"OCR extracted no readable text from image: {path}"
            )
        return text

    @staticmethod
    def _extract_examples(text: str) -> list[str]:
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Drop bullets or numbering prefixes from OCR/list formats.
            line = re.sub(r"^[\-\*\u2022]+\s*", "", line)
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            if line:
                lines.append(line)

        if not lines:
            raise ReferenceLoadError("No reference examples found in source text.")
        return lines
