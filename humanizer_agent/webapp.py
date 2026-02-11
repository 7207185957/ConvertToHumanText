"""Flask UI for split-pane AI text humanization."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request

from .agent import AITextHumanizationAgent
from .external_verification import DEFAULT_PROVIDER_IDS, SUPPORTED_PROVIDERS
from .references import ReferenceLoadError


def create_app() -> Flask:
    """Create Flask application instance."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = "humanizer-agent-local-dev"

    provider_options = [
        {"id": item.id, "label": item.label} for item in SUPPORTED_PROVIDERS
    ]

    @app.get("/")
    def index_get() -> str:
        return render_template(
            "index.html",
            input_text="",
            output_text="",
            references_text="",
            min_score=99.9,
            max_iterations=4,
            external_human_threshold=0.5,
            require_external_verification=True,
            external_providers_selected=list(DEFAULT_PROVIDER_IDS),
            provider_options=provider_options,
            report=None,
            error=None,
            success=False,
        )

    @app.post("/")
    def index_post() -> str:
        input_text = request.form.get("input_text", "").strip()
        references_text = request.form.get("references_text", "").strip()

        min_score = _parse_float(request.form.get("min_score"), default=99.9)
        max_iterations = _parse_int(request.form.get("max_iterations"), default=4, minimum=1)
        external_human_threshold = _clamp_float(
            _parse_float(request.form.get("external_human_threshold"), default=0.5),
            minimum=0.0,
            maximum=1.0,
        )
        require_external_verification = request.form.get("require_external_verification") == "on"
        external_providers_selected = _sanitize_providers(
            request.form.getlist("external_providers")
        )

        output_text = ""
        report: dict[str, Any] | None = None
        error: str | None = None
        success = False
        temp_reference_path: str | None = None

        if not input_text:
            error = "Please enter AI-generated text in the left panel."
            return render_template(
                "index.html",
                input_text=input_text,
                output_text=output_text,
                references_text=references_text,
                min_score=min_score,
                max_iterations=max_iterations,
                external_human_threshold=external_human_threshold,
                require_external_verification=require_external_verification,
                external_providers_selected=external_providers_selected,
                provider_options=provider_options,
                report=report,
                error=error,
                success=success,
            )

        uploaded_reference = request.files.get("reference_file")
        if uploaded_reference and uploaded_reference.filename:
            suffix = Path(uploaded_reference.filename).suffix.lower() or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                uploaded_reference.save(handle.name)
                temp_reference_path = handle.name

        try:
            agent = AITextHumanizationAgent(
                min_verification_score=min_score,
                max_iterations=max_iterations,
                external_providers=external_providers_selected,
                external_min_human_probability=external_human_threshold,
                require_external_verification=require_external_verification,
            )
            output = agent.convert(
                ai_generated_text=input_text,
                reference_source=temp_reference_path,
                inline_references_text=references_text or None,
            )
            output_text = output.humanized_text
            report = output.to_dict()
            success = True
        except (ReferenceLoadError, ValueError) as exc:
            error = str(exc)
        finally:
            if temp_reference_path:
                _safe_unlink(temp_reference_path)

        return render_template(
            "index.html",
            input_text=input_text,
            output_text=output_text,
            references_text=references_text,
            min_score=min_score,
            max_iterations=max_iterations,
            external_human_threshold=external_human_threshold,
            require_external_verification=require_external_verification,
            external_providers_selected=external_providers_selected,
            provider_options=provider_options,
            report=report,
            error=error,
            success=success,
        )

    return app


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_int(raw: str | None, default: int, minimum: int) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _clamp_float(raw: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, raw))


def _sanitize_providers(values: list[str]) -> list[str]:
    supported = {item.id for item in SUPPORTED_PROVIDERS}
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if normalized not in supported:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
