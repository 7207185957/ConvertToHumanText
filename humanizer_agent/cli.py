"""Command line interface for AI text humanization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import AITextHumanizationAgent
from .external_verification import DEFAULT_PROVIDER_IDS
from .references import ReferenceLoadError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AI-generated text into human-like text and verify "
            "against reference examples."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Raw AI-generated text input.")
    input_group.add_argument("--input-file", help="Path to a file with AI-generated text.")

    parser.add_argument(
        "--reference-source",
        help="Path to reference examples (txt, md, json, png, jpg, jpeg, webp, gif).",
    )
    parser.add_argument(
        "--references-text",
        help="Inline references as newline-separated examples.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=99.9,
        help="Minimum verification score required to pass. Default: 99.9",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=4,
        help="Number of rewrite attempts. Default: 4",
    )
    parser.add_argument(
        "--external-providers",
        help=(
            "Comma-separated detectors to call externally "
            f"(supported: {', '.join(DEFAULT_PROVIDER_IDS)})."
        ),
    )
    parser.add_argument(
        "--external-human-threshold",
        type=float,
        default=0.5,
        help="Minimum per-provider human probability for external pass. Default: 0.5",
    )
    parser.add_argument(
        "--require-external-pass",
        action="store_true",
        help="Require external providers to pass before considering output verified.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional path to save only the humanized text.",
    )
    parser.add_argument(
        "--report-file",
        help="Optional path to save the JSON report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON output to stdout instead of text-only output.",
    )
    return parser


def _read_input_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text.strip()
    path = Path(args.input_file)
    return path.read_text(encoding="utf-8").strip()


def _parse_external_providers(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        input_text = _read_input_text(args)
    except FileNotFoundError:
        print(f"Input file not found: {args.input_file}", file=sys.stderr)
        raise SystemExit(1)

    if not input_text:
        print("Input text is empty.", file=sys.stderr)
        raise SystemExit(1)

    agent = AITextHumanizationAgent(
        min_verification_score=args.min_score,
        max_iterations=max(1, args.max_iterations),
        external_providers=_parse_external_providers(args.external_providers),
        external_min_human_probability=max(0.0, min(1.0, args.external_human_threshold)),
        require_external_verification=args.require_external_pass,
    )
    try:
        output = agent.convert(
            ai_generated_text=input_text,
            reference_source=args.reference_source,
            inline_references_text=args.references_text,
        )
    except (ReferenceLoadError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

    report = output.to_dict()

    if args.output_file:
        Path(args.output_file).write_text(output.humanized_text + "\n", encoding="utf-8")
    if args.report_file:
        Path(args.report_file).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(output.humanized_text)
    if output.verification:
        status = "PASS" if output.verification.passed else "FAIL"
        print(
            f"\nVerification: {status} | "
            f"score={output.verification.score:.3f} "
            f"(threshold={output.verification.threshold:.3f})",
            file=sys.stderr,
        )
        print(
            f"Human-likeness confidence: {output.human_likeness_confidence:.2f}",
            file=sys.stderr,
        )
    if output.external_verification:
        status = "PASS" if output.external_verification.passed else "FAIL"
        print(
            f"External Verification: {status} | "
            f"providers={len(output.external_verification.requested_providers)} "
            f"configured={output.external_verification.configured_count} "
            f"available={output.external_verification.available_count}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
