"""External detector verification adapters.

This module supports these provider identities:
- zerogpt
- crossplag
- contentatscale
- copyleaks
- openai
- gptzero
- sapling
- writer

Each provider is configurable via environment variables:
  HUMANIZER_<PROVIDER>_API_URL
  HUMANIZER_<PROVIDER>_API_KEY
  HUMANIZER_<PROVIDER>_AUTH_HEADER (optional, default: Authorization)
  HUMANIZER_<PROVIDER>_TEXT_FIELD (optional, default: text)
  HUMANIZER_<PROVIDER>_STATIC_PAYLOAD_JSON (optional)

Optional response key overrides:
  HUMANIZER_<PROVIDER>_AI_SCORE_KEY
  HUMANIZER_<PROVIDER>_HUMAN_SCORE_KEY
  HUMANIZER_<PROVIDER>_PASS_KEY

Score keys can be dot-paths for nested JSON (for example: result.ai_score).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable
from urllib import error, request


@dataclass(frozen=True)
class ExternalProviderSpec:
    """Metadata for one supported provider."""

    id: str
    label: str
    env_prefix: str


SUPPORTED_PROVIDERS: tuple[ExternalProviderSpec, ...] = (
    ExternalProviderSpec("zerogpt", "ZeroGPT", "ZEROGPT"),
    ExternalProviderSpec("crossplag", "Crossplag", "CROSSPLAG"),
    ExternalProviderSpec("contentatscale", "Content at Scale", "CONTENT_AT_SCALE"),
    ExternalProviderSpec("copyleaks", "Copyleaks", "COPYLEAKS"),
    ExternalProviderSpec("openai", "OpenAI", "OPENAI"),
    ExternalProviderSpec("gptzero", "GPTZero", "GPTZERO"),
    ExternalProviderSpec("sapling", "Sapling", "SAPLING"),
    ExternalProviderSpec("writer", "Writer", "WRITER"),
)

DEFAULT_PROVIDER_IDS: tuple[str, ...] = tuple(spec.id for spec in SUPPORTED_PROVIDERS)


@dataclass(frozen=True)
class ExternalProviderResult:
    """Single-provider verification result."""

    provider_id: str
    provider_label: str
    configured: bool
    available: bool
    passed: bool | None
    ai_probability: float | None
    human_probability: float | None
    error: str | None
    response_status: int | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if payload["ai_probability"] is not None:
            payload["ai_probability"] = round(float(payload["ai_probability"]), 4)
        if payload["human_probability"] is not None:
            payload["human_probability"] = round(float(payload["human_probability"]), 4)
        return payload


@dataclass(frozen=True)
class ExternalVerificationSummary:
    """Aggregated result across selected external providers."""

    requested_providers: list[str]
    threshold: float
    passed: bool
    available_count: int
    configured_count: int
    aggregate_human_probability: float | None
    aggregate_ai_probability: float | None
    provider_results: list[ExternalProviderResult]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if payload["aggregate_human_probability"] is not None:
            payload["aggregate_human_probability"] = round(
                float(payload["aggregate_human_probability"]), 4
            )
        if payload["aggregate_ai_probability"] is not None:
            payload["aggregate_ai_probability"] = round(
                float(payload["aggregate_ai_probability"]), 4
            )
        payload["provider_results"] = [item.to_dict() for item in self.provider_results]
        return payload


HTTPPostFn = Callable[
    [str, dict[str, str], dict[str, Any], float],
    tuple[int, dict[str, Any]],
]


class ExternalDetectorsVerifier:
    """Execute detector checks using provider APIs."""

    def __init__(self, http_post: HTTPPostFn | None = None) -> None:
        self._http_post = http_post or _default_http_post
        self._provider_index = {spec.id: spec for spec in SUPPORTED_PROVIDERS}

    def verify(
        self,
        text: str,
        providers: list[str] | None,
        min_human_probability: float = 0.5,
        timeout_seconds: float = 18.0,
    ) -> ExternalVerificationSummary:
        requested = self._normalize_requested_providers(providers)
        results: list[ExternalProviderResult] = []

        for provider_id in requested:
            spec = self._provider_index[provider_id]
            results.append(
                self._verify_one_provider(
                    spec=spec,
                    text=text,
                    threshold=min_human_probability,
                    timeout_seconds=timeout_seconds,
                )
            )

        available = [item for item in results if item.available]
        configured = [item for item in results if item.configured]
        human_scores = [item.human_probability for item in available if item.human_probability is not None]
        ai_scores = [item.ai_probability for item in available if item.ai_probability is not None]

        aggregate_human = sum(human_scores) / len(human_scores) if human_scores else None
        aggregate_ai = sum(ai_scores) / len(ai_scores) if ai_scores else None

        # Strict pass if at least one configured provider responded and
        # all available providers marked the output as human.
        provider_decisions = [item.passed for item in available if item.passed is not None]
        passed = bool(provider_decisions) and all(provider_decisions)

        return ExternalVerificationSummary(
            requested_providers=requested,
            threshold=min_human_probability,
            passed=passed,
            available_count=len(available),
            configured_count=len(configured),
            aggregate_human_probability=aggregate_human,
            aggregate_ai_probability=aggregate_ai,
            provider_results=results,
        )

    def _normalize_requested_providers(self, providers: list[str] | None) -> list[str]:
        if not providers:
            return list(DEFAULT_PROVIDER_IDS)
        seen: set[str] = set()
        normalized: list[str] = []
        for raw in providers:
            provider_id = raw.strip().lower().replace("-", "")
            if provider_id == "content_at_scale":
                provider_id = "contentatscale"
            if provider_id not in self._provider_index:
                continue
            if provider_id in seen:
                continue
            seen.add(provider_id)
            normalized.append(provider_id)
        return normalized or list(DEFAULT_PROVIDER_IDS)

    def _verify_one_provider(
        self,
        spec: ExternalProviderSpec,
        text: str,
        threshold: float,
        timeout_seconds: float,
    ) -> ExternalProviderResult:
        url = _env_with_fallback(spec.env_prefix, "API_URL")
        if not url:
            return ExternalProviderResult(
                provider_id=spec.id,
                provider_label=spec.label,
                configured=False,
                available=False,
                passed=None,
                ai_probability=None,
                human_probability=None,
                error=(
                    f"Not configured. Set HUMANIZER_{spec.env_prefix}_API_URL "
                    f"to enable {spec.label} verification."
                ),
                response_status=None,
            )

        api_key = _env_with_fallback(spec.env_prefix, "API_KEY")
        auth_header = _env_with_fallback(spec.env_prefix, "AUTH_HEADER") or "Authorization"
        text_field = _env_with_fallback(spec.env_prefix, "TEXT_FIELD") or "text"
        static_payload_json = _env_with_fallback(spec.env_prefix, "STATIC_PAYLOAD_JSON")

        headers = {"Content-Type": "application/json"}
        if api_key:
            if auth_header.lower() == "authorization" and not api_key.lower().startswith("bearer "):
                headers[auth_header] = f"Bearer {api_key}"
            else:
                headers[auth_header] = api_key

        payload: dict[str, Any] = {text_field: text}
        if static_payload_json:
            try:
                static_payload = json.loads(static_payload_json)
                if isinstance(static_payload, dict):
                    payload.update(static_payload)
            except json.JSONDecodeError:
                pass

        try:
            status_code, data = self._http_post(url, headers, payload, timeout_seconds)
        except Exception as exc:  # pragma: no cover - broad for transport adapters.
            return ExternalProviderResult(
                provider_id=spec.id,
                provider_label=spec.label,
                configured=True,
                available=False,
                passed=None,
                ai_probability=None,
                human_probability=None,
                error=str(exc),
                response_status=None,
            )

        ai_score_key = _env_with_fallback(spec.env_prefix, "AI_SCORE_KEY")
        human_score_key = _env_with_fallback(spec.env_prefix, "HUMAN_SCORE_KEY")
        pass_key = _env_with_fallback(spec.env_prefix, "PASS_KEY")
        parsed = _parse_scores(data, ai_score_key, human_score_key, pass_key)

        human_probability = parsed["human_probability"]
        ai_probability = parsed["ai_probability"]
        passed = parsed["passed"]
        if passed is None and human_probability is not None:
            passed = human_probability >= threshold
        if passed is None and ai_probability is not None:
            passed = (1.0 - ai_probability) >= threshold

        return ExternalProviderResult(
            provider_id=spec.id,
            provider_label=spec.label,
            configured=True,
            available=True,
            passed=passed,
            ai_probability=ai_probability,
            human_probability=human_probability,
            error=None,
            response_status=status_code,
        )


def _default_http_post(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            status = int(response.getcode() or 0)
            raw = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc

    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {"raw": raw}

    if not isinstance(data, dict):
        data = {"response": data}
    return status, data


def _parse_scores(
    payload: dict[str, Any],
    ai_score_key: str | None,
    human_score_key: str | None,
    pass_key: str | None,
) -> dict[str, float | bool | None]:
    ai = _extract_probability(payload, ai_score_key, _AI_SCORE_CANDIDATES)
    human = _extract_probability(payload, human_score_key, _HUMAN_SCORE_CANDIDATES)
    passed = _extract_bool(payload, pass_key, _PASS_CANDIDATES)

    if human is None and ai is not None:
        human = max(0.0, min(1.0, 1.0 - ai))
    if ai is None and human is not None:
        ai = max(0.0, min(1.0, 1.0 - human))

    if passed is None and human is None and ai is None:
        # Last fallback: if only boolean exists under generic keys.
        passed = _extract_bool(payload, None, ("is_human", "human", "passed", "pass"))
        if passed is not None:
            human = 1.0 if passed else 0.0
            ai = 0.0 if passed else 1.0

    return {"ai_probability": ai, "human_probability": human, "passed": passed}


def _extract_probability(
    payload: dict[str, Any],
    explicit_key: str | None,
    fallback_candidates: tuple[str, ...],
) -> float | None:
    values: list[Any] = []
    if explicit_key:
        values.append(_get_by_dot_path(payload, explicit_key))
    for candidate in fallback_candidates:
        values.extend(_find_values_by_key(payload, candidate))

    for raw in values:
        parsed = _coerce_probability(raw)
        if parsed is not None:
            return parsed
    return None


def _extract_bool(
    payload: dict[str, Any],
    explicit_key: str | None,
    fallback_candidates: tuple[str, ...],
) -> bool | None:
    values: list[Any] = []
    if explicit_key:
        values.append(_get_by_dot_path(payload, explicit_key))
    for candidate in fallback_candidates:
        values.extend(_find_values_by_key(payload, candidate))

    for raw in values:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            value = raw.strip().lower()
            if value in {"true", "yes", "1", "pass", "passed", "human"}:
                return True
            if value in {"false", "no", "0", "fail", "failed", "ai"}:
                return False
    return None


def _coerce_probability(raw: Any) -> float | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        cleaned = raw.strip().replace("%", "")
        try:
            value = float(cleaned)
        except ValueError:
            return None
    else:
        return None

    if value < 0:
        return None
    if value > 1.0:
        # If API returns percent-like scores.
        if value <= 100.0:
            value = value / 100.0
        else:
            return None
    return max(0.0, min(1.0, value))


def _find_values_by_key(payload: Any, key_name: str) -> list[Any]:
    key_name = key_name.lower()
    values: list[Any] = []

    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() == key_name:
                values.append(value)
            values.extend(_find_values_by_key(value, key_name))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_find_values_by_key(value, key_name))
    return values


def _get_by_dot_path(payload: Any, dot_path: str) -> Any | None:
    current: Any = payload
    parts = [part for part in dot_path.split(".") if part]
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue
        return None
    return current


def _env_with_fallback(prefix: str, key: str) -> str | None:
    direct = os.getenv(f"HUMANIZER_{prefix}_{key}")
    if direct:
        return direct

    # Support alternate prefix without underscore for ContentAtScale legacy naming.
    collapsed = prefix.replace("_", "")
    if collapsed != prefix:
        legacy = os.getenv(f"HUMANIZER_{collapsed}_{key}")
        if legacy:
            return legacy
    return None


_AI_SCORE_CANDIDATES: tuple[str, ...] = (
    "ai_score",
    "ai_probability",
    "ai_prob",
    "probability_ai",
    "generated_probability",
    "machine_probability",
    "fake_probability",
    "gpt_probability",
)

_HUMAN_SCORE_CANDIDATES: tuple[str, ...] = (
    "human_score",
    "human_probability",
    "human_prob",
    "probability_human",
    "original_probability",
    "real_probability",
    "organic_probability",
)

_PASS_CANDIDATES: tuple[str, ...] = ("is_human", "passed", "pass")
