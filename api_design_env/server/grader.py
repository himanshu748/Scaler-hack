"""
Multi-dimensional grader for API endpoint designs.

Scores across 5 axes:
  1. Completeness        (0.30) - coverage of required operations
  2. RESTful Conventions (0.25) - correct methods, plural nouns, nesting
  3. Schema Quality      (0.20) - request/response body quality
  4. Consistency         (0.15) - naming uniformity
  5. HTTP Semantics      (0.10) - method safety / idempotency / status codes
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

WEIGHTS = {
    "completeness": 0.30,
    "restful_conventions": 0.25,
    "schema_quality": 0.20,
    "consistency": 0.15,
    "http_semantics": 0.10,
}

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
IDEMPOTENT_METHODS = {"GET", "HEAD", "OPTIONS", "PUT", "DELETE"}
VALID_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}

METHOD_STATUS_DEFAULTS = {
    "GET": 200,
    "POST": 201,
    "PUT": 200,
    "PATCH": 200,
    "DELETE": 204,
}

_PATH_PARAM_RE = re.compile(r"\{[^}]+\}")


def _normalise_path(path: str) -> str:
    """Replace path params with a placeholder for structural comparison."""
    return _PATH_PARAM_RE.sub("{id}", path.strip().rstrip("/").lower())


def _path_segments(path: str) -> List[str]:
    """Return non-param segments of a path."""
    return [s for s in path.strip("/").split("/") if not s.startswith("{")]


def _is_plural(word: str) -> bool:
    return word.endswith("s") or word.endswith("es")


# ── 1. Completeness ─────────────────────────────────────────────────


def score_completeness(
    submitted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Tuple[float, List[str]]:
    """How many ground-truth endpoints are covered by the submission."""
    suggestions: List[str] = []
    if not ground_truth:
        return 1.0, suggestions

    gt_sigs = set()
    for ep in ground_truth:
        sig = (ep["method"].upper(), _normalise_path(ep["path"]))
        gt_sigs.add(sig)

    sub_sigs = set()
    for ep in submitted:
        sig = (ep.get("method", "").upper(), _normalise_path(ep.get("path", "")))
        sub_sigs.add(sig)

    matched = gt_sigs & sub_sigs
    score = len(matched) / len(gt_sigs) if gt_sigs else 1.0

    missing = gt_sigs - sub_sigs
    for method, path in list(missing)[:3]:
        suggestions.append(f"Missing endpoint: {method} {path}")

    extra = sub_sigs - gt_sigs
    if extra:
        suggestions.append(
            f"{len(extra)} endpoint(s) not in requirements (may still be valid)"
        )

    return round(score, 4), suggestions


# ── 2. RESTful Conventions ──────────────────────────────────────────


def score_restful_conventions(
    submitted: List[Dict[str, Any]],
) -> Tuple[float, List[str]]:
    """Check plural nouns, proper nesting, no verbs in paths, param naming."""
    if not submitted:
        return 0.0, ["No endpoints submitted"]

    suggestions: List[str] = []
    checks_passed = 0
    total_checks = 0

    verb_patterns = re.compile(
        r"/(get|create|update|delete|remove|add|fetch|list|set|make)", re.I
    )
    action_suffixes = re.compile(
        r"/(cancel|retry|move|search|login|logout|approve|reject)$", re.I
    )

    for ep in submitted:
        path = ep.get("path", "")
        method = ep.get("method", "").upper()
        segments = _path_segments(path)

        # Check: resource segments should be plural nouns
        for seg in segments:
            total_checks += 1
            if _is_plural(seg) or seg in (
                "search", "health", "auth", "login", "me", "cancel",
                "retry", "move", "logout", "approve", "reject",
            ):
                checks_passed += 1
            else:
                suggestions.append(f"Consider plural noun: '{seg}' -> '{seg}s'?")

        # Check: no verbs in path (allow action suffixes like /cancel, /retry)
        total_checks += 1
        if verb_patterns.search(path) and not action_suffixes.search(path):
            suggestions.append(f"Avoid verbs in path: {path}")
        else:
            checks_passed += 1

        # Check: method is valid
        total_checks += 1
        if method in VALID_METHODS:
            checks_passed += 1
        else:
            suggestions.append(f"Invalid HTTP method: {method}")

        # Check: path params use descriptive names for nested resources
        # e.g. /posts/{post_id}/comments is better than /posts/{id}/comments/{id}
        params = _PATH_PARAM_RE.findall(path)
        if len(params) > 1:
            total_checks += 1
            param_names = [p.strip("{}") for p in params]
            if len(param_names) == len(set(param_names)):
                checks_passed += 1
            else:
                suggestions.append(
                    f"Use unique param names in nested paths: {path}"
                )

        # Check: nesting depth <= 4 segments (overly deep nesting is an anti-pattern)
        total_checks += 1
        depth = path.strip("/").count("/") + 1
        if depth <= 6:
            checks_passed += 1
        else:
            suggestions.append(f"Path too deeply nested ({depth} segments): {path}")

    score = checks_passed / total_checks if total_checks else 0.0
    return round(min(score, 1.0), 4), suggestions[:5]


# ── 3. Schema Quality ───────────────────────────────────────────────


def score_schema_quality(
    submitted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Tuple[float, List[str]]:
    """Compare request/response schemas against ground truth."""
    suggestions: List[str] = []
    if not ground_truth:
        return 1.0, suggestions

    gt_map: Dict[str, Dict[str, Any]] = {}
    for ep in ground_truth:
        key = (ep["method"].upper(), _normalise_path(ep["path"]))
        gt_map[key] = ep

    sub_map: Dict[str, Dict[str, Any]] = {}
    for ep in submitted:
        key = (ep.get("method", "").upper(), _normalise_path(ep.get("path", "")))
        sub_map[key] = ep

    total_score = 0.0
    matched = 0

    for key, gt_ep in gt_map.items():
        if key not in sub_map:
            continue
        matched += 1
        sub_ep = sub_map[key]

        ep_score = 0.0
        checks = 0

        # Request body coverage
        gt_req = set(gt_ep.get("request_body", {}).keys())
        sub_req = set(sub_ep.get("request_body", {}).keys())
        if gt_req:
            checks += 1
            overlap = len(gt_req & sub_req) / len(gt_req)
            ep_score += overlap
            if overlap < 1.0:
                missing = gt_req - sub_req
                suggestions.append(
                    f"{key[0]} {key[1]}: missing request fields {missing}"
                )
        else:
            checks += 1
            ep_score += 1.0 if not sub_req else 0.8

        # Response body coverage
        gt_resp = set(gt_ep.get("response_body", {}).keys())
        sub_resp = set(sub_ep.get("response_body", {}).keys())
        if gt_resp:
            checks += 1
            overlap = len(gt_resp & sub_resp) / len(gt_resp)
            ep_score += overlap
            if overlap < 1.0:
                missing = gt_resp - sub_resp
                suggestions.append(
                    f"{key[0]} {key[1]}: missing response fields {missing}"
                )
        else:
            checks += 1
            ep_score += 1.0

        # Query params coverage
        gt_qp = set(gt_ep.get("query_params", []))
        sub_qp = set(sub_ep.get("query_params", []))
        if gt_qp:
            checks += 1
            overlap = len(gt_qp & sub_qp) / len(gt_qp)
            ep_score += overlap
        else:
            checks += 1
            ep_score += 1.0

        total_score += ep_score / checks if checks else 0.0

    score = total_score / len(gt_map) if gt_map else 1.0
    return round(score, 4), suggestions[:5]


# ── 4. Consistency ──────────────────────────────────────────────────


def score_consistency(
    submitted: List[Dict[str, Any]],
) -> Tuple[float, List[str]]:
    """Check naming and format uniformity across all endpoints."""
    if not submitted:
        return 0.0, ["No endpoints submitted"]

    suggestions: List[str] = []
    checks_passed = 0
    total_checks = 0

    # Collect all resource names
    all_segments: List[str] = []
    for ep in submitted:
        all_segments.extend(_path_segments(ep.get("path", "")))

    # Check: consistent casing (all lowercase / snake_case / kebab-case)
    has_camel = any(re.search(r"[a-z][A-Z]", s) for s in all_segments)
    has_snake = any("_" in s for s in all_segments)
    has_kebab = any("-" in s for s in all_segments)
    naming_styles = sum([has_camel, has_snake, has_kebab, not (has_camel or has_snake or has_kebab)])

    total_checks += 1
    if naming_styles <= 1:
        checks_passed += 1
    else:
        suggestions.append("Inconsistent naming: mix of camelCase, snake_case, kebab-case")

    # Check: consistent use of trailing slashes
    paths = [ep.get("path", "") for ep in submitted]
    trailing = [p.endswith("/") for p in paths if p]
    total_checks += 1
    if all(trailing) or not any(trailing):
        checks_passed += 1
    else:
        suggestions.append("Inconsistent trailing slashes")

    # Check: all paths start with /
    total_checks += 1
    if all(p.startswith("/") for p in paths if p):
        checks_passed += 1
    else:
        suggestions.append("Some paths don't start with /")

    # Check: consistent pluralisation of resource names
    resource_names = set()
    for seg in all_segments:
        resource_names.add(seg)
    plural_count = sum(1 for n in resource_names if _is_plural(n))
    total_checks += 1
    if len(resource_names) == 0 or plural_count / len(resource_names) >= 0.7:
        checks_passed += 1
    else:
        suggestions.append("Inconsistent pluralisation of resource names")

    # Check: descriptions provided
    has_desc = sum(1 for ep in submitted if ep.get("description", "").strip())
    total_checks += 1
    if has_desc >= len(submitted) * 0.8:
        checks_passed += 1
    else:
        suggestions.append("Add descriptions to all endpoints")

    score = checks_passed / total_checks if total_checks else 0.0
    return round(score, 4), suggestions[:5]


# ── 5. HTTP Semantics ───────────────────────────────────────────────


def score_http_semantics(
    submitted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Tuple[float, List[str]]:
    """Check method safety, idempotency, and status codes."""
    if not submitted:
        return 0.0, ["No endpoints submitted"]

    suggestions: List[str] = []
    checks_passed = 0
    total_checks = 0

    gt_map: Dict[tuple, Dict[str, Any]] = {}
    for ep in ground_truth:
        key = (ep["method"].upper(), _normalise_path(ep["path"]))
        gt_map[key] = ep

    for ep in submitted:
        method = ep.get("method", "").upper()
        path = ep.get("path", "")
        status = ep.get("status_code", 200)

        # Check: safe methods should not have request body
        total_checks += 1
        if method in SAFE_METHODS and ep.get("request_body"):
            suggestions.append(f"{method} {path}: safe methods should not have a request body")
        else:
            checks_passed += 1

        # Check: POST should return 201 for creation
        total_checks += 1
        key = (method, _normalise_path(path))
        if key in gt_map:
            expected = gt_map[key].get("status_code", METHOD_STATUS_DEFAULTS.get(method, 200))
            if status == expected:
                checks_passed += 1
            else:
                suggestions.append(
                    f"{method} {path}: expected status {expected}, got {status}"
                )
        elif method in METHOD_STATUS_DEFAULTS:
            if status == METHOD_STATUS_DEFAULTS[method]:
                checks_passed += 1
            else:
                suggestions.append(
                    f"{method} {path}: conventional status is {METHOD_STATUS_DEFAULTS[method]}"
                )
        else:
            checks_passed += 1

        # Check: DELETE should return 204 (no content)
        total_checks += 1
        if method == "DELETE" and status == 204:
            checks_passed += 1
        elif method == "DELETE":
            suggestions.append(f"DELETE {path}: prefer 204 No Content")
        else:
            checks_passed += 1

    score = checks_passed / total_checks if total_checks else 0.0
    return round(min(score, 1.0), 4), suggestions[:5]


# ── Public API ──────────────────────────────────────────────────────


# ── 6. Penalty detection ────────────────────────────────────────────


def compute_penalty(
    submitted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Tuple[float, List[str]]:
    """
    Detect clearly undesirable patterns and return a penalty multiplier
    (1.0 = no penalty, <1.0 = penalised) plus suggestions.

    Penalised behaviours:
      - Empty submission
      - Massive over-submission (>3x ground truth endpoints)
      - Duplicate endpoints (same method+path)
      - Completely irrelevant paths (0% overlap with ground truth segments)
    """
    suggestions: List[str] = []

    if not submitted:
        return 0.0, ["Empty submission: provide at least one endpoint"]

    penalty = 1.0

    # Duplicate check
    sigs = [(ep.get("method", "").upper(), _normalise_path(ep.get("path", "")))
            for ep in submitted]
    dupes = len(sigs) - len(set(sigs))
    if dupes > 0:
        penalty -= 0.1 * min(dupes, 3)
        suggestions.append(f"{dupes} duplicate endpoint(s) detected -- remove them")

    # Over-submission (spam penalty)
    if len(submitted) > 3 * len(ground_truth):
        penalty -= 0.15
        suggestions.append(
            f"Too many endpoints ({len(submitted)}) vs expected (~{len(ground_truth)})"
        )

    # Relevance check: do any submitted path segments overlap with ground truth?
    gt_segments: set = set()
    for ep in ground_truth:
        gt_segments.update(_path_segments(ep["path"]))
    sub_segments: set = set()
    for ep in submitted:
        sub_segments.update(_path_segments(ep.get("path", "")))
    if gt_segments and sub_segments:
        overlap = len(gt_segments & sub_segments) / len(gt_segments)
        if overlap == 0.0:
            penalty -= 0.2
            suggestions.append("No submitted paths match expected resources")

    return round(max(penalty, 0.0), 4), suggestions


# ── Public API ──────────────────────────────────────────────────────


def grade(
    submitted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Grade a submitted API design against the ground truth.

    Returns:
        {
            "scores": {dim: float, ...},
            "penalty": float,          # 1.0 = no penalty, <1.0 = penalised
            "total": float,            # weighted score * penalty
            "suggestions": [str, ...],
        }
    """
    submitted_dicts = []
    for ep in submitted:
        if hasattr(ep, "model_dump"):
            submitted_dicts.append(ep.model_dump())
        elif isinstance(ep, dict):
            submitted_dicts.append(ep)
        else:
            submitted_dicts.append(dict(ep))

    completeness, s1 = score_completeness(submitted_dicts, ground_truth)
    restful, s2 = score_restful_conventions(submitted_dicts)
    schema, s3 = score_schema_quality(submitted_dicts, ground_truth)
    consistency, s4 = score_consistency(submitted_dicts)
    semantics, s5 = score_http_semantics(submitted_dicts, ground_truth)
    penalty, s6 = compute_penalty(submitted_dicts, ground_truth)

    scores = {
        "completeness": completeness,
        "restful_conventions": restful,
        "schema_quality": schema,
        "consistency": consistency,
        "http_semantics": semantics,
    }

    raw_total = sum(scores[k] * WEIGHTS[k] for k in scores)
    total = raw_total * penalty

    all_suggestions = s6 + s1 + s2 + s3 + s4 + s5  # penalties first
    seen = set()
    unique: List[str] = []
    for s in all_suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return {
        "scores": scores,
        "penalty": penalty,
        "total": round(total, 4),
        "suggestions": unique[:10],
    }
