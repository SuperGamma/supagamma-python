"""Compare the vendored OpenAPI snapshot with the API's current spec.

The contract tests (``test_types_contract.py``) pin every model and every query
parameter to ``tests/fixtures/openapi.json``. That only means something while the
snapshot is *current*, and nothing used to check that: the snapshot is a copy
taken by hand, so when the API moved on (new ``/v1/markets`` filters, new fields on
trades, balances and estimates) the SDK's CI stayed green against a spec that no
longer existed. This module is that missing check.

It compares the **contract**: operations, parameters, and every schema's
properties and types. Prose (summaries, descriptions, titles, examples) is
ignored, because it changes constantly and affects no caller.

The current spec comes from the monorepo, not from production:

    python tests/spec_drift.py --monorepo ../supagamma

imports the API app from that checkout and calls ``app.openapi()``, which is
offline, deterministic and needs no credentials. (``--spec FILE`` compares a spec
you already have, and ``--url`` fetches one, for when no checkout is at hand.)

    --update    rewrite the snapshot from the current spec instead of failing

Exit status: 0 = no drift, 1 = drift (the differences are printed), 2 = the current
spec could not be obtained.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SNAPSHOT = Path(__file__).parent / "fixtures" / "openapi.json"

#: Documentation-only keys. They never change what a caller can send or receive.
_PROSE = frozenset({"title", "description", "summary", "examples", "example", "externalDocs"})

#: Secrets that exist in the developer's shell. Generating a spec does not use
#: them, so they are removed from the child process instead of trusted.
_LIVE_KEYS = ("RESEND_API_KEY", "PADDLE_API_KEY", "SENTRY_DSN")

_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")

_GENERATE = """
import json, sys
sys.path.insert(0, ".")
from app.main import app
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(app.openapi(), handle, ensure_ascii=False, separators=(",", ":"))
"""


# --- getting a spec -------------------------------------------------------------


def load(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        spec: Dict[str, Any] = json.load(handle)
    return spec


def from_monorepo(monorepo: Path, python: Optional[str] = None) -> Dict[str, Any]:
    """The spec the monorepo's API would serve, generated offline.

    ``python`` must be an interpreter that has the API's dependencies installed
    (default: the one running this script).
    """
    api = monorepo / "apps" / "api"
    if not (api / "app" / "main.py").is_file():
        raise RuntimeError(f"{monorepo} does not look like the monorepo: no apps/api/app/main.py")
    env = {k: v for k, v in os.environ.items() if k not in _LIVE_KEYS}
    env.setdefault("API_SECRET_KEY", "spec-drift-check")
    env.setdefault("ENVIRONMENT", "test")
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "openapi.json"
        proc = subprocess.run(
            [python or sys.executable, "-c", _GENERATE, str(out)],
            cwd=api,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0 or not out.is_file():
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-8:]
            raise RuntimeError(
                "could not generate the spec from the monorepo (does that interpreter "
                "have apps/api's requirements installed?):\n" + "\n".join(tail)
            )
        return load(out)


def from_url(url: str) -> Dict[str, Any]:
    import httpx

    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    spec: Dict[str, Any] = response.json()
    return spec


# --- normalising and comparing ----------------------------------------------------


def _norm(node: Any, *, names: bool = False) -> Any:
    """``node`` without prose, so two specs compare equal iff their contracts do.

    ``names`` marks a mapping whose KEYS are field names (a schema's ``properties``)
    rather than spec keywords. There a key called ``description`` is a real field
    (``MarketResponse`` has one), not documentation, and must not be dropped.
    """
    if isinstance(node, dict):
        return {
            k: _norm(v, names=(k == "properties" and not names))
            for k, v in node.items()
            if names or k not in _PROSE
        }
    if isinstance(node, list):
        return [_norm(v) for v in node]
    return node


def _operations(spec: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    found: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path, item in (spec.get("paths") or {}).items():
        for method, operation in item.items():
            if method.lower() in _METHODS:
                found[(method.upper(), path)] = operation
    return found


def _params(operation: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
    return {
        (p["in"], p["name"]): _norm({k: v for k, v in p.items() if k not in ("name", "in")})
        for p in operation.get("parameters", [])
    }


def _fmt(value: Any) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= 120 else text[:117] + "..."


def _diff_operations(snapshot: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    old, new = _operations(snapshot), _operations(current)
    lines: List[str] = []
    for key in sorted(set(old) - set(new)):
        lines.append(f"- operation removed: {key[0]} {key[1]}")
    for key in sorted(set(new) - set(old)):
        lines.append(f"+ operation added:   {key[0]} {key[1]}")
    for key in sorted(set(old) & set(new)):
        label = f"{key[0]} {key[1]}"
        before, after = _params(old[key]), _params(new[key])
        for where, name in sorted(set(before) - set(after)):
            lines.append(f"- {label}: {where} parameter removed: {name}")
        for where, name in sorted(set(after) - set(before)):
            lines.append(f"+ {label}: {where} parameter added: {name}")
        for where, name in sorted(set(before) & set(after)):
            if before[(where, name)] != after[(where, name)]:
                lines.append(
                    f"~ {label}: {where} parameter {name} changed: "
                    f"{_fmt(before[(where, name)])} -> {_fmt(after[(where, name)])}"
                )
        for part in ("requestBody",):
            if _norm(old[key].get(part)) != _norm(new[key].get(part)):
                lines.append(f"~ {label}: {part} changed")
        before_codes = sorted((old[key].get("responses") or {}).keys())
        after_codes = sorted((new[key].get("responses") or {}).keys())
        if before_codes != after_codes:
            lines.append(f"~ {label}: response codes {before_codes} -> {after_codes}")
    return lines


def _diff_schemas(snapshot: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    old = (snapshot.get("components") or {}).get("schemas") or {}
    new = (current.get("components") or {}).get("schemas") or {}
    lines: List[str] = []
    for name in sorted(set(old) - set(new)):
        lines.append(f"- schema removed: {name}")
    for name in sorted(set(new) - set(old)):
        lines.append(f"+ schema added:   {name}")
    for name in sorted(set(old) & set(new)):
        before, after = _norm(old[name]), _norm(new[name])
        if before == after:
            continue
        old_props, new_props = before.get("properties") or {}, after.get("properties") or {}
        for prop in sorted(set(old_props) - set(new_props)):
            lines.append(f"- {name}: property removed: {prop}")
        for prop in sorted(set(new_props) - set(old_props)):
            lines.append(f"+ {name}: property added: {prop} {_fmt(new_props[prop])}")
        for prop in sorted(set(old_props) & set(new_props)):
            if old_props[prop] != new_props[prop]:
                lines.append(
                    f"~ {name}.{prop} changed: {_fmt(old_props[prop])} -> {_fmt(new_props[prop])}"
                )
        if sorted(before.get("required") or []) != sorted(after.get("required") or []):
            lines.append(
                f"~ {name}: required {sorted(before.get('required') or [])} -> "
                f"{sorted(after.get('required') or [])}"
            )
        rest_before = {k: v for k, v in before.items() if k not in ("properties", "required")}
        rest_after = {k: v for k, v in after.items() if k not in ("properties", "required")}
        if rest_before != rest_after:
            lines.append(f"~ {name}: definition changed: {_fmt(rest_before)} -> {_fmt(rest_after)}")
    return lines


def diff_specs(snapshot: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Human-readable differences between two specs' contracts; empty means no drift.

    ``-`` marks something the snapshot has and the current spec lost, ``+`` something
    new, ``~`` something that changed.
    """
    lines: List[str] = []
    before_version = (snapshot.get("info") or {}).get("version")
    after_version = (current.get("info") or {}).get("version")
    if before_version != after_version:
        lines.append(f"~ API version {before_version} -> {after_version}")
    lines.extend(_diff_operations(snapshot, current))
    lines.extend(_diff_schemas(snapshot, current))
    return lines


# --- command line ------------------------------------------------------------------


def write_snapshot(spec: Dict[str, Any], path: Path = SNAPSHOT) -> None:
    """Write ``spec`` in the form the server serves it: one compact line, no newline."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        json.dump(spec, handle, ensure_ascii=False, separators=(",", ":"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--monorepo", type=Path, help="path to a checkout of the monorepo")
    source.add_argument("--spec", type=Path, help="a spec file to compare against")
    source.add_argument("--url", help="fetch the spec from this URL")
    parser.add_argument("--python", help="interpreter with apps/api's requirements (--monorepo)")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT, help="the snapshot to check")
    parser.add_argument("--update", action="store_true", help="rewrite the snapshot, don't fail")
    args = parser.parse_args(argv)

    try:
        if args.monorepo:
            current = from_monorepo(args.monorepo, args.python)
        elif args.spec:
            current = load(args.spec)
        else:
            current = from_url(args.url)
    except Exception as exc:  # report, with a distinct exit status, whatever went wrong
        print(f"could not obtain the current spec: {exc}", file=sys.stderr)
        return 2

    differences = diff_specs(load(args.snapshot), current)
    if not differences:
        print(f"no drift: {args.snapshot} matches the current API contract")
        return 0
    print(f"{len(differences)} difference(s) between {args.snapshot} and the current API:\n")
    print("\n".join(differences))
    if args.update:
        write_snapshot(current, args.snapshot)
        print(f"\nsnapshot rewritten: {args.snapshot}")
        print("Now fix whatever tests/test_types_contract.py flags (models, builders).")
        return 0
    print("\nRefresh with --update, then fix what tests/test_types_contract.py flags.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
