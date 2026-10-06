"""The drift check itself, and the live check it enables.

``test_types_contract.py`` is only as good as the snapshot it pins things to, and the
snapshot is a copy that someone refreshes by hand. ``spec_drift.py`` compares it with
the API's current spec. Most of this file tests that comparison, because a checker
that reports "no drift" wrongly is worse than none: it is the thing people trust.

The live comparison runs only when you point it at a checkout of the monorepo:

    SUPAGAMMA_MONOREPO=../supagamma pytest tests/test_spec_drift.py

(``SUPAGAMMA_MONOREPO_PYTHON`` names an interpreter with ``apps/api``'s requirements
installed, if the one running pytest has none.) It is off by default because the SDK's
own CI cannot see the private monorepo, and it must never need the network or a key.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import spec_drift
from spec_drift import diff_specs, main, write_snapshot

# --- small spec builders ----------------------------------------------------------


def spec(
    *,
    version: str = "1.0.0",
    paths: Optional[Dict[str, Any]] = None,
    schemas: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {"title": "API", "version": version},
        "paths": paths or {},
        "components": {"schemas": schemas or {}},
    }


def query(name: str, **schema: Any) -> Dict[str, Any]:
    return {
        "name": name,
        "in": "query",
        "required": False,
        "schema": {"type": "integer", "title": name.title(), **schema},
    }


def operation(
    parameters: Optional[List[Dict[str, Any]]] = None,
    body: Optional[Dict[str, Any]] = None,
    codes: tuple = ("200", "422"),
) -> Dict[str, Any]:
    op: Dict[str, Any] = {
        "summary": "A summary",
        "parameters": parameters or [],
        "responses": {code: {"description": "d"} for code in codes},
    }
    if body is not None:
        op["requestBody"] = body
    return op


def base() -> Dict[str, Any]:
    return spec(
        paths={
            "/v1/things": {
                "get": operation([query("limit", minimum=1, maximum=100, default=10)]),
                "post": operation(
                    body={
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/NewThing"}
                            }
                        },
                    },
                    codes=("202", "422"),
                ),
            },
        },
        schemas={
            "Thing": {
                "type": "object",
                "title": "Thing",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string", "title": "Id"},
                    "description": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "D"},
                    "size": {"type": "integer", "title": "Size"},
                },
            },
            "NewThing": {
                "type": "object",
                "required": ["kind"],
                "properties": {"kind": {"type": "string", "enum": ["a", "b"]}},
            },
        },
    )


def drift(current: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> List[str]:
    return diff_specs(snapshot if snapshot is not None else base(), current)


def changed(mutate: Any) -> Dict[str, Any]:
    current = copy.deepcopy(base())
    mutate(current)
    return current


# --- what is NOT drift --------------------------------------------------------------


def test_identical_specs_have_no_drift():
    assert drift(base()) == []


def test_prose_is_not_drift():
    """Descriptions, summaries, titles and examples change constantly and affect no caller."""

    def rewrite(current: Dict[str, Any]) -> None:
        get = current["paths"]["/v1/things"]["get"]
        get["summary"] = "Reworded"
        get["description"] = "Now with a paragraph."
        get["parameters"][0]["description"] = "How many."
        get["parameters"][0]["schema"]["title"] = "Renamed Title"
        get["parameters"][0]["schema"]["examples"] = [5]
        thing = current["components"]["schemas"]["Thing"]
        thing["title"] = "A Thing"
        thing["description"] = "Docs for the model."
        thing["properties"]["id"]["description"] = "The id."
        thing["properties"]["id"]["example"] = "abc"
        current["info"]["title"] = "Renamed API"

    assert drift(changed(rewrite)) == []


def test_key_order_is_not_drift():
    reordered = json.loads(json.dumps(base(), sort_keys=True))
    assert drift(reordered) == []


def test_required_order_is_not_drift():
    snapshot = base()
    snapshot["components"]["schemas"]["Thing"]["required"] = ["id", "size"]
    current = copy.deepcopy(snapshot)
    current["components"]["schemas"]["Thing"]["required"] = ["size", "id"]
    assert drift(current, snapshot) == []


# --- what IS drift -------------------------------------------------------------------


def test_a_changed_api_version_is_reported():
    assert drift(
        spec(version="2.0.0", paths=base()["paths"], schemas=base()["components"]["schemas"])
    ) == ["~ API version 1.0.0 -> 2.0.0"]


def test_a_removed_and_an_added_operation():
    def move(current: Dict[str, Any]) -> None:
        del current["paths"]["/v1/things"]["post"]
        current["paths"]["/v1/things/{id}"] = {"get": operation()}

    lines = drift(changed(move))
    assert "- operation removed: POST /v1/things" in lines
    assert "+ operation added:   GET /v1/things/{id}" in lines


def test_an_added_removed_and_changed_parameter():
    def edit(current: Dict[str, Any]) -> None:
        params = current["paths"]["/v1/things"]["get"]["parameters"]
        params[0]["schema"]["maximum"] = 500  # changed
        params.append(query("offset", minimum=0))  # added

    lines = drift(changed(edit))
    assert any(line.startswith("~ GET /v1/things: query parameter limit changed") for line in lines)
    assert "+ GET /v1/things: query parameter added: offset" in lines

    def drop(current: Dict[str, Any]) -> None:
        current["paths"]["/v1/things"]["get"]["parameters"] = []

    assert drift(changed(drop)) == ["- GET /v1/things: query parameter removed: limit"]


def test_a_parameter_that_became_required_is_drift():
    def edit(current: Dict[str, Any]) -> None:
        current["paths"]["/v1/things"]["get"]["parameters"][0]["required"] = True

    assert any("parameter limit changed" in line for line in drift(changed(edit)))


def test_a_changed_request_body_and_changed_response_codes():
    def edit(current: Dict[str, Any]) -> None:
        post = current["paths"]["/v1/things"]["post"]
        post["requestBody"]["required"] = False
        post["responses"] = {"201": {"description": "d"}, "422": {"description": "d"}}

    lines = drift(changed(edit))
    assert "~ POST /v1/things: requestBody changed" in lines
    assert "~ POST /v1/things: response codes ['202', '422'] -> ['201', '422']" in lines


def test_schemas_added_and_removed():
    def edit(current: Dict[str, Any]) -> None:
        del current["components"]["schemas"]["NewThing"]
        current["components"]["schemas"]["Other"] = {"type": "object", "properties": {}}

    lines = drift(changed(edit))
    assert "- schema removed: NewThing" in lines
    assert "+ schema added:   Other" in lines


def test_properties_added_removed_and_retyped():
    def edit(current: Dict[str, Any]) -> None:
        props = current["components"]["schemas"]["Thing"]["properties"]
        del props["size"]
        props["weight"] = {"type": "number"}
        props["id"] = {"type": "integer"}

    lines = drift(changed(edit))
    assert "- Thing: property removed: size" in lines
    assert any(line.startswith("+ Thing: property added: weight") for line in lines)
    assert any(line.startswith("~ Thing.id changed") for line in lines)


def test_a_field_that_became_nullable_is_drift():
    def edit(current: Dict[str, Any]) -> None:
        current["components"]["schemas"]["Thing"]["properties"]["size"] = {
            "anyOf": [{"type": "integer"}, {"type": "null"}]
        }

    assert any(line.startswith("~ Thing.size changed") for line in drift(changed(edit)))


def test_required_changes_are_drift():
    def edit(current: Dict[str, Any]) -> None:
        current["components"]["schemas"]["Thing"]["required"] = ["id", "size"]

    assert drift(changed(edit)) == ["~ Thing: required ['id'] -> ['id', 'size']"]


def test_an_enum_change_is_drift():
    def edit(current: Dict[str, Any]) -> None:
        current["components"]["schemas"]["NewThing"]["properties"]["kind"]["enum"].append("c")

    assert any(line.startswith("~ NewThing.kind changed") for line in drift(changed(edit)))


# --- guarding the guard: a field NAMED like a prose key is still a field --------------------


def test_a_field_called_description_is_compared_like_any_other():
    """``description`` is documentation as a spec KEY but a real field as a property NAME.
    ``MarketResponse``, ``SeriesResponse`` and ``TransactionResponse`` each have one. An
    early version of the normaliser dropped every key called ``description`` wherever it
    appeared, so removing, adding or retyping that field reported "no drift"."""

    def removed(current: Dict[str, Any]) -> None:
        del current["components"]["schemas"]["Thing"]["properties"]["description"]

    assert drift(changed(removed)) == ["- Thing: property removed: description"]

    def retyped(current: Dict[str, Any]) -> None:
        current["components"]["schemas"]["Thing"]["properties"]["description"] = {"type": "integer"}

    assert any(line.startswith("~ Thing.description changed") for line in drift(changed(retyped)))

    snapshot = base()
    del snapshot["components"]["schemas"]["Thing"]["properties"]["description"]
    assert any(
        line.startswith("+ Thing: property added: description") for line in drift(base(), snapshot)
    )


@pytest.mark.parametrize("name", ["title", "summary", "example", "examples", "description"])
def test_every_prose_keyword_works_as_a_field_name(name: str):
    snapshot = spec(schemas={"M": {"type": "object", "properties": {name: {"type": "string"}}}})
    current = spec(schemas={"M": {"type": "object", "properties": {name: {"type": "integer"}}}})
    assert drift(current, snapshot) == [
        f'~ M.{name} changed: {{"type": "string"}} -> {{"type": "integer"}}'
    ]


def test_a_field_named_properties_does_not_confuse_the_normaliser():
    snapshot = spec(
        schemas={
            "M": {"type": "object", "properties": {"properties": {"type": "string", "title": "P"}}}
        }
    )
    current = copy.deepcopy(snapshot)
    current["components"]["schemas"]["M"]["properties"]["properties"]["title"] = "Other title"
    assert drift(current, snapshot) == []  # that inner title is prose
    current["components"]["schemas"]["M"]["properties"]["properties"]["type"] = "integer"
    assert drift(current, snapshot) != []


# --- the snapshot is held in the form the server serves it ---------------------------------


def test_write_snapshot_is_compact_and_round_trips(tmp_path: Path):
    path = tmp_path / "openapi.json"
    original = base()
    write_snapshot(original, path)
    text = path.read_text(encoding="utf-8")
    assert "\n" not in text and ": " not in text and ", " not in text
    assert json.loads(text) == original


def test_the_vendored_snapshot_is_in_the_served_form():
    """One compact line, so a refresh produces a reviewable diff and not a re-indent."""
    text = spec_drift.SNAPSHOT.read_text(encoding="utf-8")
    assert "\n" not in text.strip()
    assert json.loads(text)["info"]["version"]


# --- the command line --------------------------------------------------------------------------


@pytest.fixture
def files(tmp_path: Path) -> Dict[str, Path]:
    snapshot, current = tmp_path / "snapshot.json", tmp_path / "current.json"
    snapshot.write_text(json.dumps(base()), encoding="utf-8")
    return {"snapshot": snapshot, "current": current}


def test_cli_exit_0_when_there_is_no_drift(files, capsys):
    files["current"].write_text(json.dumps(base()), encoding="utf-8")
    code = main(["--spec", str(files["current"]), "--snapshot", str(files["snapshot"])])
    assert code == 0
    assert "no drift" in capsys.readouterr().out


def test_cli_exit_1_prints_the_differences_and_leaves_the_snapshot_alone(files, capsys):
    current = base()
    del current["components"]["schemas"]["Thing"]["properties"]["size"]
    files["current"].write_text(json.dumps(current), encoding="utf-8")
    before = files["snapshot"].read_text(encoding="utf-8")
    code = main(["--spec", str(files["current"]), "--snapshot", str(files["snapshot"])])
    out = capsys.readouterr().out
    assert code == 1
    assert "- Thing: property removed: size" in out and "--update" in out
    assert files["snapshot"].read_text(encoding="utf-8") == before


def test_cli_update_rewrites_the_snapshot_and_succeeds(files, capsys):
    current = base()
    current["components"]["schemas"]["Thing"]["properties"]["weight"] = {"type": "number"}
    files["current"].write_text(json.dumps(current), encoding="utf-8")
    code = main(["--spec", str(files["current"]), "--snapshot", str(files["snapshot"]), "--update"])
    assert code == 0
    assert json.loads(files["snapshot"].read_text(encoding="utf-8")) == current
    assert "snapshot rewritten" in capsys.readouterr().out
    # and now there is nothing left to report
    assert main(["--spec", str(files["current"]), "--snapshot", str(files["snapshot"])]) == 0


def test_cli_exit_2_when_the_current_spec_cannot_be_obtained(files, tmp_path, capsys):
    code = main(["--spec", str(tmp_path / "missing.json"), "--snapshot", str(files["snapshot"])])
    assert code == 2
    assert "could not obtain the current spec" in capsys.readouterr().err


def test_cli_requires_a_source():
    with pytest.raises(SystemExit):
        main([])


# --- generating the spec from the monorepo ------------------------------------------------------


def test_a_directory_that_is_not_the_monorepo_is_refused(tmp_path: Path):
    with pytest.raises(RuntimeError, match="does not look like the monorepo"):
        spec_drift.from_monorepo(tmp_path)


def test_generating_the_spec_never_passes_live_keys_to_the_child(tmp_path: Path, monkeypatch):
    (tmp_path / "apps" / "api" / "app").mkdir(parents=True)
    (tmp_path / "apps" / "api" / "app" / "main.py").write_text("", encoding="utf-8")
    for key in ("RESEND_API_KEY", "PADDLE_API_KEY", "SENTRY_DSN"):
        monkeypatch.setenv(key, "live-value")
    seen: Dict[str, Any] = {}

    def fake_run(argv: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        seen.update(env=kwargs["env"], cwd=kwargs["cwd"])
        Path(argv[-1]).write_text(json.dumps(base()), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(spec_drift.subprocess, "run", fake_run)
    assert spec_drift.from_monorepo(tmp_path)["info"]["version"] == "1.0.0"
    for key in ("RESEND_API_KEY", "PADDLE_API_KEY", "SENTRY_DSN"):
        assert key not in seen["env"], f"{key} reached the spec generator"
    assert Path(seen["cwd"]) == tmp_path / "apps" / "api"


def test_a_failed_generation_explains_what_to_do(tmp_path: Path, monkeypatch):
    (tmp_path / "apps" / "api" / "app").mkdir(parents=True)
    (tmp_path / "apps" / "api" / "app" / "main.py").write_text("", encoding="utf-8")

    def fake_run(argv: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            argv, 1, "", "ModuleNotFoundError: No module named 'fastapi'"
        )

    monkeypatch.setattr(spec_drift.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="requirements installed") as raised:
        spec_drift.from_monorepo(tmp_path)
    assert "fastapi" in str(raised.value)


# --- the live check -------------------------------------------------------------------

MONOREPO = os.environ.get("SUPAGAMMA_MONOREPO")


@pytest.mark.skipif(not MONOREPO, reason="set SUPAGAMMA_MONOREPO to a checkout of the monorepo")
def test_the_vendored_snapshot_matches_the_monorepos_current_spec():
    """The reason this module exists. Offline, deterministic, no credentials: the spec
    comes from importing the API app, not from a network call to production."""
    assert MONOREPO is not None
    current = spec_drift.from_monorepo(Path(MONOREPO), os.environ.get("SUPAGAMMA_MONOREPO_PYTHON"))
    lines = diff_specs(spec_drift.load(spec_drift.SNAPSHOT), current)
    assert not lines, (
        "tests/fixtures/openapi.json is stale against the API. Refresh it with\n"
        f"  python tests/spec_drift.py --monorepo {MONOREPO} --update\n"
        "then fix whatever tests/test_types_contract.py flags.\n\n" + "\n".join(lines)
    )
