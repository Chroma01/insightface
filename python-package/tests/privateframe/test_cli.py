from __future__ import annotations

import json
from pathlib import Path

import pytest
from insightface.app.privateframe import cli


def _analyze_args(*extra: str) -> list[str]:
    return [
        "analyze",
        "--config",
        "config.yaml",
        "--input",
        "input.mp4",
        "--workdir",
        "work",
        *extra,
    ]


def _process_args(*extra: str) -> list[str]:
    return [
        "process",
        "--config",
        "config.yaml",
        "--input",
        "input.mp4",
        "--workdir",
        "work",
        *extra,
    ]


@pytest.mark.parametrize(
    "args_factory",
    [_analyze_args, _process_args],
    ids=["analyze", "process"],
)
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--provider", "CPUExecutionProvider"),
        ("--artifacts-level", "audit"),
        ("--performance-mode", "fast"),
        ("--stride", "2"),
    ],
)
def test_cli_rejects_removed_config_shortcut_flags(args_factory, option, value):
    with pytest.raises(SystemExit) as raised:
        cli.main(args_factory(option, value, "--dry-run"))

    assert raised.value.code == 2


def test_analysis_help_mentions_dotted_config_overrides(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.command_parser().parse_args(["analyze", "--help"])

    assert raised.value.code == 0
    assert "--section.field VALUE" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["render", "process"])
def test_user_help_hides_internal_debug_video_output(command, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.command_parser().parse_args([command, "--help"])

    assert raised.value.code == 0
    assert "--debug" not in capsys.readouterr().out


def test_cli_forwards_config_fields_without_specialized_shortcuts(
    monkeypatch,
    capsys,
):
    captured = {}

    def analyze(**kwargs):
        captured.update(kwargs)
        return {"phase": "analysis"}

    monkeypatch.setattr(cli, "analyze_streaming_pipeline", analyze)

    exit_code = cli.main(
        _analyze_args(
            "--models.name",
            "raccoon_l",
            "--runtime.provider",
            "CPUExecutionProvider",
            "--runtime.scrfd_static_shape_sessions",
            "false",
            "--output.artifacts_level=audit",
            "--scan.performance_mode",
            "ultra_fast",
            "--scan.frame_stride=3",
            "--json",
        )
    )

    assert exit_code == 0
    assert captured["config_overrides"] == {
        "models.name": "raccoon_l",
        "runtime.provider": "CPUExecutionProvider",
        "runtime.scrfd_static_shape_sessions": False,
        "output.artifacts_level": "audit",
        "scan.performance_mode": "ultra_fast",
        "scan.frame_stride": 3,
    }
    assert '"ok": true' in capsys.readouterr().out


def test_cli_extracts_yaml_typed_dotted_overrides_in_both_forms():
    clean, overrides = cli._parse_dotted_config_overrides(
        _analyze_args(
            "--scan.workers",
            "8",
            "--scan.passes.0.angles=[0, 90]",
            "--render.redaction.feather={enabled: true, ratio: 0.1}",
        )
    )

    assert clean == _analyze_args()
    assert overrides == {
        "scan.workers": 8,
        "scan.passes.0.angles": [0, 90],
        "render.redaction.feather": {"enabled": True, "ratio": 0.1},
    }


def test_cli_forwards_dotted_overrides_and_invocation_root(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    def analyze(**kwargs):
        captured.update(kwargs)
        return {"phase": "analysis"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "analyze_streaming_pipeline", analyze)

    exit_code = cli.main(
        _analyze_args(
            "--scan.workers",
            "8",
            "--recognition.target_persons=[alice, bob]",
            "--json",
        )
    )

    assert exit_code == 0
    assert captured["config_overrides"] == {
        "scan.workers": 8,
        "recognition.target_persons": ["alice", "bob"],
    }
    assert captured["config_override_root"] == Path(tmp_path)
    assert '"ok": true' in capsys.readouterr().out


def test_process_forwards_dotted_overrides(monkeypatch, tmp_path, capsys):
    captured = {}

    def process(**kwargs):
        captured.update(kwargs)
        return {"analysis": {}, "render": {}}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_streaming_pipeline", process)

    exit_code = cli.main(
        [
            "process",
            "--config",
            "config.yaml",
            "--input",
            "input.mp4",
            "--workdir",
            "work",
            "--redacted",
            "output.mp4",
            "--scan.pipeline_depth=4",
            "--json",
        ]
    )

    assert exit_code == 0
    assert captured["config_overrides"] == {"scan.pipeline_depth": 4}
    assert captured["config_override_root"] == Path(tmp_path)
    assert '"ok": true' in capsys.readouterr().out


def test_render_rejects_non_render_dotted_config_override():
    with pytest.raises(SystemExit):
        cli.main(
            [
                "render",
                "--input",
                "input.mp4",
                "--redacted",
                "output.mp4",
                "--scan.workers",
                "8",
            ]
        )


def test_render_forwards_render_dotted_override(monkeypatch, tmp_path, capsys):
    captured = {}

    def render(**kwargs):
        captured.update(kwargs)
        return {"phase": "render"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "render_streaming_artifacts", render)

    exit_code = cli.main(
        [
            "render",
            "--input",
            "input.mp4",
            "--redacted",
            "output.mp4",
            "--render.redaction.method=mosaic",
            "--json",
        ]
    )

    assert exit_code == 0
    assert captured["config_overrides"] == {
        "render.redaction.method": "mosaic"
    }
    assert '"ok": true' in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ("--scan.workers", "8", "--scan.workers=9"),
        (
            "--render.redaction",
            "{}",
            "--render.redaction.method=gaussian",
        ),
        ("--scan.passes", "[]", "--scan.passes.0.input_size=640"),
    ],
)
def test_cli_rejects_duplicate_or_parent_child_config_paths(args):
    with pytest.raises(SystemExit):
        cli.main(_analyze_args(*args, "--dry-run"))


@pytest.mark.parametrize(
    "option",
    [
        "--scan.unknown=8",
        "--runtime.providers=[]",
        "--models.manifest_path=x",
        "--schema_version=2",
        "--scan.passes.-1.input_size=640",
        "--scan.passes.00.input_size=640",
    ],
)
def test_cli_rejects_unknown_internal_or_invalid_dotted_path_in_dry_run(option):
    with pytest.raises(SystemExit):
        cli.main(_analyze_args(option, "--dry-run"))


@pytest.mark.parametrize(
    "raw_value",
    [
        ".nan",
        ".inf",
        "2026-08-29",
        "!!set {one: null}",
        "!!binary YWJj",
        "{1: value}",
    ],
)
def test_cli_rejects_non_json_yaml_override_values(raw_value):
    with pytest.raises(SystemExit):
        cli.main(_analyze_args("--scan.workers", raw_value, "--dry-run"))


def test_cli_rejects_dotted_override_without_value():
    with pytest.raises(SystemExit):
        cli.main(_analyze_args("--scan.workers"))


def test_cli_does_not_abbreviate_ordinary_flags():
    with pytest.raises(SystemExit):
        cli.main(_analyze_args("--str", "4", "--dry-run"))


def test_analyze_output_dir_defaults_to_public_json_only(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    def analyze(**kwargs):
        captured.update(kwargs)
        return {"phase": "analysis"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "analyze_streaming_pipeline", analyze)

    exit_code = cli.main(
        [
            "analyze",
            "--config",
            "config.yaml",
            "--input",
            "camera.clip.mp4",
            "--output-dir",
            "exports",
            "--json",
        ]
    )

    output_dir = tmp_path / "exports"
    assert exit_code == 0
    assert captured["workdir"] == str(output_dir / ".camera.clip_privateframe_work")
    assert captured["result_path"] == str(output_dir / "camera.clip_privateframe.json")
    assert "redacted_path" not in captured
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_process_output_dir_defaults_to_paired_json_and_video(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    def process(**kwargs):
        captured.update(kwargs)
        return {"analysis": {}, "render": {}}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_streaming_pipeline", process)

    exit_code = cli.main(
        [
            "process",
            "--config",
            "config.yaml",
            "--input",
            "camera.clip.mp4",
            "--output-dir",
            "exports",
            "--json",
        ]
    )

    output_dir = tmp_path / "exports"
    assert exit_code == 0
    assert captured["workdir"] == str(output_dir / ".camera.clip_privateframe_work")
    assert captured["result_path"] == str(output_dir / "camera.clip_privateframe.json")
    assert captured["redacted_path"] == str(output_dir / "camera.clip_privateframe.mp4")
    assert captured["debug_path"] is None
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_render_output_dir_reuses_the_paired_json_and_video(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    def render(**kwargs):
        captured.update(kwargs)
        return {"phase": "render"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "render_streaming_artifacts", render)

    exit_code = cli.main(
        [
            "render",
            "--input",
            "camera.clip.mp4",
            "--output-dir",
            "exports",
            "--json",
        ]
    )

    output_dir = tmp_path / "exports"
    assert exit_code == 0
    assert captured["workdir"] == str(
        output_dir / ".camera.clip_privateframe_work"
    )
    assert captured["result_path"] == str(
        output_dir / "camera.clip_privateframe.json"
    )
    assert captured["redacted_path"] == str(
        output_dir / "camera.clip_privateframe.mp4"
    )
    assert captured["debug_path"] is None
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_output_dir_keeps_explicit_legacy_paths_and_aliases(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    def process(**kwargs):
        captured.update(kwargs)
        return {"analysis": {}, "render": {}}

    monkeypatch.setattr(cli, "run_streaming_pipeline", process)

    exit_code = cli.main(
        [
            "process",
            "--config",
            "config.yaml",
            "--input",
            "input.mp4",
            "--output-dir",
            str(tmp_path / "exports"),
            "--workdir",
            "custom-work",
            "--json-output",
            "custom.json",
            "--video-output",
            "custom.mp4",
            "--json",
        ]
    )

    assert exit_code == 0
    assert captured["workdir"] == "custom-work"
    assert captured["result_path"] == "custom.json"
    assert captured["redacted_path"] == "custom.mp4"
    assert json.loads(capsys.readouterr().out)["ok"] is True


@pytest.mark.parametrize("command", ["analyze", "process"])
def test_analysis_commands_still_require_workdir_without_output_dir(command):
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                command,
                "--config",
                "config.yaml",
                "--input",
                "input.mp4",
                "--dry-run",
            ]
        )

    assert raised.value.code == 2


def test_output_dir_dry_run_reports_resolved_default_paths(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(
        [
            "process",
            "--config",
            "config.yaml",
            "--input",
            "input.mp4",
            "--output-dir",
            "exports",
            "--dry-run",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "exports"
    assert exit_code == 0
    assert payload["output_dir"] == str(output_dir)
    assert payload["workdir"] == str(output_dir / ".input_privateframe_work")
    assert payload["result"] == str(output_dir / "input_privateframe.json")
    assert payload["redacted"] == str(output_dir / "input_privateframe.mp4")
