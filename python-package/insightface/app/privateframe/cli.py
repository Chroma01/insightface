"""CLI for artifact-first streaming analysis and deterministic rendering."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import yaml

from .base_config import validate_config_override_paths
from .output_paths import default_output_paths
from .pipeline import (
    analyze_streaming_pipeline,
    render_streaming_artifacts,
    run_streaming_pipeline,
)

_DOTTED_CONFIG_HELP = (
    "Any public YAML setting may also be overridden as "
    "--section.field VALUE (for example, --scan.workers 8). "
    "Values use YAML/JSON types; existing list items use numeric segments."
)
_DOTTED_RENDER_HELP = (
    "Public render.* YAML settings may be overridden as dotted options. "
    "Dotted values are applied after --render-config."
)


def _common(value: argparse.ArgumentParser) -> None:
    value.add_argument(
        "--json",
        action="store_true",
        help="print the command status as compact JSON; does not select output files",
    )
    value.add_argument("--dry-run", action="store_true")


def _result(value: argparse.ArgumentParser) -> None:
    value.add_argument(
        "--result",
        "--json-output",
        dest="result",
        help="analysis result JSON path",
    )


def _output_dir(value: argparse.ArgumentParser) -> None:
    value.add_argument(
        "--output-dir",
        help=(
            "directory for stable <input>_privateframe.json/.mp4 output names; "
            "also supplies a private runtime work directory"
        ),
    )


def _video_output(value: argparse.ArgumentParser) -> None:
    value.add_argument(
        "--redacted",
        "--video-output",
        dest="redacted",
        help="redacted result video path",
    )


def _render_config(value: argparse.ArgumentParser) -> None:
    value.add_argument(
        "--render-config",
        help=(
            "YAML overrides for redaction style and/or video encoding; "
            "dotted render.* options take precedence"
        ),
    )


def command_parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="insightface-privateframe",
        allow_abbrev=False,
    )
    commands = value.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser(
        "analyze",
        allow_abbrev=False,
        help="analyze a video and write the reusable result JSON only",
        description="Analyze a video and write the reusable result JSON without rendering video.",
        epilog=_DOTTED_CONFIG_HELP,
    )
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--workdir")
    _output_dir(analyze)
    _result(analyze)
    _common(analyze)

    render = commands.add_parser(
        "render",
        allow_abbrev=False,
        help="render video from an existing result JSON without model inference",
        description="Render an existing or edited PrivateFrame result JSON without rerunning models.",
        epilog=_DOTTED_RENDER_HELP,
    )
    render.add_argument("--input", required=True)
    render.add_argument("--workdir")
    _output_dir(render)
    _result(render)
    render.add_argument("--debug", help=argparse.SUPPRESS)
    _video_output(render)
    _render_config(render)
    render.add_argument("--no-verify-source", action="store_true")
    _common(render)

    process = commands.add_parser(
        "process",
        allow_abbrev=False,
        help="analyze to JSON and immediately render the paired result video",
        description="Analyze to a reusable JSON result, then render the paired result video.",
        epilog=_DOTTED_CONFIG_HELP,
    )
    process.add_argument("--config", required=True)
    process.add_argument("--input", required=True)
    process.add_argument("--workdir")
    _output_dir(process)
    process.add_argument("--debug", help=argparse.SUPPRESS)
    _video_output(process)
    _result(process)
    _render_config(process)
    _common(process)
    return value


def _reject_overlapping_config_paths(paths: list[str]) -> None:
    parts: list[tuple[str, tuple[str, ...]]] = []
    for path in paths:
        tokens = tuple(path.split("."))
        if not path or any(not token for token in tokens):
            raise ValueError(f"invalid configuration override path: {path}")
        parts.append((path, tokens))
    for index, (left_path, left) in enumerate(parts):
        for right_path, right in parts[index + 1 :]:
            shortest = min(len(left), len(right))
            if left[:shortest] == right[:shortest]:
                raise ValueError(
                    "configuration override paths overlap: "
                    f"{left_path} and {right_path}"
                )


def _validate_json_config_value(
    value: object,
    *,
    path: str,
    active_containers: set[int] | None = None,
) -> None:
    active = set() if active_containers is None else active_containers
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"configuration override --{path} must be finite")
        return
    if isinstance(value, (list, dict)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"configuration override --{path} must not be cyclic")
        active.add(identity)
        if isinstance(value, list):
            for item in value:
                _validate_json_config_value(
                    item,
                    path=path,
                    active_containers=active,
                )
        else:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"configuration override --{path} mapping keys must be strings"
                    )
                _validate_json_config_value(
                    item,
                    path=path,
                    active_containers=active,
                )
        active.remove(identity)
        return
    raise ValueError(
        f"configuration override --{path} must use JSON-compatible YAML types"
    )


def _parse_dotted_config_overrides(
    argv: list[str],
) -> tuple[list[str], dict[str, object]]:
    clean: list[str] = []
    parsed: list[tuple[str, object]] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        option, separator, attached = token.partition("=")
        if option.startswith("--") and "." in option[2:]:
            path = option[2:]
            if separator:
                raw_value = attached
            else:
                if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                    raise ValueError(f"configuration override --{path} requires a value")
                index += 1
                raw_value = argv[index]
            try:
                value = yaml.safe_load(raw_value)
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"invalid YAML value for configuration override --{path}: {exc}"
                ) from exc
            _validate_json_config_value(value, path=path)
            parsed.append((path, value))
        else:
            clean.append(token)
        index += 1
    _reject_overlapping_config_paths([path for path, _value in parsed])
    validate_config_override_paths([path for path, _value in parsed])
    return clean, dict(parsed)


def _print(value: dict[str, object], compact: bool) -> None:
    print(
        json.dumps(value, ensure_ascii=False, indent=None if compact else 2),
        flush=True,
    )


def _apply_output_defaults(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Resolve the simplified output-directory form without changing legacy paths."""

    if args.output_dir is None:
        if args.command in {"analyze", "process"} and args.workdir is None:
            parser.error(
                f"{args.command} requires --workdir unless --output-dir is used"
            )
        return

    paths = default_output_paths(args.input, args.output_dir)
    args.output_dir = str(paths.output_dir)
    if args.workdir is None:
        args.workdir = str(paths.workdir)
    if args.result is None:
        args.result = str(paths.result_json)
    if args.command in {"process", "render"} and args.redacted is None:
        args.redacted = str(paths.result_video)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = command_parser()
    try:
        clean, config_overrides = _parse_dotted_config_overrides(raw)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    args = parser.parse_args(clean)
    if config_overrides and args.command == "render":
        invalid = sorted(
            path
            for path in config_overrides
            if not path.startswith("render.")
        )
        if invalid:
            parser.error("render only accepts render.* configuration overrides")
    _apply_output_defaults(args, parser)
    if args.dry_run:
        _print(
            {
                "ok": True,
                "dry_run": True,
                **vars(args),
                "config_overrides": config_overrides,
            },
            args.json,
        )
        return 0
    try:
        command = args.command
        if command == "analyze":
            result = analyze_streaming_pipeline(
                config_path=args.config,
                input_path=args.input,
                workdir=args.workdir,
                result_path=args.result,
                config_overrides=config_overrides,
                config_override_root=Path.cwd(),
            )
        elif command == "render":
            if args.debug is None and args.redacted is None:
                raise ValueError("render requires at least one video output")
            result = render_streaming_artifacts(
                input_path=args.input,
                workdir=args.workdir,
                result_path=args.result,
                debug_path=args.debug,
                redacted_path=args.redacted,
                render_config=args.render_config,
                config_overrides=config_overrides,
                verify_source=not args.no_verify_source,
            )
        else:
            result = run_streaming_pipeline(
                config_path=args.config,
                input_path=args.input,
                debug_path=args.debug,
                redacted_path=args.redacted,
                render_config=args.render_config,
                workdir=args.workdir,
                result_path=args.result,
                config_overrides=config_overrides,
                config_override_root=Path.cwd(),
            )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    # Native PyAV/FFmpeg wrappers may become collectible only after the public
    # render function has returned.  Reclaim them before reporting success and
    # before a short-lived CLI interpreter starts tearing native libraries down.
    gc.collect()
    _print({"ok": True, **result}, args.json)
    return 0


__all__ = ["command_parser", "main"]
