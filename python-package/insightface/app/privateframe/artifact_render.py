"""Render debug or redacted MP4 files from finalized observation artifacts."""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import time
import unicodedata
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import av
import cv2
import numpy as np

from .artifacts import sha256_file
from .geometry import clip
from .packet_cache import iter_oriented_frames
from .recognition import apply_identity_policy
from .video import paths_are_distinct, probe_video, temporary_video_path


@dataclass(frozen=True)
class RenderTarget:
    mode: str
    path: Path


class _VideoWriter(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def finish(self) -> None: ...

    def commit(self) -> None: ...

    def abort(self) -> None: ...


def _release_native_writer_cycles(
    writers: list[_VideoWriter],
    *,
    pyav: bool,
) -> None:
    if not pyav:
        return
    writers.clear()
    gc.collect()


def _raise_if_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise InterruptedError("PrivateFrame operation was cancelled")


def _color(track_id: str) -> tuple[int, int, int]:
    value = (int(track_id[1:]) + 1) * 2654435761 & 0xFFFFFFFF
    return (
        80 + (value & 127),
        80 + ((value >> 8) & 127),
        80 + ((value >> 16) & 127),
    )


def _dashed_line(
    frame: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash: int,
    gap: int,
) -> None:
    length = round(np.linalg.norm(np.asarray(second) - np.asarray(first)))
    if length <= 0:
        return
    direction = (np.asarray(second, dtype=float) - np.asarray(first, dtype=float)) / length
    for start in range(0, length + 1, dash + gap):
        end = min(length, start + dash)
        a = np.asarray(first) + direction * start
        b = np.asarray(first) + direction * end
        cv2.line(
            frame,
            tuple(np.rint(a).astype(int)),
            tuple(np.rint(b).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _debug_box(
    frame: np.ndarray,
    item: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    value = clip(item["box"], frame.shape[1] - 1, frame.shape[0] - 1)
    x1, y1, x2, y2 = np.rint(value).astype(int)
    color = _color(item["track_id"])
    thickness = int(settings["debug_line_thickness"])
    if item["source"] == "detector":
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA,
        )
    else:
        dash, gap = (11, 5) if int(item.get("local_match_count", 0)) >= 1 else (2, 5)
        _dashed_line(frame, (x1, y1), (x2, y1), color, thickness, dash, gap)
        _dashed_line(frame, (x2, y1), (x2, y2), color, thickness, dash, gap)
        _dashed_line(frame, (x2, y2), (x1, y2), color, thickness, dash, gap)
        _dashed_line(frame, (x1, y2), (x1, y1), color, thickness, dash, gap)
    cv2.putText(
        frame,
        item["track_id"],
        (x1, max(15, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )


def _debug_frame_number(frame: np.ndarray, frame_idx: int) -> None:
    """Draw a prominent frame number away from video-player controls."""

    label = f"FRAME {frame_idx:06d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.65, min(1.2, min(frame.shape[:2]) / 1080.0 * 0.9))
    thickness = 2
    padding = 12
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    panel_width = text_width + padding * 2
    panel_height = text_height + baseline + padding * 2
    cv2.rectangle(
        frame,
        (0, 0),
        (panel_width, panel_height),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (padding, padding + text_height),
        font,
        scale,
        (80, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _odd_kernel(value: float) -> int:
    kernel = max(1, round(value))
    return kernel if kernel % 2 == 1 else kernel + 1


def _gaussian_blur(region: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    kernel = _odd_kernel(
        max(
            int(settings["min_kernel"]),
            round(max(region.shape[:2]) * float(settings["kernel_ratio"])),
        )
    )
    sigma = float(settings.get("sigma", 0.0))
    algorithm = str(settings.get("algorithm", "exact"))
    max_side = int(settings.get("max_side", 64))

    if algorithm == "pyramid" and max(region.shape[:2]) > max_side:
        scale = max_side / max(region.shape[:2])
        reduced_width = max(1, round(region.shape[1] * scale))
        reduced_height = max(1, round(region.shape[0] * scale))
        reduced = cv2.resize(
            region,
            (reduced_width, reduced_height),
            interpolation=cv2.INTER_AREA,
        )
        reduced_kernel = _odd_kernel(kernel * scale)
        reduced_sigma = sigma * scale if sigma > 0.0 else 0.0
        reduced = cv2.GaussianBlur(
            reduced,
            (reduced_kernel, reduced_kernel),
            sigmaX=reduced_sigma,
            sigmaY=reduced_sigma,
        )
        return cv2.resize(
            reduced,
            (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    return cv2.GaussianBlur(
        region,
        (kernel, kernel),
        sigmaX=sigma,
        sigmaY=sigma,
    )


def _blur(frame: np.ndarray, value: list[float], settings: dict[str, Any]) -> None:
    redaction = settings["redaction"]
    method = str(redaction["method"])
    scale = float(redaction["box_scale"])
    gaussian = redaction.get("gaussian", {})
    mosaic = redaction.get("mosaic", {})
    feather = redaction.get("feather", {"enabled": False})

    target = np.asarray(value, dtype=float)
    width, height = target[2] - target[0], target[3] - target[1]
    center_x = (target[0] + target[2]) * 0.5
    center_y = (target[1] + target[3]) * 0.5
    target = np.asarray(
        [
            center_x - width * scale * 0.5,
            center_y - height * scale * 0.5,
            center_x + width * scale * 0.5,
            center_y + height * scale * 0.5,
        ]
    )
    x1, y1, x2, y2 = np.rint(clip(target, frame.shape[1], frame.shape[0])).astype(int)
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    if method == "gaussian":
        redacted = _gaussian_blur(region, gaussian)
    elif method == "mosaic":
        block = max(
            int(mosaic["min_block_size"]),
            round(max(region.shape[:2]) * float(mosaic["block_size_ratio"])),
        )
        reduced_width = max(1, (region.shape[1] + block - 1) // block)
        reduced_height = max(1, (region.shape[0] + block - 1) // block)
        reduced = cv2.resize(
            region,
            (reduced_width, reduced_height),
            interpolation=cv2.INTER_AREA,
        )
        redacted = cv2.resize(
            reduced,
            (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:  # Defensive guard for callers that bypass configuration loading.
        raise ValueError(f"unsupported redaction method: {method}")

    if not bool(feather.get("enabled", False)):
        frame[y1:y2, x1:x2] = redacted
        return
    feather_pixels = max(
        int(feather.get("min_pixels", 1)),
        round(min(region.shape[:2]) * float(feather.get("ratio", 0.10))),
    )
    row_distance = np.minimum(
        np.arange(region.shape[0]),
        np.arange(region.shape[0] - 1, -1, -1),
    )[:, None]
    column_distance = np.minimum(
        np.arange(region.shape[1]),
        np.arange(region.shape[1] - 1, -1, -1),
    )[None, :]
    alpha = np.clip(
        np.minimum(row_distance, column_distance) / max(feather_pixels, 1),
        0.0,
        1.0,
    )
    # Smoothstep avoids a visible linear seam around the redaction region.
    alpha = (alpha * alpha * (3.0 - 2.0 * alpha))[..., None]
    blended = redacted.astype(np.float32) * alpha + region.astype(np.float32) * (1.0 - alpha)
    frame[y1:y2, x1:x2] = np.rint(blended).astype(np.uint8)


def _identity_should_blur(
    item: dict[str, Any],
    policy: dict[str, Any],
    recognition: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Apply a render-only identity policy with privacy-safe fallbacks."""

    mode = str(policy.get("mode", "all"))
    if mode not in {"all", "blur_only", "exempt"}:
        return True, "fail_safe_invalid_recognition_policy"
    if (
        mode != "all"
        and item.get("endpoint_repair_reason")
        == "interpolate_unanchored_endpoint"
    ):
        # The parent track has identity evidence, but this tail is deliberately
        # generated without SCRFD, Verifier, or ArcFace calls. It may extend a
        # target exemption geometrically, never biometrically.
        return True, "fail_safe_unreviewed_interpolate_endpoint"
    if mode != "all" and (not isinstance(recognition, dict) or recognition.get("enabled") is not True):
        # ``render_streaming_artifacts`` rejects this mismatch. The lower-level
        # renderer remains privacy safe when called directly with an incomplete
        # result rather than trusting any stray track mappings.
        return True, "fail_safe_missing_recognition_artifact"
    tracks = recognition.get("tracks", {}) if isinstance(recognition, dict) else {}
    record = tracks.get(str(item.get("track_id"))) if isinstance(tracks, dict) else None
    if mode != "all":
        gallery_people = recognition.get("gallery_persons")
        if gallery_people is None and isinstance(recognition.get("gallery"), dict):
            gallery_people = recognition["gallery"].get("persons")
        if not isinstance(gallery_people, list):
            return True, "fail_safe_missing_gallery_identity"
        known_people = {
            unicodedata.normalize("NFC", value.strip())
            for value in gallery_people
            if isinstance(value, str) and value.strip()
        }
        if isinstance(record, dict) and record.get("status") == "CONFIRMED":
            person_id = record.get("person_id")
            if not isinstance(person_id, str) or unicodedata.normalize("NFC", person_id) not in known_people:
                return True, "fail_safe_unknown_gallery_identity"
        raw_targets = policy.get("target_persons")
        if not isinstance(raw_targets, list) or not raw_targets:
            return True, "fail_safe_invalid_target_persons"
        targets: list[str] = []
        for value in raw_targets:
            if not isinstance(value, str) or not value.strip():
                return True, "fail_safe_invalid_target_persons"
            targets.append(unicodedata.normalize("NFC", value.strip()))
        if len(set(targets)) != len(targets) or not set(targets) <= known_people:
            return True, "fail_safe_invalid_target_persons"
    decision = apply_identity_policy(
        mode,
        record,
        targets if mode != "all" else (),
    )
    return bool(decision.should_blur), str(decision.reason)


def _bitrate(value: Any) -> int:
    text = str(value).strip().lower()
    multipliers = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    if text and text[-1] in multipliers:
        return round(float(text[:-1]) * multipliers[text[-1]])
    return int(text)


def _stream_from_template(container: Any, stream: Any) -> Any:
    method = getattr(container, "add_stream_from_template", None)
    if method is not None:
        return method(stream)
    return container.add_stream(template=stream)


def _video_arguments(settings: dict[str, Any]) -> list[str]:
    encoder = str(settings["encoder"])
    arguments = ["-c:v", encoder]
    preset = settings.get("preset")
    if preset:
        arguments.extend(["-preset", str(preset)])
    rate = settings["rate_control"]
    mode = str(rate["mode"])
    if mode in {"crf", "cq"}:
        quality = str(int(rate["quality"]))
        if "nvenc" in encoder:
            arguments.extend(["-rc:v", "vbr", "-cq:v", quality, "-b:v", "0"])
        else:
            arguments.extend(["-crf", quality])
    elif mode == "vbr":
        arguments.extend(["-b:v", str(rate["bitrate"])])
        if rate.get("max_bitrate"):
            arguments.extend(["-maxrate", str(rate["max_bitrate"])])
    elif mode == "cbr":
        bitrate = str(rate["bitrate"])
        arguments.extend(["-b:v", bitrate, "-minrate", bitrate, "-maxrate", bitrate])
        if rate.get("buffer_size"):
            arguments.extend(["-bufsize", str(rate["buffer_size"])])
    arguments.extend(["-pix_fmt", str(settings["pixel_format"])])
    if int(settings.get("keyframe_interval", 0)) > 0:
        arguments.extend(["-g", str(int(settings["keyframe_interval"]))])
    if bool(settings.get("faststart", True)):
        arguments.extend(["-movflags", "+faststart"])
    return arguments


def _pyav_video_options(settings: dict[str, Any]) -> dict[str, str]:
    encoder = str(settings["encoder"])
    options: dict[str, str] = {}
    if settings.get("preset"):
        options["preset"] = str(settings["preset"])
    rate = settings["rate_control"]
    mode = str(rate["mode"])
    if mode in {"crf", "cq"}:
        quality = str(int(rate["quality"]))
        if "nvenc" in encoder:
            options.update({"rc": "vbr", "cq": quality, "b": "0"})
        else:
            options["crf"] = quality
    elif mode == "cbr":
        options["minrate"] = str(rate["bitrate"])
        options["maxrate"] = str(rate["bitrate"])
        if rate.get("buffer_size"):
            options["bufsize"] = str(rate["buffer_size"])
    elif rate.get("max_bitrate"):
        options["maxrate"] = str(rate["max_bitrate"])
    return options


def _copy_audio(
    silent_video: Path,
    source: Path,
    destination: Path,
    *,
    requested_mode: str,
    maximum_duration: float,
) -> bool:
    """Remux source audio with the encoded video; return false if absent."""

    video_input = av.open(str(silent_video))
    audio_input = av.open(str(source))
    audio_stream = next(iter(audio_input.streams.audio), None)
    if audio_stream is None:
        video_input.close()
        audio_input.close()
        return False
    if requested_mode == "aac" and audio_stream.codec_context.name != "aac":
        video_input.close()
        audio_input.close()
        raise RuntimeError(
            "PyAV audio mode aac currently requires AAC source audio; "
            "choose copy/none or use the ffmpeg backend for transcoding"
        )
    output = av.open(str(destination), "w", options={"movflags": "+faststart"})
    try:
        output_video = _stream_from_template(output, video_input.streams.video[0])
        output_audio = _stream_from_template(output, audio_stream)
        for packet in video_input.demux(video_input.streams.video[0]):
            if packet.dts is None:
                continue
            packet.stream = output_video
            output.mux(packet)
        for packet in audio_input.demux(audio_stream):
            if packet.dts is None:
                continue
            timestamp = (
                float(packet.pts * packet.time_base)
                if packet.pts is not None and packet.time_base is not None
                else 0.0
            )
            if timestamp >= maximum_duration:
                break
            packet.stream = output_audio
            output.mux(packet)
    finally:
        output.close()
        video_input.close()
        audio_input.close()
    return True


class _PyAVWriter:
    def __init__(
        self,
        destination: Path,
        source: Path,
        width: int,
        height: int,
        fps: float,
        settings: dict[str, Any],
        audio_mode: str,
    ):
        self.destination = destination
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = temporary_video_path(destination)
        self.muxed: Path | None = None
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = 0
        self.audio_mode = audio_mode
        self.container = av.open(
            str(self.temporary),
            "w",
            options={"movflags": "+faststart"} if bool(settings.get("faststart", True)) else None,
        )
        rate = Fraction(str(fps)).limit_denominator(1_000_000)
        self.stream = self.container.add_stream(str(settings["encoder"]), rate=rate)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = str(settings["pixel_format"])
        self.stream.options = _pyav_video_options(settings)
        rate_control = settings["rate_control"]
        if str(rate_control["mode"]) in {"vbr", "cbr"}:
            self.stream.bit_rate = _bitrate(rate_control["bitrate"])
        if int(settings.get("keyframe_interval", 0)) > 0:
            self.stream.codec_context.gop_size = int(settings["keyframe_interval"])
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"renderer frame shape changed: {frame.shape} != {(self.height, self.width, 3)}")
        video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)
        self.frames += 1

    def finish(self) -> None:
        for packet in self.stream.encode(None):
            self.container.mux(packet)
        self.container.close()
        self.closed = True
        if self.audio_mode == "none":
            return
        self.muxed = temporary_video_path(self.destination)
        if not _copy_audio(
            self.temporary,
            self.source,
            self.muxed,
            requested_mode=self.audio_mode,
            maximum_duration=self.frames / self.fps,
        ):
            self.muxed = None

    def commit(self) -> None:
        ready = self.muxed or self.temporary
        os.replace(ready, self.destination)
        if ready != self.temporary and self.temporary.exists():
            self.temporary.unlink()

    def abort(self) -> None:
        if not self.closed:
            with suppress(Exception):
                self.container.close()
            self.closed = True
        for path in (self.temporary, self.muxed):
            if path is not None and path.exists():
                path.unlink()


class _FFmpegWriter:
    def __init__(
        self,
        destination: Path,
        source: Path,
        width: int,
        height: int,
        fps: float,
        settings: dict[str, Any],
        audio_mode: str,
    ):
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("ffmpeg is required for artifact video rendering")
        self.destination = destination
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = temporary_video_path(destination)
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.12g}",
            "-i",
            "pipe:0",
        ]
        if audio_mode != "none":
            command.extend(["-i", str(source), "-map", "0:v:0", "-map", "1:a:0?"])
        else:
            command.extend(["-map", "0:v:0"])
        command.extend(_video_arguments(settings))
        if audio_mode == "copy":
            command.extend(["-c:a", "copy", "-shortest"])
        elif audio_mode == "aac":
            command.extend(
                [
                    "-c:a",
                    "aac",
                    "-b:a",
                    str(settings.get("audio", {}).get("bitrate", "192k")),
                    "-shortest",
                ]
            )
        else:
            command.append("-an")
        command.append(str(self.temporary))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.width = width
        self.height = height

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"renderer frame shape changed: {frame.shape} != {(self.height, self.width, 3)}")
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg stopped while receiving video frames") from exc

    def finish(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        error = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr is not None
            else ""
        )
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg video encoder failed ({return_code}): {error.strip()}")

    def commit(self) -> None:
        os.replace(self.temporary, self.destination)

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        if self.temporary.exists():
            self.temporary.unlink()


def render_artifacts(
    *,
    source: str | Path,
    targets: list[RenderTarget],
    settings: dict[str, Any],
    analysis_result: dict[str, Any],
    verify_source: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    _raise_if_cancelled(is_cancelled)
    source_path = Path(source).expanduser().resolve()
    if not targets:
        raise ValueError("at least one render target is required")
    resolved_targets = [RenderTarget(target.mode, target.path.resolve()) for target in targets]
    if not paths_are_distinct([source_path, *(target.path for target in resolved_targets)]):
        raise ValueError("input and every render output path must be distinct")
    for target in resolved_targets:
        if target.mode not in {"debug", "redacted"}:
            raise ValueError(f"unsupported render mode: {target.mode}")

    expected_source = analysis_result["source_video"]
    if verify_source and sha256_file(source_path) != expected_source["sha256"]:
        raise ValueError("source video SHA256 does not match the analysis manifest")
    _raise_if_cancelled(is_cancelled)
    observations = analysis_result["observations"]
    recognition = analysis_result.get("recognition")
    recognition_policy = settings.get("recognition_policy", {"mode": "all", "target_persons": []})
    by_frame: dict[int, list[tuple[dict[str, Any], bool, str]]] = {}
    blurred_observations = 0
    kept_observations = 0
    fail_safe_observations = 0
    for item in observations:
        should_blur, reason = _identity_should_blur(item, recognition_policy, recognition)
        by_frame.setdefault(int(item["frame_idx"]), []).append((item, should_blur, reason))
        if should_blur:
            blurred_observations += 1
            if reason.startswith("fail_safe_"):
                fail_safe_observations += 1
        else:
            kept_observations += 1

    metadata = expected_source["metadata"]
    width, height = int(metadata["width"]), int(metadata["height"])
    fps = float(metadata["fps"])
    expected_frames = int(metadata["frame_count"])
    if progress is not None:
        progress(0, expected_frames, "render")
    _raise_if_cancelled(is_cancelled)
    audio = settings.get("audio", {})
    writer_type = _PyAVWriter if str(settings.get("backend", "pyav")) == "pyav" else _FFmpegWriter
    pyav_writer = writer_type is _PyAVWriter
    writers: list[_VideoWriter] = []
    try:
        for target in resolved_targets:
            writers.append(
                writer_type(
                    target.path,
                    source_path,
                    width,
                    height,
                    fps,
                    settings,
                    str(audio.get(target.mode, "none")),
                )
            )
    except Exception:
        try:
            for index in range(len(writers)):
                writers[index].abort()
        finally:
            _release_native_writer_cycles(writers, pyav=pyav_writer)
        raise
    count = 0
    try:
        for frame_idx, _timestamp, _pts, frame in iter_oriented_frames(source_path):
            _raise_if_cancelled(is_cancelled)
            for target_index, target in enumerate(resolved_targets):
                output = frame.copy()
                for item, should_blur, _reason in by_frame.get(frame_idx, []):
                    if target.mode == "debug":
                        _debug_box(output, item, settings)
                    elif should_blur:
                        _blur(output, item["box"], settings)
                if target.mode == "debug":
                    _debug_frame_number(output, frame_idx)
                writers[target_index].write(output)
            count += 1
            if progress is not None:
                progress(count, expected_frames, "render")
            _raise_if_cancelled(is_cancelled)
        if count != expected_frames:
            raise RuntimeError(f"rendered {count} frames, analysis manifest declares {expected_frames}")
        for index in range(len(writers)):
            writers[index].finish()
        for index in range(len(writers)):
            writers[index].commit()
    except Exception:
        try:
            for index in range(len(writers)):
                writers[index].abort()
        finally:
            _release_native_writer_cycles(writers, pyav=pyav_writer)
        raise

    outputs = []
    for target in resolved_targets:
        output_metadata = probe_video(target.path)
        outputs.append(
            {
                "mode": target.mode,
                "path": str(target.path),
                "sha256": sha256_file(target.path),
                "metadata": output_metadata.to_dict(),
            }
        )
    result = {
        "frame_count": count,
        "observations": len(observations),
        "blurred_observations": blurred_observations,
        "kept_observations": kept_observations,
        "fail_safe_observations": fail_safe_observations,
        "seconds": time.perf_counter() - started,
        "outputs": outputs,
    }
    if pyav_writer:
        # PyAV stream/container wrappers can participate in native reference
        # cycles even after every container has been closed.  Reclaim them
        # while Python and FFmpeg are still fully initialized; otherwise some
        # macOS PyAV builds can defer destruction to interpreter shutdown and
        # race a native recursive-mutex teardown after a successful render.
        _release_native_writer_cycles(writers, pyav=True)
    return result


__all__ = ["RenderTarget", "_identity_should_blur", "render_artifacts"]
