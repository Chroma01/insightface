"""SQLite-backed bounded cache of encoded video packets and GOP metadata."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np


@dataclass(frozen=True)
class CachedVideoInfo:
    codec_name: str
    extradata: bytes
    rotation_degrees: int


def _rotation(path: str | Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not read orientation metadata: {path}")
    try:
        value = round(capture.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
    finally:
        capture.release()
    if value not in {0, 90, 180, 270}:
        raise ValueError(f"unsupported video orientation: {value}")
    return value


def orient_bgr(value: np.ndarray, rotation_degrees: int) -> np.ndarray:
    if rotation_degrees == 0:
        return value
    if rotation_degrees == 90:
        return cv2.rotate(value, cv2.ROTATE_90_CLOCKWISE)
    if rotation_degrees == 180:
        return cv2.rotate(value, cv2.ROTATE_180)
    if rotation_degrees == 270:
        return cv2.rotate(value, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported rotation: {rotation_degrees}")


def crop_bgr(
    image: np.ndarray, crop: tuple[int, int, int, int]
) -> np.ndarray:
    """Return one exact, zero-padded source-coordinate BGR rectangle."""

    x1, y1, x2, y2 = crop
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid crop rectangle: {crop}")
    canvas = np.zeros((y2 - y1, x2 - x1, 3), dtype=image.dtype)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if sx2 > sx1 and sy2 > sy1:
        canvas[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1] = image[
            sy1:sy2, sx1:sx2
        ]
    return canvas


class EncodedPacketCache:
    """Append packets once, re-decode bounded historical intervals on demand.

    Rows are deleted only at a random-access boundary. SQLite may retain the
    allocated pages for reuse, but the live packet payload remains bounded.
    """

    def __init__(self, path: str | Path, source: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self._closed = False
        try:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA temp_store=MEMORY")
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS packets (
                    sequence INTEGER PRIMARY KEY,
                    pts INTEGER,
                    dts INTEGER,
                    duration INTEGER,
                    time_base_num INTEGER NOT NULL,
                    time_base_den INTEGER NOT NULL,
                    is_keyframe INTEGER NOT NULL,
                    gop_sequence INTEGER NOT NULL,
                    payload BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS packets_gop ON packets(gop_sequence, sequence);
                CREATE INDEX IF NOT EXISTS packets_pts ON packets(pts, sequence);
                CREATE TABLE IF NOT EXISTS frames (
                    frame_index INTEGER PRIMARY KEY,
                    pts INTEGER,
                    packet_sequence INTEGER NOT NULL,
                    gop_sequence INTEGER NOT NULL
                );
                """
            )
            self.rotation_degrees = _rotation(source)
        except BaseException:
            self._closed = True
            try:
                self.connection.close()
            finally:
                self._remove_runtime_files()
            raise
        self.sequence = -1
        self.gop_sequence = 0
        self.last_packet_sequence = -1
        self._pending_since_commit = 0
        self.info: CachedVideoInfo | None = None
        self.frames_decoded = 0
        self.evicted_packets = 0
        self.evicted_bytes = 0
        self.historical_decode_requests = 0
        self.historical_packets_read = 0
        self.peak_decode_range_bytes = 0

    def configure(self, stream: Any) -> None:
        self.info = CachedVideoInfo(
            codec_name=str(stream.codec_context.name),
            extradata=bytes(stream.codec_context.extradata or b""),
            rotation_degrees=self.rotation_degrees,
        )

    def append_packet(self, packet: Any) -> tuple[int, int] | None:
        payload = bytes(packet)
        if not payload:
            return None
        self.sequence += 1
        if bool(packet.is_keyframe):
            self.gop_sequence = self.sequence
        time_base = Fraction(packet.time_base)
        self.connection.execute(
            "INSERT INTO packets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.sequence,
                packet.pts,
                packet.dts,
                packet.duration,
                time_base.numerator,
                time_base.denominator,
                int(bool(packet.is_keyframe)),
                self.gop_sequence,
                sqlite3.Binary(payload),
            ),
        )
        self.last_packet_sequence = self.sequence
        self._pending_since_commit += 1
        if self._pending_since_commit >= 120:
            self.connection.commit()
            self._pending_since_commit = 0
        return self.sequence, self.gop_sequence

    def record_frame(self, frame_index: int, pts: int | None, packet_sequence: int, gop_sequence: int) -> None:
        if pts is None:
            raise RuntimeError(
                "video frame has no PTS; reliable GOP history decoding is unsupported"
            )
        if pts is not None:
            owner = self.connection.execute(
                "SELECT sequence, gop_sequence FROM packets WHERE pts = ? ORDER BY sequence LIMIT 1",
                (pts,),
            ).fetchone()
            if owner is not None:
                packet_sequence, gop_sequence = int(owner[0]), int(owner[1])
        self.connection.execute(
            "INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?)",
            (frame_index, pts, packet_sequence, gop_sequence),
        )
        self.frames_decoded = max(self.frames_decoded, frame_index + 1)

    def commit(self) -> None:
        self.connection.commit()
        self._pending_since_commit = 0

    def evict_before_frame(self, frame_index: int) -> None:
        row = self.connection.execute(
            "SELECT gop_sequence FROM frames WHERE frame_index >= ? ORDER BY frame_index LIMIT 1",
            (max(0, frame_index),),
        ).fetchone()
        if row is None:
            return
        keep_gop = int(row[0])
        aggregate = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM packets WHERE sequence < ?",
            (keep_gop,),
        ).fetchone()
        self.evicted_packets += int(aggregate[0])
        self.evicted_bytes += int(aggregate[1])
        self.connection.execute("DELETE FROM packets WHERE sequence < ?", (keep_gop,))
        self.connection.execute("DELETE FROM frames WHERE gop_sequence < ?", (keep_gop,))
        self.connection.commit()

    def decode_range(
        self,
        first_frame: int,
        last_frame: int,
        *,
        crop: tuple[int, int, int, int] | None = None,
        crops: dict[int, tuple[int, int, int, int]] | None = None,
    ) -> dict[int, np.ndarray]:
        if first_frame > last_frame:
            return {}
        if crop is not None and crops is not None:
            raise ValueError("crop and crops are mutually exclusive")
        if self.info is None:
            raise RuntimeError("packet cache has not been configured")
        start = self.connection.execute(
            "SELECT gop_sequence FROM frames WHERE frame_index <= ? ORDER BY frame_index DESC LIMIT 1",
            (first_frame,),
        ).fetchone()
        end = self.connection.execute(
            "SELECT packet_sequence FROM frames WHERE frame_index >= ? ORDER BY frame_index LIMIT 1",
            (last_frame,),
        ).fetchone()
        if start is None or end is None:
            raise KeyError(f"frames {first_frame}..{last_frame} are outside the encoded cache")
        rows = self.connection.execute(
            "SELECT sequence, pts, dts, duration, time_base_num, time_base_den, payload "
            "FROM packets WHERE sequence >= ? AND sequence <= ? ORDER BY sequence",
            (int(start[0]), self.last_packet_sequence),
        )
        wanted_pts = {
            int(row[1]): int(row[0])
            for row in self.connection.execute(
                "SELECT frame_index, pts FROM frames WHERE frame_index BETWEEN ? AND ?",
                (first_frame, last_frame),
            )
            if row[1] is not None
        }
        decoder = av.CodecContext.create(self.info.codec_name, "r")
        decoder.extradata = self.info.extradata
        output: dict[int, np.ndarray] = {}

        def receive(frames: list[Any]) -> None:
            for frame in frames:
                if frame.pts is None or int(frame.pts) not in wanted_pts:
                    continue
                index = wanted_pts[int(frame.pts)]
                image = orient_bgr(
                    frame.to_ndarray(format="bgr24"), self.info.rotation_degrees
                )
                selected_crop = crops[index] if crops is not None else crop
                if selected_crop is not None:
                    image = crop_bgr(image, selected_crop)
                output[index] = image

        self.historical_decode_requests += 1
        try:
            for _sequence, pts, dts, duration, num, den, payload in rows:
                self.historical_packets_read += 1
                packet = av.Packet(bytes(payload))
                packet.pts = pts
                packet.dts = dts
                packet.duration = duration
                packet.time_base = Fraction(int(num), int(den))
                receive(decoder.decode(packet))
                if len(output) == len(wanted_pts):
                    break
        finally:
            rows.close()
        if len(output) != len(wanted_pts):
            receive(decoder.decode(None))
        missing = sorted(set(range(first_frame, last_frame + 1)) - set(output))
        if missing:
            raise RuntimeError(f"packet cache failed to decode frames: {missing[:8]}")
        self.peak_decode_range_bytes = max(
            self.peak_decode_range_bytes,
            sum(int(image.nbytes) for image in output.values()),
        )
        return output

    def oldest_frame_index(self) -> int | None:
        """Return the oldest frame whose random-access GOP is still cached."""

        row = self.connection.execute(
            "SELECT MIN(frame_index) FROM frames"
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def live_payload_bytes(self) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(SUM(LENGTH(payload)), 0) FROM packets"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        if self._closed:
            self._remove_runtime_files()
            return
        self._closed = True
        try:
            self.connection.commit()
            self._pending_since_commit = 0
        finally:
            try:
                self.connection.close()
            finally:
                self._remove_runtime_files()

    def _remove_runtime_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                self.path.with_name(f"{self.path.name}{suffix}").unlink()
            except FileNotFoundError:
                pass


def iter_cached_frames(
    source: str | Path, cache: EncodedPacketCache
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Demux once, append encoded packets, and yield display-oriented BGR frames."""

    container = av.open(str(source))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    cache.configure(stream)
    average_rate = float(stream.average_rate or 0.0)
    frame_index = 0
    try:
        for packet in container.demux(stream):
            marker = cache.append_packet(packet)
            frames = packet.decode()
            for frame in frames:
                if marker is None:
                    marker = (cache.last_packet_sequence, cache.gop_sequence)
                cache.record_frame(frame_index, frame.pts, marker[0], marker[1])
                timestamp = float(frame.pts * frame.time_base) if frame.pts is not None else (
                    frame_index / average_rate if average_rate > 0.0 else 0.0
                )
                yield frame_index, timestamp, orient_bgr(
                    frame.to_ndarray(format="bgr24"), cache.rotation_degrees
                )
                frame_index += 1
    finally:
        try:
            cache.commit()
        finally:
            container.close()


def iter_oriented_frames(
    source: str | Path,
) -> Iterator[tuple[int, float, int | None, np.ndarray]]:
    """Decode one display-oriented pass without creating a packet cache."""

    container = av.open(str(source))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    rotation_degrees = _rotation(source)
    average_rate = float(stream.average_rate or 0.0)
    try:
        for frame_index, frame in enumerate(container.decode(stream)):
            timestamp = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None
                else frame_index / average_rate
                if average_rate > 0.0
                else 0.0
            )
            yield (
                frame_index,
                timestamp,
                int(frame.pts) if frame.pts is not None else None,
                orient_bgr(
                    frame.to_ndarray(format="bgr24"), rotation_degrees
                ),
            )
    finally:
        container.close()


__all__ = [
    "EncodedPacketCache",
    "crop_bgr",
    "iter_cached_frames",
    "iter_oriented_frames",
    "orient_bgr",
]
