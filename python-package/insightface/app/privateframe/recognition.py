"""Fail-safe, track-level face recognition primitives.

The module deliberately keeps identity inference separate from detection,
tracking, admission, and rendering.  Streaming code can retain a bounded set
of aligned candidates and inject them into :class:`RecognitionEngine` only
after canonical tracks have been formed.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from .geometry import area_ratio, containment, iou, normalized_center_distance
from .models import detect_faces

ARC_FACE_TEMPLATE_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
ARC_FACE_TEMPLATE_112.setflags(write=False)

SUPPORTED_GALLERY_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
RECOGNITION_MODES = frozenset({"all", "blur_only", "exempt"})

# Gallery references are user-supplied still images, so they use a deliberately
# small, fixed upright-only detector policy instead of inheriting the much more
# expensive multi-angle video scan or local-revalidation policy. The 640 view
# supplies accurate landmarks for ordinary faces; 128 recovers very large
# faces. SCRFD performs global multi-scale NMS before PrivateFrame applies its
# stable output cap and the one-face Gallery validator receives the results.
GALLERY_DETECTOR_INPUT_SIZES = (640, 128)
GALLERY_DETECTOR_CONFIDENCE_THRESHOLD = 0.50

# Video recognition alignment prefers already-computed local SCRFD landmarks
# and can fall back to the paired full-frame SCRFD box and landmarks. A single
# source gate applies to every sampling profile; profile differences concern
# only how many candidates are selected and how identity is decided. Gallery
# references follow the separate upright policy above.
LOCAL_LANDMARK_CONFIDENCE_THRESHOLD = 0.70
LOCAL_LANDMARK_MAX_CENTER_DISTANCE = 0.35
LOCAL_LANDMARK_MAX_AREA_RATIO = 2.0
LOCAL_LANDMARK_MIN_IOU = 0.30
LOCAL_LANDMARK_MIN_CONTAINMENT = 0.60

TEMPORAL_EVIDENCE_MIN_SELECTED_FRAMES = 3
TEMPORAL_EVIDENCE_MIN_ADJACENT_GAP_SECONDS = 1.0
TEMPORAL_EVIDENCE_SIMILARITY_OFFSET = 0.10
SINGLE_FRAME_SIMILARITY_OFFSET = 0.05

RECOGNITION_LANDMARK_SOURCES = frozenset({"local_scrfd", "global_scrfd"})


def _diagnostic_failure_reason(stage: str, error: BaseException) -> str:
    """Return a stable, message-free reason for a recoverable model failure."""

    return f"{stage}:{type(error).__name__}"


def detect_gallery_faces_upright(
    analysis: Any,
    image: np.ndarray,
    *,
    max_detections: int,
) -> list[dict[str, Any]]:
    """Detect Gallery faces at fixed 640/128 upright views without review."""

    result = detect_faces(
        analysis,
        image,
        input_sizes=GALLERY_DETECTOR_INPUT_SIZES,
        confidence_threshold=GALLERY_DETECTOR_CONFIDENCE_THRESHOLD,
        max_detections=max_detections,
    )
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise TypeError("Gallery SCRFD detector must return a sequence")
    if not all(isinstance(value, Mapping) for value in result):
        raise TypeError("Gallery SCRFD detections must be mappings")
    return [dict(value) for value in result]


def local_landmark_box_agreement(
    original_box: Any,
    local_box: Any,
) -> tuple[bool, dict[str, float]]:
    """Require a local-SCRFD face to be the same full-frame detection."""

    original = np.asarray(original_box, dtype=np.float64).reshape(4)
    local = np.asarray(local_box, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(original)) or not np.all(np.isfinite(local)):
        raise ValueError("recognition landmark boxes must be finite")
    overlap = float(iou(original, local))
    inside = max(
        float(containment(original, local)),
        float(containment(local, original)),
    )
    metrics = {
        "iou": overlap,
        "containment": inside,
        "center_distance": float(
            normalized_center_distance(original, local)
        ),
        "area_ratio": float(area_ratio(original, local)),
    }
    accepted = bool(
        metrics["center_distance"] <= LOCAL_LANDMARK_MAX_CENTER_DISTANCE
        and metrics["area_ratio"] <= LOCAL_LANDMARK_MAX_AREA_RATIO
        and (
            metrics["iou"] >= LOCAL_LANDMARK_MIN_IOU
            or metrics["containment"] >= LOCAL_LANDMARK_MIN_CONTAINMENT
        )
    )
    return accepted, metrics


def _landmark_array(value: Any) -> np.ndarray:
    landmarks = np.asarray(value, dtype=np.float64)
    if landmarks.shape != (5, 2):
        raise ValueError(f"ArcFace alignment requires five 2D landmarks, received {landmarks.shape}")
    if not np.all(np.isfinite(landmarks)):
        raise ValueError("ArcFace landmarks must be finite")
    centered = landmarks - np.mean(landmarks, axis=0, keepdims=True)
    if float(np.sum(centered * centered)) <= np.finfo(np.float64).eps:
        raise ValueError("ArcFace landmarks are degenerate")
    return landmarks


def estimate_arcface_similarity(landmarks: Any) -> np.ndarray:
    """Estimate the least-squares similarity mapping into the ArcFace 112 template."""

    source = _landmark_array(landmarks)
    destination = ARC_FACE_TEMPLATE_112.astype(np.float64)
    source_mean = np.mean(source, axis=0)
    destination_mean = np.mean(destination, axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    covariance = destination_centered.T @ source_centered / len(source)
    left, singular_values, right_transposed = np.linalg.svd(covariance)
    correction = np.eye(2, dtype=np.float64)
    if np.linalg.det(left) * np.linalg.det(right_transposed) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_transposed
    variance = float(np.sum(source_centered * source_centered) / len(source))
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = destination_mean - scale * (rotation @ source_mean)
    matrix = np.column_stack((scale * rotation, translation)).astype(np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("ArcFace similarity transform is not finite")
    return matrix


def arcface_align_112(frame_bgr: np.ndarray, landmarks: Any) -> np.ndarray:
    """Return the conventional 112x112 ArcFace BGR alignment crop."""

    image = np.asarray(frame_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"ArcFace source image must be HWC BGR, received {image.shape}")
    if image.size == 0:
        raise ValueError("ArcFace source image must not be empty")
    matrix = estimate_arcface_similarity(landmarks)
    return cv2.warpAffine(
        image,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def recognition_candidate_quality(
    aligned_bgr: np.ndarray,
    detector_box: Any,
    landmarks: Any,
    confidence: float,
    source_shape: Sequence[int],
    *,
    scan_angle_degrees: int = 0,
) -> tuple[float, bool, dict[str, float]]:
    """Rank identity anchors and reject geometrically unreliable alignments.

    These constants are an internal recognizer profile, not user-facing
    configuration. The gate is intentionally conservative because a rejected
    candidate makes a track UNKNOWN (and therefore blurred), while a poor crop
    can create an unsafe false exemption.
    """

    aligned = np.asarray(aligned_bgr)
    if aligned.shape != (112, 112, 3):
        raise ValueError("recognition quality requires one aligned 112x112 BGR crop")
    if len(source_shape) < 2:
        raise ValueError("recognition source shape must contain height and width")
    height, width = int(source_shape[0]), int(source_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("recognition source dimensions must be positive")
    points = _landmark_array(landmarks)
    box = np.asarray(detector_box, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(box)):
        raise ValueError("recognition detector box must be finite")
    minimum_side = max(0.0, min(float(box[2] - box[0]), float(box[3] - box[1])))
    detector_confidence = float(np.clip(confidence, 0.0, 1.0))

    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_left = float(np.linalg.norm(nose - left_eye))
    eye_right = float(np.linalg.norm(nose - right_eye))
    mouth_left = float(np.linalg.norm(nose - left_mouth))
    mouth_right = float(np.linalg.norm(nose - right_mouth))
    if min(eye_left, eye_right, mouth_left, mouth_right) <= 1e-6:
        pose = 0.0
    else:
        pose = min(
            min(eye_left, eye_right) / max(eye_left, eye_right),
            min(mouth_left, mouth_right) / max(mouth_left, mouth_right),
        )

    matrix = estimate_arcface_similarity(points)
    projected = np.column_stack((points, np.ones(5))) @ matrix.T
    residual = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (projected - ARC_FACE_TEMPLATE_112.astype(np.float64)) ** 2,
                    axis=1,
                )
            )
        )
    )
    inverse = cv2.invertAffineTransform(matrix)
    # A fixed 16x16 target grid estimates crop coverage to within far less than
    # the deliberately broad 0.55 gate, without adding a second full warp.
    axis = np.linspace(0.0, 111.0, 16, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis)
    target = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1), np.ones(grid_x.size)))
    source = target @ inverse.T
    coverage = float(
        np.mean(
            (source[:, 0] >= 0.0) & (source[:, 0] < width) & (source[:, 1] >= 0.0) & (source[:, 1] < height)
        )
    )

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    geometry_score = math.exp(-((residual / 8.0) ** 2))
    detail_score = float(np.clip(math.log1p(laplacian_variance) / math.log1p(500.0), 0.0, 1.0))
    size_score = float(np.clip(minimum_side / 112.0, 0.0, 1.0))
    view_score = 1.0 if int(scan_angle_degrees) == 0 else 0.75
    quality = max(
        1e-6,
        detector_confidence
        * coverage
        * (0.5 + 0.5 * pose)
        * (0.5 + 0.5 * geometry_score)
        * (0.7 + 0.3 * detail_score)
        * (0.7 + 0.3 * size_score)
        * view_score,
    )
    eligible = bool(
        detector_confidence >= 0.50
        and minimum_side >= 32.0
        and coverage >= 0.55
        and pose >= 0.20
        and residual <= 10.0
        and laplacian_variance >= 20.0
        and contrast >= 12.0
    )
    return (
        quality,
        eligible,
        {
            "confidence": detector_confidence,
            "minimum_side": minimum_side,
            "pose_score": pose,
            "alignment_residual": residual,
            "source_coverage": coverage,
            "laplacian_variance": laplacian_variance,
            "contrast": contrast,
            "view_score": view_score,
        },
    )


def l2_normalize_embedding(value: Any) -> np.ndarray:
    """Validate and L2-normalize one embedding vector."""

    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("face embedding must be a non-empty finite vector")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= np.finfo(np.float32).eps:
        raise ValueError("face embedding must have a non-zero L2 norm")
    result = np.ascontiguousarray(vector / norm, dtype=np.float32)
    result.setflags(write=False)
    return result


def _embed_aligned_face(recognizer: Any, aligned_bgr: np.ndarray) -> np.ndarray:
    """Infer and normalize one aligned crop through legacy ArcFaceONNX."""

    aligned = np.asarray(aligned_bgr)
    expected = (112, 112, 3)
    if aligned.shape != expected:
        raise ValueError(
            f"aligned ArcFace crop must have shape {expected}, received {aligned.shape}"
        )
    get_feat = getattr(recognizer, "get_feat", None)
    if not callable(get_feat):
        raise TypeError("ArcFace recognizer must provide get_feat(image)")
    input_size = tuple(getattr(recognizer, "input_size", ()))
    if input_size != (112, 112):
        raise RuntimeError(
            "PrivateFrame requires a 112x112 ArcFace recognizer, "
            f"received {input_size}"
        )
    output = np.asarray(get_feat(aligned))
    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] <= 0:
        raise RuntimeError(
            "ArcFace get_feat() must return one non-empty embedding row, "
            f"received {output.shape}"
        )
    return l2_normalize_embedding(output[0])


@dataclass(frozen=True)
class RecognitionProfile:
    name: str
    max_frames_per_track: int
    minimum_margin: float = 0.08
    minimum_cluster_similarity: float = 0.35


RECOGNITION_PROFILES: Mapping[str, RecognitionProfile] = MappingProxyType(
    {
        # A one-frame track cannot expose a switch; the single-frame decision
        # therefore compensates with a stricter cosine threshold and margin.
        "fast": RecognitionProfile("fast", 1),
        "balanced": RecognitionProfile("balanced", 3),
        "accurate": RecognitionProfile("accurate", 5),
    }
)


def resolve_recognition_profile(
    name: str = "balanced",
    max_frames_per_track: int | None = None,
) -> RecognitionProfile:
    try:
        profile = RECOGNITION_PROFILES[str(name)]
    except KeyError as error:
        raise ValueError(f"recognition profile must be one of {sorted(RECOGNITION_PROFILES)}") from error
    if max_frames_per_track is None:
        return profile
    maximum = int(max_frames_per_track)
    if maximum <= 0:
        raise ValueError("max_frames_per_track must be positive")
    return replace(profile, max_frames_per_track=maximum)


@dataclass(frozen=True)
class RecognitionCandidate:
    """A bounded, aligned crop retained by Streaming for later inference."""

    frame_index: int
    quality: float
    aligned_face: np.ndarray
    landmark_source: str = "local_scrfd"

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("recognition candidate frame_index must be non-negative")
        if not math.isfinite(float(self.quality)) or float(self.quality) <= 0.0:
            raise ValueError("recognition candidate quality must be finite and positive")
        if np.asarray(self.aligned_face).shape != (112, 112, 3):
            raise ValueError("recognition candidate must contain a 112x112 BGR crop")
        if self.landmark_source not in RECOGNITION_LANDMARK_SOURCES:
            raise ValueError(
                "recognition candidate landmark_source must be local_scrfd or global_scrfd"
            )


@dataclass(frozen=True)
class TemporalThresholdEvidence:
    """Audit the temporal independence behind one track's cosine threshold."""

    selected_count: int
    span_seconds: float
    minimum_adjacent_gap_seconds: float | None
    decision_similarity_threshold: float
    effective_similarity_threshold: float
    threshold_reason: str

    def annotate(self, decision: IdentityDecision) -> IdentityDecision:
        """Attach the effective threshold evidence without changing identity."""

        return replace(
            decision,
            selected_frame_count=self.selected_count,
            effective_similarity_threshold=self.effective_similarity_threshold,
            temporal_evidence_span_seconds=self.span_seconds,
            minimum_adjacent_gap_seconds=self.minimum_adjacent_gap_seconds,
            threshold_reason=self.threshold_reason,
        )


def temporal_threshold_evidence(
    frame_indices: Sequence[int],
    *,
    frames_per_second: float,
    base_similarity_threshold: float,
) -> TemporalThresholdEvidence:
    """Choose the base threshold only for three independently timed samples."""

    fps = _finite_number(
        frames_per_second,
        field="frames_per_second",
        positive=True,
    )
    threshold = _cosine_value(
        base_similarity_threshold,
        field="base_similarity_threshold",
    )
    if threshold < 0.0:
        raise ValueError("base_similarity_threshold must be between 0 and 1")
    ordered = sorted(int(value) for value in frame_indices)
    if any(value < 0 for value in ordered):
        raise ValueError("temporal evidence frame indices must be non-negative")

    span_seconds = (
        float(ordered[-1] - ordered[0]) / fps if len(ordered) >= 2 else 0.0
    )
    minimum_gap = (
        min(
            float(current - previous) / fps
            for previous, current in pairwise(ordered)
        )
        if len(ordered) >= 2
        else None
    )
    independent = bool(
        len(ordered) >= TEMPORAL_EVIDENCE_MIN_SELECTED_FRAMES
        and minimum_gap is not None
        and minimum_gap >= TEMPORAL_EVIDENCE_MIN_ADJACENT_GAP_SECONDS
    )
    if independent:
        decision_threshold = threshold
        reason = "independent_temporal_evidence"
    else:
        decision_threshold = min(
            1.0,
            threshold + TEMPORAL_EVIDENCE_SIMILARITY_OFFSET,
        )
        reason = (
            "insufficient_selected_frames"
            if len(ordered) < TEMPORAL_EVIDENCE_MIN_SELECTED_FRAMES
            else "insufficient_temporal_separation"
        )

    # decide_track_identity keeps the existing stricter single-sample rule.
    # Record the final cosine gate here so the artifact states the threshold
    # that was actually applied, while passing the pre-singleton value below.
    effective_threshold = (
        min(1.0, decision_threshold + SINGLE_FRAME_SIMILARITY_OFFSET)
        if len(ordered) == 1
        else decision_threshold
    )
    if len(ordered) == 1:
        reason += "+single_frame_offset"
    return TemporalThresholdEvidence(
        selected_count=len(ordered),
        span_seconds=span_seconds,
        minimum_adjacent_gap_seconds=minimum_gap,
        decision_similarity_threshold=decision_threshold,
        effective_similarity_threshold=effective_threshold,
        threshold_reason=reason,
    )


def select_temporally_distributed(
    candidates: Sequence[RecognitionCandidate],
    max_frames: int,
) -> list[RecognitionCandidate]:
    """Choose one landmark proposal per time, distributed over the track.

    Local-SCRFD proposals take precedence inside a temporal bucket. A global
    SCRFD proposal is used when the bucket has no local proposal; quality then
    breaks ties within the preferred source. Collapsing same-frame proposals
    before bucketing prevents two crops from one instant from being counted as
    independent temporal evidence.
    """

    maximum = int(max_frames)
    if maximum <= 0:
        raise ValueError("max_frames must be positive")
    source_rank = {"local_scrfd": 0, "global_scrfd": 1}
    ordered_all = sorted(
        candidates,
        key=lambda item: (
            item.frame_index,
            source_rank[item.landmark_source],
            -item.quality,
        ),
    )
    ordered: list[RecognitionCandidate] = []
    for candidate in ordered_all:
        if ordered and candidate.frame_index == ordered[-1].frame_index:
            continue
        ordered.append(candidate)
    if len(ordered) <= maximum:
        return ordered
    first = ordered[0].frame_index
    last = ordered[-1].frame_index
    if first == last:
        return sorted(
            ordered,
            key=lambda item: (
                source_rank[item.landmark_source],
                -item.quality,
                item.frame_index,
            ),
        )[:maximum]
    buckets: list[list[RecognitionCandidate]] = [[] for _ in range(maximum)]
    span = last - first
    for candidate in ordered:
        bucket = min(
            maximum - 1,
            int((candidate.frame_index - first) * maximum / (span + 1)),
        )
        buckets[bucket].append(candidate)
    selected = [
        min(
            bucket,
            key=lambda item: (
                source_rank[item.landmark_source],
                -item.quality,
                item.frame_index,
            ),
        )
        for bucket in buckets
        if bucket
    ]
    selected_ids = {id(value) for value in selected}
    while len(selected) < maximum:
        remaining = [value for value in ordered if id(value) not in selected_ids]
        value = max(
            remaining,
            key=lambda item: (
                min(abs(item.frame_index - chosen.frame_index) for chosen in selected),
                -source_rank[item.landmark_source],
                item.quality,
                -item.frame_index,
            ),
        )
        selected.append(value)
        selected_ids.add(id(value))
    return sorted(selected, key=lambda item: item.frame_index)


@dataclass(frozen=True)
class EmbeddingSample:
    frame_index: int
    embedding: np.ndarray
    quality: float = 1.0

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("embedding frame_index must be non-negative")
        if not math.isfinite(float(self.quality)) or float(self.quality) <= 0.0:
            raise ValueError("embedding quality must be finite and positive")


@dataclass(frozen=True)
class IdentityPrototype:
    person_id: str
    embedding: np.ndarray
    reference_count: int
    rejected_count: int = 0


@dataclass(frozen=True)
class TrackEmbedding:
    embedding: np.ndarray | None
    support: int
    frame_indices: tuple[int, ...]


def _normalized_matrix(values: Sequence[Any]) -> np.ndarray:
    vectors = [l2_normalize_embedding(value) for value in values]
    if not vectors:
        raise ValueError("at least one embedding is required")
    widths = {value.size for value in vectors}
    if len(widths) != 1:
        raise ValueError("all embeddings must have the same width")
    return np.stack(vectors)


def _complete_link_clusters(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    clusters = [[index] for index in range(len(vectors))]
    similarities = vectors @ vectors.T
    while True:
        best: tuple[float, int, int] | None = None
        for first in range(len(clusters)):
            for second in range(first + 1, len(clusters)):
                cross = similarities[np.ix_(clusters[first], clusters[second])]
                minimum = float(np.min(cross))
                if minimum < threshold:
                    continue
                candidate = (float(np.mean(cross)), first, second)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            break
        _score, first, second = best
        clusters[first] = sorted(clusters[first] + clusters[second])
        del clusters[second]
    return clusters


def _weighted_prototype(
    vectors: np.ndarray,
    indices: Sequence[int],
    weights: np.ndarray,
) -> np.ndarray:
    combined = np.average(vectors[list(indices)], axis=0, weights=weights[list(indices)])
    return l2_normalize_embedding(combined)


def build_identity_prototype(
    person_id: str,
    embeddings: Sequence[Any],
    weights: Sequence[float] | None = None,
    *,
    minimum_cluster_similarity: float = 0.35,
) -> IdentityPrototype:
    """Build a robust gallery identity prototype from its largest coherent cluster."""

    normalized_id = _person_id(person_id)
    vectors = _normalized_matrix(embeddings)
    sample_weights = _validated_weights(weights, len(vectors))
    threshold = _cosine_value(minimum_cluster_similarity, field="minimum_cluster_similarity")
    clusters = _complete_link_clusters(vectors, threshold)
    ranked = sorted(
        clusters,
        # Coherent reference count, never detector confidence, determines
        # which identity cluster is dominant.
        key=lambda values: (len(values), sum(sample_weights[values]), -min(values)),
        reverse=True,
    )
    if len(ranked) > 1:
        # Detector confidence may weight already-consistent references, but it
        # must never choose between incompatible identities. Two disagreeing
        # references are always ambiguous. With three or more references, only
        # a coherent cluster containing at least two thirds of all references
        # may reject the remaining outliers.
        minimum_dominant_count = math.ceil(len(vectors) * 2.0 / 3.0)
        if len(ranked[0]) < max(2, minimum_dominant_count):
            raise ValueError(f"identity {normalized_id!r} has inconsistent reference images")
    kept = ranked[0]
    return IdentityPrototype(
        person_id=normalized_id,
        embedding=_weighted_prototype(vectors, kept, sample_weights),
        reference_count=len(kept),
        rejected_count=len(vectors) - len(kept),
    )


@dataclass(frozen=True)
class _TrackClusters:
    ordered: tuple[EmbeddingSample, ...]
    vectors: np.ndarray
    indices: tuple[tuple[int, ...], ...]
    prototypes: tuple[np.ndarray, ...]


def _cluster_track_embeddings(
    samples: Sequence[EmbeddingSample],
    minimum_cluster_similarity: float,
) -> _TrackClusters | None:
    if not samples:
        return None
    ordered = tuple(sorted(samples, key=lambda item: item.frame_index))
    vectors = _normalized_matrix([item.embedding for item in ordered])
    weights = _validated_weights([item.quality for item in ordered], len(ordered))
    threshold = _cosine_value(
        minimum_cluster_similarity,
        field="minimum_cluster_similarity",
    )
    clusters = _complete_link_clusters(vectors, threshold)
    clusters.sort(
        key=lambda values: (sum(weights[values]), len(values), -min(values)),
        reverse=True,
    )
    indices = tuple(tuple(values) for values in clusters)
    prototypes = tuple(
        _weighted_prototype(vectors, values, weights) for values in indices
    )
    return _TrackClusters(ordered, vectors, indices, prototypes)


def aggregate_track_embeddings(
    samples: Sequence[EmbeddingSample],
    *,
    minimum_cluster_similarity: float = 0.35,
) -> TrackEmbedding:
    """Aggregate a coherent track; any second quality cluster is a conflict."""

    analysis = _cluster_track_embeddings(samples, minimum_cluster_similarity)
    if analysis is None:
        return TrackEmbedding(None, 0, ())
    main = analysis.indices[0]
    conflicting = len(analysis.indices) > 1
    return TrackEmbedding(
        embedding=None if conflicting else analysis.prototypes[0],
        support=len(main),
        frame_indices=tuple(
            analysis.ordered[index].frame_index for index in main
        ),
    )


class IdentityStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class IdentityDecision:
    status: IdentityStatus
    person_id: str | None = None
    top1_person_id: str | None = None
    top1_similarity: float | None = None
    runner_up_person_id: str | None = None
    runner_up_similarity: float | None = None
    margin: float | None = None
    support: int = 0
    frame_indices: tuple[int, ...] = ()
    reason: str = ""
    selected_frame_count: int | None = None
    effective_similarity_threshold: float | None = None
    temporal_evidence_span_seconds: float | None = None
    minimum_adjacent_gap_seconds: float | None = None
    threshold_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the stable, JSON-native identity artifact representation."""

        return {
            "status": self.status.value,
            "person_id": self.person_id,
            "top1_person_id": self.top1_person_id,
            "top1_similarity": self.top1_similarity,
            "runner_up_person_id": self.runner_up_person_id,
            "runner_up_similarity": self.runner_up_similarity,
            "margin": self.margin,
            "support": self.support,
            "frame_indices": list(self.frame_indices),
            "reason": self.reason,
            "selected_frame_count": self.selected_frame_count,
            "effective_similarity_threshold": self.effective_similarity_threshold,
            "temporal_evidence_span_seconds": self.temporal_evidence_span_seconds,
            "minimum_adjacent_gap_seconds": self.minimum_adjacent_gap_seconds,
            "threshold_reason": self.threshold_reason,
        }


def _prototype_mapping(
    prototypes: Mapping[str, IdentityPrototype | np.ndarray],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    width: int | None = None
    for raw_id, raw_prototype in prototypes.items():
        person_id = _person_id(raw_id)
        if person_id in result:
            raise ValueError(f"duplicate NFC-normalized person id: {person_id!r}")
        vector = l2_normalize_embedding(
            raw_prototype.embedding if isinstance(raw_prototype, IdentityPrototype) else raw_prototype
        )
        if width is None:
            width = vector.size
        elif vector.size != width:
            raise ValueError("gallery prototypes must have the same width")
        result[person_id] = vector
    if not result:
        raise ValueError("at least one gallery prototype is required")
    return result


def _rank_identities(
    embedding: np.ndarray,
    prototypes: Mapping[str, np.ndarray],
) -> list[tuple[str, float]]:
    if any(value.size != embedding.size for value in prototypes.values()):
        raise ValueError("track and gallery embedding widths do not match")
    return sorted(
        ((person_id, float(embedding @ value)) for person_id, value in prototypes.items()),
        key=lambda item: (-item[1], item[0]),
    )


def decide_track_identity(
    samples: Sequence[EmbeddingSample],
    prototypes: Mapping[str, IdentityPrototype | np.ndarray],
    similarity_threshold: float,
    *,
    minimum_margin: float = 0.08,
    minimum_cluster_similarity: float = 0.35,
    single_frame_similarity_offset: float = SINGLE_FRAME_SIMILARITY_OFFSET,
    single_frame_minimum_margin: float = 0.12,
) -> IdentityDecision:
    """Return one fail-safe identity decision for a canonical track."""

    threshold = _cosine_value(similarity_threshold, field="similarity_threshold")
    if threshold < 0.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    margin_required = _finite_number(minimum_margin, field="minimum_margin")
    if not 0.0 <= margin_required <= 2.0:
        raise ValueError("minimum_margin must be between 0 and 2")
    single_offset = _finite_number(
        single_frame_similarity_offset,
        field="single_frame_similarity_offset",
    )
    if not 0.0 <= single_offset <= 1.0:
        raise ValueError("single_frame_similarity_offset must be between 0 and 1")
    single_margin = _finite_number(
        single_frame_minimum_margin,
        field="single_frame_minimum_margin",
    )
    if not 0.0 <= single_margin <= 2.0:
        raise ValueError("single_frame_minimum_margin must be between 0 and 2")
    gallery = _prototype_mapping(prototypes)
    frame_indices = [sample.frame_index for sample in samples]
    if len(set(frame_indices)) != len(frame_indices):
        return IdentityDecision(
            IdentityStatus.UNKNOWN,
            reason="duplicate_frame_samples",
        )

    clustered = _cluster_track_embeddings(
        samples, minimum_cluster_similarity
    )
    if clustered is None:
        return IdentityDecision(status=IdentityStatus.UNKNOWN, reason="no_embedding")

    # Whole-track exemption is deliberately unanimous over every selected,
    # quality-qualified sample. A minority sample can be a brief identity
    # switch even when it remains close enough to the dominant appearance to
    # fall into the same complete-link cluster. Majority voting alone would
    # therefore be unsafe: two target frames could hide one stranger frame.
    independently_confirmed: dict[
        str, list[tuple[EmbeddingSample, list[tuple[str, float]], float | None]]
    ] = {}
    unconfirmed_frames: list[int] = []
    per_frame_threshold = (
        min(1.0, threshold + single_offset)
        if len(clustered.ordered) == 1
        else threshold
    )
    per_frame_margin = (
        max(margin_required, single_margin)
        if len(clustered.ordered) == 1
        else margin_required
    )
    for sample, vector in zip(clustered.ordered, clustered.vectors):
        ranked_sample = _rank_identities(vector, gallery)
        sample_margin = (
            ranked_sample[0][1] - ranked_sample[1][1]
            if len(ranked_sample) > 1
            else None
        )
        if ranked_sample[0][1] < per_frame_threshold or (
            sample_margin is not None and sample_margin < per_frame_margin
        ):
            unconfirmed_frames.append(sample.frame_index)
            continue
        independently_confirmed.setdefault(ranked_sample[0][0], []).append(
            (sample, ranked_sample, sample_margin)
        )
    if len(independently_confirmed) > 1:
        return IdentityDecision(
            status=IdentityStatus.CONFLICT,
            support=len(clustered.ordered),
            frame_indices=tuple(
                item.frame_index for item in clustered.ordered
            ),
            reason="multiple_confirmed_identities",
        )
    if unconfirmed_frames and len(clustered.ordered) > 1:
        separated_cluster = len(clustered.indices) > 1
        return IdentityDecision(
            status=(
                IdentityStatus.CONFLICT
                if separated_cluster
                else IdentityStatus.UNKNOWN
            ),
            support=len(clustered.ordered) - len(unconfirmed_frames),
            frame_indices=tuple(
                item.frame_index for item in clustered.ordered
            ),
            reason=(
                "unconfirmed_track_cluster"
                if separated_cluster
                else "unconfirmed_track_sample"
            ),
        )

    # A whole-track exemption is allowed across pose-separated embedding
    # clusters only when every quality-qualified cluster independently clears
    # Gallery threshold+margin for the same person. An UNKNOWN minority cluster
    # can be a short identity switch, so it vetoes the entire track.
    if len(clustered.indices) > 1:
        cluster_matches: list[
            tuple[
                str,
                float,
                str | None,
                float | None,
                float | None,
                tuple[int, ...],
            ]
        ] = []
        all_cluster_frames = tuple(
            item.frame_index for item in clustered.ordered
        )
        for indices, prototype in zip(
            clustered.indices, clustered.prototypes
        ):
            ranked_cluster = _rank_identities(prototype, gallery)
            cluster_person, cluster_score = ranked_cluster[0]
            runner_person, runner_score = (
                ranked_cluster[1]
                if len(ranked_cluster) > 1
                else (None, None)
            )
            cluster_margin = (
                cluster_score - runner_score
                if runner_score is not None
                else None
            )
            singleton = len(indices) == 1
            cluster_threshold = (
                min(1.0, threshold + single_offset)
                if singleton
                else threshold
            )
            cluster_margin_required = (
                max(margin_required, single_margin)
                if singleton
                else margin_required
            )
            if cluster_score < cluster_threshold or (
                cluster_margin is not None
                and cluster_margin < cluster_margin_required
            ):
                return IdentityDecision(
                    status=IdentityStatus.CONFLICT,
                    support=len(clustered.ordered),
                    frame_indices=all_cluster_frames,
                    reason="unconfirmed_track_cluster",
                )
            cluster_matches.append(
                (
                    cluster_person,
                    cluster_score,
                    runner_person,
                    runner_score,
                    cluster_margin,
                    tuple(
                        clustered.ordered[index].frame_index
                        for index in indices
                    ),
                )
            )
        confirmed_people = {value[0] for value in cluster_matches}
        if len(confirmed_people) != 1:
            return IdentityDecision(
                status=IdentityStatus.CONFLICT,
                support=len(clustered.ordered),
                frame_indices=all_cluster_frames,
                reason="multiple_confirmed_identities",
            )
        weakest = min(cluster_matches, key=lambda value: value[1])
        margins = [
            value[4] for value in cluster_matches if value[4] is not None
        ]
        person_id = next(iter(confirmed_people))
        return IdentityDecision(
            status=IdentityStatus.CONFIRMED,
            person_id=person_id,
            top1_person_id=person_id,
            top1_similarity=weakest[1],
            runner_up_person_id=weakest[2],
            runner_up_similarity=weakest[3],
            margin=min(margins) if margins else None,
            support=len(clustered.ordered),
            frame_indices=all_cluster_frames,
            reason="confirmed_all_clusters",
        )

    aggregate = aggregate_track_embeddings(
        samples,
        minimum_cluster_similarity=minimum_cluster_similarity,
    )
    if aggregate.embedding is None:
        return IdentityDecision(status=IdentityStatus.UNKNOWN, reason="no_embedding")

    ranked = _rank_identities(aggregate.embedding, gallery)
    top_id, top_score = ranked[0]
    runner_id, runner_score = ranked[1] if len(ranked) > 1 else (None, None)
    margin = top_score - runner_score if runner_score is not None else None
    common = {
        "top1_person_id": top_id,
        "top1_similarity": top_score,
        "runner_up_person_id": runner_id,
        "runner_up_similarity": runner_score,
        "margin": margin,
        "support": aggregate.support,
        "frame_indices": aggregate.frame_indices,
    }

    aggregate_threshold = min(1.0, threshold + single_offset) if len(samples) == 1 else threshold
    aggregate_margin = max(margin_required, single_margin) if len(samples) == 1 else margin_required
    if top_score < aggregate_threshold:
        return IdentityDecision(
            status=IdentityStatus.UNKNOWN,
            reason="below_similarity_threshold",
            **common,
        )
    if margin is not None and margin < aggregate_margin:
        return IdentityDecision(
            status=IdentityStatus.UNKNOWN,
            reason="insufficient_margin",
            **common,
        )
    matches = independently_confirmed.get(top_id, [])
    required_support = len(samples)
    if len(matches) < required_support:
        return IdentityDecision(
            status=IdentityStatus.UNKNOWN,
            reason="insufficient_consistent_support",
            **common,
        )
    return IdentityDecision(
        status=IdentityStatus.CONFIRMED,
        person_id=top_id,
        top1_person_id=top_id,
        top1_similarity=top_score,
        runner_up_person_id=runner_id,
        runner_up_similarity=runner_score,
        margin=margin,
        support=len(matches),
        frame_indices=tuple(sorted(sample.frame_index for sample, _ranked, _margin in matches)),
        reason=(
            "confirmed_single_frame"
            if len(samples) == 1
            else "confirmed_unanimous_support"
        ),
    )


@dataclass(frozen=True)
class GalleryImage:
    person_id: str
    path: Path
    relative_file_name: str


@dataclass(frozen=True)
class GalleryFileFingerprint:
    """Non-biometric, location-independent gallery content identity."""

    person_id: str
    relative_file_name: str
    content_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "person_id": self.person_id,
            "relative_file_name": self.relative_file_name,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class GalleryScan:
    root: Path
    person_ids: tuple[str, ...]
    images: tuple[GalleryImage, ...]


@dataclass(frozen=True)
class GalleryReference:
    person_id: str
    file_name: str
    relative_file_name: str
    content_sha256: str
    embedding: np.ndarray
    quality: float


@dataclass(frozen=True)
class GalleryRejection:
    person_id: str
    file_name: str
    reason: str
    relative_file_name: str = ""
    content_sha256: str = ""


@dataclass(frozen=True)
class Gallery:
    root: Path
    person_ids: tuple[str, ...]
    prototypes: Mapping[str, IdentityPrototype]
    references: tuple[GalleryReference, ...]
    rejections: tuple[GalleryRejection, ...]
    file_fingerprints: tuple[GalleryFileFingerprint, ...]
    content_fingerprint: str

    @property
    def fingerprint(self) -> str:
        return self.content_fingerprint

    def fingerprint_dict(self) -> dict[str, Any]:
        """Return stable gallery identity without paths, pixels, or embeddings."""

        return {
            "sha256": self.content_fingerprint,
            "files": [value.to_dict() for value in self.file_fingerprints],
        }


def scan_gallery(path: str | Path) -> GalleryScan:
    """Scan only direct, non-hidden, non-symlink images under first-level people."""

    root = Path(path).expanduser()
    if root.is_symlink():
        raise ValueError("gallery root must not be a symlink")
    if not root.is_dir():
        raise FileNotFoundError(f"gallery directory does not exist: {root}")
    root = root.resolve()
    people: dict[str, Path] = {}
    for entry in sorted(root.iterdir(), key=lambda value: value.name):
        if entry.name.startswith(".") or entry.is_symlink() or not entry.is_dir():
            continue
        person_id = _person_id(entry.name)
        if person_id in people:
            raise ValueError(
                "gallery has directory names that collide after Unicode NFC "
                f"normalization: {people[person_id].name!r}, {entry.name!r}"
            )
        people[person_id] = entry
    images: list[GalleryImage] = []
    for person_id, directory in people.items():
        normalized_names: dict[str, str] = {}
        for entry in sorted(directory.iterdir(), key=lambda value: value.name):
            if entry.name.startswith(".") or entry.is_symlink() or not entry.is_file():
                continue
            if entry.suffix.lower() in SUPPORTED_GALLERY_EXTENSIONS:
                normalized_name = unicodedata.normalize("NFC", entry.name)
                if normalized_name in normalized_names:
                    raise ValueError(
                        "gallery file names collide after Unicode NFC "
                        f"normalization: {normalized_names[normalized_name]!r}, "
                        f"{entry.name!r}"
                    )
                normalized_names[normalized_name] = entry.name
                images.append(
                    GalleryImage(
                        person_id,
                        entry.resolve(),
                        f"{person_id}/{normalized_name}",
                    )
                )
    return GalleryScan(root, tuple(people), tuple(images))


def _read_gallery_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gallery_content_fingerprint(
    values: Sequence[GalleryFileFingerprint],
) -> str:
    payload = json.dumps(
        [value.to_dict() for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _detect_gallery_faces(detector: Any, image: np.ndarray) -> Sequence[Mapping[str, Any]]:
    if callable(detector):
        result = detector(image)
    elif callable(getattr(detector, "detect", None)):
        result = detector.detect(image)
    else:
        raise TypeError("gallery detector must be callable or expose detect(image)")
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise TypeError("gallery detector must return a sequence")
    if not all(isinstance(value, Mapping) for value in result):
        raise TypeError("gallery detector results must be mappings")
    return result


def build_gallery(
    path: str | Path,
    detector: Any,
    recognizer: Any,
    *,
    target_persons: Sequence[str] = (),
    image_loader: Callable[[Path], np.ndarray | None] = _read_gallery_image,
    minimum_cluster_similarity: float = 0.35,
) -> Gallery:
    """Validate gallery images and build one prototype per usable identity."""

    scanned = scan_gallery(path)
    targets = _person_ids(target_persons)
    missing = sorted(set(targets) - set(scanned.person_ids))
    if missing:
        raise ValueError(f"target persons are absent from gallery: {missing}")
    references: list[GalleryReference] = []
    rejections: list[GalleryRejection] = []
    file_fingerprints = tuple(
        sorted(
            (
                GalleryFileFingerprint(
                    item.person_id,
                    item.relative_file_name,
                    _sha256_path(item.path),
                )
                for item in scanned.images
            ),
            key=lambda value: (
                value.person_id,
                value.relative_file_name,
                value.content_sha256,
            ),
        )
    )
    fingerprint_by_relative_name = {value.relative_file_name: value for value in file_fingerprints}

    def reject(item: GalleryImage, reason: str) -> None:
        fingerprint = fingerprint_by_relative_name[item.relative_file_name]
        rejections.append(
            GalleryRejection(
                item.person_id,
                item.path.name,
                reason,
                item.relative_file_name,
                fingerprint.content_sha256,
            )
        )

    seen_content_by_person: dict[str, set[str]] = {}
    for item in scanned.images:
        fingerprint = fingerprint_by_relative_name[item.relative_file_name]
        seen_content = seen_content_by_person.setdefault(item.person_id, set())
        if fingerprint.content_sha256 in seen_content:
            reject(item, "duplicate_content")
            continue
        seen_content.add(fingerprint.content_sha256)
        try:
            image = image_loader(item.path)
        except (OSError, ValueError, cv2.error):
            image = None
        if image is None or np.asarray(image).ndim != 3 or np.asarray(image).shape[2] != 3:
            reject(item, "unreadable_image")
            continue
        try:
            detections = _detect_gallery_faces(detector, image)
        except MemoryError:
            raise
        except Exception as error:
            reject(
                item,
                _diagnostic_failure_reason(
                    "detector_inference_error",
                    error,
                ),
            )
            continue
        if len(detections) == 0:
            reject(item, "no_face")
            continue
        if len(detections) != 1:
            reject(item, "multiple_faces")
            continue
        detection = detections[0]
        if "landmarks" not in detection:
            reject(item, "missing_landmarks")
            continue
        try:
            landmarks = _landmark_array(detection["landmarks"])
        except (TypeError, ValueError):
            reject(item, "invalid_landmarks")
            continue
        raw_confidence = detection.get("confidence")
        if (
            isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, (int, float))
            or not math.isfinite(float(raw_confidence))
            or not 0.0 < float(raw_confidence) <= 1.0
        ):
            reject(item, "invalid_confidence")
            continue
        if "box" not in detection:
            reject(item, "missing_box")
            continue
        confidence = float(raw_confidence)
        aligned: np.ndarray | None = None
        try:
            aligned = arcface_align_112(image, landmarks)
            quality, eligible, _details = recognition_candidate_quality(
                aligned,
                detection["box"],
                landmarks,
                confidence,
                image.shape,
            )
        except (TypeError, ValueError, cv2.error):
            reject(item, "invalid_geometry")
            if aligned is not None and aligned.flags.writeable:
                aligned.fill(0)
            continue
        if not eligible:
            reject(item, "low_quality")
            if aligned.flags.writeable:
                aligned.fill(0)
            continue
        try:
            embedding = _embed_aligned_face(recognizer, aligned)
        except MemoryError:
            raise
        except Exception as error:
            reject(
                item,
                _diagnostic_failure_reason(
                    "recognizer_inference_error",
                    error,
                ),
            )
            continue
        finally:
            if aligned.flags.writeable:
                aligned.fill(0)
        references.append(
            GalleryReference(
                item.person_id,
                item.path.name,
                item.relative_file_name,
                fingerprint.content_sha256,
                embedding,
                quality,
            )
        )

    by_person: dict[str, list[GalleryReference]] = {person_id: [] for person_id in scanned.person_ids}
    for reference in references:
        by_person[reference.person_id].append(reference)
    prototypes: dict[str, IdentityPrototype] = {}
    for person_id, values in by_person.items():
        if not values:
            continue
        prototypes[person_id] = build_identity_prototype(
            person_id,
            [value.embedding for value in values],
            [value.quality for value in values],
            minimum_cluster_similarity=minimum_cluster_similarity,
        )
    invalid_targets = sorted(set(targets) - set(prototypes))
    if invalid_targets:
        raise ValueError(f"target persons have no usable reference images: {invalid_targets}")
    if not prototypes:
        raise ValueError("gallery has no usable reference images")
    return Gallery(
        root=scanned.root,
        person_ids=scanned.person_ids,
        prototypes=MappingProxyType(prototypes),
        references=tuple(references),
        rejections=tuple(rejections),
        file_fingerprints=file_fingerprints,
        content_fingerprint=_gallery_content_fingerprint(file_fingerprints),
    )


@dataclass(frozen=True)
class PolicyDecision:
    should_blur: bool
    reason: str


def _identity_from_artifact(
    value: IdentityDecision | Mapping[str, Any] | None,
) -> IdentityDecision:
    if isinstance(value, IdentityDecision):
        return value
    if not isinstance(value, Mapping):
        return IdentityDecision(IdentityStatus.UNKNOWN, reason="missing_identity_artifact")
    raw_status = value.get("status", "")
    try:
        status = raw_status if isinstance(raw_status, IdentityStatus) else IdentityStatus(str(raw_status))
    except ValueError:
        return IdentityDecision(IdentityStatus.UNKNOWN, reason="invalid_identity_artifact")
    person = value.get("person_id")
    if person is not None and not isinstance(person, str):
        return IdentityDecision(IdentityStatus.UNKNOWN, reason="invalid_identity_artifact")
    return IdentityDecision(status, person_id=person, reason="artifact")


def apply_identity_policy(
    mode: str,
    identity: IdentityDecision | Mapping[str, Any] | None,
    target_persons: Sequence[str],
) -> PolicyDecision:
    """Map identity evidence to rendering behavior without touching geometry."""

    policy_mode = _recognition_mode(mode)
    if policy_mode == "all":
        return PolicyDecision(True, "policy_all")
    normalized_identity = _identity_from_artifact(identity)
    targets = set(_person_ids(target_persons))
    if not targets:
        raise ValueError(f"recognition mode {policy_mode} requires target persons")
    if normalized_identity.status is not IdentityStatus.CONFIRMED or normalized_identity.person_id is None:
        return PolicyDecision(True, f"fail_safe_{normalized_identity.status.value.lower()}")
    try:
        matched = _person_id(normalized_identity.person_id) in targets
    except (TypeError, ValueError):
        return PolicyDecision(True, "fail_safe_invalid_identity")
    if policy_mode == "blur_only":
        return PolicyDecision(matched, "target_match" if matched else "confirmed_non_target")
    return PolicyDecision(not matched, "target_exempt" if matched else "confirmed_non_target")


@dataclass(frozen=True)
class RecognitionEngine:
    enabled: bool
    mode: str
    profile: RecognitionProfile
    target_persons: tuple[str, ...] = ()
    similarity_threshold: float | None = None
    recognizer: Any = None
    gallery: Gallery | Any = None

    @property
    def max_frames_per_track(self) -> int:
        return self.profile.max_frames_per_track

    @staticmethod
    def unknown_decision(reason: str) -> IdentityDecision:
        text = str(reason).strip()
        if not text:
            raise ValueError("unknown identity reason must not be empty")
        return IdentityDecision(IdentityStatus.UNKNOWN, reason=text)

    def identify_track(
        self,
        candidates: Sequence[RecognitionCandidate],
        *,
        frames_per_second: float,
    ) -> IdentityDecision:
        if not self.enabled:
            return replace(
                IdentityDecision(
                    status=IdentityStatus.UNKNOWN,
                    reason="policy_all",
                ),
                selected_frame_count=0,
                threshold_reason="policy_all",
            )
        temporal_evidence: TemporalThresholdEvidence | None = None
        selected: Sequence[RecognitionCandidate] = ()
        try:
            selected = select_temporally_distributed(
                candidates,
                self.profile.max_frames_per_track,
            )
            temporal_evidence = temporal_threshold_evidence(
                [value.frame_index for value in selected],
                frames_per_second=frames_per_second,
                base_similarity_threshold=float(self.similarity_threshold),
            )
            samples = [
                EmbeddingSample(
                    frame_index=value.frame_index,
                    embedding=_embed_aligned_face(
                        self.recognizer,
                        value.aligned_face,
                    ),
                    quality=value.quality,
                )
                for value in selected
            ]
            return temporal_evidence.annotate(
                decide_track_identity(
                    samples,
                    self.gallery.prototypes,
                    temporal_evidence.decision_similarity_threshold,
                    minimum_margin=self.profile.minimum_margin,
                    minimum_cluster_similarity=(
                        self.profile.minimum_cluster_similarity
                    ),
                )
            )
        except MemoryError:
            raise
        except Exception as error:
            decision = self.unknown_decision(
                _diagnostic_failure_reason(
                    "track_recognition_error",
                    error,
                )
            )
            if temporal_evidence is not None:
                return temporal_evidence.annotate(decision)
            return replace(
                decision,
                selected_frame_count=len(selected),
                threshold_reason="track_recognition_error",
            )


def create_recognition_engine(
    settings: Mapping[str, Any],
    *,
    recognizer: Any,
    gallery_detector: Any,
    gallery_builder: Callable[..., Any] = build_gallery,
) -> RecognitionEngine:
    """Create recognition lazily; ``mode=all`` touches no model or gallery state."""

    mode = _recognition_mode(settings.get("mode", "all"))
    if mode == "all":
        # Profile, thresholds, targets and gallery are selective-only inputs.
        # Ignore even stale/invalid values to preserve the zero-recognition
        # contract of the default policy.
        return RecognitionEngine(False, mode, RECOGNITION_PROFILES["balanced"])
    profile = resolve_recognition_profile(
        str(settings.get("profile", "balanced")),
        settings.get("max_frames_per_track"),
    )

    targets_raw = settings.get("target_persons")
    if not isinstance(targets_raw, Sequence) or isinstance(targets_raw, (str, bytes)):
        raise TypeError("selective recognition target_persons must be a sequence")
    targets = _person_ids(targets_raw)
    if not targets:
        raise ValueError("selective recognition requires target_persons")
    gallery_dir = settings.get("gallery_dir")
    if not isinstance(gallery_dir, (str, Path)) or not str(gallery_dir).strip():
        raise ValueError("selective recognition requires gallery_dir")
    if "similarity_threshold" not in settings:
        raise ValueError("selective recognition requires similarity_threshold")
    threshold = _cosine_value(settings["similarity_threshold"], field="similarity_threshold")
    if not callable(getattr(recognizer, "get_feat", None)):
        raise TypeError("selective recognition requires an ArcFace recognizer with get_feat()")
    if gallery_detector is None:
        raise ValueError("selective recognition requires a gallery detector")
    gallery = gallery_builder(
        gallery_dir,
        gallery_detector,
        recognizer,
        target_persons=targets,
        minimum_cluster_similarity=profile.minimum_cluster_similarity,
    )
    if not isinstance(getattr(gallery, "prototypes", None), Mapping):
        raise TypeError("gallery builder must return an object with prototype mappings")
    return RecognitionEngine(
        True,
        mode,
        profile,
        targets,
        threshold,
        recognizer,
        gallery,
    )


def _finite_number(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        suffix = " finite and positive" if positive else " finite"
        raise ValueError(f"{field} must be{suffix}")
    return result


def _cosine_value(value: Any, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between -1 and 1")
    return result


def _validated_weights(values: Sequence[float] | None, count: int) -> np.ndarray:
    if values is None:
        return np.ones(count, dtype=np.float64)
    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights.size != count:
        raise ValueError(f"expected {count} embedding weights, received {weights.size}")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("embedding weights must be finite and positive")
    return weights


def _person_id(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("person id must be a string")
    result = unicodedata.normalize("NFC", value)
    if not result or result in {".", ".."}:
        raise ValueError("person id must not be empty or a path component")
    if "/" in result or "\\" in result or "\x00" in result:
        raise ValueError("person id must not contain path separators")
    return result


def _person_ids(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_person_id(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError("person ids collide after Unicode NFC normalization")
    return result


def _recognition_mode(value: Any) -> str:
    mode = str(value)
    if mode not in RECOGNITION_MODES:
        raise ValueError(f"recognition mode must be one of {sorted(RECOGNITION_MODES)}")
    return mode


__all__ = [
    "ARC_FACE_TEMPLATE_112",
    "GALLERY_DETECTOR_CONFIDENCE_THRESHOLD",
    "GALLERY_DETECTOR_INPUT_SIZES",
    "LOCAL_LANDMARK_CONFIDENCE_THRESHOLD",
    "LOCAL_LANDMARK_MAX_AREA_RATIO",
    "LOCAL_LANDMARK_MAX_CENTER_DISTANCE",
    "LOCAL_LANDMARK_MIN_CONTAINMENT",
    "LOCAL_LANDMARK_MIN_IOU",
    "RECOGNITION_LANDMARK_SOURCES",
    "RECOGNITION_MODES",
    "RECOGNITION_PROFILES",
    "SINGLE_FRAME_SIMILARITY_OFFSET",
    "SUPPORTED_GALLERY_EXTENSIONS",
    "TEMPORAL_EVIDENCE_MIN_ADJACENT_GAP_SECONDS",
    "TEMPORAL_EVIDENCE_MIN_SELECTED_FRAMES",
    "TEMPORAL_EVIDENCE_SIMILARITY_OFFSET",
    "EmbeddingSample",
    "Gallery",
    "GalleryFileFingerprint",
    "GalleryImage",
    "GalleryReference",
    "GalleryRejection",
    "GalleryScan",
    "IdentityDecision",
    "IdentityPrototype",
    "IdentityStatus",
    "PolicyDecision",
    "RecognitionCandidate",
    "RecognitionEngine",
    "RecognitionProfile",
    "TemporalThresholdEvidence",
    "TrackEmbedding",
    "aggregate_track_embeddings",
    "apply_identity_policy",
    "arcface_align_112",
    "build_gallery",
    "build_identity_prototype",
    "create_recognition_engine",
    "decide_track_identity",
    "detect_gallery_faces_upright",
    "estimate_arcface_similarity",
    "l2_normalize_embedding",
    "local_landmark_box_agreement",
    "recognition_candidate_quality",
    "resolve_recognition_profile",
    "scan_gallery",
    "select_temporally_distributed",
    "temporal_threshold_evidence",
]
