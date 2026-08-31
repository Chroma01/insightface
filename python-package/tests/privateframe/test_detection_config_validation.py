from copy import deepcopy

import pytest
from insightface.app.privateframe.base_config import (
    validate_config_keys,
    validate_revalidation_passes,
    validate_scan_passes,
)


def _valid_config():
    return {
        "scan": {
            "passes": [
                {
                    "name": "full_frame",
                    "angles": [0, 90, -90],
                    "input_size": 640,
                    "horizontal_padding_ratio": 0.0,
                    "vertical_padding_ratio": 1.0,
                    "confidence_threshold": 0.5,
                }
            ]
        },
        "revalidation": {
            "input_size": 160,
            "angles": [0, 90, -90],
            "confidence_threshold": 0.18,
            "crop_expansion": 2.0,
        },
    }


def test_detection_pass_boundaries_are_valid():
    config = _valid_config()
    config["scan"]["passes"][0]["input_size"] = 32
    config["scan"]["passes"][0]["confidence_threshold"] = 0.0
    config["revalidation"]["confidence_threshold"] = 1.0

    validate_scan_passes(config)
    validate_revalidation_passes(config)


@pytest.mark.parametrize("value", [1, 2, 4])
def test_scan_frame_stride_accepts_positive_integers(value):
    config = _valid_config()
    config["scan"]["frame_stride"] = value

    validate_scan_passes(config)

    assert config["scan"]["frame_stride"] == value


@pytest.mark.parametrize("value", [True, 1.0, "2", None])
def test_scan_frame_stride_rejects_coercion(value):
    config = _valid_config()
    config["scan"]["frame_stride"] = value

    with pytest.raises(TypeError, match="scan.frame_stride must be an integer"):
        validate_scan_passes(config)


@pytest.mark.parametrize("value", [0, -1])
def test_scan_frame_stride_must_be_positive(value):
    config = _valid_config()
    config["scan"]["frame_stride"] = value

    with pytest.raises(ValueError, match="scan.frame_stride must be positive"):
        validate_scan_passes(config)


def test_scan_frame_stride_rejects_more_than_four_frames() -> None:
    config = _valid_config()
    config["scan"]["frame_stride"] = 5

    with pytest.raises(ValueError, match="between 1 and 4"):
        validate_scan_passes(config)


def test_scan_frame_stride_defaults_to_every_frame():
    config = _valid_config()

    validate_scan_passes(config)

    assert config["scan"]["frame_stride"] == 1


@pytest.mark.parametrize(
    ("performance_mode", "expected_stride"),
    [("normal", 1), ("fast", 2), ("ultra_fast", 4)],
)
def test_scan_performance_mode_materializes_canonical_stride(
    performance_mode,
    expected_stride,
):
    config = _valid_config()
    config["scan"]["performance_mode"] = performance_mode

    validate_scan_passes(config)

    assert config["scan"]["frame_stride"] == expected_stride
    assert "performance_mode" not in config["scan"]


def test_explicit_scan_frame_stride_wins_over_performance_mode():
    config = _valid_config()
    config["scan"].update(
        {
            "performance_mode": "ultra_fast",
            "frame_stride": 3,
        }
    )

    validate_scan_passes(config)

    assert config["scan"]["frame_stride"] == 3
    assert "performance_mode" not in config["scan"]


@pytest.mark.parametrize("value", ["ultra-fast", "standard", "FAST", "", None, 2])
def test_scan_performance_mode_is_strictly_validated(value):
    config = _valid_config()
    config["scan"]["performance_mode"] = value
    # An explicit stride wins selection, but does not make an invalid preset a
    # valid configuration value.
    config["scan"]["frame_stride"] = 2

    with pytest.raises((TypeError, ValueError), match="scan.performance_mode"):
        validate_scan_passes(config)


def test_derived_config_schema_accepts_frame_stride_override():
    validate_config_keys(
        {
            "schema_version": 1,
            "base_config": "base.yaml",
            "scan": {"frame_stride": 2},
        },
        allow_base_config=True,
    )


def test_derived_config_schema_accepts_performance_mode_and_manual_stride():
    validate_config_keys(
        {
            "schema_version": 1,
            "base_config": "base.yaml",
            "scan": {
                "performance_mode": "fast",
                "frame_stride": 3,
            },
        },
        allow_base_config=True,
    )


def test_derived_config_schema_rejects_misspelled_frame_stride():
    with pytest.raises(ValueError, match="scan.frame_strides"):
        validate_config_keys(
            {
                "schema_version": 1,
                "base_config": "base.yaml",
                "scan": {"frame_strides": 2},
            },
            allow_base_config=True,
        )


def test_scan_frame_stride_must_fit_initial_pre_roll():
    config = _valid_config()
    config["scan"]["frame_stride"] = 4
    config["tracking"] = {
        "endpoint_extension": 2,
        "association_max_scan_gap": 12,
        "association_max_gap_seconds": 1.0,
    }

    with pytest.raises(ValueError, match="tracking.endpoint_extension"):
        validate_scan_passes(config)


def test_association_scan_gap_is_independent_of_frame_stride():
    config = _valid_config()
    config["scan"]["frame_stride"] = 3
    config["tracking"] = {
        "endpoint_extension": 8,
        "association_max_scan_gap": 2,
        "association_max_gap_seconds": 1.0,
    }

    validate_scan_passes(config)


def test_scan_frame_stride_accepts_relationship_boundaries():
    config = _valid_config()
    config["scan"]["frame_stride"] = 3
    config["tracking"] = {
        "endpoint_extension": 2,
        "association_max_scan_gap": 3,
        "association_max_gap_seconds": 1.0,
    }

    validate_scan_passes(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("endpoint_extension", True),
        ("endpoint_extension", "8"),
        ("association_max_scan_gap", False),
        ("association_max_scan_gap", 12.0),
    ],
)
def test_scan_frame_stride_relationships_reject_coercion(field, value):
    config = _valid_config()
    config["scan"]["frame_stride"] = 2
    config["tracking"] = {
        "endpoint_extension": 8,
        "association_max_scan_gap": 12,
        "association_max_gap_seconds": 1.0,
    }
    config["tracking"][field] = value

    with pytest.raises(TypeError, match=field):
        validate_scan_passes(config)


@pytest.mark.parametrize("value", [0, -32, 31, 33])
def test_scan_input_size_must_be_a_positive_multiple_of_32(value):
    config = _valid_config()
    config["scan"]["passes"][0]["input_size"] = value

    with pytest.raises(ValueError, match="positive multiple of 32"):
        validate_scan_passes(config)


@pytest.mark.parametrize("value", [True, 64.0, "64", None])
def test_scan_input_size_must_be_an_integer(value):
    config = _valid_config()
    config["scan"]["passes"][0]["input_size"] = value

    with pytest.raises(TypeError, match="input_size must be an integer"):
        validate_scan_passes(config)


@pytest.mark.parametrize("value", [[180], [45], [-180]])
def test_scan_angles_reject_unsupported_inverse_transforms(value):
    config = _valid_config()
    config["scan"]["passes"][0]["angles"] = value

    with pytest.raises(ValueError, match="one of -90, 0, or 90"):
        validate_scan_passes(config)


@pytest.mark.parametrize("value", [[], "0", [0.0], [True]])
def test_scan_angles_require_a_nonempty_integer_list(value):
    config = _valid_config()
    config["scan"]["passes"][0]["angles"] = value

    with pytest.raises(TypeError, match="angles"):
        validate_scan_passes(config)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_scan_confidence_threshold_must_be_finite_and_bounded(value):
    config = _valid_config()
    config["scan"]["passes"][0]["confidence_threshold"] = value

    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        validate_scan_passes(config)


@pytest.mark.parametrize("value", [True, "0.5", None])
def test_scan_confidence_threshold_must_be_numeric(value):
    config = _valid_config()
    config["scan"]["passes"][0]["confidence_threshold"] = value

    with pytest.raises(TypeError, match="confidence_threshold must be a number"):
        validate_scan_passes(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizontal_padding_ratio", -0.01),
        ("horizontal_padding_ratio", float("nan")),
        ("vertical_padding_ratio", 1.01),
        ("vertical_padding_ratio", float("inf")),
    ],
)
def test_scan_padding_must_be_finite_and_at_most_one_source_dimension(
    field,
    value,
):
    config = _valid_config()
    config["scan"]["passes"][0][field] = value

    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        validate_scan_passes(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_size", 0, "positive multiple of 32"),
        ("input_size", 161, "positive multiple of 32"),
        ("angles", [180], "one of -90, 0, or 90"),
        ("confidence_threshold", -0.01, r"finite and in \[0, 1\]"),
        ("confidence_threshold", float("nan"), r"finite and in \[0, 1\]"),
    ],
)
def test_revalidation_shared_detector_settings_are_strict(field, value, message):
    config = _valid_config()
    config["revalidation"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_revalidation_passes(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_size", True),
        ("angles", [0.0]),
        ("confidence_threshold", "0.18"),
    ],
)
def test_revalidation_shared_detector_settings_reject_coercion(field, value):
    config = _valid_config()
    config["revalidation"][field] = value

    with pytest.raises(TypeError, match=field):
        validate_revalidation_passes(config)


def test_revalidation_optional_pass_uses_the_same_input_size_contract():
    config = _valid_config()
    config["revalidation"]["passes"] = [
        {"name": "close_review", "input_size": 150, "crop_expansion": 1.5}
    ]

    with pytest.raises(ValueError, match="positive multiple of 32"):
        validate_revalidation_passes(config)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_revalidation_crop_expansion_must_be_finite_and_positive(value):
    config = deepcopy(_valid_config())
    config["revalidation"]["crop_expansion"] = value

    with pytest.raises(ValueError, match="finite and positive"):
        validate_revalidation_passes(config)
