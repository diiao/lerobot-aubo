# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import replace

import numpy as np
import pytest

from lerobot.bamboo_sorting.contracts import (
    ACTION_SCHEMA_VERSION,
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_SCHEMA_VERSION,
    INSTRUCTION_SPECS,
)
from lerobot.bamboo_sorting.observation_contract import (
    BASE_TIMESTAMP_STREAMS,
    DEPTH_STREAM_KEY,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_STATE_FIELD_NAMES,
    CameraIntrinsics,
    EmbodiedObservationV1,
    SensorTimestamp,
    build_observation_manifest,
    validate_temporal_alignment,
)


def _timestamp(value: float) -> SensorTimestamp:
    return SensorTimestamp(
        sync_timestamp_s=value,
        host_receive_monotonic_s=value,
        sync_method="host_receive",
    )


def _observation(*, with_depth: bool = False) -> EmbodiedObservationV1:
    instruction = INSTRUCTION_SPECS["pick_any_collection"]
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    timestamps = {
        stream: _timestamp(10.0 + index * 0.001) for index, stream in enumerate(BASE_TIMESTAMP_STREAMS)
    }
    kwargs = {}
    if with_depth:
        valid = np.ones((3, 4), dtype=bool)
        depth = np.full((3, 4), 0.5, dtype=np.float32)
        xyz = np.zeros((3, 4, 3), dtype=np.float32)
        xyz[..., 2] = depth
        timestamps[DEPTH_STREAM_KEY] = _timestamp(10.002)
        kwargs = {
            "wrist_depth_m": depth,
            "wrist_depth_valid": valid,
            "wrist_xyz_m": xyz,
            "wrist_depth_intrinsics": CameraIntrinsics(
                width_px=4,
                height_px=3,
                fx_px=90.0,
                fy_px=90.0,
                cx_px=1.5,
                cy_px=1.0,
                distortion_model="none",
                distortion_coefficients=(),
            ),
        }

    t_ee_camera = np.eye(4, dtype=np.float64)
    t_ee_camera[:3, 3] = [0.0, 0.1, 0.2]
    t_base_ee = np.eye(4, dtype=np.float64)
    t_base_ee[:3, 3] = [0.5, -0.5, 0.3]
    intrinsics = CameraIntrinsics(
        width_px=5,
        height_px=4,
        fx_px=100.0,
        fy_px=100.0,
        cx_px=2.0,
        cy_px=1.5,
        distortion_model="none",
        distortion_coefficients=(),
    )
    return EmbodiedObservationV1(
        global_rgb=image.copy(),
        grasp_rgb=image.copy(),
        wrist_rgb=image.copy(),
        robot_state=[0.0] * 12 + [0.0],
        timestamps=timestamps,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        instruction_schema_version=INSTRUCTION_SCHEMA_VERSION,
        instruction_id=instruction.instruction_id,
        instruction_text=instruction.text,
        instruction_language=INSTRUCTION_LANGUAGE,
        scene_id="scene-0001",
        calibration_version="calib-dev-unverified",
        tool_version="tool-dev-unverified",
        model_input_ref="model-input-dev-unverified",
        action_schema_version=ACTION_SCHEMA_VERSION,
        camera_intrinsics={stream: intrinsics for stream in BASE_TIMESTAMP_STREAMS[:3]},
        t_base_global_camera=np.eye(4, dtype=np.float64),
        t_base_grasp_camera=np.eye(4, dtype=np.float64),
        t_ee_camera=t_ee_camera,
        t_base_ee=t_base_ee,
        **kwargs,
    )


def test_observation_manifest_freezes_run06_state_and_dynamic_extrinsic() -> None:
    manifest = build_observation_manifest(depth_enabled=True)

    assert manifest["schema_version"] == OBSERVATION_SCHEMA_VERSION
    assert OBSERVATION_STATE_FIELD_NAMES == (
        "J1",
        "J2",
        "J3",
        "J4",
        "J5",
        "J6",
        "ee.x",
        "ee.y",
        "ee.z",
        "ee.wx",
        "ee.wy",
        "ee.wz",
        "gripper_pos",
    )
    assert manifest["stored_transforms"] == [
        "t_base_global_camera",
        "t_base_grasp_camera",
        "t_ee_camera",
        "t_base_ee",
    ]
    assert manifest["derived_transforms"] == ["t_base_camera"]
    assert "wrist_depth_m" in manifest["depth_fields"]
    assert "wrist_depth_intrinsics" in manifest["depth_fields"]


def test_observation_derives_time_varying_base_camera_transform() -> None:
    observation = _observation()

    np.testing.assert_allclose(observation.t_base_camera[:3, 3], [0.5, -0.4, 0.5])
    assert "t_base_camera" not in observation.__dataclass_fields__


def test_observation_accepts_complete_depth_bundle_and_timing() -> None:
    observation = _observation(with_depth=True)

    metrics = observation.validate_timing(reference_time_s=10.010, max_skew_ms=10.0, max_age_ms=20.0)

    assert observation.depth_enabled
    assert metrics.max_pairwise_skew_ms == pytest.approx(3.0)
    assert metrics.oldest_sample_age_ms == pytest.approx(10.0)


def test_observation_rejects_partial_depth_bundle() -> None:
    observation = _observation(with_depth=True)

    with pytest.raises(ValueError, match="must be present together"):
        replace(observation, wrist_xyz_m=None)


def test_depth_layout_uses_its_own_intrinsics_instead_of_rgb_dimensions() -> None:
    observation = _observation(with_depth=True)

    assert observation.wrist_rgb.shape[:2] == (4, 5)
    assert observation.wrist_depth_m.shape == (3, 4)

    wrong_size = replace(observation.wrist_depth_intrinsics, width_px=5)
    with pytest.raises(ValueError, match="must match wrist_depth_intrinsics"):
        replace(observation, wrist_depth_intrinsics=wrong_size)


def test_observation_rejects_invalid_depth_or_xyz_at_valid_pixels() -> None:
    observation = _observation(with_depth=True)
    bad_depth = observation.wrist_depth_m.copy()
    bad_depth[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite and positive"):
        replace(observation, wrist_depth_m=bad_depth)

    bad_xyz = observation.wrist_xyz_m.copy()
    bad_xyz[0, 0, 2] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        replace(observation, wrist_xyz_m=bad_xyz)


def test_observation_allows_nonfinite_depth_only_where_mask_is_false() -> None:
    observation = _observation(with_depth=True)
    depth = observation.wrist_depth_m.copy()
    xyz = observation.wrist_xyz_m.copy()
    valid = observation.wrist_depth_valid.copy()
    valid[0, 0] = False
    depth[0, 0] = np.nan
    xyz[0, 0] = np.nan

    replaced = replace(
        observation,
        wrist_depth_m=depth,
        wrist_depth_valid=valid,
        wrist_xyz_m=xyz,
    )

    assert replaced.depth_enabled


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"robot_state": [0.0] * 12}, "requires 13"),
        ({"robot_state": [float("nan")] + [0.0] * 12}, "must be finite"),
        ({"robot_state": [0.0] * 12 + [50.0]}, "legacy 0/100"),
        ({"global_rgb": np.zeros((4, 5, 3), dtype=np.float32)}, "uint8 HxWx3"),
        ({"scene_id": ""}, "non-empty"),
        ({"observation_schema_version": "EmbodiedObservationV2"}, "Unsupported observation schema"),
        ({"action_schema_version": "ActionSchemaV2"}, "Unsupported action schema"),
    ],
)
def test_observation_rejects_structural_contract_drift(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_observation(), **changes)


def test_observation_rejects_invalid_transform() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = 2.0

    with pytest.raises(ValueError, match="rotation must be orthonormal"):
        replace(_observation(), t_ee_camera=transform)


def test_observation_rejects_missing_or_mismatched_camera_intrinsics() -> None:
    observation = _observation()

    with pytest.raises(ValueError, match="Missing camera intrinsics"):
        replace(observation, camera_intrinsics={"global_rgb": observation.camera_intrinsics["global_rgb"]})

    wrong_size = replace(observation.camera_intrinsics["global_rgb"], width_px=6)
    intrinsics = dict(observation.camera_intrinsics)
    intrinsics["global_rgb"] = wrong_size
    with pytest.raises(ValueError, match="dimensions must match"):
        replace(observation, camera_intrinsics=intrinsics)


def test_timestamp_rejects_ambiguous_device_clock() -> None:
    with pytest.raises(ValueError, match="provided together"):
        SensorTimestamp(
            sync_timestamp_s=10.0,
            host_receive_monotonic_s=10.0,
            sync_method="host_receive",
            device_timestamp_s=123.0,
        )

    with pytest.raises(ValueError, match="requires a raw device timestamp"):
        SensorTimestamp(
            sync_timestamp_s=10.0,
            host_receive_monotonic_s=10.1,
            sync_method="device_to_host_calibrated",
        )


def test_timestamp_accepts_explicit_device_to_host_calibration() -> None:
    timestamp = SensorTimestamp(
        sync_timestamp_s=10.0,
        host_receive_monotonic_s=10.1,
        sync_method="device_to_host_calibrated",
        device_timestamp_s=123.0,
        device_clock_id="mech_eye_device_clock",
    )

    assert timestamp.sync_timestamp_s == 10.0


def test_host_receive_timestamp_cannot_claim_a_different_sync_time() -> None:
    with pytest.raises(ValueError, match="must use host_receive_monotonic_s"):
        SensorTimestamp(
            sync_timestamp_s=10.0,
            host_receive_monotonic_s=10.1,
            sync_method="host_receive",
        )


def test_timing_rejects_missing_stale_skewed_and_future_streams() -> None:
    timestamps = {stream: _timestamp(10.0) for stream in BASE_TIMESTAMP_STREAMS}

    with pytest.raises(ValueError, match="Missing timestamp streams"):
        validate_temporal_alignment(
            {"global_rgb": _timestamp(10.0)},
            required_streams=BASE_TIMESTAMP_STREAMS,
            reference_time_s=10.01,
            max_skew_ms=50.0,
            max_age_ms=100.0,
        )

    stale = dict(timestamps)
    stale["global_rgb"] = _timestamp(9.0)
    with pytest.raises(ValueError, match="Cross-stream skew"):
        validate_temporal_alignment(
            stale,
            required_streams=BASE_TIMESTAMP_STREAMS,
            reference_time_s=10.01,
            max_skew_ms=50.0,
            max_age_ms=100.0,
        )

    with pytest.raises(ValueError, match="Oldest sample age"):
        validate_temporal_alignment(
            timestamps,
            required_streams=BASE_TIMESTAMP_STREAMS,
            reference_time_s=10.2,
            max_skew_ms=50.0,
            max_age_ms=100.0,
        )

    future = dict(timestamps)
    future["wrist_rgb"] = _timestamp(10.02)
    with pytest.raises(ValueError, match="cannot be in the future"):
        validate_temporal_alignment(
            future,
            required_streams=BASE_TIMESTAMP_STREAMS,
            reference_time_s=10.01,
            max_skew_ms=50.0,
            max_age_ms=100.0,
        )
