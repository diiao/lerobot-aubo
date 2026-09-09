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
from pathlib import Path

import pytest

from lerobot.bamboo_sorting.capture_manifest import (
    RAW_CAPTURE_MANIFEST_SCHEMA_VERSION,
    RawArtifactRecord,
    RawCaptureManifestV1,
    reserve_capture_directory,
)
from lerobot.bamboo_sorting.contracts import INSTRUCTION_LANGUAGE, INSTRUCTION_SPECS


def _manifest(**changes: object) -> RawCaptureManifestV1:
    values = {
        "schema_version": RAW_CAPTURE_MANIFEST_SCHEMA_VERSION,
        "capture_id": "capture-dev-0001",
        "session_id": "session-dev-0001",
        "capture_purpose": "depth_go_no_go_static",
        "created_at_utc": "2026-09-09T08:00:00Z",
        "authorization_evidence_ref": "operator-approval-dev-0001",
        "scene_id": "scene-dev-0001",
        "calibration_version": "calib-dev-unverified",
        "tool_version": "tool-dev-unverified",
        "capture_profile_id": "profile-dev-unverified",
        "go_no_go_threshold_version": "depth-gate-v1",
        "representative_frame_rule_version": "representative-frame-v1",
        "depth_requested": True,
        "raw_data_immutable": True,
        "finalized": False,
        "sensor_identifiers": {"wrist_rgbd": "serial-unverified"},
        "capture_parameters": {"requested_frame_count": 20, "depth_unit": "m"},
    }
    values.update(changes)
    return RawCaptureManifestV1(**values)


def test_prepared_manifest_freezes_schema_versions_without_artifacts() -> None:
    manifest = _manifest()
    serialized = manifest.to_dict()

    assert serialized["schema_version"] == RAW_CAPTURE_MANIFEST_SCHEMA_VERSION
    assert serialized["raw_data_immutable"] is True
    assert serialized["artifacts"] == []


def test_teleop_demonstration_requires_canonical_language_from_day_one() -> None:
    with pytest.raises(ValueError, match="requires canonical instruction fields"):
        _manifest(capture_purpose="teleop_demonstration")

    instruction = INSTRUCTION_SPECS["pick_any_collection"]
    manifest = _manifest(
        capture_purpose="teleop_demonstration",
        instruction_id=instruction.instruction_id,
        instruction_text=instruction.text,
        instruction_language=INSTRUCTION_LANGUAGE,
        instruction_text_sha256=instruction.text_sha256,
    )

    assert manifest.instruction_text == "Pick one strip and place it in the collection area."


def test_control_instruction_cannot_enter_demonstration_manifest() -> None:
    instruction = INSTRUCTION_SPECS["control_fixed"]

    with pytest.raises(ValueError, match="not valid for demonstrations"):
        _manifest(
            capture_purpose="teleop_demonstration",
            instruction_id=instruction.instruction_id,
            instruction_text=instruction.text,
            instruction_language=INSTRUCTION_LANGUAGE,
            instruction_text_sha256=instruction.text_sha256,
        )


def test_finalized_manifest_requires_hashed_artifacts() -> None:
    artifact = RawArtifactRecord(
        logical_name="wrist_rgbd_raw",
        relative_path="streams/wrist_rgbd_000.npz",
        sha256="a" * 64,
        byte_count=1024,
        sample_count=20,
    )
    manifest = _manifest(finalized=True, artifacts=(artifact,))

    assert manifest.artifacts == (artifact,)
    with pytest.raises(ValueError, match="requires at least one artifact"):
        _manifest(finalized=True)
    with pytest.raises(ValueError, match="only be attached"):
        _manifest(artifacts=(artifact,))


@pytest.mark.parametrize("relative_path", ["/tmp/raw.npz", "../raw.npz", "streams/../raw.npz"])
def test_artifact_path_must_remain_inside_capture_directory(relative_path: str) -> None:
    with pytest.raises(ValueError, match="remain inside"):
        RawArtifactRecord(
            logical_name="wrist_rgbd_raw",
            relative_path=relative_path,
            sha256="a" * 64,
            byte_count=1,
            sample_count=1,
        )


def test_manifest_rejects_non_utc_non_json_or_mutable_raw_contract() -> None:
    with pytest.raises(ValueError, match="UTC timezone"):
        _manifest(created_at_utc="2026-09-09T08:00:00+08:00")
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        _manifest(capture_parameters={"bad": float("nan")})
    with pytest.raises(ValueError, match="raw_data_immutable=true"):
        _manifest(raw_data_immutable=False)


def test_reserve_capture_directory_never_reuses_existing_path(tmp_path: Path) -> None:
    capture_directory = reserve_capture_directory(tmp_path, "capture-dev-0001")

    assert capture_directory.is_dir()
    with pytest.raises(FileExistsError):
        reserve_capture_directory(tmp_path, "capture-dev-0001")
