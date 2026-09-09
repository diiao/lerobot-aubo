#!/usr/bin/env python

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

"""Contracts for the AUBO depth-enhanced VLA bamboo-sorting research."""

from .authorization_contract import (
    CONTROL_SOURCES,
    EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
    ExecutionAuthorizationV1,
    build_authorization_manifest,
)
from .capture_manifest import (
    CAPTURE_PURPOSES,
    RAW_CAPTURE_MANIFEST_SCHEMA_VERSION,
    RawArtifactRecord,
    RawCaptureManifestV1,
    reserve_capture_directory,
)
from .contracts import (
    ACTION_CONTROL_MODE,
    ACTION_FIELD_NAMES,
    ACTION_FIELD_SPECS,
    ACTION_SCHEMA_VERSION,
    CONTROL_FIXED_ID,
    INSTRUCTION_LANGUAGE,
    INSTRUCTION_SCHEMA_VERSION,
    INSTRUCTION_SPECS,
    ActionFieldSpec,
    InstructionSpec,
    build_action_manifest,
    build_instruction_manifest,
    validate_action_vector,
    validate_instruction_fields,
)
from .observation_contract import (
    BASE_TIMESTAMP_STREAMS,
    DEPTH_STREAM_KEY,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_STATE_FIELD_NAMES,
    OBSERVATION_STATE_FIELD_SPECS,
    RGB_STREAM_KEYS,
    CameraIntrinsics,
    EmbodiedObservationV1,
    ObservationStateFieldSpec,
    SensorTimestamp,
    TemporalAlignmentMetrics,
    build_observation_manifest,
    validate_temporal_alignment,
)

__all__ = [
    "CONTROL_SOURCES",
    "EXECUTION_AUTHORIZATION_SCHEMA_VERSION",
    "ExecutionAuthorizationV1",
    "build_authorization_manifest",
    "CAPTURE_PURPOSES",
    "RAW_CAPTURE_MANIFEST_SCHEMA_VERSION",
    "RawArtifactRecord",
    "RawCaptureManifestV1",
    "reserve_capture_directory",
    "ACTION_CONTROL_MODE",
    "ACTION_FIELD_NAMES",
    "ACTION_FIELD_SPECS",
    "ACTION_SCHEMA_VERSION",
    "CONTROL_FIXED_ID",
    "INSTRUCTION_LANGUAGE",
    "INSTRUCTION_SCHEMA_VERSION",
    "INSTRUCTION_SPECS",
    "ActionFieldSpec",
    "InstructionSpec",
    "build_action_manifest",
    "build_instruction_manifest",
    "validate_action_vector",
    "validate_instruction_fields",
    "BASE_TIMESTAMP_STREAMS",
    "DEPTH_STREAM_KEY",
    "OBSERVATION_SCHEMA_VERSION",
    "OBSERVATION_STATE_FIELD_NAMES",
    "OBSERVATION_STATE_FIELD_SPECS",
    "RGB_STREAM_KEYS",
    "CameraIntrinsics",
    "EmbodiedObservationV1",
    "ObservationStateFieldSpec",
    "SensorTimestamp",
    "TemporalAlignmentMetrics",
    "build_observation_manifest",
    "validate_temporal_alignment",
]
