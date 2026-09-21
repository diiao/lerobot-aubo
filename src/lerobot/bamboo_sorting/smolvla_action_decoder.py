"""Explicit regression-score to binary-gripper decoding before the safety gate.

Raw model scores are retained and are not ActionSchemaV1 commands. Only the
last coordinate is decoded; Cartesian and joint predictions are unmodified.
"""

import math
from numbers import Real
from collections.abc import Sequence
from dataclasses import dataclass

from .contracts import validate_action_vector

SMOLVLA_GRIPPER_DECODER_VERSION = "SmolVLAGripperDecodeV1"
GRIPPER_DECISION_THRESHOLD = 50.0


@dataclass(frozen=True)
class DecodedSmolVLAChunk:
    decoder_version: str
    raw_actions: tuple[tuple[float, ...], ...]
    actions: tuple[tuple[float, ...], ...]


def decode_smolvla_gripper(raw_actions: Sequence[Sequence[object]]) -> DecodedSmolVLAChunk:
    """Decode finite 0/100-trained regression scores at their fixed midpoint.

    Values outside [0, 100] are possible regression scores. They are recorded,
    not clipped. This function has no task coordinates, state, or time input.
    """
    if isinstance(raw_actions, (str, bytes)) or len(raw_actions) == 0:
        raise ValueError("raw_actions must be a nonempty action chunk")
    raw = []
    decoded = []
    for row in raw_actions:
        if isinstance(row, (str, bytes)) or len(row) != 8:
            raise ValueError("each raw action must have 8 numeric values")
        values = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("raw action values must be numeric and not bool")
            try:
                number = float(value)
            except (ValueError, TypeError) as exc:
                raise ValueError("raw action values must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError("raw action values must be finite")
            values.append(number)
        raw.append(tuple(values))
        command = (*values[:7], 100.0 if values[7] >= GRIPPER_DECISION_THRESHOLD else 0.0)
        decoded.append(validate_action_vector(command))
    return DecodedSmolVLAChunk(SMOLVLA_GRIPPER_DECODER_VERSION, tuple(raw), tuple(decoded))
