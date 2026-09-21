"""Joint capture evidence and lifecycle; importing this module opens no devices.

Reuse the existing timestamp and suction-controller journals, adapting only
the suction field name at that journal boundary. Joint labels are checked
against the accepted SDK target before a frame is added. These are raw
capture records, not historical C0 finalized manifests or physical proof.
"""

import copy
import math
from pathlib import Path

import numpy as np

from .aubo_joint_contract import JOINT_FIELDS, JOINT_NAMES, expand_joint_command, joint_command, joint_contract_record
from .c0_batch_capture import (
    C0FormalFrameObserverV1, C0SensorTimestampJournalV1, _write_json_atomic_exclusive,
)
from .c0_gripper_capture_journal import C0GripperPendingCaptureJournalV1


def write_joint_json(path, record):
    _write_json_atomic_exclusive(Path(path), record)


class JointFrameObserver:
    def __init__(self, episode_index):
        identity = f"joint-episode-{episode_index:04d}"
        self.shared = C0FormalFrameObserverV1(
            timestamp_journal=C0SensorTimestampJournalV1(episode_id=identity, episode_index=episode_index),
            gripper_journal=C0GripperPendingCaptureJournalV1(
                episode_id=identity, episode_index=episode_index, fps=25),
        )
        self.frames = []
        self.pending = None
        self.failed = False

    @staticmethod
    def _journal_args(kwargs):
        result = dict(kwargs)
        # The original journal requires its historical C0 suction field.
        # This detached view is never sent to the robot or saved as an action.
        if "action" in result:
            result["action"] = {"ee.gripper_pos": joint_command(result["action"])["gripper_pos"]}
        return result

    def before_send(self, **kwargs):
        if self.failed or self.pending is not None:
            raise ValueError("joint observer is not ready")
        command = joint_command(kwargs["action"])
        if command != joint_command(kwargs["robot_action_to_send"]):
            raise ValueError("joint dataset label differs from command to send")
        if kwargs["candidate_frame_index"] != len(self.frames):
            raise ValueError("joint frame index is not contiguous")
        self.shared.before_send(**self._journal_args(kwargs))
        self.pending = {"frame_index": len(self.frames), "action": [command[k] for k in JOINT_FIELDS],
                        "sequence_before": getattr(kwargs["robot"], "_joint_sequence", None)}

    def on_send_success(self, **kwargs):
        if self.pending is None or self.failed:
            raise ValueError("joint command was not staged")
        self.failed = True  # Only clear after all evidence agrees.
        returned = joint_command(kwargs["sent_action"])
        if [returned[k] for k in JOINT_FIELDS] != self.pending["action"]:
            raise ValueError("returned joint command differs from label")
        trace = copy.deepcopy(getattr(kwargs["robot"], "last_joint_command_trace", None))
        before = self.pending["sequence_before"]
        if (not isinstance(trace, dict) or type(before) is not int
                or trace.get("sequence") != before + 1 or trace.get("return_code") != 0):
            raise ValueError("fresh accepted joint command trace is required")
        full = expand_joint_command(dict(zip(JOINT_FIELDS, self.pending["action"], strict=True)))
        expected_joints = [full[k] for k in JOINT_NAMES]
        if trace.get("joint_target_deg") != expected_joints:
            raise ValueError("SDK joint target differs from dataset label")
        expected_rad = [math.radians(v) for v in expected_joints]
        if trace.get("sdk_target_rad") != expected_rad:
            raise ValueError("SDK joint target unit conversion differs from label")
        accepted = trace.get("accepted_monotonic_s")
        if isinstance(accepted, bool) or not isinstance(accepted, (float, int)) or not math.isfinite(accepted):
            raise ValueError("joint trace timestamp is invalid")
        self.shared.on_send_success(**self._journal_args(kwargs))
        self.pending["joint_command_trace"] = trace
        self.failed = False

    def on_add_frame_success(self, **kwargs):
        if self.failed or self.pending is None or "joint_command_trace" not in self.pending:
            raise ValueError("joint frame lacks successful command evidence")
        self.shared.on_add_frame_success(**self._journal_args(kwargs))
        self.frames.append(self.pending)
        self.pending = None

    def on_send_failure(self, **kwargs):
        self.failed = True
        self.shared.on_send_failure(**self._journal_args(kwargs))

    def on_add_frame_failure(self, **kwargs):
        self.failed = True
        self.shared.on_add_frame_failure(**self._journal_args(kwargs))

    def validate_buffer(self, dataset):
        buffer = dataset.episode_buffer
        if self.failed or self.pending is not None or not self.frames or buffer["size"] != len(self.frames):
            raise ValueError("joint episode is incomplete; refusing save")
        if self.shared.timestamp_journal.frame_count != len(self.frames):
            raise ValueError("timestamp count differs from joint frames")
        if self.shared.gripper_journal.episode_failed or len(self.shared.gripper_journal.cycles) != len(self.frames):
            raise ValueError("suction evidence is incomplete")
        expected = np.asarray([f["action"] for f in self.frames], dtype=np.float32)
        if not np.array_equal(np.asarray(buffer["action"], dtype=np.float32), expected):
            raise ValueError("recorded action buffer differs from accepted joint commands")

    def record(self):
        return {"joint_frames": copy.deepcopy(self.frames),
                "sensor_and_suction_evidence": self.shared.to_manifest_record()}


class JointCaptureSession:
    """Variable-length capture; stop/rerecord never saves an unfinished episode."""

    def __init__(self, *, dataset_root, evidence_root, episode_count, prepare_episode, outcome_provider):
        self.dataset_root = Path(dataset_root).resolve()
        self.evidence_root = Path(evidence_root).resolve()
        if type(episode_count) is not int or episode_count < 1:
            raise ValueError("episode_count must be positive")
        self.episode_count = episode_count
        self.prepare_episode = prepare_episode
        self.outcome_provider = outcome_provider
        self.attempts = []
        self.active = None
        self.saved_count = 0
        self.finalized = False

    def episode_frame_observer(self):
        if self.active is not None or self.saved_count >= self.episode_count:
            raise ValueError("joint session is not ready for an episode")
        metadata = self.prepare_episode(self.saved_count)
        if not isinstance(metadata, dict) or any(not isinstance(metadata.get(k), str) or not metadata[k].strip()
                                               for k in ("scene_id", "placement_reference")):
            raise ValueError("scene_id and physical placement_reference are required")
        self.active = {"episode_index": self.saved_count, "metadata": copy.deepcopy(metadata),
                       "status": "recording", "observer": JointFrameObserver(self.saved_count)}
        self.attempts.append(self.active)
        return self.active["observer"]

    def _require_active(self, dataset, episode_index):
        if Path(dataset.root).resolve() != self.dataset_root:
            raise ValueError("joint session dataset root mismatch")
        if self.active is None or episode_index != self.active["episode_index"]:
            raise ValueError("joint session episode mismatch")
        return self.active

    def on_episode_rerecord(self, *, dataset, episode_index, **_):
        self._require_active(dataset, episode_index)["status"] = "abandoned_rerecorded"
        self.active = None

    def on_episode_incomplete(self, *, dataset, episode_index, reason, **_):
        attempt = self._require_active(dataset, episode_index)
        attempt.update(status="incomplete", reason=reason)
        self.active = None
        dataset.clear_episode_buffer()

    def save_episode(self, *, dataset, episode_index, **_):
        attempt = self._require_active(dataset, episode_index)
        attempt["observer"].validate_buffer(dataset)
        outcome = self.outcome_provider(episode_index)
        if outcome not in ("single_success", "empty", "multi_pick", "slip", "blocked", "uncertain"):
            raise ValueError("invalid human outcome")
        attempt["human_outcome"] = outcome
        path = self.evidence_root / f"episode-{episode_index:04d}.json"
        if path.exists():
            raise ValueError("joint episode sidecar already exists")
        attempt["status"] = "save_uncertain"
        dataset.save_episode()  # Never retry a save that raised.
        self.saved_count += 1
        attempt["status"] = "saved_pending_finalize"
        write_joint_json(path, self._record_attempt(attempt))
        self.active = None

    def _record_attempt(self, attempt):
        return {**{k: copy.deepcopy(v) for k, v in attempt.items() if k != "observer"},
                **attempt["observer"].record(), "contract": joint_contract_record(),
                "dataset_root": str(self.dataset_root), "physical_grasp_success_proven": False}

    def on_dataset_finalize_success(self, *, dataset, recorded_episode_count):
        if Path(dataset.root).resolve() != self.dataset_root or recorded_episode_count != self.saved_count:
            raise ValueError("joint finalize dataset/count mismatch")
        if dataset.num_episodes != self.saved_count:
            raise ValueError("joint finalized episode count mismatch")
        record = {"contract": joint_contract_record(), "dataset_root": str(self.dataset_root),
                  "dataset_finalized": True, "saved_episode_count": self.saved_count,
                  "requested_episode_count": self.episode_count,
                  "capture_complete": self.saved_count == self.episode_count and self.active is None,
                  "attempts": [self._record_attempt(a) for a in self.attempts],
                  "training_authorized": False, "policy_execution_authorized": False}
        write_joint_json(self.evidence_root / "session.json", record)
        self.finalized = True

    def on_dataset_finalize_failure(self, *, error, **_):
        write_joint_json(self.evidence_root / "finalize_failure.json", {"error": str(error),
                         "dataset_finalized": False, "saved_episode_count": self.saved_count})
