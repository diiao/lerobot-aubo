"""Offline geometric association of fresh instance masks, not mask propagation.

Frame-local IDs and classes do not establish physical identity. These initial
heuristics require real-scene review, especially with similar neighboring strips.
"""

import copy
import math

import cv2
import numpy as np


class TargetAssociation:
    PARAMETERS = {
        "max_center_distance_px": 40.0, "max_axis_angle_deg": 45.0,
        "min_area_ratio": 0.25, "max_area_ratio": 4.0,
        "min_seed_area_ratio": 0.2, "max_seed_area_ratio": 5.0,
        "min_overlap": 0.05, "min_score": 0.35, "ambiguity_margin": 0.15,
        "score_weights": {"overlap": 0.6, "center": 0.25, "axis": 0.15},
    }

    def __init__(self, target_id, *, alignment="translation"):
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("stable target_id required")
        self.target_id = target_id
        if alignment not in ("translation", "rigid"):
            raise ValueError("unknown shape alignment")
        self.alignment = alignment
        self.mask = self.center = self.axis = None
        self.velocity = np.zeros(2)
        self.last_frame = None
        self.seed_area = None

    @staticmethod
    def _candidates(instance_map, segments):
        if (instance_map.shape != (480, 640) or not np.issubdtype(instance_map.dtype, np.integer)
                or np.any(instance_map < 0)):
            raise ValueError("integer 480x640 instance map required")
        ids = [s["mask_id"] for s in segments]
        if len(ids) != len(set(ids)) or any(type(i) is not int or i <= 0 for i in ids):
            raise ValueError("unique positive instance IDs required")
        if not set(np.unique(instance_map)) <= {0, *ids}:
            raise ValueError("instance map contains IDs absent from segments")
        return [(segment, instance_map == segment["mask_id"]) for segment in segments
                if np.any(instance_map == segment["mask_id"])]

    @staticmethod
    def _geometry(mask):
        y, x = np.nonzero(mask)
        points = np.column_stack((x, y)).astype(float)
        center = points.mean(axis=0)
        centered = points - center
        _, axes = np.linalg.eigh(centered.T @ centered / max(len(points), 1))
        return center, axes[:, -1]

    def select(self, frame_index, instance_map, segments, instance_id):
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("nonnegative frame index required")
        selected = [(s, m) for s, m in self._candidates(instance_map, segments) if s["mask_id"] == instance_id]
        if len(selected) != 1:
            raise ValueError("explicit selected instance must be visible")
        segment, mask = selected[0]
        self.mask = mask.copy()
        self.center, self.axis = self._geometry(mask)
        self.velocity = np.zeros(2)
        self.last_frame, self.seed_area = frame_index, int(mask.sum())
        return mask.copy(), {"frame": frame_index, "target_id": self.target_id, "status": "seed",
                             "mask_id": instance_id, "class_id": segment["class_id"], "reviewed": False}

    def _lost(self, frame_index, reason, candidates=None):
        self.mask = None
        self.last_frame = frame_index
        return None, {"frame": frame_index, "target_id": self.target_id, "status": "lost", "reason": reason,
                      "candidates": candidates or [], "mask_id": None, "reviewed": False}

    def step(self, frame_index, instance_map, segments):
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("nonnegative frame index required")
        candidates = self._candidates(instance_map, segments)
        if self.mask is None:
            return self._lost(frame_index, "explicit_selection_required")
        if frame_index != self.last_frame + 1:
            return self._lost(frame_index, "nonconsecutive_frame")
        parameters = self.PARAMETERS
        predicted_center = self.center + self.velocity
        shift = np.float32([[1, 0, self.velocity[0]], [0, 1, self.velocity[1]]])
        predicted_mask = cv2.warpAffine(self.mask.astype(np.uint8), shift, (640, 480),
                                        flags=cv2.INTER_NEAREST).astype(bool)
        ranked, details = [], []
        for segment, mask in candidates:
            center, axis = self._geometry(mask)
            distance = float(np.linalg.norm(center - predicted_center))
            angle = math.degrees(math.acos(float(np.clip(abs(axis @ self.axis), 0, 1))))
            area_ratio = float(mask.sum() / self.mask.sum())
            seed_area_ratio = float(mask.sum() / self.seed_area)
            overlaps = [np.count_nonzero(mask & prior) / max(1, np.count_nonzero(mask | prior))
                        for prior in (self.mask, predicted_mask)]
            if self.alignment == "rigid":
                # Compare shape after aligning centroids and the unoriented long
                # axis. This transformed mask is NEVER returned as a prediction.
                aligned_axis = axis if axis @ self.axis >= 0 else -axis
                rotation = math.degrees(math.atan2(self.axis[0] * aligned_axis[1] - self.axis[1] * aligned_axis[0],
                                                   float(self.axis @ aligned_axis)))
                transform = cv2.getRotationMatrix2D(tuple(self.center), -rotation, 1.0)
                transform[:, 2] += center - self.center
                aligned = cv2.warpAffine(self.mask.astype(np.uint8), transform, (640, 480),
                                        flags=cv2.INTER_NEAREST).astype(bool)
                overlaps.append(np.count_nonzero(mask & aligned) / max(1, np.count_nonzero(mask | aligned)))
            overlap = float(max(overlaps))
            weights = parameters["score_weights"]
            score = (weights["overlap"] * overlap
                     + weights["center"] * max(0, 1 - distance / parameters["max_center_distance_px"])
                     + weights["axis"] * max(0, 1 - angle / parameters["max_axis_angle_deg"]))
            eligible = (distance <= parameters["max_center_distance_px"] and angle <= parameters["max_axis_angle_deg"]
                        and parameters["min_area_ratio"] <= area_ratio <= parameters["max_area_ratio"]
                        and parameters["min_seed_area_ratio"] <= seed_area_ratio <= parameters["max_seed_area_ratio"]
                        and overlap >= parameters["min_overlap"] and score >= parameters["min_score"])
            detail = {"mask_id": segment["mask_id"], "class_id": segment["class_id"], "score": score,
                      "center_distance_px": distance, "axis_angle_deg": angle, "area_ratio": area_ratio,
                      "seed_area_ratio": seed_area_ratio, "overlap": overlap, "eligible": eligible}
            details.append(detail)
            if eligible:
                ranked.append((score, segment, mask, center, axis))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            return self._lost(frame_index, "no_compatible_candidate", details)
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < parameters["ambiguity_margin"]:
            return self._lost(frame_index, "ambiguous_candidates", details)
        _, segment, mask, center, axis = ranked[0]
        self.velocity = 0.5 * self.velocity + 0.5 * (center - self.center)
        self.mask, self.center, self.axis, self.last_frame = mask.copy(), center, axis, frame_index
        return mask.copy(), {"frame": frame_index, "target_id": self.target_id, "status": "matched",
                             "mask_id": segment["mask_id"], "class_id": segment["class_id"],
                             "candidates": details, "reviewed": False}


class TargetReconfirmation:
    """Bounded re-confirmation after empty detections; never after ambiguity.

    Keep the last accepted geometry privately during a gap. No target mask is
    emitted until two distinct source images support both the anchor and a
    consistent candidate trajectory. Geometric agreement is not physical proof.
    """

    PARAMETERS = {
        "max_gap_frames": 5, "required_confirmations": 2, "min_score": 0.6,
        "max_anchor_center_distance_px": 30.0, "max_axis_angle_deg": 15.0,
        "min_area_ratio": 0.5, "max_area_ratio": 2.0, "min_overlap": 0.5,
    }

    def __init__(self, target_id):
        self.track = TargetAssociation(target_id, alignment="rigid")
        self.target_id = target_id
        self.last_input = self.last_timestamp = self.accepted_timestamp = None
        self.pending = False
        self.candidate_track = None
        self.confirmations = 0
        self.candidate_timestamp = None

    @staticmethod
    def _timestamp(frame_index, source_timestamp):
        stamp = frame_index if source_timestamp is None else source_timestamp
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            raise ValueError("finite source timestamp required")
        return stamp

    def select(self, frame_index, instance_map, segments, instance_id, *, source_timestamp=None):
        stamp = self._timestamp(frame_index, source_timestamp)
        result = self.track.select(frame_index, instance_map, segments, instance_id)
        self.last_input = frame_index
        self.last_timestamp = self.accepted_timestamp = stamp
        self.pending = False
        self.candidate_track = None
        self.confirmations = 0
        self.candidate_timestamp = None
        return result

    def _lost(self, frame_index, reason, candidates=None):
        self.pending = False
        self.candidate_track = None
        self.confirmations = 0
        return self.track._lost(frame_index, reason, candidates)

    def _waiting(self, frame_index, status, reason, candidates=None):
        return None, {"frame": frame_index, "target_id": self.target_id, "status": status, "reason": reason,
                      "mask_id": None, "candidates": candidates or [], "reviewed": False,
                      "gap_frames": frame_index - self.track.last_frame, "confirmations": self.confirmations}

    def step(self, frame_index, instance_map, segments, *, source_timestamp=None):
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("nonnegative frame index required")
        stamp = self._timestamp(frame_index, source_timestamp)
        candidates = self.track._candidates(instance_map, segments)
        consecutive = self.last_input is not None and frame_index == self.last_input + 1
        monotonic = self.last_timestamp is None or stamp >= self.last_timestamp
        self.last_input, self.last_timestamp = frame_index, stamp
        if self.track.mask is None:
            return self._lost(frame_index, "explicit_selection_required")
        if not consecutive or not monotonic:
            return self._lost(frame_index, "frame_or_source_time_discontinuity")
        gap = frame_index - self.track.last_frame
        if self.pending and gap > self.PARAMETERS["max_gap_frames"]:
            return self._lost(frame_index, "reconfirmation_timeout")
        if not candidates:
            self.pending = True
            self.candidate_track = None
            self.confirmations = 0
            self.candidate_timestamp = None
            return self._waiting(frame_index, "unobserved", "empty_detection")
        if not self.pending:
            selected, record = self.track.step(frame_index, instance_map, segments)
            if selected is not None:
                self.accepted_timestamp = stamp
            return selected, record

        # Repeated video ticks are not additional camera evidence.
        if stamp <= self.accepted_timestamp:
            return self._waiting(frame_index, "confirming", "no_new_source_image")
        probe = copy.deepcopy(self.track)
        probe.last_frame = frame_index - 1
        probe.velocity *= gap
        selected, candidate = probe.step(frame_index, instance_map, segments)
        if selected is None:
            return self._lost(frame_index, "reconfirmation_" + candidate["reason"], candidate["candidates"])
        detail = next(x for x in candidate["candidates"] if x["mask_id"] == candidate["mask_id"])
        center_distance = float(np.linalg.norm(probe.center - self.track.center))
        parameters = self.PARAMETERS
        if not (detail["score"] >= parameters["min_score"]
                and center_distance <= parameters["max_anchor_center_distance_px"]
                and detail["axis_angle_deg"] <= parameters["max_axis_angle_deg"]
                and parameters["min_area_ratio"] <= detail["area_ratio"] <= parameters["max_area_ratio"]
                and detail["overlap"] >= parameters["min_overlap"]):
            return self._lost(frame_index, "reconfirmation_anchor_mismatch", candidate["candidates"])

        if self.candidate_track is None:
            self.candidate_track = TargetAssociation(self.target_id, alignment="rigid")
            self.candidate_track.select(frame_index, instance_map, segments, candidate["mask_id"])
        else:
            _, continuity = self.candidate_track.step(frame_index, instance_map, segments)
            if continuity["status"] != "matched" or continuity["mask_id"] != candidate["mask_id"]:
                return self._lost(frame_index, "reconfirmation_candidate_discontinuity", candidate["candidates"])
        if self.candidate_timestamp is None or stamp > self.candidate_timestamp:
            self.confirmations += 1
            self.candidate_timestamp = stamp
        if self.confirmations < parameters["required_confirmations"]:
            return self._waiting(frame_index, "confirming", "awaiting_distinct_source_confirmation", candidate["candidates"])

        self.track = self.candidate_track
        self.pending = False
        self.candidate_track = None
        self.accepted_timestamp = stamp
        record = {**candidate, "status": "reconfirmed", "gap_frames": gap, "confirmations": self.confirmations,
                  "anchor_center_distance_px": center_distance, "source_timestamp": stamp}
        self.confirmations = 0
        self.candidate_timestamp = None
        return selected.copy(), record
