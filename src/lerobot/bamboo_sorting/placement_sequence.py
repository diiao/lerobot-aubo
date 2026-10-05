"""RGB placement differences for human review, never verified layer/grasp truth."""

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to save {path}")


def validate_roi(roi, shape):
    """ROI is x0,y0,x1,y1 in original pixels, with exclusive right/bottom bounds."""
    height, width = shape[:2]
    if roi is None:
        return 0, 0, width, height
    if len(roi) != 4 or any(type(value) is not int for value in roi):
        raise ValueError("roi must contain four integer pixel coordinates")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("roi must be inside the original image with positive width/height")
    return x0, y0, x1, y1


def difference_candidate(previous, current, *, threshold=25, min_area=80, roi=None, min_chroma_change=0.):
    """Return all sufficiently large changed components; do not invent hidden pixels."""
    if previous.dtype != np.uint8 or current.dtype != np.uint8:
        raise ValueError("frames must be uint8 RGB")
    if previous.shape != current.shape or current.ndim != 3 or current.shape[2] != 3:
        raise ValueError("frames must have identical H x W x 3 shapes")
    if not 1 <= threshold <= 255 or min_area < 1:
        raise ValueError("threshold must be 1..255 and min_area must be positive")
    if not np.isfinite(min_chroma_change) or min_chroma_change < 0:
        raise ValueError("min_chroma_change must be finite and nonnegative")
    x0, y0, x1, y1 = validate_roi(roi, current.shape)
    delta = cv2.absdiff(previous, current).max(axis=2)
    changed = delta >= threshold
    inside = np.zeros(changed.shape, dtype=bool)
    inside[y0:y1, x0:x1] = True
    changed &= inside
    before_filter = int(changed.sum())
    if min_chroma_change:
        # Lab separates brightness (L) from color (a,b). This suppresses
        # brightness-only shadows, but can also remove same-color crossings.
        old_lab = cv2.cvtColor(previous, cv2.COLOR_RGB2LAB).astype(np.float32)
        new_lab = cv2.cvtColor(current, cv2.COLOR_RGB2LAB).astype(np.float32)
        chroma = np.linalg.norm(new_lab[:, :, 1:] - old_lab[:, :, 1:], axis=2)
        changed &= chroma >= min_chroma_change
    filtered_pixels = before_filter - int(changed.sum())
    changed = changed.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed, connectivity=8)
    kept = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    mask = np.isin(labels, kept)
    warnings = []
    if not kept:
        warnings.append("no_new_object_detected")
    if len(kept) > 1:
        warnings.append("multiple_changed_components")
    if mask.sum() / int(inside.sum()) > 0.35:
        warnings.append("large_scene_change")
    if mask[y0, x0:x1].any() or mask[y1 - 1, x0:x1].any() or mask[y0:y1, x0].any() or mask[y0:y1, x1 - 1].any():
        warnings.append("candidate_touches_roi_boundary")
    if min_chroma_change:
        warnings.append("chroma_filter_may_remove_same_color_crossings")
    return mask, {"changed_pixels": int(mask.sum()), "component_count": len(kept),
                  "chroma_filtered_pixels": filtered_pixels, "warnings": warnings}


class PlacementSequence:
    """One physical placement group; retain original frames and every derived stage."""

    def __init__(self, output, *, placement_id, split, configuration, threshold=25, min_area=80,
                 roi=None, min_chroma_change=0.):
        if not placement_id.strip():
            raise ValueError("placement_id must not be empty")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.threshold, self.min_area = threshold, min_area
        self.roi, self.min_chroma_change = roi, min_chroma_change
        self.previous = None
        self.instances = []
        self.relations = []
        self.layer_order_unresolved = bool(min_chroma_change)
        self.manifest = {
            "schema": "bamboo_placement_candidates", "placement_id": placement_id, "split": split,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration,
            "parameters": {"threshold": threshold, "min_area": min_area,
                           "roi_xyxy": roi, "min_chroma_change": min_chroma_change},
            "annotation_status": "needs_review", "training_ready": False,
            "assumptions": ["frame_000_is_empty", "one_strip_added_per_step", "older_strips_do_not_move",
                            "new_strip_is_placed_from_above", "camera_and_lighting_are_fixed"],
            "frames": [], "finished": False,
        }
        self.save()

    def save(self):
        write_json(self.output / "sequence.json", self.manifest)

    def append(self, rgb, *, source=None):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("frame must be uint8 RGB")
        x0, y0, x1, y1 = validate_roi(self.roi, rgb.shape)
        index = len(self.manifest["frames"])
        prefix = f"frame_{index:03d}"
        record = {"index": index, "image": prefix + ".png", "source": source,
                  "saved_at_utc": datetime.now(timezone.utc).isoformat()}
        overlay = rgb.copy()
        if self.previous is None:
            record["role"] = "empty_baseline"
        else:
            mask, quality = difference_candidate(self.previous, rgb, threshold=self.threshold, min_area=self.min_area,
                                                 roi=self.roi, min_chroma_change=self.min_chroma_change)
            self.layer_order_unresolved |= bool(quality["warnings"])
            record.update(quality)
            if self.min_chroma_change:
                unfiltered, _ = difference_candidate(self.previous, rgb, threshold=self.threshold,
                                                     min_area=self.min_area, roi=self.roi)
                record["unfiltered_mask"] = prefix + "_unfiltered.png"
                write_image(self.output / record["unfiltered_mask"], unfiltered.astype(np.uint8) * 255)
            record["candidate_mask"] = prefix + "_candidate.png"
            write_image(self.output / record["candidate_mask"], mask.astype(np.uint8) * 255)
            # Placement order is only a hypothesis: RGB shadows or movement can
            # corrupt both this footprint and every relation derived from it.
            if mask.any():
                identity = f"strip_{index:03d}"
                for previous in self.instances:
                    overlap = int(np.count_nonzero(mask & previous["footprint"]))
                    if overlap:
                        self.relations.append({"above": identity, "below": previous["id"],
                                               "overlap_pixels": overlap})
                    previous["visible"] &= ~mask
                self.instances.append({"id": identity, "footprint": mask.copy(), "visible": mask.copy(),
                                       "source_mask": record["candidate_mask"]})
            covered = {relation["below"] for relation in self.relations}
            instances = []
            for instance in self.instances:
                visible_name = f"{prefix}_{instance['id']}_visible.png"
                write_image(self.output / visible_name, instance["visible"].astype(np.uint8) * 255)
                instances.append({"id": instance["id"], "footprint_candidate": instance["source_mask"],
                                  "visible_mask_candidate": visible_name, "graspable": None})
            labels = {"annotation_status": "needs_review", "training_ready": False,
                      "placement_id": self.manifest["placement_id"], "frame": record["image"],
                      "instances": instances, "relations_candidate": list(self.relations),
                      "top_ids_candidate": None if self.layer_order_unresolved else [
                          item["id"] for item in self.instances if item["id"] not in covered and item["visible"].any()],
                      "layer_order_status": "unresolved" if self.layer_order_unresolved else "candidate_only",
                      "warnings": quality["warnings"],
                      "note": "Historical footprints are not verified amodal masks. Review shadows, motion and contact; graspability is unknown."}
            record["labels"] = prefix + "_labels.json"
            write_json(self.output / record["labels"], labels)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
        if self.roi is not None:
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 180, 255), 1)
        write_image(self.output / record["image"], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        record["review_image"] = prefix + "_review.png"
        write_image(self.output / record["review_image"], cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        self.manifest["frames"].append(record)
        self.previous = rgb.copy()
        self.save()
        return record

    def finish(self, reason):
        self.manifest["finished"] = reason == "completed"
        self.manifest["stop_reason"] = reason
        self.save()
