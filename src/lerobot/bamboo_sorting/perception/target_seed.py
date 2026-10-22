"""Decode explicit human-reviewed visible target seeds.

No hardware or model access. Polygons describe only visible pixels; callers
must not fill occluded regions.
"""

import cv2
import numpy as np

from ..joint_target_separation import visible_seed_masks


def reviewed_masks(snapshot, labels, *, allow_multiple=False):
    if (labels.get('schema') != 'joint_target_shadow_selection'
            or labels.get('snapshot_id') != snapshot['snapshot_id']
            or labels.get('human_reviewed') is not True):
        raise ValueError('explicit user review bound to this snapshot required')
    targets = labels.get('targets', [])
    if allow_multiple:
        return list(visible_seed_masks(targets).items())
    if len(targets) != 2 or len({t.get('target_id') for t in targets}) != 2:
        raise ValueError('two distinct targets required')
    masks = []
    for target in targets:
        if not isinstance(target.get('target_id'), str) or not target['target_id'].strip():
            raise ValueError('nonempty target identity required')
        mask = np.zeros((480, 640), np.uint8)
        polygons = target.get('polygons', [])
        if not polygons:
            raise ValueError('visible polygons required; do not fill occluded regions')
        for polygon in polygons:
            points = np.asarray(polygon, dtype=float)
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all()
                    or np.any(points < 0) or np.any(points[:, 0] > 639) or np.any(points[:, 1] > 479)):
                raise ValueError('polygon vertices must lie inside captured RGB')
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        if not mask.any():
            raise ValueError('empty visible target mask')
        masks.append(mask.astype(bool))
    if np.any(masks[0] & masks[1]):
        raise ValueError('visible target masks overlap; select only the visible region of each object')
    return [(target['target_id'], mask) for target, mask in zip(targets, masks)]
