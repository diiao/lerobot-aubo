"""Experimental two-strip foreground separation, with no hardware access.

Frame-local instance IDs are discarded. Each CURRENT foreground pixel is assigned
to the nearest of two independently maintained strip centerlines. Huber line fits
update only the reference geometry; no reference pixels become output pixels.
This geometric hypothesis is not identity truth. The live caller must explicitly
select the paired-foreground trial; geometry output never authorizes execution.
"""
import cv2
import numpy as np

from .joint_target_association import TargetAssociation


class PairedStripSeparation:
    """Seeded narrow-strip geometry for offline review and explicit trials.

Suitable for separated elongated strips and small inter-frame motion. Crossing,
large displacement, or foreground errors can invalidate pixel ownership. Empty
ownership keeps the geometry reference but returns an empty current mask.
"""

    def __init__(self, seeds):
        if len(seeds) != 2 or any(not isinstance(k, str) or not k for k in seeds):
            raise ValueError('two named target seeds required')
        self.targets = tuple(seeds)
        self.references = {}
        occupied = np.zeros((480, 640), bool)
        for target, mask in seeds.items():
            if mask.shape != (480, 640) or mask.dtype != np.bool_ or not mask.any():
                raise ValueError('nonempty bool 480x640 seed required')
            if np.any(occupied & mask):
                raise ValueError('seed regions must not overlap')
            occupied |= mask
            center, axis = TargetAssociation._geometry(mask)
            y, x = np.nonzero(mask)
            centered = np.column_stack((x, y)) - center
            along = centered @ axis
            normal = np.array([-axis[1], axis[0]])
            self.references[target] = {
                'center': center, 'axis': axis,
                'half_length': max(1., float(np.max(np.abs(along)))),
                'half_width': max(1., float(np.quantile(np.abs(centered @ normal), .95))),
            }
        self.last_frame = -1

    def step(self, frame_index, instance_map, segments):
        if type(frame_index) is not int or frame_index != self.last_frame + 1:
            raise ValueError('consecutive offline frame index required')
        candidates = TargetAssociation._candidates(instance_map, segments)
        foreground = np.logical_or.reduce([m for _, m in candidates]) if candidates else np.zeros((480, 640), bool)
        y, x = np.nonzero(foreground)
        points = np.column_stack((x, y)).astype(float)
        distances = []
        for target in self.targets:
            ref = self.references[target]
            delta = points - ref['center']
            normal = np.array([-ref['axis'][1], ref['axis'][0]])
            perpendicular = delta @ normal
            beyond_endpoint = np.maximum(0., np.abs(delta @ ref['axis']) - ref['half_length'])
            distances.append((perpendicular**2 + beyond_endpoint**2) / ref['half_width']**2)
        owners = np.argmin(np.array(distances), axis=0)
        masks, evidence = {}, {}
        for index, target in enumerate(self.targets):
            mask = np.zeros((480, 640), bool)
            selected = owners == index
            mask[y[selected], x[selected]] = True
            ref = self.references[target]
            before = {'center': ref['center'].tolist(), 'axis': ref['axis'].tolist()}
            # A point or pair cannot reliably update an elongated object's axis.
            # They are still returned as current foreground, without invented fill.
            if np.count_nonzero(selected) >= 3:
                vx, vy, px, py = cv2.fitLine(points[selected].astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
                axis = np.array([vx, vy], dtype=float)
                if axis @ ref['axis'] < 0:
                    axis = -axis
                normal = np.array([-axis[1], axis[0]])
                # Do not shift the axial reference center toward the visible end
                # just because the other end has become occluded.
                ref['center'] = ref['center'] + normal * ((np.array([px, py]) - ref['center']) @ normal)
                ref['axis'] = axis
            masks[target] = mask
            evidence[target] = {'current_pixels': int(mask.sum()), 'reference_before': before,
                                'reference_after': {'center': ref['center'].tolist(), 'axis': ref['axis'].tolist()},
                                'status': 'current_foreground' if mask.any() else 'no_assigned_foreground'}
        self.last_frame = frame_index
        return masks, {'frame': frame_index, 'method': 'paired_current_foreground_centerlines',
                       'execution_eligible': False, 'human_reviewed': False, 'targets': evidence}


def visible_seed_masks(targets, instance_map=None):
    """Decode explicit visible polygons or selected CURRENT instance IDs."""
    masks = {}
    occupied = np.zeros((480, 640), bool)
    for target in targets:
        name = target.get('target_id')
        if not isinstance(name, str) or not name.strip() or name in masks:
            raise ValueError('distinct nonempty target identities required')
        polygons = target.get('polygons', [])
        ids = target.get('instance_ids', [])
        if bool(polygons) == bool(ids):
            raise ValueError('choose visible polygons OR current instance IDs for each seed')
        mask = np.zeros((480, 640), np.uint8)
        if ids:
            if instance_map is None or any(type(i) is not int or i <= 0 or not np.any(instance_map == i) for i in ids):
                raise ValueError('seed instance ID absent from current prediction')
            mask[np.isin(instance_map, ids)] = 1
        for polygon in polygons:
            points = np.asarray(polygon, dtype=float)
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                    or not np.isfinite(points).all() or np.any(points < 0)
                    or np.any(points[:, 0] > 639) or np.any(points[:, 1] > 479)):
                raise ValueError('visible polygon vertices must lie inside captured RGB')
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        mask = mask.astype(bool)
        if not mask.any() or np.any(mask & occupied):
            raise ValueError('visible seed masks must be nonempty and must not overlap')
        occupied |= mask
        masks[name] = mask
    if not masks:
        raise ValueError('at least one visible target seed required')
    return masks


class MultiStripSeparation:
    """Instance-first draft association for one pick, including crossing strips.

    Current instances supported by one visible seed retain their full shape.
    Merged instances use distances to the last visible regions, with centerline
    distance only as a secondary cue. Ties stay unassigned. This does not infer
    depth order: every split is explicitly flagged for human review. References
    guide ownership; only current candidate pixels can enter output masks.
    """

    def __init__(self, seeds, *, motion_radius_px=8):
        if not seeds or any(not isinstance(t, str) or not t.strip() for t in seeds):
            raise ValueError('named visible seeds required')
        occupied = np.zeros((480, 640), bool)
        for mask in seeds.values():
            if mask.shape != occupied.shape or mask.dtype != np.bool_ or not mask.any():
                raise ValueError('nonempty bool 480x640 seed required')
            if np.any(mask & occupied):
                raise ValueError('visible seed masks must not overlap')
            occupied |= mask
        self.targets = tuple(sorted(seeds))
        self.references = {t: seeds[t].copy() for t in self.targets}
        self.motion_radius_px = motion_radius_px
        self.last_frame = -1

    def step(self, frame_index, instance_map, segments):
        if type(frame_index) is not int or frame_index != self.last_frame + 1:
            raise ValueError('consecutive offline frame index required')
        candidates = TargetAssociation._candidates(instance_map, segments)
        masks = {t: np.zeros((480, 640), bool) for t in self.targets}
        distances = {t: cv2.distanceTransform((~self.references[t]).astype(np.uint8), cv2.DIST_L2, 5)
                     for t in self.targets}
        assignments, merged, unassigned = [], [], 0
        for segment, current in candidates:
            supports = [t for t in self.targets if np.any(current & self.references[t])]
            if len(supports) > 1:
                overlap = {t: int(np.count_nonzero(current & self.references[t])) for t in supports}
                dominant = max(supports, key=overlap.get)
                # Small motion at a crossing may overlap a few pixels of the
                # other seed. Preserve a strongly supported CURRENT instance
                # rather than splitting it along yesterday's crossing edge.
                if overlap[dominant] >= .8 * sum(overlap.values()) and overlap[dominant] >= .5 * current.sum():
                    supports = [dominant]
            if not supports:
                supports = [t for t in self.targets if np.any(current & (distances[t] <= self.motion_radius_px))]
            if len(supports) == 1:
                masks[supports[0]] |= current
                assignments.append({'mask_id': segment['mask_id'], 'target_id': supports[0], 'method': 'current_instance_seed_support'})
            elif supports:
                y, x = np.nonzero(current)
                costs = []
                for target in supports:
                    center, axis = TargetAssociation._geometry(self.references[target])
                    normal = np.array([-axis[1], axis[0]])
                    line_distance = np.abs((np.column_stack((x, y)) - center) @ normal)
                    costs.append(distances[target][y, x] + .05 * line_distance)
                costs = np.asarray(costs)
                owner = np.argmin(costs, axis=0)
                ordered = np.sort(costs, axis=0)
                ambiguous = (ordered[1] - ordered[0]) < .25
                # Explicit visible seeds decide overlap pixels; geometry must
                # never override these disjoint identity anchors.
                for i, target in enumerate(supports):
                    anchored = self.references[target][y, x]
                    owner[anchored] = i
                    ambiguous[anchored] = False
                for i, target in enumerate(supports):
                    chosen = (owner == i) & ~ambiguous
                    masks[target][y[chosen], x[chosen]] = True
                unassigned += int(ambiguous.sum())
                merged.append({'mask_id': segment['mask_id'], 'targets': supports,
                               'ambiguous_pixels': int(ambiguous.sum()), 'requires_crossing_review': True})
            else:
                unassigned += int(current.sum())
                assignments.append({'mask_id': segment['mask_id'], 'target_id': None, 'method': 'no_seed_support'})
        # Do not learn a potentially incorrect ownership split as a new seed.
        split_targets = {t for item in merged for t in item['targets']}
        for target in self.targets:
            if masks[target].any() and target not in split_targets:
                self.references[target] = masks[target].copy()
        self.last_frame = frame_index
        return masks, {'frame': frame_index, 'method': 'multi_instance_visible_seed_support',
                       'execution_eligible': False, 'human_reviewed': False,
                       'motion_radius_px': self.motion_radius_px,
                       'assignments': assignments, 'merged_instances': merged,
                       'unassigned_foreground_pixels': unassigned,
                       'targets': {t: {'current_pixels': int(masks[t].sum()),
                                       'status': 'current_foreground' if masks[t].any() else 'no_assigned_foreground'}
                                   for t in self.targets}}
