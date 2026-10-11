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
