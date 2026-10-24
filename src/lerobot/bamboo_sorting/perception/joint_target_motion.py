"""Opt-in RGB motion references. Original reviewed seeds remain immutable.

Track visible texture, not segmentation drafts. Reject unreliable reference
updates; live callers retain the last reference with required=False. References
are search regions, and policy outlines still require current segmentation.
Keep one previously successful source pair to recover from a poor new reference.
Carry matched feature coordinates with their image instead of redetecting on occluders.
"""
import cv2
import numpy as np

from ..mask_codec import encode_mask


class _InsufficientMotionPoints(ValueError):
    """A reference lacks enough usable point correspondences for a new fit."""


class TargetMotionReferences:
    def __init__(self, rgb, masks):
        gray = self._gray(rgb)
        self.original = {t: m.copy() for t, m in masks.items()}
        self.references = {t: (gray.copy(), m.copy()) for t, m in masks.items()}
        self.reference_frames = {t: 'initial_snapshot' for t in masks}
        self.recovery_references = {}
        self.reference_points = {t: None for t in masks}
        self.recovery_points = {}

    @staticmethod
    def _gray(rgb):
        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
            raise ValueError('motion tracking requires RGB uint8 480x640')
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _move(previous, current, mask, *, points=None):
        if points is None:
            points = cv2.goodFeaturesToTrack(previous, maxCorners=100, qualityLevel=.01,
                minDistance=4, mask=mask.astype(np.uint8)*255, blockSize=3)
        if points is None or len(points) < 6:
            raise _InsufficientMotionPoints('insufficient visible texture')
        options = dict(winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        forward, good, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
        if forward is None:
            raise ValueError('optical flow unavailable')
        backward, back_good, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **options)
        if backward is None:
            raise ValueError('reverse optical flow unavailable')
        keep = good.ravel().astype(bool) & back_good.ravel().astype(bool)
        keep &= np.linalg.norm(points-backward, axis=2).ravel() <= 1.5
        p, q = points.reshape(-1, 2)[keep], forward.reshape(-1, 2)[keep]
        if len(p) < 6:
            raise _InsufficientMotionPoints('insufficient consistent motion points')
        transform, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
            ransacReprojThreshold=1.5, maxIters=1000, confidence=.99)
        if transform is None:
            raise ValueError('rigid motion fit failed')
        chosen = inliers.ravel().astype(bool)
        if chosen.sum() < 6 or chosen.mean() < .6:
            raise ValueError(f'ambiguous motion fit: inliers={int(chosen.sum())}/{len(chosen)}, '
                             f'fraction={chosen.mean():.3f}')
        y, x = np.nonzero(mask)
        coordinates = np.column_stack([x, y]).astype(float)
        _, axes = np.linalg.eigh(np.cov(coordinates.T))
        axis = axes[:, -1]
        span = np.ptp(p[chosen] @ axis) / max(np.ptp(coordinates @ axis), 1.)
        scale = np.linalg.norm(transform[:, 0])
        angle = np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
        center = coordinates.mean(0)
        displacement = np.linalg.norm(transform[:, :2] @ center + transform[:, 2] - center)
        if span < .35 or not .97 <= scale <= 1.03 or abs(angle) > 15 or displacement > 35:
            raise ValueError('motion exceeds texture/rigid-step bounds: '
                             f'span={span:.3f}, scale={scale:.4f}, rotation_deg={angle:.3f}, '
                             f'translation_px={displacement:.3f}')
        # Preserve size; rotate about the tracked center, never grow or shrink
        # identity from the segmented fragment of an occluded strip.
        translated_center = transform[:, :2] @ center + transform[:, 2]
        transform[:, :2] /= scale
        transform[:, 2] = translated_center - transform[:, :2] @ center
        moved = cv2.warpAffine(mask.astype(np.uint8), transform, (640, 480),
            flags=cv2.INTER_NEAREST).astype(bool)
        if moved.sum() < .9 * mask.sum():
            raise ValueError('target moved outside image')
        return moved, dict(points=int(chosen.sum()),span_fraction=float(span),
            translation_px=float(displacement),rotation_deg=float(angle),scale=float(scale),
            _matched_points=q[chosen].reshape(-1, 1, 2).copy())

    def update(self, rgb, targets, *, frame_id, required=True):
        gray = self._gray(rgb)
        proposed, details, failures, sources = {}, {}, {}, {}
        next_points, source_points = {}, {}
        for target in targets:
            previous, mask = self.references[target]
            source_frame = self.reference_frames[target]
            points = self.reference_points[target]
            try:
                moved, detail = self._move(previous, gray, mask, points=points)
            except ValueError as exc:
                recovery = self.recovery_references.get(target)
                # Ambiguous fits or rigid-motion violations remain rejected.
                # Only retry an unusable set of feature correspondences.
                if recovery is None or not isinstance(exc, _InsufficientMotionPoints):
                    failures[target] = str(exc)
                    continue
                # The image and mask must stay paired. Never attach the latest
                # mask to an older image, or turn segmentation into a new seed.
                previous, mask, source_frame = recovery
                points = self.recovery_points[target]
                try:
                    moved, detail = self._move(previous, gray, mask, points=points)
                    # The backup may predate the last accepted position. Check
                    # that position too, so recovery cannot jump across it by
                    # more than the existing per-update motion bounds.
                    centers, axes = [], []
                    for region in (self.references[target][1], moved):
                        y, x = np.nonzero(region)
                        xy = np.column_stack((x, y))
                        centers.append(xy.mean(0))
                        _, vectors = np.linalg.eigh(np.cov(xy.T))
                        axes.append(vectors[:, -1])
                    step = float(np.linalg.norm(centers[1] - centers[0]))
                    rotation = float(np.degrees(np.arccos(np.clip(abs(axes[0] @ axes[1]), 0, 1))))
                    if step > 35 or rotation > 15:
                        raise ValueError('recovery exceeds last accepted step: '
                                         f'translation_px={step:.3f}, rotation_deg={rotation:.3f}')
                except ValueError as recovery_exc:
                    failures[target] = f'{exc}; recovery from {source_frame}: {recovery_exc}'
                    continue
                detail.update(recovered=True, primary_failure=str(exc))
            next_points[target] = detail.pop('_matched_points', None)
            source_points[target] = points
            detail['reference_frame_id'] = source_frame
            proposed[target] = moved
            details[target] = detail
            sources[target] = (previous, mask, source_frame)
        # Independent strip identities cannot claim the same visible region.
        for _ in targets:
            candidates = {t:self.references[t][1] if t in failures else proposed[t] for t in targets}
            conflicts = set()
            for i, target in enumerate(targets):
                for other in targets[i+1:]:
                    if np.any(candidates[target] & candidates[other]):
                        conflicts.update((target, other))
            if not conflicts:break
            for target in conflicts:failures[target] = 'overlapping motion identities'
        for target, mask in proposed.items():
            if target not in failures:
                # Retain the SOURCE that actually passed all checks, not a
                # failed latest reference. Conflicting proposals commit neither.
                self.recovery_references[target] = sources[target]
                self.recovery_points[target] = source_points[target]
                self.reference_points[target] = next_points[target]
                self.references[target] = (gray.copy(), mask)
                self.reference_frames[target] = frame_id
        if failures and required:
            raise ValueError('target motion tracking lost: ' + str(failures))
        return dict(schema='joint_target_motion_reference', frame_id=frame_id,
            human_reviewed=False, source='rgb_optical_flow',
            masks_png={t:encode_mask(self.references[t][1]) for t in targets},
            diagnostics=details, failures=failures)
