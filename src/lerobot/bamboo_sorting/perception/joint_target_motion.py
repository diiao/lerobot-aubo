"""Opt-in RGB motion references. Original reviewed seeds remain immutable.

Track visible texture, not segmentation drafts. Fail rather than guessing through
occlusion or large motion. References are only search regions: the remote policy
still requires foreground pixels from the current segmentation.
"""
import cv2
import numpy as np

from ..mask_codec import encode_mask


class TargetMotionReferences:
    def __init__(self, rgb, masks):
        gray = self._gray(rgb)
        self.original = {t: m.copy() for t, m in masks.items()}
        self.references = {t: (gray.copy(), m.copy()) for t, m in masks.items()}

    @staticmethod
    def _gray(rgb):
        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
            raise ValueError('motion tracking requires RGB uint8 480x640')
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _move(previous, current, mask):
        points = cv2.goodFeaturesToTrack(previous, maxCorners=100, qualityLevel=.01,
            minDistance=4, mask=mask.astype(np.uint8)*255, blockSize=3)
        if points is None or len(points) < 6:
            raise ValueError('insufficient visible texture')
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
            raise ValueError('insufficient consistent motion points')
        transform, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
            ransacReprojThreshold=1.5, maxIters=1000, confidence=.99)
        if transform is None:
            raise ValueError('rigid motion fit failed')
        chosen = inliers.ravel().astype(bool)
        if chosen.sum() < 6 or chosen.mean() < .6:
            raise ValueError('ambiguous motion fit')
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
            raise ValueError('motion exceeds texture/rigid-step bounds')
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
            translation_px=float(displacement),rotation_deg=float(angle),scale=float(scale))

    def update(self, rgb, targets, *, frame_id, required=True):
        gray = self._gray(rgb)
        proposed, details, failures = {}, {}, {}
        for target in targets:
            previous, mask = self.references[target]
            try:
                moved, detail = self._move(previous, gray, mask)
                proposed[target] = moved
                details[target] = detail
            except ValueError as exc:
                failures[target] = str(exc)
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
                self.references[target] = (gray.copy(), mask)
        if failures and required:
            raise ValueError('target motion tracking lost: ' + str(failures))
        return dict(schema='joint_target_motion_reference', frame_id=frame_id,
            human_reviewed=False, source='rgb_optical_flow',
            masks_png={t:encode_mask(self.references[t][1]) for t in targets},
            diagnostics=details, failures=failures)
