"""Fresh segmentation plus explicit human seed; no hardware or motion commands.

Automatic masks remain reviewed=False. Association failure stops the request;
no old mask is returned. Explicit multi-target misses carry diagnostic evidence.
"""
import base64
import time

import cv2
import numpy as np

from ..mask_codec import decode_mask, encode_mask
from ..perception.joint_target_association import TargetAssociation
from ..policy.joint_target import outline_target_images
from ..policy.joint_target_policy import target_policy_contract_record


class TargetForegroundMissing(ValueError):
    """Only a same-frame multi-target empty assignment; never a model/IO error."""

    code = 'target_foreground_missing'

    def __init__(self, evidence):
        super().__init__('target association failed: selected target has no current foreground: '
                         + evidence['target_id'])
        self.evidence = evidence


class FrozenStripSegmenter:
    def __init__(self, checkpoint):
        import torch
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
        self.torch = torch
        self.processor = Mask2FormerImageProcessor.from_pretrained(checkpoint, local_files_only=True, do_resize=False)
        self.model, loading = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint, local_files_only=True, output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError(f'segmentation checkpoint mismatch: {loading}')
        if self.model.config.id2label != {0: 'top_strip', 1: 'covered_strip'}:
            raise ValueError('segmentation class mapping mismatch')
        self.model.requires_grad_(False).eval().to('cuda')
        self(np.zeros((480, 640, 3), np.uint8))

    def __call__(self, rgb):
        with self.torch.inference_mode():
            batch = self.processor(images=[rgb], do_resize=False, return_tensors='pt').to('cuda')
            result = self.processor.post_process_instance_segmentation(
                self.model(**batch), threshold=.5, mask_threshold=.5,
                overlap_mask_area_threshold=.8, target_sizes=[(480, 640)])[0]
            raw = result['segmentation'].cpu().numpy()
        instances = np.zeros((480, 640), np.uint16)
        segments = []
        for index, row in enumerate(result['segments_info'], 1):
            instances[raw == row['id']] = index
            segments.append(dict(mask_id=index, class_id=int(row['label_id']), score=float(row['score'])))
        return instances, segments


class FreshTargetInputs:
    def __init__(self, seed, segmenter):
        if (seed.get('schema') != 'joint_target_live_seed' or seed.get('human_reviewed') is not True
                or not seed.get('snapshot_id') or not seed.get('target_id')):
            raise ValueError('explicit reviewed target seed required')
        self.seed = dict(seed)
        self.target_id = seed['target_id']
        self.segmenter = segmenter
        mask = decode_mask(seed['mask_png'])
        self.excluded_mask = decode_mask(seed['excluded_mask_png'], allow_empty=seed.get('target_selection_mode') == 'multi_foreground')
        if np.any(mask & self.excluded_mask):
            raise ValueError('selected and excluded seed regions overlap')
        self.selection_mode = seed.get('target_selection_mode', 'highest_score')
        if self.selection_mode not in ('highest_score', 'paired_foreground', 'multi_foreground'):
            raise ValueError('unknown target selection mode')
        self.separator = None
        if self.selection_mode == 'paired_foreground':
            from ..perception.joint_target_separation import PairedStripSeparation
            targets = seed.get('paired_targets', [])
            if (len(targets) != 2 or len({t.get('target_id') for t in targets}) != 2
                    or self.target_id not in {t.get('target_id') for t in targets}):
                raise ValueError('two distinct paired target identities required')
            masks = {t['target_id']: decode_mask(t['mask_png']) for t in targets}
            if (not np.array_equal(masks[self.target_id], mask)
                    or not np.array_equal(np.logical_or.reduce([m for t,m in masks.items() if t!=self.target_id]), self.excluded_mask)):
                raise ValueError('paired masks must match selected and excluded reviewed seeds')
            self.separator = PairedStripSeparation({t:masks[t] for t in sorted(masks)})
        if self.selection_mode == 'multi_foreground':
            from ..perception.joint_target_separation import MultiStripSeparation
            targets = seed.get('visible_targets', [])
            masks = {t['target_id']: decode_mask(t['mask_png']) for t in targets}
            other = np.zeros((480, 640), bool)
            for target, value in masks.items():
                if target != self.target_id:
                    other |= value
            if (len(masks) != len(targets) or self.target_id not in masks
                    or not np.array_equal(masks[self.target_id], mask) or not np.array_equal(other, self.excluded_mask)):
                raise ValueError('visible seeds must match selected and excluded reviewed masks')
            self.separator = MultiStripSeparation(masks)
        # Live selection ranks every current candidate without rejection gates.
        # Scores rank candidates; they do not certify identity or mask quality.
        # Offline association retains its original rejection rules by default.
        self.tracker = TargetAssociation(self.target_id, alignment='rigid', enforce_geometry_limits=False,
                                         enforce_match_thresholds=False)
        self.tracker.select(0, mask.astype(np.uint16), [dict(mask_id=1, class_id=0)], 1)
        self.index = 0
        self.last_frame = None
        self.transport_started = False

    def prepare(self, frame, metadata, *, motion_reference=None):
        if (not isinstance(metadata, dict) or metadata.get('target_id') != self.target_id
                or not metadata.get('frame_id') or metadata['frame_id'] == self.last_frame
                or metadata.get('phase') not in ('approach', 'transport')):
            raise ValueError('fresh frame and fixed target identity required')
        transport = metadata['phase'] == 'transport'
        if self.transport_started and not transport:
            raise ValueError('transport phase cannot revert within a pick')
        self.last_frame = metadata['frame_id']
        images = {key: frame['observation.images.' + key] for key in ('global_rgb', 'grasp_rgb')}
        for value in images.values():
            if value.shape != (480, 640, 3) or value.dtype != np.uint8:
                raise ValueError('raw RGB schema mismatch')
        started = time.perf_counter()
        evidence = dict(frame_id=self.last_frame, target_id=self.target_id, phase=metadata['phase'],
                        human_reviewed=False, seed_snapshot_id=self.seed['snapshot_id'],
                        target_selection_mode=self.selection_mode)
        motion_masks = None
        if self.seed.get('motion_tracking') and not transport:
            if (self.selection_mode != 'multi_foreground' or not isinstance(motion_reference, dict)
                    or motion_reference.get('schema') != 'joint_target_motion_reference'
                    or motion_reference.get('frame_id') != metadata['frame_id']
                    or motion_reference.get('source') != 'rgb_optical_flow'
                    or motion_reference.get('human_reviewed') is not False
                    or set(motion_reference.get('masks_png', {})) != set(self.separator.targets)):
                raise ValueError('same-frame motion references required')
            motion_masks = {t: decode_mask(m) for t, m in motion_reference['masks_png'].items()}
            # References come from RGB motion, never previous segmentation drafts.
            # The human seed remains unchanged, and only CURRENT foreground can
            # become the policy outline.
            from ..perception.joint_target_separation import MultiStripSeparation
            self.separator = MultiStripSeparation(motion_masks)
            self.separator.last_frame = self.index-1
            evidence['motion_reference'] = {k:motion_reference[k] for k in
                ('frame_id', 'source', 'human_reviewed', 'diagnostics', 'failures')}
        elif motion_reference is not None:
            raise ValueError('unexpected motion reference')
        if transport:
            self.transport_started = True
            prepared = dict(target_global_rgb=images['global_rgb'].copy(), grasp_rgb=images['grasp_rgb'].copy())
            evidence.update(source='raw_rgb_after_accepted_close', mask_png=None)
        else:
            instances, segments = self.segmenter(images['global_rgb'])
            self.index += 1
            if self.separator is not None:
                masks, separation = self.separator.step(self.index-1, instances, segments)
                mask = masks[self.target_id]
                ok, encoded = cv2.imencode('.png', instances)
                if not ok:
                    raise ValueError('instance evidence encoding failed')
                evidence.update(source='same_frame_segmentation', mask_png=encode_mask(mask, allow_empty=True),
                    association=dict(frame=self.index, target_id=self.target_id,
                                     status='separated' if mask.any() else 'missing', reviewed=False),
                    association_geometry_limits_enforced=False, association_match_thresholds_enforced=False,
                    association_selection_mode=separation['method'],
                    separation=separation,
                    paired_masks_png={t:encode_mask(m) if m.any() else None for t,m in masks.items()},
                    segmentation_instances_png=base64.b64encode(encoded).decode('ascii'),
                    segmentation_segments=segments,
                    raw_foreground_other_seed_overlap_px=int(np.count_nonzero((instances>0) & self.excluded_mask)),
                    other_reviewed_target_overlap_px=int(np.count_nonzero(mask & self.excluded_mask)))
                if self.selection_mode == 'multi_foreground':
                    evidence['visible_masks_png'] = evidence.pop('paired_masks_png')
                evidence['preparation_s'] = time.perf_counter()-started
                if not mask.any():
                    if self.selection_mode == 'multi_foreground':
                        raise TargetForegroundMissing(evidence)
                    raise ValueError('target association failed: selected target has no current foreground: ' + self.target_id)
                if motion_masks is not None:
                    failures = motion_reference['failures']
                    # Failed optical flow can only fall back to the last location
                    # when this frame's segmentation still supports that region.
                    # A tiny nonempty fragment cannot certify a moved identity.
                    support = {t: float(np.count_nonzero(masks[t] & m) / m.sum())
                               for t, m in motion_masks.items()}
                    evidence['motion_reference']['current_foreground_support'] = support
                    for target in motion_masks:
                        if support[target] < (.5 if target in failures else .25):
                            raise ValueError('target motion reference lacks current foreground support: ' + target)
                prepared = outline_target_images(images, mask)
                return {'observation.state':frame['observation.state'], 'task':frame['task'],
                        'target_contract':target_policy_contract_record(), 'target_input':dict(metadata),
                        **{'observation.images.'+name:value for name,value in prepared.items()}}, evidence
            mask, association = self.tracker.step(self.index, instances, segments)
            if mask is None:
                raise ValueError('target association failed: ' + str(association))
            # Diagnostic only for supervised trials: preserve the complete fresh
            # mask and report overlap, without rejecting or hiding contamination.
            conflict = int(np.count_nonzero(mask & self.excluded_mask))
            prepared = outline_target_images(images, mask)
            evidence.update(source='same_frame_segmentation', association=association, mask_png=encode_mask(mask),
                            association_geometry_limits_enforced=self.tracker.enforce_geometry_limits,
                            association_match_thresholds_enforced=self.tracker.enforce_match_thresholds,
                            association_selection_mode='highest_score_without_rejection_thresholds',
                            association_parameters={
                                'score_weights':self.tracker.PARAMETERS['score_weights'],
                                'center_score_scale_px':self.tracker.PARAMETERS['max_center_distance_px'],
                                'axis_score_scale_deg':self.tracker.PARAMETERS['max_axis_angle_deg']},
                            other_reviewed_target_overlap_px=conflict)
        evidence['preparation_s'] = time.perf_counter() - started
        result = {'observation.state': frame['observation.state'], 'task': frame['task'],
                  'target_contract': target_policy_contract_record(), 'target_input': dict(metadata),
                  **{'observation.images.' + name: value for name, value in prepared.items()}}
        return result, evidence
