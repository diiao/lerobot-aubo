"""Offline wrist-angle sample preparation, measurement and regression.

Extracted from the command entry so other diagnostics can reuse the pure
business functions. No hardware, model or raw-data writes.
"""

import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def default_roi(width, height):
    return [[round(width*x), round(height*y)] for x,y in ((.25,.15),(.75,.15),(.75,.9),(.25,.9))]


def interpolate_target(times, actions, timestamp, max_gap):
    """No extrapolation or interpolation across a recording gap/close boundary."""
    right = int(np.searchsorted(times, timestamp))
    if right < len(times) and abs(times[right] - timestamp) < 1e-7:
        left = right
    else:
        left = right - 1
    if left < 0 or right >= len(times):
        return None
    if times[right] - times[left] > max_gap or np.any(actions[left:right+1, 6] >= 50):
        return None
    weight = 0 if left == right else (timestamp-times[left])/(times[right]-times[left])
    value = actions[left, 5] * (1-weight) + actions[right, 5] * weight
    return float(value), [left, right]


def candidate_window(config, source, close, episode=None):
    bounds = source.get('episode_windows', {}).get(str(episode),
              source.get('before_close_s', config.get('before_close_s', [3, 0])))
    if len(bounds) != 2 or not np.isfinite(bounds).all() or not bounds[0] > bounds[1] >= 0:
        raise ValueError('before_close_s must be [earlier, later], earlier > later >= 0')
    return close-bounds[0], close-bounds[1]


def demonstration_candidates(source, config, skipped):
    import pyarrow.parquet as pq

    root, evidence = Path(source['root']), Path(source['evidence_root'])
    metadata = {r['episode_index']: r for p in sorted((root/'meta/episodes').rglob('*.parquet'))
                for r in pq.read_table(p).to_pylist()}
    captures = [c for c in read_json(evidence/'session.json')['attempts']
                if c['status'] == 'saved_pending_finalize']
    for capture in captures:
        ep = capture['episode_index']
        if 'episode_indices' in source and ep not in source['episode_indices']:
            continue
        episode_id = f'{root}/episode-{ep}'
        meta = metadata[ep]
        path = root/f'data/chunk-{meta["data/chunk_index"]:03d}/file-{meta["data/file_index"]:03d}.parquet'
        data = {r['frame_index']: r for r in pq.read_table(path).to_pylist() if r['episode_index'] == ep}
        frames = capture['joint_frames']
        clocks = capture['sensor_and_suction_evidence']['timestamp_journal']['frames']
        times = np.array([f['joint_command_trace']['accepted_monotonic_s'] for f in frames])
        actions = np.array([f['action'] for f in frames])
        if (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                or any(f['joint_command_trace']['return_code'] != 0 for f in frames)):
            raise ValueError(f'invalid accepted command timeline: {episode_id}')
        closed = np.flatnonzero(actions[:, 6] >= 50)
        measurement_only = config.get('measurement_only', False)
        if not len(closed) and not measurement_only:
            skipped.append(dict(episode_id=episode_id, reason='no_accepted_close'))
            continue
        close = float(times[closed[0]]) if len(closed) else None
        first_wrist = clocks[0]['timestamps']['grasp_rgb']['host_receive_monotonic_s']
        last_wrist = clocks[-1]['timestamps']['grasp_rgb']['host_receive_monotonic_s']
        if config.get('full_preclose', False):
            start, end = first_wrist, close if close is not None else last_wrist+1e-6
        else:
            start, end = candidate_window(config, source, close if close is not None else last_wrist, ep)
        previous = -np.inf
        video_key = 'videos/observation.images.grasp_rgb'
        video = root/f'{video_key}/chunk-{meta[video_key+"/chunk_index"]:03d}/file-{meta[video_key+"/file_index"]:03d}.mp4'
        for frame, clock in zip(frames, clocks, strict=True):
            fi = frame['frame_index']
            row = data[fi]
            if clock['frame_index'] != fi or not np.allclose(row['action'], frame['action'], atol=1e-4, rtol=0):
                raise ValueError(f'frame/action mismatch: {episode_id}/{fi}')
            sensor = [clock['timestamps'][k]['host_receive_monotonic_s']
                      for k in ('global_rgb', 'grasp_rgb', 'robot_state')]
            t0, wrist = min(sensor), sensor[1]
            if not start <= wrist < end or wrist-previous < config['sample_interval_s']-1e-7:
                continue
            previous = wrist
            base = dict(group='expert', episode_id=episode_id, frame_index=fi,
                        **source.get('episode_context', {}).get(str(ep), {}),
                        video=str(video), video_time_s=meta[video_key+'/from_timestamp']+row['timestamp'],
                        wrist_source_monotonic_s=wrist, observation_anchor_monotonic_s=t0,
                        sensor_times=sensor, alignment_error_s=0., accepted_close_monotonic_s=close,
                        sequence_start_monotonic_s=first_wrist,
                        observed_state=row['observation.state'], source_session=str(evidence/'session.json'))
            if base['video_time_s'] >= meta[video_key+'/to_timestamp']:
                raise ValueError(f'video timestamp outside episode: {episode_id}/{fi}')
            if measurement_only:
                yield dict(base, measurement_only=True, response_available=False,
                           stage='preclose_unreviewed' if close is not None else 'no_close_unreviewed')
                continue
            if max(sensor)-min(sensor) > config['max_alignment_error_s']:
                skipped.append(dict(episode_id=episode_id, frame_index=fi, reason='sensor_skew'))
                continue
            for horizon in config['horizons_s']:
                target = interpolate_target(times, actions, t0+horizon, config['max_target_gap_s'])
                if t0+horizon >= end or target is None:
                    skipped.append(dict(episode_id=episode_id, frame_index=fi, horizon_s=horizon,
                                        reason='future_target_gap_or_window_end'))
                    continue
                value, bracket = target
                yield dict(base, horizon_s=horizon, future_target_j6_deg=value,
                           delta_j6_deg=value-float(row['observation.state'][5]),
                           target_bracket_frame_indices=[frames[i]['frame_index'] for i in bracket],
                           target_bracket_monotonic_s=[float(times[i]) for i in bracket])


def model_image_candidates(source, config, skipped):
    """Measure saved camera observations without requiring predictions or closure."""
    root = Path(source['root'])
    trace = read_jsonl(root/'trace.jsonl')
    closes = [r['target_phase_command']['first_accepted_close_s'] for r in trace
              if r.get('target_phase_command', {}).get('first_accepted_close_s') is not None]
    close = min(closes) if closes else None
    side = read_jsonl(root/'videos/grasp_rgb_frames.jsonl')
    if not side:
        skipped.append(dict(episode_id=str(root), reason='empty_video_sidecar'))
        return
    first, last = side[0]['source_timestamp'], side[-1]['source_timestamp']
    if config.get('full_preclose', False):
        start, end = first, close if close is not None else last+1e-6
    else:
        start, end = candidate_window(config, source, close if close is not None else last)
    previous, seen = -np.inf, set()
    plan = read_json(root/'plan.json') if (root/'plan.json').exists() else {}
    complete = read_json(root/'complete.json') if (root/'complete.json').exists() else {}
    configuration = {k: plan.get(k) for k in ('model', 'target_selection_mode', 'action_selection',
                       'transport_age_aligned', 'camera_set_sha256', 'checkpoint_sha256')}
    for frame in side:
        t = frame['source_timestamp']
        if not start <= t < end:
            continue
        if frame.get('filled_missing_tick') or frame.get('repeated_source') or t in seen:
            skipped.append(dict(episode_id=str(root), frame_index=frame['frame_index'], reason='repeated_or_filled_video_frame'))
            continue
        seen.add(t)
        if t-previous < config['sample_interval_s']-1e-7:
            continue
        previous = t
        yield dict(group='model', episode_id=str(root), target_id=plan.get('target_id'),
                   trial_outcome=source.get('trial_outcome', 'unknown'), trial_end_reason=complete.get('reason'),
                   configuration=configuration, frame_index=frame['frame_index'],
                   video=str(root/'videos/grasp_rgb.mp4'), video_time_s=frame['video_time_s'],
                   wrist_source_monotonic_s=t, accepted_close_monotonic_s=close,
                   sequence_start_monotonic_s=first, alignment_error_s=0.,
                   measurement_only=True, response_available=False,
                   stage='preclose_unreviewed' if close is not None else 'no_close_unreviewed',
                   source_trace=str(root/'trace.jsonl'))


def model_candidates(source, config, skipped):
    if config.get('measurement_only', False):
        yield from model_image_candidates(source, config, skipped)
        return
    root = Path(source['root'])
    trace = read_jsonl(root/'trace.jsonl')
    fps = source.get('action_fps')
    if not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError('model source requires action_fps from the training contract, not video/control fps')
    closes = [r['target_phase_command']['first_accepted_close_s'] for r in trace
              if r.get('target_phase_command', {}).get('first_accepted_close_s') is not None]
    if not closes:
        skipped.append(dict(episode_id=str(root), reason='no_accepted_close'))
        return
    close = min(closes)
    start, end = candidate_window(config, source, close)
    side = read_jsonl(root/'videos/grasp_rgb_frames.jsonl')
    # Filled ticks and repeated frames are not new camera observations.
    side = [r for r in side if not r.get('filled_missing_tick') and not r.get('repeated_source')]
    previous, used = -np.inf, set()
    for row in trace:
        if 'prediction' not in row or row['prediction'].get('target_input', {}).get('phase') != 'approach':
            continue
        sensor = row['sensor_times']
        t0, wrist = min(sensor), sensor[1]
        if not start <= wrist < end or wrist-previous < config['sample_interval_s']-1e-7:
            continue
        previous = wrist
        candidate = min(side, key=lambda s: abs(s['source_timestamp']-wrist)) if side else None
        reason = None
        if candidate is None or abs(candidate['source_timestamp']-wrist) > config['max_alignment_error_s']:
            reason = 'missing_aligned_video_frame'
        elif max(sensor)-min(sensor) > config['max_alignment_error_s']:
            reason = 'sensor_skew'
        elif not start <= candidate['source_timestamp'] < end:
            reason = 'aligned_frame_outside_window'
        elif candidate['frame_index'] in used:
            reason = 'duplicate_video_observation'
        if reason:
            skipped.append(dict(episode_id=str(root), prediction_index=row['index'], reason=reason))
            continue
        used.add(candidate['frame_index'])
        chunk = np.asarray(row['prediction']['action_chunk'], float)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
            raise ValueError('finite N x 7 action chunk required')
        base = dict(group='model', episode_id=str(root), prediction_index=row['index'],
                    trial_outcome=source.get('trial_outcome', 'unknown'),
                    frame_index=candidate['frame_index'], video=str(root/'videos/grasp_rgb.mp4'),
                    video_time_s=candidate['video_time_s'], wrist_source_monotonic_s=candidate['source_timestamp'],
                    requested_wrist_monotonic_s=wrist, observation_anchor_monotonic_s=t0, sensor_times=sensor,
                    alignment_error_s=candidate['source_timestamp']-wrist,
                    accepted_close_monotonic_s=close, observed_state=row['observed_state'],
                    prediction_received_monotonic_s=row['received_at'],
                    prediction_latency_s=row['received_at']-t0, action_fps=fps,
                    source_trace=str(root/'trace.jsonl'))
        for horizon in config['horizons_s']:
            target = interpolate_target(np.arange(len(chunk))/fps, chunk, horizon, 1.01/fps)
            # A future row after any planned closure is already transport, even
            # if a later row happens to predict open again.
            crosses_close = np.any(chunk[:min(len(chunk), int(np.ceil(horizon*fps))+1), 6] >= 50)
            if t0+horizon >= end or target is None or crosses_close:
                skipped.append(dict(episode_id=str(root), prediction_index=row['index'], horizon_s=horizon,
                                    reason='future_target_outside_chunk_or_window_or_close'))
                continue
            value, bracket = target
            yield dict(base, horizon_s=horizon, future_target_j6_deg=value,
                       delta_j6_deg=value-float(row['observed_state'][5]), target_chunk_indices=bracket)


def extract_sample_images(samples, output):
    """Decode by video PTS, including offsets in concatenated episode videos."""
    import av

    videos = {}
    for row in samples:
        videos.setdefault(row['video'], {}).setdefault(row['video_time_s'], []).append(row)
    for video, by_time in videos.items():
        wanted = sorted(by_time)
        cursor = 0
        with av.open(video) as container:
            stream = container.streams.video[0]
            container.seek(int(wanted[0]/stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                actual = float(frame.pts*stream.time_base)
                if actual < wanted[cursor]-1e-4:
                    continue
                if abs(actual-wanted[cursor]) > 1e-4:
                    raise ValueError(f'video PTS mismatch: {video}: {actual} != {wanted[cursor]}')
                rows = by_time[wanted[cursor]]
                image = frame.to_image()
                name = f'images/{rows[0]["id"]}.png'
                image.save(output/name)
                for row in rows:
                    row.update(image=name, decoded_video_time_s=actual, image_size=list(image.size))
                cursor += 1
                if cursor == len(wanted):
                    break
        if cursor != len(wanted):
            raise ValueError(f'video ended before requested samples: {video}')


def stratify_expert_sources(sources, strata):
    """Select from first accepted global masks, before any wrist validity filter.

    The angle matches the historical coverage audit: atan2(axis_x, axis_y),
    directed toward positive image y. It is not the measured wrist angle.
    """
    pool, selected = [], []
    for si, (source, group) in enumerate(sources):
        if group != 'expert':
            continue
        annotations = Path(source['target_annotations'])
        for episode in read_json(annotations)['episodes']:
            ep = episode['episode_index']
            if 'episode_indices' in source and ep not in source['episode_indices']:
                continue
            first = episode['frames'][0]
            if episode.get('reviewed') is not True or first.get('reviewed') is not True:
                raise ValueError('expert stratification requires reviewed initial target masks')
            mask_path = (annotations.parent/first['mask_path']).resolve()
            mask = np.asarray(Image.open(mask_path).convert('L')) > 0
            y, x = np.nonzero(mask)
            if len(x) < 3:
                raise ValueError(f'nonempty initial target mask required: {mask_path}')
            points = np.column_stack((x, y)).astype(float)
            _, axes = np.linalg.eigh(np.cov(points.T))
            axis = axes[:, -1]
            if axis[1] < 0:
                axis = -axis
            pool.append(dict(source_index=si, episode_index=ep, scene_id=episode['scene_id'],
                             target_id=episode['target_id'], initial_mask=str(mask_path),
                             initial_angle_deg=float(np.degrees(np.arctan2(axis[0], axis[1])))))
    for stratum in strata:
        lo, hi = stratum['angle_range_deg']
        candidates = sorted([r for r in pool if lo <= r['initial_angle_deg'] < hi],
                            key=lambda r: (r['initial_angle_deg'], r['source_index'], r['episode_index']))
        if 'expected_pool_count' in stratum and len(candidates) != stratum['expected_pool_count']:
            raise ValueError(f'angle stratum coverage changed: {stratum["name"]}: {len(candidates)}')
        count = stratum.get('take')
        if count is not None and (type(count) is not int or count < 1):
            raise ValueError('stratum take must be positive or null (all)')
        indices = range(len(candidates)) if count is None else np.linspace(
            0, len(candidates)-1, min(count, len(candidates)), dtype=int)
        selected.extend(dict(candidates[i], angle_stratum=stratum['name']) for i in indices)
    identities = [(r['source_index'], r['episode_index']) for r in selected]
    if len(set(identities)) != len(identities):
        raise ValueError('overlapping expert angle strata')
    result = []
    for si, (source, group) in enumerate(sources):
        if group != 'expert':
            result.append((source, group))
            continue
        chosen = [r for r in selected if r['source_index'] == si]
        if chosen:
            result.append((dict(source, episode_indices=[r['episode_index'] for r in chosen],
                                episode_context={str(r['episode_index']): {k:v for k,v in r.items()
                                    if k not in ('source_index', 'episode_index')} for r in chosen}), group))
    return result, dict(pool_episodes=len(pool), selected_episodes=selected, strata=strata,
                        selection='all required stratum; evenly spaced angle ranks within other strata; independent of wrist validity')


def prepare_manifest(sources_path, output):
    config = {'horizons_s': [.5, 1.], 'sample_interval_s': .2,
              'max_alignment_error_s': .04, 'max_target_gap_s': .12, **read_json(sources_path)}
    for key in ('sample_interval_s', 'max_alignment_error_s', 'max_target_gap_s'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    horizons = config['horizons_s']
    if not horizons or len(set(horizons)) != len(horizons) or not np.isfinite(horizons).all() or min(horizons) <= 0:
        raise ValueError('unique positive finite horizons_s required')
    base = sources_path.resolve().parent
    sources = [(s, 'expert') for s in config.get('demonstrations', [])]
    if config.get('training_sources'):
        path = (base/config['training_sources']).resolve()
        training = read_json(path)
        for split in ('train', 'validation'):
            for original in training[split]:
                source = dict(original)
                for key in ('root', 'evidence_root', 'target_annotations'):
                    source[key] = str((path.parent/source[key]).resolve())
                source['episode_indices'] = [r['episode_index'] for r in read_json(Path(source['target_annotations']))['episodes']]
                sources.append((source, 'expert'))
    sources.extend((s, 'model') for s in config.get('runs', []))
    sources = [(dict(source, **{k: str((base/source[k]).resolve()) for k in
                               ('root', 'evidence_root', 'target_annotations') if k in source}), group)
               for source, group in sources]
    sampling = None
    if config.get('expert_angle_strata'):
        sources, sampling = stratify_expert_sources(sources, config['expert_angle_strata'])
    samples, skipped, seen = [], [], set()
    for source_index, (original, group) in enumerate(sources):
        source = dict(original)
        for key in ('root', 'evidence_root'):
            if key in source:
                source[key] = str((base/source[key]).resolve())
        defaults = {**config.get('defaults', {}), **source.get('defaults', {})}
        if 'gap_line' not in defaults:
            raise ValueError('one shared gap_line per camera setup is required in defaults')
        producer = demonstration_candidates if group == 'expert' else model_candidates
        for row in producer(source, config, skipped):
            identity = (row['episode_id'], row.get('prediction_index', row['frame_index']), row.get('horizon_s'))
            if identity in seen:
                raise ValueError(f'duplicate source sample: {identity}')
            seen.add(identity)
            samples.append(dict(row, id=f's{source_index:03d}-{len(samples):06d}',
                                **defaults, descent_segment_reviewed=False))
    if not samples:
        raise ValueError(f'no aligned preclose samples; exclusions: {skipped[:5]}')
    output.mkdir(parents=True, exist_ok=False)
    (output/'images').mkdir()
    extract_sample_images(samples, output)
    for row in samples:
        if 'roi_polygon' not in row:
            row['roi_polygon'] = default_roi(*row['image_size'])
            row['roi_source'] = 'default_central_region_requires_review'
    manifest = dict(schema='joint_wrist_angle_samples', sources=str(sources_path.resolve()), samples=samples,
                    exclusions=skipped, preparation_config=config, expert_sampling=sampling, human_reviewed=False,
                    window_definition='full recording before first accepted close (or recording end without closure)' if config.get('full_preclose')
                        else 'candidate interval before first accepted close; confirm descent in review',
                    measurement_only=config.get('measurement_only', False),
                    response_definition='target J6 at min(sensor_times)+horizon minus observed state J6; degrees',
                    interpolation='linear between bracketing targets; no extrapolation, closure or long-gap bridging',
                    hardware_access=False, model_loaded=False)
    write_new(output/'manifest.json', manifest)
    return manifest


def measure_alignment(rgb, gap_line, roi_polygon, gripper_polygons=(), min_axis_span_roi_fraction=.5):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('uint8 RGB required')
    height, width = rgb.shape[:2]
    line, polygon = np.asarray(gap_line, float), np.asarray(roi_polygon, float)
    exclusions = [np.asarray(p, float) for p in gripper_polygons]
    for points, minimum in ((line, 2), (polygon, 3), *((p, 3) for p in exclusions)):
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum
                or not np.isfinite(points).all() or np.any(points < 0)
                or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height)):
            raise ValueError('reference line and ROI must lie inside the image')
    if line.shape != (2, 2) or np.linalg.norm(line[1] - line[0]) < 1:
        raise ValueError('two distinct gap line points required')
    axis = (line[1] - line[0]) / np.linalg.norm(line[1] - line[0])
    roi = np.zeros((height, width), np.uint8)
    cv2.fillPoly(roi, [np.rint(polygon).astype(np.int32)], 1)
    excluded = np.zeros_like(roi)
    for points in exclusions:
        cv2.fillPoly(excluded, [np.rint(points).astype(np.int32)], 1)
    if not 0 < min_axis_span_roi_fraction <= 1:
        raise ValueError('min_axis_span_roi_fraction must be in (0, 1]')
    min_span = float(np.ptp(polygon[:, 1])*min_axis_span_roi_fraction)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # A deliberately inspectable color proposal, not a learned segmentation.
    foreground = cv2.inRange(hsv, np.array([5, 30, 50]), np.array([40, 255, 255])) & (roi * 255) & ((1-excluded)*255)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    candidates, rejected = [], []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 80:
            continue
        y, x = np.nonzero(labels == index)
        points = np.column_stack((x, y)).astype(float)
        values, axes = np.linalg.eigh(np.cov(points.T))
        span = float(np.ptp(points @ axes[:, -1]))
        # A near-camera strip can be broad. Length and jaw exclusion carry the
        # main rejection; the old 9:1 covariance test also rejected real strips.
        if span > min_span and values[-1] >= 3 * max(values[0], 1):
            candidates.append(points)
        else:
            rejected.append(dict(axis_span_px=span, pixels=len(points)))
    overlay = rgb.copy()
    overlay[excluded > 0] = (overlay[excluded > 0]*.55 + np.array([255, 70, 30])*.45).astype(np.uint8)
    for points in exclusions:
        cv2.polylines(overlay, [np.rint(points).astype(np.int32)], True, (255, 70, 30), 2)
    cv2.polylines(overlay, [np.rint(polygon).astype(np.int32)], True, (0, 180, 255), 1)
    cv2.line(overlay, tuple(np.rint(line[0]).astype(int)), tuple(np.rint(line[1]).astype(int)), (0, 255, 0), 2)
    evidence = dict(min_axis_span_px=min_span, gripper_excluded_pixels=int(np.sum(excluded & roi)),
                    rejected_short_or_broad_components=rejected, candidate_count=len(candidates))
    if len(candidates) != 1:
        return {'valid': False, 'reason': 'missing_or_occluded' if not candidates else 'multiple_visible_components',
                **evidence}, overlay
    points = candidates[0]
    vx, vy, px, py = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
    direction = np.array([vx, vy], float)
    if direction @ axis < 0:
        direction = -direction
    angle = np.degrees(np.arctan2(axis[0]*direction[1] - axis[1]*direction[0], axis @ direction))
    center = np.array([px, py])
    normal = np.array([-axis[1], axis[0]])
    # Offset at the reference midpoint's along-gap coordinate, rather than a
    # visible fragment centroid that shifts merely because of occlusion.
    along = float((line.mean(0) - center) @ axis)
    crossing = center + direction * along / float(direction @ axis) if abs(direction @ axis) > .1 else None
    if crossing is None:
        return {'valid': False, 'reason': 'near_perpendicular_to_gap', **evidence}, overlay
    offset = float((crossing - line.mean(0)) @ normal)
    projected = (points - center) @ direction
    ends = [center + direction * x for x in (projected.min(), projected.max())]
    cv2.line(overlay, tuple(np.rint(ends[0]).astype(int)), tuple(np.rint(ends[1]).astype(int)), (255, 0, 255), 2)
    residual = np.abs((points - center) @ np.array([-direction[1], direction[0]]))
    return {'valid': True, 'angle_deg': float(angle), 'lateral_offset_px': offset,
            **evidence, 'axis_span_px': float(np.ptp(projected)), 'visible_pixels': len(points),
            'median_axis_distance_px': float(np.median(residual))}, overlay


def response_summary(rows):
    """Descriptive, equal-episode-weight slopes; no causal classification."""
    summaries = []
    groups = sorted({(r['group'], r['horizon_s']) for r in rows})
    for group, horizon in groups:
        part = [r for r in rows if (r['group'], r['horizon_s']) == (group, horizon)]
        x = np.asarray([r['measurement']['angle_deg'] for r in part])
        y = np.asarray([r['delta_j6_deg'] for r in part])
        if len(part) < 3 or np.ptp(x) < 1:
            summaries.append(dict(group=group, horizon_s=horizon, valid=False, reason='insufficient_angle_variation'))
            continue
        counts = {r['episode_id']: sum(s['episode_id'] == r['episode_id'] for s in part) for r in part}
        weights = np.array([1/counts[r['episode_id']] for r in part])
        design = np.column_stack((x, np.ones(len(x))))
        slope, intercept = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
        residual = y - (slope*x + intercept)
        summaries.append(dict(group=group, horizon_s=horizon, valid=True, samples=len(part), episodes=len(counts),
                              slope_deg_per_image_degree=float(slope), intercept_deg=float(intercept),
                              weighted_rmse_deg=float(np.sqrt(np.average(residual**2, weights=weights))),
                              image_angle_range_deg=[float(x.min()), float(x.max())]))
    return {'groups': summaries, 'causal_diagnosis': None,
            'scope': 'descriptive associations only; image geometry is not world yaw or a J6 correction'}


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def measurement_coverage(rows, warning_gap=.2):
    """Count camera frames once, not once per requested response horizon."""
    sample_rows = len(rows)
    unique = {}
    for row in rows:
        key = (row.get('episode_id', ''), row['image'])
        if key in unique and unique[key]['measurement'] != row['measurement']:
            raise ValueError('different measurements for the same source frame')
        unique[key] = row
    def counts(part):
        valid = sum(r['measurement']['valid'] for r in part)
        reasons = {}
        for r in part:
            if not r['measurement']['valid']:
                reason = r['measurement']['reason']
                reasons[reason] = reasons.get(reason, 0)+1
        return dict(frames=len(part), program_valid_frames=valid,
                    program_valid_rate=valid/len(part) if part else None, invalid_reasons=reasons)
    rows = list(unique.values())
    by_episode = [dict(episode_id=ep, group=group, trial_outcome=outcome, **counts([
        r for r in rows if (r.get('episode_id', ''), r.get('group', ''), r.get('trial_outcome', 'unknown')) == (ep, group, outcome)]))
        for ep, group, outcome in sorted({(r.get('episode_id', ''), r.get('group', ''),
                                          r.get('trial_outcome', 'unknown')) for r in rows})]
    outcomes = {outcome: counts([r for r in rows if r.get('group') == 'model' and
                                r.get('trial_outcome', 'unknown') == outcome])
                for outcome in ('success', 'failure', 'unknown')}
    a, b = outcomes['success']['program_valid_rate'], outcomes['failure']['program_valid_rate']
    gap = abs(a-b) if a is not None and b is not None else None
    strata = {name: counts([r for r in rows if r.get('angle_stratum') == name])
              for name in sorted({r['angle_stratum'] for r in rows if 'angle_stratum' in r})}
    return dict(**counts(rows), sample_rows=sample_rows, by_episode=by_episode,
                by_model_outcome=outcomes, by_expert_angle_stratum=strata,
                outcome_rate_gap=gap, warning_gap_threshold=warning_gap,
                selection_bias_warning=gap is not None and gap > warning_gap,
                scope='program eligibility among extracted unique frames, not correctness or physical success; '
                      'a small rate gap does not rule out bias; selection and timing exclusions precede this denominator')


def measure_manifest(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'joint_wrist_angle_samples' or not manifest.get('samples'):
        raise ValueError('explicit wrist image/ROI/reference manifest required')
    samples = manifest['samples']
    if len({r['id'] for r in samples}) != len(samples):
        raise ValueError('unique sample IDs required')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, row in enumerate(samples):
        row = {**manifest.get('defaults', {}), **row}
        image_path = (manifest_path.parent / row['image']).resolve()
        rgb = np.array(Image.open(image_path).convert('RGB'))
        row.setdefault('roi_polygon', default_roi(rgb.shape[1], rgb.shape[0]))
        measurement, overlay = measure_alignment(rgb, row['gap_line'], row['roi_polygon'],
            row.get('gripper_polygons', ()), row.get('min_axis_span_roi_fraction', .5))
        name = f'sample-{index:04d}.png'
        Image.fromarray(overlay).save(output / name)
        rows.append({**row, 'image': str(image_path), 'measurement': measurement, 'preview': name})
    report = {'schema': 'joint_wrist_angle_measurements', 'manifest': str(manifest_path.resolve()),
              'samples': rows, 'human_reviewed': False, 'hardware_access': False, 'model_loaded': False,
              'color_proposal_hsv_bounds': [[5, 30, 50], [40, 255, 255]],
              'coordinate_convention': 'signed image angle to directed gap line; signed pixel offset to its left normal',
              'causal_diagnosis': None, 'measurement_method': 'fixed_gripper_exclusion_and_axis_span',
              'coverage': measurement_coverage(rows)}
    write_new(output/'measurements.json', report)
    write_measurement_review(report, output)
    return report


def write_measurement_review(report, output):
    """Render the saved report; never change measurements or review decisions."""
    cards = []
    for index, row in enumerate(report['samples']):
        context = {k: row[k] for k in ('group', 'episode_id', 'frame_index', 'prediction_index',
                    'horizon_s', 'delta_j6_deg', 'alignment_error_s', 'angle_stratum',
                    'initial_angle_deg', 'trial_outcome') if k in row}
        if row.get('accepted_close_monotonic_s') is not None:
            context['seconds_before_close'] = row['accepted_close_monotonic_s']-row['wrist_source_monotonic_s']
        cards.append(f'<article><h2>{html.escape(str(row["id"]))}</h2><img src="{row["preview"]}" width="640">'
                     f'<p>{html.escape(json.dumps(context, ensure_ascii=False))}</p>'
                     f'<p>{html.escape(json.dumps(row["measurement"], ensure_ascii=False))}</p>'
                     f'<label><input type="checkbox" data-index="{index}" {"" if row["measurement"]["valid"] else "disabled"}>'
                     '我已确认这是下探帧，遮罩未混入夹指，拟合线落在目标竹条上，参考线与测量可接受</label></article>')
    payload = json.dumps(report, ensure_ascii=False).replace('<', '\\u003c')
    page = ('<!doctype html><meta charset="utf-8"><title>腕部角度测量复核</title>'
            '<style>body{font:16px sans-serif;max-width:900px;margin:24px auto}article{margin:24px 0}img{max-width:100%}</style>'
            '<h1>腕部角度测量草稿</h1><p>绿色：夹指缝参考；紫色：拟合竹条轴；蓝色：测量区域；橙红：固定夹指排除区域。'
            '先确认整批共享参考线和测量区域；默认区域不保证适合本相机。'
            '闭合前窗口只是下探候选，须结合原录像核对。图像角度不是实物偏航，不能直接变成 J6 补偿。'
            '程序有效不等于测对；无效帧保留，不进入回归。</p>'
            + ('<h2>有效率与选择偏差</h2><p>' + ('警告：成功／失败轮有效率差距超过20个百分点，存在选择偏差风险。'
                if report.get('coverage', {}).get('selection_bias_warning') else
                '即使有效率相近，也不能证明没有选择偏差；每个画面只计一次，不按时间窗重复计数。') + '</p><pre style="white-space:pre-wrap">'
                + html.escape(json.dumps(report.get('coverage', {}), ensure_ascii=False, indent=2)) + '</pre>')
            + ''.join(cards) + '<button id="export">导出已实际复核的测量</button><script>const report=' + payload + ';'
            'document.getElementById("export").onclick=()=>{const samples=Array.from(document.querySelectorAll("input:checked"))'
            '.map(e=>report.samples[Number(e.dataset.index)]).map(r=>({id:r.id,measurement:r.measurement,descent_segment_reviewed:true}));'
            'const data={schema:"joint_wrist_angle_review",manifest:report.manifest,samples,human_reviewed:true,reviewed_at:new Date().toISOString()};'
            'const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:"application/json"}));'
            'a.download="wrist_angle_review.json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};</script>')
    (output/'review.html').write_text(page)


def regress_reviewed(folder, decisions_path):
    report = json.loads((folder/'measurements.json').read_text())
    if report.get('measurement_method') != 'fixed_gripper_exclusion_and_axis_span':
        raise ValueError('legacy color-only measurements must be regenerated and reviewed before regression')
    review = json.loads(decisions_path.read_text())
    if (review.get('schema') != 'joint_wrist_angle_review' or review.get('manifest') != report['manifest']
            or review.get('human_reviewed') is not True or not review.get('reviewed_at')):
        raise ValueError('explicit measurement review required before regression')
    by_id = {r['id']: r for r in report['samples']}
    selected = []
    for item in review['samples']:
        row = by_id[item['id']]
        if item['measurement'] != row['measurement'] or not row['measurement']['valid']:
            raise ValueError('review does not match a valid saved measurement')
        if row.get('descent_segment_reviewed') is False and item.get('descent_segment_reviewed') is not True:
            raise ValueError('candidate preclose frames require explicit descent segment review')
        if (not row.get('episode_id') or row.get('group') not in ('expert', 'model')
                or not isinstance(row.get('horizon_s'), (int, float)) or row['horizon_s'] <= 0
                or not np.isfinite([row['horizon_s'], row.get('delta_j6_deg', np.nan)]).all()):
            raise ValueError('time-aligned episode, group, horizon and J6 delta required')
        selected.append(row)
    if not selected or len({r['id'] for r in selected}) != len(selected):
        raise ValueError('nonempty unique reviewed samples required')
    result = response_summary(selected)
    result.update(reviewed_sample_ids=[r['id'] for r in selected],
                  excluded_sample_ids=[r['id'] for r in report['samples'] if r not in selected])
    write_new(folder/'response_summary.json', result)
    return result
