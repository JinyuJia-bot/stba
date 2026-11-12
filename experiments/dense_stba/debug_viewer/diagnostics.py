"""Coverage and failure accounting from saved arrays; no inference or fitting."""
import numpy as np

from .rois import resolve_roi


LEGEND = {
    'pass': {'label': 'Pass', 'color': [44, 180, 110, 255]},
    'fail': {'label': 'Fail', 'color': [226, 87, 76, 255]},
    'success': {'label': 'Valid reconstructed trajectory', 'color': [44, 180, 110, 255]},
    'roma_initial': {'label': 'Recorded RoMa score / FB / validity / bounds gate', 'color': [233, 93, 75, 255]},
    'collision': {'label': 'Recorded competing seed gate', 'color': [228, 164, 48, 255]},
    'flow_or_anchors': {'label': 'Recorded combined temporal / anchor gate (edge4)', 'color': [173, 94, 213, 255]},
    'reference_flow': {'label': 'Reference temporal observation count insufficient', 'color': [232, 128, 82, 255]},
    'multiview': {'label': 'No accepted multiview seed', 'color': [149, 113, 80, 255]},
    'initialization': {'label': 'Not selected by 3D initialization', 'color': [72, 148, 210, 255]},
    'trajectory': {'label': 'Initialized but final trajectory invalid', 'color': [56, 96, 164, 255]},
    'not_attempted': {'label': 'Recorded edge not attempted', 'color': [140, 140, 140, 255]},
    'missing': {'label': 'Required diagnostic field missing', 'color': [92, 92, 92, 255]},
}


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _state(run, camera_index):
    n = len(run.point_ids)
    if not 0 <= camera_index < len(run.cameras):
        raise ValueError('camera_index out of range')
    obs, result = run.obs, run.result
    missing = []
    def field(mapping, key, source):
        if key not in mapping:
            missing.append(source + '.' + key)
            return None
        return np.asarray(mapping[key])
    config = run.method_meta.get('config', run.meta.get('config', {}))
    cfg = config.get('association', {})
    threshold = dict(min_track_observations=int(cfg.get('min_track_observations', 8)),
                     anchor_min_support=int(cfg.get('anchor_min_support', 2)),
                     anchor_error_max=float(cfg.get('anchor_error_max', 1.5)),
                     confidence_min=float(cfg.get('confidence_min', .15)), roma_fb_max=float(cfg.get('roma_fb_max', 2.)))
    edge = field(obs, 'edge_status', 'obs')
    edge = None if edge is None else edge[:, camera_index]
    status = field(obs, 'status', 'obs')
    duration = None if status is None else (status[:, camera_index] == 1).sum(-1)
    refduration = None if status is None else (status[:, 0] == 1).sum(-1)
    temporal = field(obs, 'temporal_success', 'obs')
    if temporal is None and refduration is not None:
        temporal = refduration >= threshold['min_track_observations']
    anchor_error = field(obs, 'anchor_error', 'obs')
    anchor_support = field(obs, 'anchor_support', 'obs')
    anchor_frames = field(obs, 'anchor_frames', 'obs')
    seed_frame = field(obs, 'seed_frame', 'obs')
    anchor_count = disagreement = None
    if all(a is not None for a in [anchor_error, anchor_support, anchor_frames, seed_frame]):
        independent = anchor_support[:, camera_index] & (anchor_frames != int(seed_frame))[None, :]
        anchor_count = independent.sum(-1)
        disagreement = np.max(np.where(independent, anchor_error[:, camera_index], -np.inf), axis=-1)
        disagreement = np.where(anchor_count > 0, disagreement, np.inf)
    confidence = field(obs, 'roma_confidence', 'obs')
    confidence = None if confidence is None else confidence[:, camera_index]
    fb = field(obs, 'roma_fb_error', 'obs')
    fb = None if fb is None else fb[:, camera_index]
    conditions = {
        'query': np.ones(n, bool),
        'roma_initial_valid': None if edge is None or camera_index == 0 else np.isin(edge, [1, 3, 4]),
        'collision_free': None if edge is None or camera_index == 0 else np.isin(edge, [1, 4]),
        'flow': None if duration is None else duration >= threshold['min_track_observations'],
        'anchor_support': None if anchor_count is None or camera_index == 0 else anchor_count >= threshold['anchor_min_support'],
        'anchor_error': None if disagreement is None or camera_index == 0 else np.isfinite(disagreement) & (disagreement <= threshold['anchor_error_max']),
        'target_edge_accepted': None if edge is None else edge == 1,
        'temporal_success': temporal,
        'multiview_seed': field(obs, 'multiview_seed', 'obs'),
        'all_three_seed': field(obs, 'all_three_seed', 'obs'),
        'initialization': field(result, 'seed_eligible', 'result'),
        'trajectory_valid': field(result, 'trajectory_valid', 'result'),
    }
    conditions['anchors'] = None if conditions['anchor_support'] is None or conditions['anchor_error'] is None else conditions['anchor_support'] & conditions['anchor_error']
    overlapping = {
        'roma_score_below_threshold': None if confidence is None or camera_index == 0 else confidence < threshold['confidence_min'],
        'roma_fb_above_threshold_or_nonfinite': None if fb is None or camera_index == 0 else ~np.isfinite(fb) | (fb > threshold['roma_fb_max']),
        'recorded_collision': None if edge is None or camera_index == 0 else edge == 3,
        'target_observation_count_insufficient': None if duration is None else duration < threshold['min_track_observations'],
        'independent_anchor_support_insufficient': None if anchor_count is None or camera_index == 0 else anchor_count < threshold['anchor_min_support'],
        # No supported anchor means missing evidence, not a measured large error.
        'independent_anchor_error_above_threshold': None if disagreement is None or camera_index == 0 else (anchor_count > 0) & (~np.isfinite(disagreement) | (disagreement > threshold['anchor_error_max'])),
        'reference_temporal_insufficient': None if temporal is None else ~temporal.astype(bool),
        'multiview_insufficient': None if conditions['multiview_seed'] is None else ~conditions['multiview_seed'].astype(bool),
        'initialization_rejected': None if conditions['initialization'] is None else ~conditions['initialization'].astype(bool),
        'trajectory_invalid': None if conditions['trajectory_valid'] is None else ~conditions['trajectory_valid'].astype(bool),
    }
    # Reverse priority assignment preserves the recorded first edge gate. No
    # order between duration/support/error is invented for edge_status == 4.
    category = np.full(n, 'success', dtype='<U32')
    for key, label in [('trajectory_valid', 'trajectory'), ('initialization', 'initialization'),
                       ('multiview_seed', 'multiview'), ('temporal_success', 'reference_flow')]:
        value = conditions[key]
        if value is None:
            category[:] = 'missing'
        else:
            category[~value.astype(bool)] = label
    if edge is None:
        category[:] = 'missing'
    elif camera_index != 0:
        for code, label in [(0, 'not_attempted'), (2, 'roma_initial'), (3, 'collision'), (4, 'flow_or_anchors')]:
            category[edge == code] = label
        category[~np.isin(edge, [0, 1, 2, 3, 4])] = 'missing'
    return dict(conditions=conditions, overlapping=overlapping, first_failure=category, edge=edge,
                duration=duration, anchor_count=anchor_count, disagreement=disagreement,
                confidence=confidence, fb=fb, missing=sorted(set(missing)), thresholds=threshold)


def _category(state, stage):
    if stage == 'first_failure':
        return state['first_failure']
    if stage == 'edge_status':
        labels = {0: 'not_attempted', 1: 'pass', 2: 'roma_initial', 3: 'collision', 4: 'flow_or_anchors'}
        return np.array([labels.get(int(v), 'missing') for v in state['edge']]) if state['edge'] is not None else np.full(len(state['first_failure']), 'missing')
    if stage not in state['conditions']:
        raise ValueError('Unsupported coverage stage: ' + str(stage))
    value = state['conditions'][stage]
    return np.full(len(state['first_failure']), 'missing') if value is None else np.where(value, 'pass', 'fail')


def coverage(run, roi=None, camera_index=1, stage='first_failure'):
    selected, spec = resolve_roi(run, roi)
    state = _state(run, camera_index)
    total = int(selected.sum())
    count = lambda mask: None if mask is None else int((selected & mask.astype(bool)).sum())
    stages = {key: {'count': count(mask), 'denominator': total,
                    'fraction': count(mask) / total if mask is not None and total else None}
              for key, mask in state['conditions'].items()}
    cumulative = selected.copy()
    pipeline = []
    for key in ['query', 'roma_initial_valid', 'collision_free', 'flow', 'anchor_support', 'anchor_error', 'target_edge_accepted']:
        value = state['conditions'][key]
        if value is None:
            pipeline.append({'stage': key, 'count': None, 'missing': True})
            continue
        cumulative &= value.astype(bool)
        pipeline.append({'stage': key, 'count': int(cumulative.sum()), 'missing': False})
    first = state['first_failure'][selected]
    edge4 = None if state['edge'] is None else state['edge'] == 4
    subdivisions = {}
    edge4_unknown = np.zeros(len(selected), bool) if edge4 is None else edge4.copy()
    for key in ['target_observation_count_insufficient', 'independent_anchor_support_insufficient', 'independent_anchor_error_above_threshold']:
        condition = state['overlapping'][key]
        subdivisions[key] = None if edge4 is None or condition is None else count(edge4 & condition)
        if condition is not None:
            edge4_unknown &= ~condition
    return {'schema_version': 1, 'run_id': run.run_id, 'method': run.method, 'roi': spec,
            'camera_index': camera_index, 'camera_id': run.cameras[camera_index]['camera_id'],
            'reference_camera_id': run.cameras[0]['camera_id'], 'stage': stage,
            'image_shape': [run.cameras[0]['size'][1], run.cameras[0]['size'][0]],
            'stage_counts': stages, 'target_pipeline_cumulative_counts': pipeline,
            'recorded_first_gate_counts': {str(k): int((first == k).sum()) for k in np.unique(first)},
            'overlapping_condition_counts': {k: count(v) for k, v in state['overlapping'].items()},
            'edge4_subdivision': {'recorded_count': count(edge4), 'conditions_can_overlap': True,
                'counts': subdivisions, 'not_explained_by_available_fields': count(edge4_unknown) if edge4 is not None else None,
                'interpretation': 'Saved combined gate; subdivisions are reconstructed from saved counts/independent anchors and are not recorded first-subgate order'},
            'thresholds': state['thresholds'], 'missing_fields': state['missing'], 'legend': LEGEND,
            'points': {'point_ids': np.asarray(run.point_ids)[selected].astype(int).tolist(),
                       'seed_uv': np.asarray(run.obs['seed_uv'])[selected].tolist(),
                       'category': _category(state, stage)[selected].tolist()},
            'interpretation': ['All ROI coordinates belong to the reference seed image, never the displayed current frame.',
                'Stage counts are independent conditions; only target_pipeline_cumulative_counts impose pipeline order.',
                'first_failure uses the selected camera edge gate before downstream multiview/initialization/trajectory eligibility; an earlier camera failure can coexist with a valid trajectory from other cameras.',
                'RoMa initial validity is inferred exactly from saved edge status; individual model-valid/bounds rejection was not separately persisted.',
                'Flow duration counts saved status==1 before cross-view masking, not final visible==1.',
                'Seed-frame anchor is excluded. No ground-truth identity or semantic label is used.']}


def point_list(run, roi=None, camera_index=1, reason=None, sort='point_id', descending=False,
               offset=0, limit=200, stage='first_failure'):
    if offset < 0 or not 1 <= limit <= 1000:
        raise ValueError('offset must be nonnegative and limit must be 1..1000')
    selected, _ = resolve_roi(run, roi)
    state = _state(run, camera_index)
    categories = _category(state, stage)
    if reason:
        if reason in state['overlapping']:
            value = state['overlapping'][reason]
            selected &= False if value is None else value
        else:
            selected &= categories == reason
    fields = {'point_id': np.asarray(run.point_ids), 'temporal_observations': state['duration'],
              'independent_anchor_support': state['anchor_count'], 'max_anchor_error': state['disagreement'],
              'roma_confidence': state['confidence'], 'roma_fb_error': state['fb']}
    if sort not in fields:
        raise ValueError('Unknown point sort field')
    indices = np.flatnonzero(selected)
    if fields[sort] is not None:
        values = fields[sort][indices].astype(float)
        # Missing values always last, in either sort direction.
        order = np.lexsort((np.asarray(run.point_ids)[indices], np.where(np.isfinite(values), -values if descending else values, np.inf)))
        indices = indices[order]
    rows = []
    for i in indices[offset:offset + limit]:
        row = {key: None if value is None else _finite(value[i]) for key, value in fields.items()}
        row['point_id'] = int(run.point_ids[i])
        row.update(seed_uv=np.asarray(run.obs['seed_uv'])[i].tolist(), category=str(categories[i]),
                   recorded_first_gate=str(state['first_failure'][i]),
                   edge_status=None if state['edge'] is None else int(state['edge'][i]))
        rows.append(row)
    return {'total': len(indices), 'offset': offset, 'limit': limit, 'sort': sort,
            'descending': bool(descending), 'points': rows}
