"""Point diagnostics replayed from verified existing dense-flow caches only.

Never calls flow estimation, tracking, RoMa or optimization. Diagnostic values
are sampled from frozen cache files and explicitly distinguished from recordings.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np

from ..temporal_tracks import bilinear_sample, STATUS


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class CacheMissing(RuntimeError):
    pass


class FrozenFlowCache:
    """Verify image/settings identity before reading each existing NPZ pair."""
    def __init__(self, run, camera_index):
        self.run, self.camera_index = run, camera_index
        self.camera = run.cameras[camera_index]
        self.size = tuple(self.camera['size'])
        self.length = np.asarray(run.obs['status']).shape[-1]
        self.ids = list(self.camera.get('raw_frame_ids', range(self.length)))
        cache = getattr(run, 'cache_dir', None)
        if cache is None:
            raise CacheMissing('Run has no associated temporal cache directory')
        self.root = Path(cache).parent / 'temporal' / ('cam' + str(self.camera['camera_id']))
        if not self.root.is_dir():
            raise CacheMissing('No existing temporal flow cache: ' + str(self.root))
        matching = []
        for path in sorted(self.root.glob(f'sequence_{self.ids[0]}_{self.ids[-1]}_*.json')):
            metadata = json.loads(path.read_text())
            if metadata.get('raw_frame_ids') == self.ids and metadata.get('camera_id') == self.camera['camera_id'] and metadata.get('size') == list(self.size):
                matching.append((path, metadata))
        if not matching:
            raise CacheMissing('No sequence cache metadata matches camera, raw-frame IDs and image size')
        identities = {json.dumps({k: m[k] for k in ['settings', 'opencv', 'size', 'camera_id']}, sort_keys=True) for _, m in matching}
        if len(identities) != 1:
            raise CacheMissing('Multiple incompatible flow configurations exist; cannot attribute original cache uniquely')
        self.metadata_path, self.metadata = matching[0]
        self.identity = identities.pop()
        self.files = {}

    @lru_cache(maxsize=16)
    def image_hash(self, frame):
        rgb = np.asarray(self.run.image(self.camera_index, frame))
        if rgb.dtype != np.uint8 or rgb.shape != (self.size[1], self.size[0], 3):
            raise CacheMissing('Source RGB image shape/type differs from cached camera declaration')
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return hashlib.sha256(gray.tobytes()).hexdigest()

    @lru_cache(maxsize=16)
    def pair(self, a, b):
        if not 0 <= a < b < self.length:
            raise CacheMissing('Requested cache pair outside sequence')
        key = hashlib.sha256((self.identity + self.image_hash(a) + self.image_hash(b)).encode()).hexdigest()[:20]
        path = self.root / f'raw_{self.ids[a]}_{self.ids[b]}_{key}.npz'
        if not path.is_file():
            alternatives = list(self.root.glob(f'raw_{self.ids[a]}_{self.ids[b]}_*.npz'))
            reason = 'Source/settings fingerprint does not match existing pair' if alternatives else 'Pair not present in frozen cache'
            raise CacheMissing(reason + ': ' + str(path))
        with np.load(path, allow_pickle=False) as archive:
            forward, backward = archive['forward'], archive['backward']
        expected = (self.size[1], self.size[0], 2)
        if forward.shape != expected or backward.shape != expected:
            raise CacheMissing('Cached flow shape mismatch: ' + str(path))
        self.files[str(path)] = {'path': str(path), 'source_frames': [a, b], 'raw_frame_ids': [self.ids[a], self.ids[b]],
            'image_hashes': [self.image_hash(a), self.image_hash(b)], 'fingerprint': key,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source_identity_verified': True}
        return forward, backward

    def step(self, source, target, xy):
        low, high = sorted([source, target])
        forward, backward = self.pair(low, high)
        field, reverse = (forward, backward) if source < target else (backward, forward)
        xy = np.asarray(xy, np.float32).reshape(1, 2)
        displacement, valid = bilinear_sample(field, xy)
        target_xy = xy + displacement
        inverse, inside = bilinear_sample(reverse, target_xy)
        return {'source_frame': source, 'target_frame': target, 'source_uv': xy[0],
                'displacement': displacement[0], 'predicted_uv': target_xy[0],
                'reverse_displacement_at_prediction': inverse[0],
                'source_valid': bool(valid[0]), 'target_reverse_valid': bool(inside[0]),
                'fb_error_px': float(np.linalg.norm(displacement[0] + inverse[0])) if valid[0] and inside[0] else None}


def _first_failure(status, seed, direction):
    if status[seed] != 1:
        return {'frame': seed, 'status': int(status[seed]), 'reason': STATUS.get(int(status[seed]), 'unknown')}
    for frame in range(seed + direction, len(status) if direction == 1 else -1, direction):
        if status[frame] != 1:
            return {'frame': frame, 'status': int(status[frame]), 'reason': STATUS.get(int(status[frame]), 'unknown')}
    return None


def flow_diagnostics(run, point_id, camera_index=1, frame=None):
    """Return recorded history and read-only cache replay for one seed identity."""
    if not 0 <= camera_index < len(run.cameras):
        raise ValueError('camera_index out of range')
    index = run.index(int(point_id))
    obs = run.obs
    if 'status' not in obs or 'uv' not in obs:
        return {'point_id': int(point_id), 'missing': ['obs.status or obs.uv'], 'cache_replay': {'available': False}}
    status = np.asarray(obs['status'])[index, camera_index]
    tracks = np.asarray(obs['uv'])[index, camera_index]
    seed = int(np.asarray(obs['seed_frame']))
    length = len(status)
    frame = seed if frame is None else int(frame)
    if not 0 <= frame < length:
        raise ValueError('frame outside sequence')
    fb = np.asarray(obs['temporal_fb_error'])[index, camera_index] if 'temporal_fb_error' in obs else None
    visible = np.asarray(obs['visible'])[index, camera_index] if 'visible' in obs else None
    distance = np.linalg.norm(np.diff(tracks, axis=0), axis=-1)
    accepted_pair = (status[:-1] == 1) & (status[1:] == 1) & np.isfinite(distance)
    order = np.flatnonzero(accepted_pair)
    order = order[np.argsort(-distance[order], kind='stable')][:10]
    output = {'schema_version': 1, 'run_id': run.run_id, 'method': run.method,
        'point_id': int(point_id), 'camera_index': camera_index, 'camera_id': run.cameras[camera_index]['camera_id'],
        'seed_frame': seed, 'selected_frame': frame,
        'recorded': {'uv': tracks, 'status': status, 'status_legend': STATUS, 'temporal_fb_error_px': fb,
            'visible_after_cross_view_gate': visible,
            'first_failure_forward': _first_failure(status, seed, 1),
            'first_failure_backward': _first_failure(status, seed, -1),
            'cycle_error_available': 'cycle_error' in obs,
            'largest_accepted_adjacent_displacements': [{'source_frame': int(i), 'target_frame': int(i + 1), 'displacement_px': float(distance[i])} for i in order],
            'jump_interpretation': 'Ranked image displacement only; no jump rejection threshold or identity error is inferred'},
        'cache_replay': {'available': False, 'semantics': '从冻结缓存重放的诊断值 / diagnostic replay from frozen cache; not original recorded cycle errors'},
        'missing': [], 'read_only': True,
        'interpretation': 'Accepted optical-flow status is not ground-truth visibility. Cross-view masking can remove a temporally tracked observation.'}
    if 'cycle_error' in obs:
        output['recorded']['cycle_error_px'] = np.asarray(obs['cycle_error'])[index, camera_index]
    else:
        output['missing'].append('obs.cycle_error was not persisted; replay is explicitly separate')
    try:
        cache = FrozenFlowCache(run, camera_index)
    except (CacheMissing, OSError, ValueError, KeyError) as error:
        output['cache_replay']['missing_reason'] = str(error)
        return _json(output)
    replay = output['cache_replay']
    replay.update(metadata_path=str(cache.metadata_path), metadata=cache.metadata,
                  source_verification='Each sampled pair filename fingerprint is recomputed from actual source grayscale pixels and recorded settings/OpenCV/camera/size. Files are SHA256 fingerprinted; no estimation runs.')
    cfg = {'fb_max': .75, 'fb_relative': .03, 'cycle_max': 1.5, 'cycle_fb_max': 2.}
    tag = 'observations_lsm' if run.method == 'B4' else 'observations'
    metadata_path = Path(run.cache_dir) / (tag + '.json')
    if metadata_path.is_file():
        cfg.update(json.loads(metadata_path.read_text()).get('tracking_config', {}))
        replay['tracking_config_source'] = str(metadata_path)
    else:
        replay['tracking_config_source'] = 'Documented original defaults; per-run frontend metadata missing'
    replay['thresholds'] = cfg
    for direction, label in [(1, 'forward'), (-1, 'backward')]:
        target = frame + direction
        if not 0 <= target < length or not np.isfinite(tracks[frame]).all():
            replay[label] = {'available': False, 'missing_reason': 'Outside sequence or selected recorded proposal is nonfinite'}
            continue
        try:
            step = cache.step(frame, target, tracks[frame])
            tolerance = cfg['fb_max'] + cfg['fb_relative'] * np.linalg.norm(step['displacement'])
            step.update(available=True, fb_tolerance_px=float(tolerance),
                        recorded_target_uv=tracks[target], recorded_target_status=int(status[target]),
                        proposal_difference_from_recorded_px=float(np.linalg.norm(step['predicted_uv'] - tracks[target])))
            replay[label] = step
            replay['available'] = True
        except (CacheMissing, OSError, ValueError, KeyError) as error:
            replay[label] = {'available': False, 'missing_reason': str(error)}
    stride = int(cache.metadata['settings'].get('skip_stride', 5))
    low = min((frame // stride) * stride, ((length - 1) // stride - 1) * stride) if stride > 1 else -1
    high = low + stride
    if low < 0 or high >= length:
        replay['direct_vs_composed'] = {'available': False, 'missing_reason': 'No complete skip-frame interval'}
    else:
        # Tracking walks away from the seed. Choose that same direction when
        # this interval is on a single side; crossing-seed intervals are only
        # descriptive and explicitly not an original cycle decision.
        source, target = (high, low) if high <= seed else (low, high)
        direction = 1 if target > source else -1
        try:
            if not np.isfinite(tracks[source]).all():
                raise CacheMissing('Recorded skip-interval source UV unavailable')
            direct = cache.step(source, target, tracks[source])
            xy = tracks[source].copy()
            steps = []
            for t in range(source, target, direction):
                value = cache.step(t, t + direction, xy)
                steps.append(value)
                if not value['source_valid'] or not value['target_reverse_valid']:
                    raise CacheMissing('Composed path leaves valid cached field support')
                xy = value['predicted_uv']
            reliable = bool(status[source] == 1 and direct['source_valid'] and direct['target_reverse_valid'] and direct['fb_error_px'] is not None and direct['fb_error_px'] <= cfg['cycle_fb_max'])
            replay['direct_vs_composed'] = {'available': True, 'source_frame': source, 'target_frame': target,
                'direct': direct, 'composed_uv': xy, 'composed_adjacent_steps': steps,
                'direct_vs_composed_error_px': float(np.linalg.norm(xy - direct['predicted_uv'])),
                'direct_vs_recorded_endpoint_error_px': float(np.linalg.norm(tracks[target] - direct['predicted_uv'])),
                'direct_fb_reliable': reliable, 'cycle_threshold_px': cfg['cycle_max'],
                'original_cycle_interval_eligible': bool(high <= seed or low >= seed),
                'recorded_source_and_endpoint_tracked': bool(status[source] == 1 and status[target] == 1),
                'composed_path_policy': 'Read-only adjacent flow composition; original rejection/termination is not rerun. This path may continue after an original failure.'}
            replay['available'] = True
        except (CacheMissing, OSError, ValueError, KeyError) as error:
            replay['direct_vs_composed'] = {'available': False, 'missing_reason': str(error)}
    replay['verified_cache_files'] = list(cache.files.values())
    replay['verified_cache_paths'] = list(cache.files)
    return _json(output)
