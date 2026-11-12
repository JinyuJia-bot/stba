"""Small read-only adapter over actual, frozen experiment artifacts.

NPZ compression cannot provide random rows without decompressing that member.
Only requested fields are decompressed; bounded field caches and a two-run LRU
avoid eagerly materializing every million-observation array. No files are edited.
"""
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
import threading
import zipfile
import cv2
import numpy as np

DENSE_ROOT=Path(__file__).resolve().parents[1]
METHODS=('B0','B1','B2','B3','B4')


def to_jsonable(value):
    if isinstance(value,dict):return {str(k):to_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [to_jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return to_jsonable(value.tolist())
    if isinstance(value,np.generic):return to_jsonable(value.item())
    if isinstance(value,Path):return str(value)
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open() as handle:return json.load(handle)


class LazyNPZ(Mapping):
    """Dictionary-style, field-lazy NPZ reader; every returned array is read-only."""
    def __init__(self,path,max_cache_bytes=96*1024**2):
        self.path=Path(path);self.max_cache_bytes=int(max_cache_bytes)
        self._cache=OrderedDict();self._rows=OrderedDict();self._bytes=0;self._lock=threading.RLock()
        self.field_load_count=0;self.row_load_count=0
        with np.load(self.path,allow_pickle=False) as archive:self._keys=tuple(archive.files)
    def __iter__(self):return iter(self._keys)
    def __len__(self):return len(self._keys)
    def __contains__(self,key):return key in self._keys
    def __getitem__(self,key):
        if key not in self:raise KeyError(f'{self.path.name}: missing field {key!r}; available={list(self._keys)}')
        with self._lock:
            if key in self._cache:
                value=self._cache.pop(key);self._cache[key]=value;return value
            with np.load(self.path,allow_pickle=False) as archive:value=archive[key]
            value.setflags(write=False);self.field_load_count+=1
            if value.nbytes<=self.max_cache_bytes:
                while self._cache and self._bytes+value.nbytes>self.max_cache_bytes:
                    _,old=self._cache.popitem(last=False);self._bytes-=old.nbytes
                self._cache[key]=value;self._bytes+=value.nbytes
            return value
    @property
    def cached_fields(self):return tuple(self._cache)
    @property
    def cache_bytes(self):return self._bytes
    def field_info(self,key):
        """Read only the NPY header inside a compressed member, not its payload."""
        if key not in self:raise KeyError(key)
        with zipfile.ZipFile(self.path) as archive:
            with archive.open(key+'.npy') as handle:
                version=np.lib.format.read_magic(handle)
                if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(handle)
                else:shape,fortran,dtype=np.lib.format.read_array_header_2_0(handle)
        return dict(shape=list(shape),dtype=str(dtype),fortran_order=bool(fortran))
    def row(self,key,index):
        """Read one first-axis row without materializing a large compressed field.

        Deflate must still scan the prefix; ZipExtFile.seek discards it in bounded
        chunks. Only the requested row is allocated and cached. This is especially
        useful for a single-point inspector on million-observation runs.
        """
        if key not in self:raise KeyError(key)
        index=int(index);token=(key,index)
        with self._lock:
            if key in self._cache:
                result=np.array(self._cache[key][index],copy=True);result.setflags(write=False);return result
            if token in self._rows:
                result=self._rows.pop(token);self._rows[token]=result;return result
            with zipfile.ZipFile(self.path) as archive:
                with archive.open(key+'.npy') as handle:
                    version=np.lib.format.read_magic(handle)
                    if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(handle)
                    else:shape,fortran,dtype=np.lib.format.read_array_header_2_0(handle)
                    if not shape:raise IndexError(f'{key} is scalar, not a point-indexed field')
                    if dtype.hasobject:raise ValueError('Pickled/object arrays are not allowed')
                    if index<0:index+=shape[0]
                    if not 0<=index<shape[0]:raise IndexError(index)
                    if fortran:
                        result=np.array(self[key][index],copy=True)
                    else:
                        row_shape=shape[1:];count=int(np.prod(row_shape,dtype=np.int64)) if row_shape else 1
                        nbytes=count*dtype.itemsize;start=handle.tell()+index*nbytes
                        handle.seek(start);payload=handle.read(nbytes)
                        if len(payload)!=nbytes:raise ValueError(f'Truncated NPZ member {key}')
                        result=np.frombuffer(payload,dtype=dtype,count=count).reshape(row_shape).copy()
            result.setflags(write=False);self.row_load_count+=1;self._rows[token]=result
            while len(self._rows)>128:self._rows.popitem(last=False)
            return result
    def close(self):
        with self._lock:self._cache.clear();self._rows.clear();self._bytes=0


def _resolve(path,root,directory=None):
    path=Path(path)
    if path.exists():return path.resolve()
    for base in [directory,root]:
        if base is not None and (Path(base)/path).exists():return (Path(base)/path).resolve()
    marker='experiments/dense_stba/'
    if marker in path.as_posix():
        moved=Path(root)/path.as_posix().split(marker,1)[1]
        if moved.exists():return moved.resolve()
    raise FileNotFoundError(f'Declared source is unavailable: {path}')


def frame_identity(run,camera_index,frame):
    camera_index=int(camera_index);frame=int(frame)
    if not 0<=camera_index<len(run.cameras):raise IndexError('Camera index outside manifest')
    cam=run.cameras[camera_index];count=int(cam.get('n_frames',len(cam.get('image_paths',[]))))
    if not 0<=frame<count:raise IndexError(f'Frame {frame} outside [0,{count})')
    raw_ids=cam.get('raw_frame_ids');paths=cam.get('image_paths',[])
    return dict(camera_index=camera_index,camera_id=cam['camera_id'],local_frame=frame,
        raw_frame_id=int(raw_ids[frame]) if raw_ids is not None else None,
        image_path=str(paths[frame]) if paths else None,frames_path=cam.get('frames_path'),
        timestamp_source=cam.get('timestamp_kind','nominal camera timestamps'),
        manifest_sha256=run.source_validation.get('manifest_sha256'))


class LoadedRun:
    def __init__(self,root,directory,run_id,method,verify_sources=True):
        self.root=Path(root);self.directory=Path(directory);self.run_id=run_id;self.method=method
        self.meta=read_json(self.directory/'meta.json')
        self.method_meta=read_json(self.directory/f'{method}.json')
        self.warnings=[];self.source_validation={};self._images=OrderedDict();self._image_lock=threading.RLock()
        self._roma_lock=threading.RLock()
        self._reference_seed=None
        manifest_path=_resolve(self.method_meta.get('manifest_path',self.meta['manifest_path']),self.root,self.directory)
        self.manifest_path=manifest_path;self.manifest=read_json(manifest_path);self.cameras=self.manifest['cameras']
        obs_path=self.method_meta.get('observations_path',self.meta.get('observations_path'))
        if not obs_path:raise ValueError('Method metadata does not declare its frozen observations file')
        obs_path=_resolve(obs_path,self.root,self.directory)
        self.observations_path=obs_path
        self.obs=LazyNPZ(obs_path);self.result=LazyNPZ(self.directory/f'{method}.npz')
        if verify_sources:
            for label,path,expected in [
                ('observations',obs_path,self.method_meta.get('observations_sha256',self.meta.get('observations_sha256'))),
                ('manifest',manifest_path,self.method_meta.get('manifest_sha256',self.meta.get('manifest_sha256')))]:
                actual=sha256(path);self.source_validation[label+'_sha256']=actual
                self.source_validation[label+'_expected_sha256']=expected
                if expected and actual!=expected:raise ValueError(f'{label} SHA256 mismatch: frozen experiment source changed')
                if not expected:self.warnings.append(f'{label}: recorded SHA256 is missing; identity cannot be certified')
            source_hashes={}
            for path,expected in self.manifest.get('calibration_sources',{}).items():
                try:resolved=_resolve(path,self.root,self.directory)
                except FileNotFoundError:
                    self.warnings.append(f'Original calibration source unavailable; using recorded fixed manifest: {path}');continue
                actual=sha256(resolved)
                if actual!=expected:raise ValueError(f'Fixed calibration source SHA256 mismatch: {path}')
                source_hashes[str(resolved)]=actual
            self.source_validation['calibration_sources']=source_hashes
        else:self.warnings.append('Source SHA validation explicitly disabled for this store')
        self.point_ids=np.asarray(self.result['point_ids'])
        if len(np.unique(self.point_ids))!=len(self.point_ids):raise ValueError('Point IDs must be unique in a run')
        if 'point_ids' in self.obs and not np.array_equal(self.point_ids,self.obs['point_ids']):raise ValueError('Frozen observations and result point IDs differ')
        if 'seed_uv' in self.result and 'seed_uv' in self.obs and not np.array_equal(self.result['seed_uv'],self.obs['seed_uv']):
            raise ValueError('Frozen observations and result seed pixel identities differ')
        if 'camera_ids' in self.result and not np.array_equal(self.result['camera_ids'],[c['camera_id'] for c in self.cameras]):
            raise ValueError('Result camera order differs from fixed manifest')
        self.id_to_index={int(value):i for i,value in enumerate(self.point_ids)}
        name=str(self.meta.get('name',self.directory.name))
        base='cache_density1' if 'density1' in Path(run_id).parts[0] else 'cache'
        self.cache_dir=self.root/base/name
        if not self.cache_dir.exists():self.warnings.append(f'Optional frontend cache missing: {self.cache_dir}')
        candidates=[self.cache_dir/'roma',self.root/'cache'/name/'roma']
        self.roma_cache_dir=next((p for p in candidates if p.exists()),candidates[0])
        self.frontend_meta={}
        frontend_path=self.cache_dir/('observations_lsm.json' if 'lsm' in obs_path.stem else 'observations.json')
        if frontend_path.exists():self.frontend_meta=read_json(frontend_path)
        self.source_validation['raw_or_lsm_observations']='lsm' if 'lsm' in obs_path.stem else 'raw'
        self.source_validation['verified']=bool(verify_sources and
            self.source_validation.get('observations_expected_sha256') and self.source_validation.get('manifest_expected_sha256'))
    def index(self,point_id):
        try:return self.id_to_index[int(point_id)]
        except (KeyError,ValueError,TypeError):raise KeyError(f'Point ID {point_id!r} was not queried in {self.run_id}') from None
    def field_slice(self,source,key,index,default=None):
        data=self.obs if source=='obs' else self.result if source=='result' else None
        if data is None:raise ValueError("source must be 'obs' or 'result'")
        if key not in data:return default
        # Copy the small row so callers do not keep an uncached large field alive.
        if np.isscalar(index):return data.row(key,index)
        result=np.array(data[key][index],copy=True);result.setflags(write=False);return result
    def image(self,camera_index,frame):
        identity=frame_identity(self,camera_index,frame);key=(int(camera_index),int(frame))
        with self._image_lock:
            if key in self._images:
                result=self._images.pop(key);self._images[key]=result;return result
            cam=self.cameras[key[0]]
            if identity['image_path']:
                path=_resolve(identity['image_path'],self.root,self.manifest_path.parent)
                bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
                if bgr is None:raise FileNotFoundError(path)
                image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            elif identity['frames_path']:
                array=np.load(_resolve(identity['frames_path'],self.root,self.manifest_path.parent),mmap_mode='r',allow_pickle=False)
                image=np.array(array[key[1]],copy=True)
            else:raise FileNotFoundError('Manifest has no single-frame image path or frames.npy source')
            if image.dtype!=np.uint8 or image.shape!=(cam['size'][1],cam['size'][0],3):raise ValueError('Frame shape/dtype disagrees with fixed working calibration')
            image.setflags(write=False);self._images[key]=image
            while len(self._images)>3:self._images.popitem(last=False)
            return image
    def summary(self):
        cfg=self.method_meta.get('config',self.meta.get('config',{}))
        # This identity belongs to the reference seed image, independently of
        # the currently displayed frame, method, stride or target-camera shift.
        if self._reference_seed is None:
            seed=int(self.meta.get('seed_frame',self.manifest.get('seed_frame',0)))
            reference=int(self.meta.get('reference_camera_index',0))
            identity=frame_identity(self,reference,seed)
            identity.update(image_size=self.cameras[reference]['size'],image_sha256=None)
            try:
                if identity['image_path']:
                    source=_resolve(identity['image_path'],self.root,self.manifest_path.parent)
                    identity.update(image_path=str(source),image_sha256=sha256(source),image_hash_kind='file_bytes_sha256')
                else:
                    image=self.image(reference,seed)
                    identity.update(image_sha256=hashlib.sha256(image.tobytes(order='C')).hexdigest(),
                        image_hash_kind='decoded_rgb_uint8_C_order_sha256')
            except (OSError,ValueError) as error:
                identity['image_hash_unavailable_reason']=str(error)
            self._reference_seed=identity
        return to_jsonable(dict(run_id=self.run_id,method=self.method,directory=self.directory,
            label=self.meta.get('name',self.directory.name),camera_ids=[c['camera_id'] for c in self.cameras],
            image_sizes=[c['size'] for c in self.cameras],frame_counts=[c.get('n_frames',len(c.get('image_paths',[]))) for c in self.cameras],
            seed_frame=self.meta.get('seed_frame',self.manifest.get('seed_frame')),query_count=len(self.point_ids),
            reference_seed=self._reference_seed,
            stride=cfg.get('association',{}).get('stride'),window=self.meta.get('window'),
            valid_trajectories=self.method_meta.get('valid_trajectories'),valid_3D_samples=self.method_meta.get('valid_3D_samples'),
            beta_frames=self.method_meta.get('beta_frames'),initial_beta_frames=self.method_meta.get('initial_beta_frames'),
            observability=self.method_meta.get('observability'),timing=self.method_meta.get('timing'),
            config=cfg,source_validation=self.source_validation,obs_fields=list(self.obs),result_fields=list(self.result),
            missing_optional_fields=[key for key in ['control_initial','objective_per_observation'] if key not in self.result],
            gt_available=(self.manifest_path.parent/'gt.npz').exists(),warnings=self.warnings))
    def close(self):
        self.obs.close();self.result.close()
        with self._image_lock:self._images.clear()
        for archive in getattr(self,'_debug_roma_archives',{}).values():archive.close()
        if hasattr(self,'_debug_roma_archives'):self._debug_roma_archives.clear()


class DatasetStore:
    def __init__(self,root=DENSE_ROOT,max_runs=2,verify_sources=True):
        self.root=Path(root).resolve();self.max_runs=min(2,max(1,int(max_runs)))
        self.verify_sources=verify_sources;self._runs=OrderedDict();self._lock=threading.RLock()
    def catalog(self):
        entries=[]
        for path in sorted(self.root.rglob('meta.json')):
            try:meta=read_json(path)
            except (OSError,ValueError):continue
            if 'manifest_path' not in meta or 'methods' not in meta:continue
            methods=[m for m in METHODS if (path.parent/f'{m}.npz').exists() and (path.parent/f'{m}.json').exists()]
            if not methods:continue
            run_id=path.parent.relative_to(self.root).as_posix();cfg=meta.get('config',{})
            stride=cfg.get('association',{}).get('stride');group=run_id.split('/')[0]
            entries.append(dict(id=run_id,label=f"{meta.get('name',path.parent.name)} · stride {stride} · {group}",
                group=group,case=meta.get('name',path.parent.name),methods=methods,stride=stride,
                query_count=meta.get('seed_grid_size'),seed_frame=meta.get('seed_frame'),window=meta.get('window'),
                camera_ids=meta.get('camera_ids'),preliminary=('validation' in group)))
        return entries
    def load(self,run_id,method='B3'):
        if method not in METHODS:raise ValueError(f'Unknown method {method}')
        run_id=str(run_id).strip('/')
        directory=(self.root/run_id).resolve()
        if not directory.is_relative_to(self.root) or not (directory/'meta.json').is_file():raise KeyError(f'Unknown run ID {run_id}')
        if not (directory/f'{method}.npz').is_file() or not (directory/f'{method}.json').is_file():raise KeyError(f'{method} is not available for {run_id}')
        key=(run_id,method)
        with self._lock:
            if key in self._runs:
                run=self._runs.pop(key);self._runs[key]=run;return run
            run=LoadedRun(self.root,directory,run_id,method,self.verify_sources)
            self._runs[key]=run
            while len(self._runs)>self.max_runs:
                _,old=self._runs.popitem(last=False);old.close()
            return run
    @property
    def cached_runs(self):return tuple(self._runs)
    def close(self):
        with self._lock:
            for run in self._runs.values():run.close()
            self._runs.clear()
