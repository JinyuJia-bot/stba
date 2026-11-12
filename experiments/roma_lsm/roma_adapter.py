"""Audited RoMa v1 symmetric dense-field adapter and immutable inference cache.

No model.sample(), keypoints, sparse correspondence selection, or camera fitting.
The installed v1 package is pinned in provenance by its actual source-file hashes.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

try:
    from .coordinates import normalized_to_pixel, inside_image, pixel_grid, pixel_to_normalized
    from .prepare_data import sha256
except ImportError:
    from coordinates import normalized_to_pixel, inside_image, pixel_grid, pixel_to_normalized
    from prepare_data import sha256


ADAPTER_SCHEMA = 1


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _resize_field(value, size):
    """Bilinear scalar-field interpolation, explicitly align_corners=False."""
    value=np.asarray(value)
    scalar=value.ndim==2
    if scalar:
        value=value[...,None]
    tensor=torch.as_tensor(np.ascontiguousarray(value),dtype=torch.float32).permute(2,0,1)[None]
    result=functional.interpolate(tensor,size=(size[1],size[0]),mode='bilinear',align_corners=False)
    result=result[0].permute(1,2,0).numpy()
    return result[...,0] if scalar else result


def decode_symmetric_native(warp, certainty, size_a, size_b):
    """Decode actual v1 [1,H,2W,4] to two independent logical dense fields.

    Native left half: (A grid xy, target B xy); right: (target A xy, B grid xy).
    All four native values are normalized align_corners=False coordinates.
    Native target fields are bilinearly queried on each SOURCE work-image grid,
    then converted using that TARGET's own dimensions, with the -0.5 correction.
    Interpolated output is initialization, never claimed as independently refined.
    Unsupported interpolation or any invalid native contributor invalidates output.
    """
    warp,certainty=_numpy(warp),_numpy(certainty)
    if warp.ndim==4:
        if warp.shape[0]!=1 or certainty.shape[0]!=1:
            raise ValueError('Adapter requires one image pair per model.match call')
        warp,certainty=warp[0],certainty[0]
    if warp.ndim!=3 or warp.shape[-1]!=4 or warp.shape[1]%2 or certainty.shape!=warp.shape[:2]:
        raise ValueError('Expected symmetric native warp Hx2Wx4 and certainty Hx2W')
    height,width2=warp.shape[:2]
    width=width2//2
    native_size=(width,height)
    expected=pixel_to_normalized(pixel_grid(native_size),native_size)
    for actual in [warp[:,:width,:2],warp[:,width:,2:]]:
        if not np.allclose(actual,expected,atol=2e-6,rtol=0):
            raise ValueError('Unexpected RoMa source lattice/layout; refusing ambiguous decode')
    result={}
    for direction,normalized,score,source_size,target_size in [
        ('AB',warp[:,:width,2:],certainty[:,:width],size_a,size_b),
        ('BA',warp[:,width:,:2],certainty[:,width:],size_b,size_a)]:
        native_valid=np.isfinite(normalized).all(-1)&np.isfinite(score)&(score>0)
        # Preserve raw scores, do not treat them as calibrated probabilities.
        uv=normalized_to_pixel(_resize_field(normalized,source_size),target_size).astype(np.float32)
        confidence=_resize_field(score,source_size)
        query_native=normalized_to_pixel(pixel_to_normalized(pixel_grid(source_size),source_size),native_size)
        valid=(_resize_field(native_valid.astype(np.float32),source_size)>=1-1e-6)
        valid &= inside_image(query_native,native_size)&inside_image(uv,target_size)
        result[f'uv_{direction}']=uv
        result[f'confidence_{direction}']=confidence.astype(np.float32)
        result[f'valid_{direction}']=valid
    return result


def adapter_self_test():
    """Analytical identity/direction tests including unequal image dimensions."""
    native_size=(28,20)
    grid=pixel_to_normalized(pixel_grid(native_size),native_size).astype(np.float32)
    warp=np.concatenate([np.concatenate([grid,grid],-1),np.concatenate([grid,grid],-1)],1)[None]
    certainty=np.ones(warp.shape[:3],np.float32)
    for a,b in [((14,10),(14,10)),((13,9),(17,12)),(native_size,native_size)]:
        out=decode_symmetric_native(warp,certainty,a,b)
        for direction,source,target in [('AB',a,b),('BA',b,a)]:
            truth=normalized_to_pixel(pixel_to_normalized(pixel_grid(source),source),target)
            np.testing.assert_allclose(out[f'uv_{direction}'],truth,atol=3e-6,rtol=0)
    moved=warp.copy()
    moved[0,:,:28,2]+=0.07
    moved[0,:,28:,1]-=0.04
    out=decode_symmetric_native(moved,certainty,(14,10),(17,12))
    np.testing.assert_allclose(out['uv_AB'][...,0]-decode_symmetric_native(warp,certainty,(14,10),(17,12))['uv_AB'][...,0],.07*17/2,atol=2e-6)
    np.testing.assert_allclose(out['uv_BA'][...,1]-decode_symmetric_native(warp,certainty,(14,10),(17,12))['uv_BA'][...,1],-.04*10/2,atol=2e-6)
    certainty[:,5,5]=0
    out=decode_symmetric_native(warp,certainty,native_size,native_size)
    assert not out['valid_AB'][5,5] and out['valid_BA'][5,5]
    return {'passed':True,'checks':['identity','unequal_image_sizes','bidirectional_layout','half_pixel','native_invalid_preservation']}


def provenance(weights, backbone, settings):
    import romatch
    root=Path(romatch.__file__).resolve().parent
    distribution=importlib.metadata.distribution('romatch')
    direct=distribution.read_text('direct_url.json')
    files={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*.py'))}
    return dict(adapter_schema=ADAPTER_SCHEMA,adapter_source_sha256=sha256(__file__),
                algorithm='RoMa v1 outdoor; official pretrained model; not RoMa v2',
                package_version=distribution.version,package_root=str(root),
                package_direct_url=json.loads(direct) if direct else None,
                actual_python_source_sha256=files,
                weights={str(weights):sha256(weights),str(backbone):sha256(backbone)},settings=settings,
                python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                coordinate_convention='xy integer pixel centers; grid_sample align_corners=False; uv=(normalized+1)*target_size/2-0.5',
                network_layout='symmetric Hx2Wx4: left=[A source grid,B target], right=[A target,B source grid]',
                resampling='Bilinear normalized-target-field resampling from native lattice to each source work image; align_corners=False. Different target dimensions handled independently. Source queries outside native center extent invalidated; native-invalid interpolation contributors invalidate output.',
                certainty_meaning='Uncalibrated model score: sigmoid of certainty logit with built-in low-resolution attenuation; outside[-1,1] target predictions score0 before clamp. Not localization variance or calibrated probability.',
                preprocessing='PIL RGB images separately resized by official model transform to square coarse resolution and square upsample resolution; no padding, no crop, no external rectification.',
                source_audit='matcher.py match: source gridlinspace(-1+1/N,1-1/N,N); symmetric native width concatenation. Official to_pixel_coordinates helper omits -0.5, therefore intentionally not used.',
                adapter_tests=adapter_self_test())


def run(manifest_path, out, device='cuda', coarse=560, upsample=560, selected_pairs=None):
    adapter_self_test()  # Mandatory before any heavyweight model inference.
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this execution context; use permitted host GPU execution, do not silently run CPU')
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(20261111)
    np.random.seed(20261111)
    torch.set_num_threads(4)
    manifest=json.loads(Path(manifest_path).read_text())
    cameras={str(c['camera_id']):c for c in manifest['cameras']}
    weights=Path('/home/jjy/.cache/torch/hub/checkpoints/roma_outdoor.pth')
    backbone=Path('/home/jjy/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth')
    settings=dict(coarse_resolution=[coarse,coarse],upsample_resolution=[upsample,upsample],
                  upsample_preds=True,symmetric=True,use_custom_corr=False,with_padding=False,
                  do_compile=False,amp_dtype='float16',seed=20261111,device=device,
                  matmul_precision='highest',torch_threads=4)
    prov=provenance(weights,backbone,settings)
    prov['gpu']=torch.cuda.get_device_name(0) if device.startswith('cuda') else 'CPU'
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    (out/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    model=None
    results=[]
    for pair in manifest['pairs']:
        a,b=str(pair['reference_id']),str(pair['target_id'])
        pair_name=f'{a}_{b}'
        if selected_pairs and pair_name not in selected_pairs:
            continue
        ca,cb=cameras[a],cameras[b]
        paths=[ca['image_path'],cb['image_path']]
        sizes=[tuple(Image.open(p).size) for p in paths]
        if sizes != [tuple(ca['size']),tuple(cb['size'])]:
            raise ValueError('Manifest sizes must match actual working image dimensions')
        identity={'provenance':prov,'images':{p:sha256(p) for p in paths}}
        fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        dest=out/f'pair_{pair_name}.npz'
        meta_path=out/f'pair_{pair_name}.json'
        if dest.exists():
            old=json.loads(meta_path.read_text())
            if old['fingerprint'] != fingerprint:
                raise RuntimeError(f'Stale cache {dest}; choose a new output directory or explicitly remove old files')
            print(json.dumps({'pair':pair_name,'cache_hit':True,'path':str(dest)}),flush=True)
            results.append(old)
            continue
        model_load_seconds=0.
        if model is None:
            import romatch
            start=time.perf_counter()
            # Explicit local tensors disable implicit network downloads.
            model=romatch.roma_outdoor(device=device,
                    weights=torch.load(weights,map_location='cpu',weights_only=True),
                    dinov2_weights=torch.load(backbone,map_location='cpu',weights_only=True),
                    coarse_res=coarse,upsample_res=upsample,amp_dtype=torch.float16,
                    symmetric=True,use_custom_corr=False,upsample_preds=True,
                    with_padding=False,do_compile=False)
            model.eval()
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            model_load_seconds=time.perf_counter()-start
            print(json.dumps({'model_loaded_seconds':model_load_seconds,'gpu':prov['gpu']}),flush=True)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        with torch.inference_mode():
            warp,certainty=model.match(paths[0],paths[1],device=device)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        inference_seconds=time.perf_counter()-start
        peak_allocated=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0
        peak_reserved=torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0
        transfer_start=time.perf_counter()
        native_warp,native_certainty=_numpy(warp),_numpy(certainty)
        transfer_seconds=time.perf_counter()-transfer_start
        del warp,certainty
        start=time.perf_counter()
        fields=decode_symmetric_native(native_warp,native_certainty,*sizes)
        decode_seconds=time.perf_counter()-start
        metadata=dict(pair=pair_name,fingerprint=fingerprint,cache_hit=False,reference_id=pair['reference_id'],
                      target_id=pair['target_id'],size_A=sizes[0],size_B=sizes[1],
                      native_warp_shape=list(native_warp.shape),native_certainty_shape=list(native_certainty.shape),
                      model_load_seconds=model_load_seconds,inference_seconds=inference_seconds,
                      device_to_host_seconds=transfer_seconds,decode_seconds=decode_seconds,
                      peak_gpu_allocated_bytes=peak_allocated,peak_gpu_reserved_bytes=peak_reserved,
                      logical_valid_AB_pixels=int(fields['valid_AB'].sum()),
                      logical_valid_BA_pixels=int(fields['valid_BA'].sum()),
                      coordinate_convention=prov['coordinate_convention'],
                      resampling=prov['resampling'],provenance_path=str(out/'provenance.json'))
        save_start=time.perf_counter()
        np.savez_compressed(dest,native_warp=native_warp,native_certainty=native_certainty,
                            **fields,metadata_json=np.asarray(json.dumps(metadata)))
        metadata['save_seconds']=time.perf_counter()-save_start
        meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(metadata),flush=True)
        results.append(metadata)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='experiments/roma_lsm/data_manifest.json')
    parser.add_argument('--out',default='experiments/roma_lsm/cache')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--coarse',type=int,default=560)
    parser.add_argument('--upsample',type=int,default=560)
    parser.add_argument('--pairs',nargs='*')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(adapter_self_test(),indent=2))
    else:
        run(args.manifest,args.out,args.device,args.coarse,args.upsample,args.pairs)
