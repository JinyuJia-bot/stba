"""Independently audit actual inputs, bounded iterates, projections and artifacts."""
from collections import Counter
import json
import subprocess
import numpy as np
import cv2
from run import HERE, ROOT, PREVIOUS, VIEW_FIELDS, read_csv, save, sha


def main():
    original=json.loads((HERE/'root_original_hashes.json').read_text())
    assert all(sha(ROOT/path)==digest for path,digest in original.items())
    manifest=json.loads((HERE/'manifest.json').read_text())
    assert all(sha(path)==digest for path,digest in manifest['source_sha256'].items())
    result={'root_source_and_binary_unchanged':True,'source_data_unchanged':True,'runs':{},'videos':{}}
    for sub in ['', 'holdout07']:
        base=HERE/sub;old=PREVIOUS/sub
        for file in ['cameras.csv','views.csv','observations.csv']:
            assert (base/'input'/file).read_bytes()==(old/'input'/file).read_bytes()
        before=read_csv(base/'input/views.csv')
        obs=read_csv(base/'input/observations.csv')
        keys={(int(r['track_id']),int(r['camera_id']),int(r['frame'])):r for r in obs}
        current=read_csv(base/'baseline/trajectories.csv');previous=read_csv(old/'baseline/trajectories.csv')
        baseline_delta=max(abs(float(a[k])-float(b[k])) for a,b in zip(current,previous) for k in ['time','x','y','z'])
        assert len(current)==len(previous) and baseline_delta<1e-9
        for audit_file in sorted(base.glob('*.bounds.csv')):
            name=audit_file.name.removesuffix('.bounds.csv');out=base/name
            audit=read_csv(audit_file)
            lo=min(float(r['min_beta']) for r in audit);hi=max(float(r['max_beta']) for r in audit)
            assert lo>=-1 and hi<=1
            views=read_csv(out/'views.csv')
            delta=max(abs(float(a[k])-float(b[k])) for a,b in zip(before,views) for k in VIEW_FIELDS)
            assert len(views)==len(before) and delta==0
            cameras=read_csv(out/'cameras.csv');beta=[float(r['beta']) for r in cameras]
            assert beta[0]==0 and max(map(abs,beta))<=1
            rows=read_csv(out/'trajectories.csv')
            assert len(rows)==len(keys)
            assert {tuple(int(r[k]) for k in ['track_id','camera_id','frame']) for r in rows}==keys.keys()
            sse=0.
            for row in rows:
                key=tuple(int(row[k]) for k in ['track_id','camera_id','frame'])
                v=views[key[1]];ob=keys[key]
                rotation=cv2.Rodrigues(np.array([float(v[k]) for k in ['rx','ry','rz']]))[0]
                q=rotation@np.array([float(row[k]) for k in ['x','y','z']])+np.array([float(v[k]) for k in ['tx','ty','tz']])
                assert q[2]>0
                uv=q[:2]/q[2]*np.array([float(v[k]) for k in ['fx','fy']])+np.array([float(v[k]) for k in ['cx','cy']])
                sse+=np.sum((uv-np.array([float(ob[k]) for k in ['u','v']]))**2)
            metrics=read_csv(out/'metrics.csv')[-1]
            rmse=np.sqrt(sse/len(rows));assert abs(rmse-float(metrics['reprojection_rmse']))<1e-9
            assert metrics['usable']=='1' and metrics['order_valid']=='1'
            dct=read_csv(out/'dct_metrics.csv');assert len(dct)==17 and all(r['usable']=='1' for r in dct)
            result['runs'][str(out.relative_to(HERE))]=dict(audited_states=len(audit),stage_counts=dict(Counter(r['stage'] for r in audit)),
                minimum_beta=lo,maximum_beta=hi,final_beta=beta,calibration_max_change=delta,
                observations=len(rows),independently_recomputed_rmse=rmse,baseline_xyz_time_difference=baseline_delta,
                dct_usable_tracks=17,solver_report=(out/'solver_report.txt').read_text())
    for name in ['skeleton','heldout07']:
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,nb_frames,r_frame_rate',
                    '-of','json',str(HERE/f'plots/{name}.mp4')]))['streams'][0]
        assert probe['nb_frames']=='60' and probe['codec_name']=='h264'
        result['videos'][name]=probe
    result['input_same_as_unbounded']=True
    result['all_checks_passed']=True
    save(HERE/'validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
    print('Total audited states:',sum(x['audited_states'] for x in result['runs'].values()))


if __name__=='__main__':main()
