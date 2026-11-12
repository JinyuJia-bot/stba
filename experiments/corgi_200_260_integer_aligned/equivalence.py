"""Same physical observations: integer reindexing must preserve times and geometry.

This control does not crop the reindexed frames again. The main experiment does
crop aligned [200,260), and therefore includes different endpoint observations.
"""
import json
import os
import shutil
import subprocess
import time
import numpy as np

from run import HERE, PREVIOUS, SOLVER, FPS, IDS, read_csv, save, sha


def write(path, rows):
    import csv
    with path.open('x', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    folder=HERE/'equivalence'
    folder.mkdir()
    inp=folder/'input';inp.mkdir()
    source=PREVIOUS/'input'
    cams=json.loads((HERE/'calibration.json').read_text())
    shifts=[c['integer_frame_shift'] for c in cams]
    cameras=read_csv(source/'cameras.csv')
    old_beta=[float(r['beta']) for r in cameras]
    for row,k in zip(cameras,shifts):row['beta']=float(row['beta'])-k
    write(inp/'cameras.csv',cameras)
    shutil.copy2(source/'views.csv',inp/'views.csv')
    rows=read_csv(source/'observations.csv')
    maximum_time_difference=0.
    for row in rows:
        c=int(row['camera_id']);f=int(row['frame']);g=f-shifts[c]
        old=(f-old_beta[c])/FPS;new=(g-float(cameras[c]['beta']))/FPS
        maximum_time_difference=max(maximum_time_difference,abs(old-new))
        row['frame']=g
    assert maximum_time_difference<1e-14
    write(inp/'observations.csv',rows)
    cmd=[str(SOLVER),'--input',str(inp),'--output',str(folder/'result'),
         '--config',str(HERE/'config.yaml'),'--mode','fixed','--no-dct']
    started=time.time()
    with (folder/'solver.log').open('x') as f:
        status=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    save(folder/'command.json',dict(command=cmd,returncode=status.returncode,seconds=time.time()-started,
         solver_sha256=sha(SOLVER)))
    assert status.returncode==0
    prior={tuple(int(r[k]) for k in ['track_id','camera_id','frame']):r
           for r in read_csv(PREVIOUS/'baseline/trajectories.csv')}
    distance=[]
    for row in read_csv(folder/'result/trajectories.csv'):
        c=int(row['camera_id']);key=(int(row['track_id']),c,int(row['frame'])+shifts[c])
        a=prior[key]
        distance.append(float(np.linalg.norm([float(row[k])-float(a[k]) for k in ['x','y','z']])))
    assert len(distance)==len(prior)
    old=read_csv(PREVIOUS/'baseline/metrics.csv')[-1]
    new=read_csv(folder/'result/metrics.csv')[-1]
    max_distance=max(distance)
    rmse_delta=abs(float(old['reprojection_rmse'])-float(new['reprojection_rmse']))
    assert max_distance<1e-7 and rmse_delta<1e-7,(max_distance,rmse_delta)
    result=dict(observations=len(distance),max_physical_time_difference_seconds=maximum_time_difference,
        max_xyz_difference_native_units=max_distance,undistorted_rmse_px_difference=rmse_delta,
        previous_cost=float(old['cost']),reindexed_cost=float(new['cost']),
        passed=True,meaning='Integer reindexing with matching beta preserves the same physical reconstruction. Main-window differences are not evidence that the previous time formula ignored multi-frame offsets.')
    save(folder/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
