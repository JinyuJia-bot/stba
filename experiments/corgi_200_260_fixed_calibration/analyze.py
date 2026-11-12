"""Audit, evaluate and visualize the fixed-calibration corgi window experiment."""
from pathlib import Path
import csv
import json
import subprocess
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from run import HERE, ROOT, IDS, FPS, START, STOP, VIEW_FIELDS, read_csv, save, sha

LINKS = [(0,2),(1,2),(2,3),(3,4),(3,5),(5,6),(6,7),(3,8),(8,9),(9,10),
         (4,11),(11,12),(12,13),(4,14),(14,15),(15,16)]
COLORS = {'baseline':'#677789', 'result':'#d96b18'}
LABELS = {'baseline':'Official sync fixed', 'result':'STBA optimized sync'}
PLOTS = HERE / 'plots'
Z = dict(np.load(HERE / 'observations.npz'))
CAMS = json.loads((HERE / 'calibration.json').read_text())
FRAMES = Z['frame']
NAMES = Z['joint_names'].tolist()


def result_dir(base, name):
    choice = base / 'selected_results.json'
    selected = json.loads(choice.read_text()) if choice.exists() else {}
    return base / selected.get(name, name)


def stats(e):
    e = np.asarray(e)
    if not len(e):
        return {'count':0}
    return dict(count=len(e), rmse=float(np.sqrt(np.mean(e**2))), median=float(np.median(e)),
                p95=float(np.percentile(e,95)), maximum=float(np.max(e)),
                mean_huber6=float(np.mean(np.where(e <= 6, e**2, 12*e-36))))


def project(x, original):
    cam = CAMS[IDS.index(original)]
    x = np.asarray(x).reshape(-1,3)
    projected = cv2.projectPoints(x, np.array(cam['rvec']), np.array(cam['T']),
                                 np.array(cam['K']), np.array(cam['D']))[0].reshape(-1,2)
    depth = (x @ np.array(cam['R']).T + np.array(cam['T']))[:,2]
    return projected, depth


def curves(path):
    by_joint = {}
    for row in read_csv(path):
        by_joint.setdefault(int(row['track_id']), []).append([float(row[k]) for k in ['time','x','y','z']])
    output = {}
    for joint, data in by_joint.items():
        values = np.array(data)
        values = values[np.argsort(values[:,0])]
        unique, first, count = np.unique(values[:,0], return_index=True, return_counts=True)
        for start, n in zip(first, count):
            if n > 1:
                assert np.ptp(values[start:start+n,1:],axis=0).max() < 1e-8
        output[joint] = values[first]
    return output


def interpolate(curve, times):
    times = np.atleast_1d(times)
    return np.column_stack([np.interp(times,curve[:,0],curve[:,k],left=np.nan,right=np.nan) for k in [1,2,3]])


def audit_fit(base):
    metadata = json.loads((base/'input/metadata.json').read_text())
    original_ids = metadata['camera_original_ids']
    original = read_csv(base/'input/views.csv')
    input_rows = read_csv(base/'input/observations.csv')
    expected_keys = {(int(r['track_id']),int(r['camera_id']),int(r['frame'])) for r in input_rows}
    output = {}
    for name in ['baseline','result']:
        result = result_dir(base,name)
        exported = read_csv(result/'views.csv')
        assert len(original) == len(exported)
        delta = max(abs(float(a[k])-float(b[k])) for a,b in zip(original,exported) for k in VIEW_FIELDS)
        assert delta == 0
        for local, row in enumerate(exported):
            cam = CAMS[IDS.index(original_ids[local])]
            k = np.array(cam['K'])
            expected = [k[0,0],k[1,1],k[0,2],k[1,2],*cam['rvec'],*cam['T']]
            assert np.array_equal([float(row[key]) for key in VIEW_FIELDS[3:]], expected)
        rows = read_csv(result/'trajectories.csv')
        observed_keys = {(int(r['track_id']),int(r['camera_id']),int(r['frame'])) for r in rows}
        assert observed_keys == expected_keys and len(rows) == len(expected_keys)
        errors, keys, predictions, xyz = [], [], [], []
        negative = 0
        for row in rows:
            joint, local, frame = (int(row[k]) for k in ['track_id','camera_id','frame'])
            original_id = original_ids[local]
            point = np.array([float(row[k]) for k in ['x','y','z']])
            uv, depth = project(point,original_id)
            observed = Z['xy_full_distorted'][IDS.index(original_id),frame-START,joint]
            errors.append(np.linalg.norm(uv[0]-observed))
            keys.append([original_id,frame,joint]); predictions.append(uv[0]); xyz.append(point)
            negative += int(depth[0] <= 0)
        dct = read_csv(result/'dct_metrics.csv')
        assert len(dct) == 17 and all(r['usable']=='1' for r in dct)
        np.savez_compressed(base/f'{name}_fit.npz', keys=keys, predictions=predictions, xyz=xyz, error=errors)
        output[name] = dict(calibration_max_abs_change=delta, exact_match_to_published_K_rvec_T=True,
            observation_coverage=len(rows)/len(input_rows), observations=len(rows),
            negative_depth=negative, distorted_full_resolution_px=stats(errors),
            result_directory=str(result), solver_report=(result/'solver_report.txt').read_text(),
            stages=read_csv(result/'metrics.csv'), dct_usable_tracks=len(dct),
            offset_seconds=[float(r['beta'])/float(r['fps']) for r in read_csv(result/'cameras.csv')])
    return output


def support(joint, query, originals, offsets):
    count = 0
    for original, offset in zip(originals, offsets):
        raw = (query+offset)*FPS
        if abs(raw-round(raw)) < 1e-7:
            lo = hi = int(round(raw))
        else:
            lo, hi = int(np.floor(raw)), int(np.ceil(raw))
        if START <= lo <= hi < STOP and Z['valid'][IDS.index(original),lo-START:hi-START+1,joint].all():
            count += 1
    return count


def heldout():
    base = HERE/'holdout07'
    originals = json.loads((base/'input/metadata.json').read_text())['camera_original_ids']
    assert 7 not in originals
    held = IDS.index(7)
    times = FRAMES/FPS - CAMS[held]['official_offset_seconds']
    predictions, points, masks, details = {}, {}, {}, {}
    for name in ['baseline','result']:
        offset = [float(r['beta'])/float(r['fps']) for r in read_csv(result_dir(base,name)/'cameras.csv')]
        local_support = np.array([[support(j,t,originals,offset)>=2 for j in range(17)] for t in times])
        for kind, filename in [('discrete','trajectories.csv'),('dct','dct.csv')]:
            key = name+'_'+kind
            c = curves(result_dir(base,name)/filename)
            xyz = np.array([interpolate(c[j], times) for j in range(17)]).transpose(1,0,2)
            valid = np.isfinite(xyz).all(axis=-1)
            predicted = np.full((len(FRAMES),17,2),np.nan)
            projected, depth = project(xyz[valid],7)
            predicted[valid] = projected
            good_depth = np.zeros_like(valid); good_depth[valid] = depth>0
            mask = Z['valid'][held] & valid & good_depth & local_support
            predictions[key], points[key], masks[key] = predicted, xyz, mask
            details[key] = dict(available=int(mask.sum()), negative_depth=int(np.sum(valid & ~good_depth)),
                                insufficient_local_source_support=int(np.sum(Z['valid'][held] & ~local_support)))
    common = np.logical_and.reduce(list(masks.values()))
    observed = Z['xy_full_distorted'][held]
    result = {}
    for key, predicted in predictions.items():
        error = np.linalg.norm(predicted-observed,axis=-1)
        result[key] = stats(error[common])
        result[key]['per_joint'] = {NAMES[j]:stats(error[:,j][common[:,j]]) for j in range(17)}
    np.savez_compressed(base/'predictions.npz',frame=FRAMES, time_s=times, observed=observed, common=common,
                        **predictions, **{'xyz_'+k:v for k,v in points.items()})
    return dict(camera_original_id=7,eligible_observations=int(Z['valid'][held].sum()),
        common_observations=int(common.sum()), common_fraction=float(common.sum()/Z['valid'][held].sum()),
        error_units='Original 3840x2160 pixels; divide by 2 for displayed 1920x1080 images',
        reference='No camera07 observations enter either time estimation or trajectory optimization. Its clock stays at published sync.',
        interpolation='Linear interpolation of discrete or exported DCT samples at held-out times; no extrapolation; >=2 locally supported source views',
        per_method_coverage=details, metrics=result)


def summary_plot(summary):
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    axes[0,0].bar(['Fixed official sync','STBA'],[summary['all8'][n]['distorted_full_resolution_px']['rmse'] for n in LABELS],
                  color=list(COLORS.values()))
    axes[0,0].set(title='All 8 views: fitted residual (not a test)',ylabel='RMSE, original-resolution px')
    metric = summary['heldout']['metrics']
    x = np.arange(3)
    for i,name in enumerate(LABELS):
        axes[0,1].bar(x+(i-.5)*.32,[metric[name+'_discrete'][k] for k in ['rmse','median','p95']],.32,
                      label=LABELS[name],color=COLORS[name])
    axes[0,1].set_xticks(x,['RMSE','Median','P95'])
    axes[0,1].set(title=f'Camera07 held out: {summary["heldout"]["common_observations"]} common points',ylabel='Original-resolution px')
    axes[0,1].legend(fontsize=8)
    for name in LABELS:
        axes[1,0].plot(IDS,np.array(summary['all8'][name]['offset_seconds'])*1000,'o-',color=COLORS[name],label=LABELS[name])
    axes[1,0].set(title='Offsets relative to cam04 (official sync is not GT)',xlabel='Original camera ID',ylabel='Subtract from frame/60, ms')
    axes[1,0].set_xticks(IDS); axes[1,0].legend(fontsize=8)
    for name in LABELS:
        axes[1,1].plot(range(17),[metric[name+'_discrete']['per_joint'][j].get('rmse',np.nan) for j in NAMES],
                       '.-',color=COLORS[name],label=LABELS[name])
    axes[1,1].set(title='Held-out error by joint',ylabel='RMSE, original-resolution px')
    axes[1,1].set_xticks(range(17),NAMES,rotation=70,fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Corgi [200,260): official camera calibration fixed throughout',fontsize=14)
    fig.tight_layout();fig.savefig(PLOTS/'summary.png',dpi=170);plt.close(fig)


def skeletons():
    times = FRAMES/FPS
    all_xyz, all_discrete = [], []
    for name in LABELS:
        c = curves(result_dir(HERE,name)/'dct.csv');d = curves(result_dir(HERE,name)/'trajectories.csv')
        all_xyz.append(np.array([interpolate(c[j],times) for j in range(17)]).transpose(1,0,2))
        all_discrete.append(np.array([interpolate(d[j],times) for j in range(17)]).transpose(1,0,2))
    xyz = np.array(all_xyz); discrete = np.array(all_discrete)
    np.savez_compressed(HERE/'reconstruction.npz',frame=FRAMES,time_s=times,methods=list(LABELS),
                        keypoint_names=NAMES,xyz_dct=xyz,xyz_discrete=discrete,reference_camera=4)
    with (HERE/'reconstruction.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['method','kind','reference_camera','frame','time_s','track_id','name','x','y','z','valid'])
        for m,name in enumerate(LABELS):
            for kind,data in [('discrete',discrete),('dct',xyz)]:
                for i,f in enumerate(FRAMES):
                    for j in range(17):
                        point=data[m,i,j]
                        w.writerow([name,kind,4,f,times[i],j,NAMES[j],*point,int(np.isfinite(point).all())])
    # Display basis only; exported world coordinates are untouched.
    display = xyz[:,:,:,[0,2,1]].copy();display[:,:,:,2]*=-1
    values=display[np.isfinite(display).all(axis=-1)]
    center=(values.min(axis=0)+values.max(axis=0))/2
    radius=np.ptp(values,axis=0).max()*.55
    fig=plt.figure(figsize=(11,5.6))
    axes=[fig.add_subplot(1,2,i+1,projection='3d') for i in range(2)]
    def draw(i):
        for m,(name,ax) in enumerate(zip(LABELS,axes)):
            ax.clear();p=display[m,i];valid=np.isfinite(p).all(axis=1)
            for a,b in LINKS:
                if valid[a] and valid[b]:
                    ax.plot(*p[[a,b]].T,color=COLORS[name],lw=1.7)
            ax.scatter(*p[valid].T,color=COLORS[name],s=4)
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
                   zlim=(center[2]-radius,center[2]+radius),xlabel='World X',ylabel='World Z',zlabel='-World Y',title=LABELS[name])
            ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=30)
        fig.suptitle(f'Corgi reference-camera frame {FRAMES[i]} | 17 DCT joints | fixed calibration')
    fig.text(.5,.03,'Native SelfCap world units; display axes X,Z,-Y. Sparse reconstruction, no independent 3D ground truth.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.03,right=.97,bottom=.12,top=.87,wspace=.02)
    for frame in [200,230,259]:
        draw(frame-START);fig.savefig(PLOTS/f'skeleton_{frame}.png',dpi=160)
    writer=FFMpegWriter(fps=15,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','22'])
    with writer.saving(fig,str(PLOTS/'skeleton.mp4'),dpi=100):
        for i in range(len(FRAMES)):
            draw(i);writer.grab_frame()
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True)
    for row,joint in enumerate([2,7,13]):
        for axis in range(3):
            ax=axes[row,axis]
            for m,name in enumerate(LABELS):
                ax.plot(FRAMES,xyz[m,:,joint,axis],color=COLORS[name],label=LABELS[name]+' DCT',lw=1.6)
            curve=curves(result_dir(HERE,'result')/'trajectories.csv')[joint]
            use=(curve[:,0]>=times[0])&(curve[:,0]<=times[-1])
            ax.scatter(curve[use,0]*FPS,curve[use,axis+1],s=1.2,alpha=.28,c='#268ac4',label='STBA discrete')
            ax.set_title(NAMES[joint]+' / '+'XYZ'[axis]);ax.grid(alpha=.2)
            if row==2:ax.set_xlabel('Frame on camera04 reference clock')
            if axis==0:ax.set_ylabel('Native world-coordinate unit')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Discrete STBA samples and DCT curves; fixed-sync baseline is a comparison, not GT')
    fig.tight_layout();fig.savefig(PLOTS/'trajectories.png',dpi=170);plt.close(fig)


def overlay_video():
    p=dict(np.load(HERE/'holdout07/predictions.npz'))
    # Use one constant image crop for every frame and both methods.
    cloud=(Z['xy_full_distorted'][IDS.index(7)][Z['valid'][IDS.index(7)]]-.5)/2
    x0,y0=np.maximum(np.floor(cloud.min(axis=0)-70),0).astype(int)
    x1,y1=np.minimum(np.ceil(cloud.max(axis=0)+70),[1920,1080]).astype(int)
    size=(600,560)
    def render(index):
        frame=int(FRAMES[index]);raw=cv2.imread(str(HERE/f'frames/cam07/{frame:04d}.png'))
        assert raw is not None
        mask=p['common'][index];joints=np.flatnonzero(mask)
        observed=(p['observed'][index]-.5)/2
        panels=[]
        for name in LABELS:
            im=raw.copy();pred=(p[name+'_discrete'][index]-.5)/2
            for a,b in LINKS:
                if mask[a] and mask[b]:
                    cv2.line(im,tuple(np.rint(pred[a]).astype(int)),tuple(np.rint(pred[b]).astype(int)),(200,50,220),1,cv2.LINE_AA)
            for j in joints:
                a=tuple(np.rint(observed[j]).astype(int));b=tuple(np.rint(pred[j]).astype(int))
                cv2.line(im,a,b,(230,230,230),1,cv2.LINE_AA)
                cv2.circle(im,a,2,(50,235,40),-1,cv2.LINE_AA)
                cv2.drawMarker(im,b,(200,50,220),cv2.MARKER_CROSS,7,1,cv2.LINE_AA)
            crop=im[y0:y1,x0:x1]
            scale=min(580/crop.shape[1],450/crop.shape[0])
            patch=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            canvas=np.full((size[1],size[0],3),24,np.uint8)
            yy=75+(450-patch.shape[0])//2;xx=(600-patch.shape[1])//2
            canvas[yy:yy+patch.shape[0],xx:xx+patch.shape[1]]=patch
            rmse=np.sqrt(np.mean(np.sum((p[name+'_discrete'][index,mask]-p['observed'][index,mask])**2,axis=1))) if mask.any() else np.nan
            for title,y,scale_text in [(LABELS[name],28,.7),(f'Cam07 held out | frame {frame} | RMSE {rmse:.2f} full-res px',56,.48),
                                       ('Green: detected 2D | Magenta: projected 3D',548,.52)]:
                cv2.putText(canvas,title,(12,y),cv2.FONT_HERSHEY_SIMPLEX,scale_text,(245,245,245),1,cv2.LINE_AA)
            panels.append(canvas)
        return np.hstack(panels)
    rawvideo=PLOTS/'heldout07_raw.mp4'
    writer=cv2.VideoWriter(str(rawvideo),cv2.VideoWriter_fourcc(*'mp4v'),15,(1200,560))
    assert writer.isOpened()
    for i in range(len(FRAMES)):
        frame=render(i);writer.write(frame)
        if int(FRAMES[i]) in [200,230,259]:
            cv2.imwrite(str(PLOTS/f'heldout07_{FRAMES[i]}.jpg'),frame)
    writer.release()
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(rawvideo),
                    '-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(PLOTS/'heldout07.mp4')],check=True)
    rawvideo.unlink()
    save(PLOTS/'display_metadata.json',dict(video_fps=15,original_fps=60,playback='quarter speed, all 60 frames',
        crop_half_resolution=[int(x0),int(y0),int(x1),int(y1)],
        metric_units='Full-resolution 3840x2160 pixels; image displayed from 1920x1080 source',
        legend='Green detected 2D, magenta projected discrete 3D from seven other views; no camera07 fitting',
        skeleton='17 DCT joints at original cam04 frame/60 times; X,Z,-Y display basis; output XYZ unchanged'))


def report(summary):
    h=summary['heldout'];m=h['metrics'];a=summary['all8']
    lines=['# Corgi 200–260 帧：固定相机内外参 STBA 实验','',
        '- 窗口 **[200,260)**，原始 60 FPS，200–259 共 60 帧；不使用 30 FPS 衍生视频。',
        '- 使用原相机 **04、07、10、13、16、18、21、23**，狗的 **17 个 APT36K 关节点**；有效二维观测 **6807/8160**。',
        '- 直接沿用官方原始分辨率 K、Rodrigues R、T 和 D。输入与全部输出 K/R/T 数值完全相等；D 用于去畸变及回投影，不参与优化。',
        '- 官方 YAML 的 Rot 矩阵与其 R 旋转向量存在约 3e-8 的舍入差；采用官方 R 原值，不重新拟合旋转。',
        '- 初始时间来自官方 sync.json，参考 cam04；full 模式估计时间与轨迹，fixed 基线只估轨迹。',
        '- λ=4000、Huber=6 原分辨率 px，等效于先前半分辨率 λ=1000、Huber=3 px；参数未按本次误差挑选。',
        '- 这是稀疏骨架/轨迹实验，不是动态稠密点云重建。没有独立三维或曝光时间 GT，不能报告毫米精度。','',
        '## 数值结果','',
        '**以下误差均为原始 3840×2160 像素坐标；换算成 1920×1080 展示图像误差需除以 2。**','',
        '| 评价 | 固定官方同步 | STBA 优化同步 |','|---|---:|---:|',
        f'| 全 8 路拟合 RMSE（带畸变回投影，px） | {a["baseline"]["distorted_full_resolution_px"]["rmse"]:.3f} | {a["result"]["distorted_full_resolution_px"]["rmse"]:.3f} |',
        f'| 留出 cam07 RMSE（px） | {m["baseline_discrete"]["rmse"]:.3f} | {m["result_discrete"]["rmse"]:.3f} |',
        f'| 留出 cam07 中位数（px） | {m["baseline_discrete"]["median"]:.3f} | {m["result_discrete"]["median"]:.3f} |',
        f'| 留出 cam07 P95（px） | {m["baseline_discrete"]["p95"]:.3f} | {m["result_discrete"]["p95"]:.3f} |',
        f'| 留出 cam07 DCT RMSE（px） | {m["baseline_dct"]["rmse"]:.3f} | {m["result_dct"]["rmse"]:.3f} |','',
        f'留出评价采用同一批 **{h["common_observations"]}/{h["eligible_observations"]}** 个有效目标。cam07 的二维观测从时间估计与轨迹拟合中完全排除；其评价时钟固定为官方 sync。要求目标时刻至少两路连续有效源观测，并且不外推轨迹。DCT 误差含导出网格线性插值误差。','',
        '这能检验对留出视角二维检测的预测，但检测器也可能错配左右肢体或预测遮挡点。既有相机选择参考过第230帧可见性，同一时间窗用于拟合及留出视角检查，不能称为盲测、独立三维精度或跨时间泛化验证。','',
        '## 优化状态','',
        '初次全局优化均达到200步上限，原结果保存在 `result/`。仅按未收敛状态对两组均从已保存XYZ/时间偏移续算最多1000步；使用当前根目录同一 `libstba_core.a`，不重跑成对搜索、不更改目标或相机参数。最终采用 `result_refined/`，不按留出误差挑选迭代结果。','']
    for group,key in [('8路重建','all8'),('7路拟合、留出07','holdout_fit')]:
        for method in LABELS:
            lines += [f'- {group} / {LABELS[method]}：`{summary[key][method]["solver_report"].splitlines()[-1]}`']
    lines += ['', '## 8路估计时间','', '| 原相机 | 官方相对修正 ms | STBA 修正 ms |', '|---|---:|---:|']
    for i,c in enumerate(IDS):
        lines.append(f'| {c:02d} | {a["baseline"]["offset_seconds"][i]*1000:+.3f} | {a["result"]["offset_seconds"][i]*1000:+.3f} |')
    lines += ['', '公式为 `t=frame/60-offset`。与官方修正的差不能称为真实同步误差。', '',
        '## 查看与复现','',
        '- [三维骨架图](plots/skeleton_230.png)、[三维骨架视频](plots/skeleton.mp4)、[STBA/DCT轨迹图](plots/trajectories.png)。',
        '- [留出07对照图](plots/heldout07_230.jpg)、[留出07对照视频](plots/heldout07.mp4)、[数值概览](plots/summary.png)。视频均按15 FPS展示，原60 FPS片段为四分之一速度。',
        f'- [输入](input/)、[STBA输出]({result_dir(HERE,"result").name}/)、[固定同步基线](baseline/)、[统一参考时刻的三维CSV](reconstruction.csv)、[NPZ](reconstruction.npz)。',
        '- `frames/` 为所选480张半分辨率原图的只读来源链接；没有覆盖或修改数据源。`frame_manifest.csv` 保留来源路径。',
        '- `summary.json`、`holdout07/predictions.npz`、`*.command.json`、日志和标定校验文件保留数值及实际运行命令。',
        '- 使用仓库根目录当前 `build/stba`，非旧实验副本。`manifest.json` 记录二进制和输入来源 SHA256。',
        '- 用 `tmp/walking_stba/venv/bin/python run.py prepare` 提取，再分别 `run.py solve all8`、`run.py solve holdout07`；构建 `refine/` 后分别 `run.py continue all8`、`run.py continue holdout07`，最后 `analyze.py`。实际路径/参数见 command JSON。提取与求解拒绝覆盖已有结果。', '',
        '![结果概览](plots/summary.png)', '']
    (HERE/'README.md').write_text('\n'.join(lines))


def main():
    PLOTS.mkdir(exist_ok=True)
    manifest=json.loads((HERE/'manifest.json').read_text())
    assert all(sha(p)==digest for p,digest in manifest['source_sha256'].items())
    summary=dict(all8=audit_fit(HERE),holdout_fit=audit_fit(HERE/'holdout07'))
    summary['heldout']=heldout()
    summary['source_files_unchanged']=True
    summary['independent_ground_truth']=False
    save(HERE/'summary.json',summary)
    print(json.dumps(summary['heldout']['metrics'],indent=2),flush=True)
    summary_plot(summary)
    skeletons()
    overlay_video()
    report(summary)
    print('Artifacts saved:',HERE,flush=True)


if __name__=='__main__':
    main()
