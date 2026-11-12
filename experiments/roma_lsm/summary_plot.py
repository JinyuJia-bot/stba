"""Compact comparison built only from saved evaluation tables."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def main():
    cases=['translation_0_1','affine_brightness_0_1','affine_brightness_0_2','occlusion_weak_noise_0_1']
    labels=['Translation','Affine + brightness\ntarget 1','Affine + brightness\ntarget 2','Noise / weak texture\n/ occlusion']
    methods=['A0','A1_affine','A2_affine'];names=['RoMa (shared quality gates)','+ photometric affine LSM','+ soft epipolar geometry']
    colors=['#4C78A8','#F58518','#54A24B'];x=np.arange(4)
    reports=[json.loads((ROOT/f'results/synthetic_{c}/metrics.json').read_text()) for c in cases]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for i,(method,name,color) in enumerate(zip(methods,names,colors)):
        means=[r['methods'][method]['scopes']['common_primary']['endpoint_error_px']['mean'] for r in reports]
        coverage=[100*r['methods'][method]['scopes']['own']['endpoint_error_px']['correct_coverage_lt_1_px'] for r in reports]
        for ax,values in zip(axes,[means,coverage]):
            bars=ax.bar(x+(i-1)*.24,values,.24,color=color,label=name)
            ax.bar_label(bars,fmt='%.3f' if ax==axes[0] else '%.1f',fontsize=7,padding=2)
    axes[0].set_ylabel('Mean endpoint error [working pixels]');axes[0].set_ylim(0,.065)
    axes[0].set_title('Accuracy on IDENTICAL valid pixels within each case')
    axes[1].set_ylabel('Correct matches / original GT-evaluable ROI [%]');axes[1].set_ylim(0,108)
    axes[1].set_title('Correct-match coverage (<1px), method-specific support')
    for ax in axes:
        ax.set_xticks(x,labels,fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Actual dense RoMa predictions, then stride-1 LSM; fixed calibrated cameras\nPhotometric fitting is not guaranteed to improve true correspondence accuracy',fontsize=12)
    fig.savefig(ROOT/'results/summary.png',dpi=180,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
