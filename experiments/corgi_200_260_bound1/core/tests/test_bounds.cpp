#include "core/stba.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace stba;
void Check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
Dataset Fixture(int n, double reference_beta=0.0) {
    Dataset d;
    for (int c=0;c<n;++c) {
        d.cameras.push_back({30, c==0?reference_beta:(c==1?.3:-.2)});
        View v;v.camera=c;v.translation={c==0?0.0:(c==1?-1.2:1.2),0,0};
        d.views.push_back(v);
    }
    for (int j=0;j<3;++j) {
        Track track;track.id=j;
        for (int c=0;c<n;++c)
            for (int f=0;f<16;++f) {
                const double true_beta=c==0?reference_beta:(c==1?.82:-.74);
                double t=(f-true_beta)/30;
                Vec3 x(.5*std::sin(8*t+j),.4*std::cos(9*t+.3*j),5+.2*std::sin(7*t+.8*j));
                track.observations.push_back({c,f,c,Project(d.views[c],x),1});
            }
        d.tracks.push_back(track);
    }
    return d;
}
void CheckResult(const Solution &s,double bound) {
    Check(s.usable && s.order_valid,"Bounded reconstruction must be usable and ordered");
    for(const auto &c:s.data.cameras)Check(std::abs(c.beta)<=bound,"Final camera outside box");
    std::size_t count=0;
    for(const auto &t:s.trajectories)for(const auto &node:t.samples)count+=node.observations.size();
    Check(count==s.data.cameras.size()*3*16,"Bounds must not discard observations");
}
int main() {
    try {
        Options o;o.motion_weight=100;o.max_iterations=200;
        auto d=Fixture(2);
        auto free=Optimize(d,{0,1},o);
        Check(free.usable && free.data.cameras[1].beta>.5,"Fixture must prefer an offset beyond the test box");
        o.offset_bound_frames=.5;
        auto bounded=Optimize(d,{0,1},o);
        CheckResult(bounded,.5);
        Check(std::abs(bounded.data.cameras[1].beta-.5)<1e-6,"Ceres must reach the active hard boundary");
        Check(bounded.cost>=free.cost-1e-5,"Constrained local optimum cannot beat converged unconstrained fixture");
        d.cameras[1].beta=.7;
        bool rejected=false;
        try { (void)Optimize(d,{0,1},o); } catch(const std::runtime_error &) { rejected=true; }
        Check(rejected,"Out-of-box warm input must be rejected, never silently clipped as a result");
        o.search_radius=1.5;o.coarse_step=.25;o.fine_step=.05;
        auto full=Reconstruct(Fixture(3),o,true);
        CheckResult(full.incremental,.5);CheckResult(full.global,.5);
        for(const auto &trial:full.trials)Check(std::abs(trial.beta)<=.5,"Incremental candidate escaped box");
        for(const auto &view:full.global.data.views) {
            const auto original=Fixture(3).views[view.camera];
            Check(view.intrinsics==original.intrinsics && view.rotation==original.rotation &&
                  view.translation==original.translation,"Calibration changed");
        }
        bool gauge_rejected=false;
        try { (void)Reconstruct(Fixture(2,.1),o,true); } catch(const std::invalid_argument &) {gauge_rejected=true;}
        Check(gauge_rejected,"Nonzero global reference must be rejected by temporary absolute box implementation");
        std::cout<<"free_beta="<<free.data.cameras[1].beta<<" bounded_beta="<<bounded.data.cameras[1].beta
                 <<" full_order=";
        for(int c:full.order)std::cout<<c<<',';
        std::cout<<"\nHard-bound tests passed\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
