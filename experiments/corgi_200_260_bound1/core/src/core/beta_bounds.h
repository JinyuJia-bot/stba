#pragma once
#include "core/stba.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

namespace stba {
inline bool Bounded(const Options &o) {
    if (!std::isfinite(o.offset_bound_frames) || o.offset_bound_frames < 0)
        throw std::invalid_argument("offset_bound_frames must be finite and nonnegative");
    return o.offset_bound_frames > 0;
}
inline bool BetaInside(double beta, const Options &o) {
    return !Bounded(o) || (std::isfinite(beta) && std::abs(beta) <= o.offset_bound_frames);
}
inline double ProjectBeta(double beta, const Options &o) {
    return Bounded(o) ? std::clamp(beta, -o.offset_bound_frames, o.offset_bound_frames) : beta;
}
inline void CheckBetaBounds(const Dataset &d, const std::vector<int> &active,
                            const Options &o, const char *stage, int iteration = -1) {
    if (!Bounded(o)) return;
    double lower = o.offset_bound_frames, upper = -o.offset_bound_frames;
    for (int c : active) {
        const double beta = d.cameras.at(c).beta;
        if (!BetaInside(beta, o))
            throw std::runtime_error(std::string(stage) + ": residual beta outside hard box");
        lower = std::min(lower, beta);
        upper = std::max(upper, beta);
    }
    if (const char *path = std::getenv("STBA_BOUND_AUDIT_PATH")) {
        std::ofstream file(path, std::ios::app);
        if (!file) throw std::runtime_error("Cannot write bound audit file");
        file << std::setprecision(17) << stage << ',' << iteration << ',' << active.size()
             << ',' << lower << ',' << upper << '\n';
    }
}
inline std::vector<int> AllCameras(const Dataset &d) {
    std::vector<int> active(d.cameras.size());
    std::iota(active.begin(), active.end(), 0);
    return active;
}
// Feasible INITIALIZATION only. Never clamp an optimized solution after the solve.
inline void ProjectGraphInitialization(Dataset &d, const Options &o) {
    if (!Bounded(o)) return;
    for (auto &c : d.cameras) c.beta = ProjectBeta(c.beta, o);
    CheckBetaBounds(d, AllCameras(d), o, "graph_initialization");
}
} // namespace stba
