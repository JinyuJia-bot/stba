#include "core/stba.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace stba {
namespace {

// Use the same set of observations for every candidate, so costs are comparable.
std::size_t CommonObservationCount(const Dataset &data, int a, int b, int &common_tracks,
                                   int &dynamic_tracks) {
    std::size_t count = 0;
    for (const auto &track : data.tracks) {
        bool sees_a = false, sees_b = false;
        std::size_t observations = 0;
        for (const auto &obs : track.observations) {
            sees_a = sees_a || obs.camera == a;
            sees_b = sees_b || obs.camera == b;
            if (obs.camera == a || obs.camera == b)
                ++observations;
        }
        if (sees_a && sees_b) {
            ++common_tracks;
            if (!track.is_static)
                ++dynamic_tracks;
            count += observations;
        }
    }
    return count;
}

std::size_t UsedObservationCount(const Solution &solution) {
    std::size_t count = 0;
    for (const auto &trajectory : solution.trajectories)
        for (const auto &sample : trajectory.samples)
            count += sample.observations.size();
    return count;
}

double AverageBaseline(const Dataset &data, int a, int b) {
    std::vector<const View *> first, second;
    for (const auto &view : data.views) {
        if (view.camera == a)
            first.push_back(&view);
        if (view.camera == b)
            second.push_back(&view);
    }
    if (first.empty() || second.empty())
        return 0;
    const auto &camera_a = data.cameras[static_cast<std::size_t>(a)];
    const auto &camera_b = data.cameras[static_cast<std::size_t>(b)];
    double total = 0;
    for (const View *left : first) {
        const View *right =
            *std::min_element(second.begin(), second.end(), [&](const View *x, const View *y) {
                const double t =
                    (static_cast<double>(std::max(left->frame, 0)) - camera_a.beta) / camera_a.fps;
                return std::abs((static_cast<double>(std::max(x->frame, 0)) - camera_b.beta) /
                                    camera_b.fps -
                                t) <
                       std::abs((static_cast<double>(std::max(y->frame, 0)) - camera_b.beta) /
                                    camera_b.fps -
                                t);
            });
        total += (CameraCenter(*left) - CameraCenter(*right)).norm();
    }
    return total / static_cast<double>(first.size());
}

} // namespace

PairAlignment AlignPair(const Dataset &data, int a, int b, const Options &options) {
    if (a < 0 || b < 0 || a == b || static_cast<std::size_t>(std::max(a, b)) >= data.cameras.size())
        throw std::invalid_argument("AlignPair: invalid camera ids");
    if (!std::isfinite(options.search_radius) || options.search_radius < 0 ||
        !std::isfinite(options.coarse_step) || options.coarse_step <= 0 ||
        !std::isfinite(options.fine_step) || options.fine_step <= 0 ||
        options.fine_step > options.coarse_step)
        throw std::invalid_argument("Invalid pairwise search radius or step");

    PairAlignment result;
    result.a = a;
    result.b = b;
    result.cost = std::numeric_limits<double>::infinity();
    int dynamic_tracks = 0;
    const std::size_t expected_count =
        CommonObservationCount(data, a, b, result.common_tracks, dynamic_tracks);
    if (!dynamic_tracks)
        return result; // Static points alone cannot align clocks.
    result.baseline = AverageBaseline(data, a, b);
    if (result.baseline <= 1e-8)
        return result;

    Options fixed = options;
    fixed.optimize_offsets = false;
    fixed.optimize_poses = false;
    fixed.optimize_intrinsics = false;
    fixed.reference_camera = a;
    const std::size_t bi = static_cast<std::size_t>(b);
    const std::size_t ai = static_cast<std::size_t>(a);
    const double center = data.cameras[bi].beta;
    double best_beta = center;
    auto evaluate = [&](double beta) {
        Dataset candidate = data;
        candidate.cameras[bi].beta = beta;
        try {
            const Solution solution = Optimize(candidate, {a, b}, fixed);
            if (solution.usable && solution.order_valid &&
                UsedObservationCount(solution) == expected_count && solution.cost < result.cost) {
                result.valid = true;
                result.cost = solution.cost;
                best_beta = beta;
            }
        } catch (const std::runtime_error &) {
            // A particular offset may have no valid positive-depth initialization.
        }
    };
    auto sweep = [&](double lower, double upper, double step) {
        const double intervals = std::ceil((upper - lower) / step);
        if (intervals > 10000)
            throw std::invalid_argument("Pairwise search exceeds 10000 intervals; increase step");
        const int count = static_cast<int>(intervals);
        for (int i = 0; i <= count; ++i)
            evaluate(std::min(upper, lower + static_cast<double>(i) * step));
    };
    // Include the supplied initial estimate even when the grid does not land on it.
    evaluate(center);
    sweep(center - options.search_radius, center + options.search_radius, options.coarse_step);
    if (!result.valid)
        return result;
    const double lower = std::max(center - options.search_radius, best_beta - options.coarse_step);
    const double upper = std::min(center + options.search_radius, best_beta + options.coarse_step);
    sweep(lower, upper, options.fine_step);
    // Clock offset convention is beta_b/fps_b - beta_a/fps_a, in seconds.
    result.delta_seconds =
        best_beta / data.cameras[bi].fps - data.cameras[ai].beta / data.cameras[ai].fps;
    return result;
}

} // namespace stba
