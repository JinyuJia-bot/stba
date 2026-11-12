#include "core/stba.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace stba {
namespace {

bool EdgeLess(const PairAlignment &a, const PairAlignment &b) {
    if (a.edge_weight != b.edge_weight)
        return a.edge_weight < b.edge_weight;
    if (a.a != b.a)
        return a.a < b.a;
    return a.b < b.b;
}

// These few lines of union-find are all the graph infrastructure needed here.
std::vector<std::size_t> MinimumTree(int cameras, const std::vector<PairAlignment> &pairs) {
    std::vector<std::size_t> sorted;
    for (std::size_t i = 0; i < pairs.size(); ++i)
        if (pairs[i].valid && std::isfinite(pairs[i].edge_weight))
            sorted.push_back(i);
    std::stable_sort(sorted.begin(), sorted.end(),
                     [&](std::size_t a, std::size_t b) { return EdgeLess(pairs[a], pairs[b]); });
    std::vector<int> parent(static_cast<std::size_t>(cameras));
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) {
        while (parent[static_cast<std::size_t>(x)] != x) {
            parent[static_cast<std::size_t>(x)] =
                parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
            x = parent[static_cast<std::size_t>(x)];
        }
        return x;
    };
    std::vector<std::size_t> tree;
    for (std::size_t edge : sorted) {
        const int a = find(pairs[edge].a), b = find(pairs[edge].b);
        if (a == b)
            continue;
        parent[static_cast<std::size_t>(a)] = b;
        tree.push_back(edge);
    }
    if (tree.size() + 1 != static_cast<std::size_t>(cameras))
        throw std::runtime_error("Camera graph is disconnected: check shared dynamic tracks, "
                                 "calibration and offset search range");
    return tree;
}

double OrientedOffset(const PairAlignment &edge, int from) {
    return from == edge.a ? edge.delta_seconds : -edge.delta_seconds;
}

void NormalizeReference(Dataset &data, int reference, double target_beta) {
    const auto &ref = data.cameras[static_cast<std::size_t>(reference)];
    const double shift_seconds = (target_beta - ref.beta) / ref.fps;
    for (auto &camera : data.cameras)
        camera.beta += camera.fps * shift_seconds;
}

// Every boundary is an actual observation-time coincidence, also for unequal fps.
// Within one open interval, the adjacency graph of the motion prior is unchanged.
std::vector<double> SlotBoundaries(const Dataset &data, const std::vector<int> &active, int added,
                                   double lower, double upper) {
    std::vector<double> boundaries{lower, upper};
    const double fps = data.cameras[static_cast<std::size_t>(added)].fps;
    // Optimize constrains the union of dynamic camera/frame events, including
    // observations on different tracks. Enumerate that same union here: using
    // only shared-track coincidences would miss cells under sparse visibility.
    std::vector<std::pair<int, int>> active_events;
    std::vector<int> added_frames;
    for (const auto &track : data.tracks) {
        if (track.is_static)
            continue;
        for (const auto &obs : track.observations) {
            if (std::find(active.begin(), active.end(), obs.camera) != active.end())
                active_events.emplace_back(obs.camera, obs.frame);
            if (obs.camera == added)
                added_frames.push_back(obs.frame);
        }
    }
    std::sort(active_events.begin(), active_events.end());
    active_events.erase(std::unique(active_events.begin(), active_events.end()),
                        active_events.end());
    std::sort(added_frames.begin(), added_frames.end());
    added_frames.erase(std::unique(added_frames.begin(), added_frames.end()), added_frames.end());
    std::vector<double> existing;
    for (const auto &[camera_id, frame] : active_events) {
        const auto &camera = data.cameras[static_cast<std::size_t>(camera_id)];
        existing.push_back((static_cast<double>(frame) - camera.beta) / camera.fps);
    }
    std::sort(existing.begin(), existing.end());
    for (int frame : added_frames) {
        const double min_time = (static_cast<double>(frame) - upper) / fps;
        const double max_time = (static_cast<double>(frame) - lower) / fps;
        auto it = std::lower_bound(existing.begin(), existing.end(), min_time);
        for (; it != existing.end() && *it <= max_time; ++it)
            boundaries.push_back(static_cast<double>(frame) - fps * *it);
    }
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end(),
                                 [](double a, double b) { return std::abs(a - b) < 1e-8; }),
                     boundaries.end());
    if (boundaries.size() > 10000)
        throw std::runtime_error(
            "More than 10000 temporal slots; shorten the clip or reduce search_radius");
    return boundaries;
}

} // namespace

std::vector<int> CameraOrder(const Dataset &data, std::vector<PairAlignment> &pairs) {
    const int count = static_cast<int>(data.cameras.size());
    if (count < 2)
        throw std::invalid_argument("STBA needs at least two cameras");
    std::vector<std::vector<double>> offsets(
        static_cast<std::size_t>(count),
        std::vector<double>(static_cast<std::size_t>(count),
                            std::numeric_limits<double>::quiet_NaN()));
    for (const auto &edge : pairs)
        if (edge.valid) {
            offsets[static_cast<std::size_t>(edge.a)][static_cast<std::size_t>(edge.b)] =
                edge.delta_seconds;
            offsets[static_cast<std::size_t>(edge.b)][static_cast<std::size_t>(edge.a)] =
                -edge.delta_seconds;
        }
    for (auto &edge : pairs) {
        if (!edge.valid)
            continue;
        double cycle_error = 0;
        int triangles = 0;
        for (int k = 0; k < count; ++k) {
            if (k == edge.a || k == edge.b)
                continue;
            const double jk =
                offsets[static_cast<std::size_t>(edge.b)][static_cast<std::size_t>(k)];
            const double ik =
                offsets[static_cast<std::size_t>(edge.a)][static_cast<std::size_t>(k)];
            if (std::isfinite(jk) && std::isfinite(ik)) {
                cycle_error += std::abs(edge.delta_seconds + jk - ik);
                ++triangles;
            }
        }
        // Eq. (6). A sparse graph with no triangles has no cycle statistic;
        // use one frame's duration as an explicit fallback, rather than a zero edge.
        if (!triangles)
            cycle_error = 1.0 / data.cameras[static_cast<std::size_t>(edge.b)].fps;
        edge.edge_weight =
            edge.cost * cycle_error / (static_cast<double>(edge.common_tracks) * edge.baseline);
    }
    const auto tree = MinimumTree(count, pairs);
    const auto &seed = pairs[tree.front()];
    std::vector<int> order{seed.a, seed.b};
    while (order.size() < data.cameras.size()) {
        bool found = false;
        // Tree edges retain Kruskal cost order. Always attach to the current component.
        for (std::size_t index : tree) {
            const auto &edge = pairs[index];
            const bool has_a = std::find(order.begin(), order.end(), edge.a) != order.end();
            const bool has_b = std::find(order.begin(), order.end(), edge.b) != order.end();
            if (has_a == has_b)
                continue;
            order.push_back(has_a ? edge.b : edge.a);
            found = true;
            break;
        }
        if (!found)
            throw std::runtime_error("Cannot traverse camera spanning tree");
    }
    return order;
}

PipelineResult Reconstruct(const Dataset &input, const Options &options, bool refine_global) {
    Validate(input);
    if (options.reference_camera < 0 ||
        static_cast<std::size_t>(options.reference_camera) >= input.cameras.size())
        throw std::invalid_argument("Invalid reference_camera");
    PipelineResult result;
    const int count = static_cast<int>(input.cameras.size());
    for (int a = 0; a < count; ++a)
        for (int b = a + 1; b < count; ++b)
            result.pairs.push_back(AlignPair(input, a, b, options));
    result.order = CameraOrder(input, result.pairs);
    const auto tree = MinimumTree(count, result.pairs);
    Dataset data = input;
    std::vector<int> processed{result.order.front()};
    // Propagate pair estimates through the tree to create a common time gauge.
    for (std::size_t i = 1; i < result.order.size(); ++i) {
        const int added = result.order[i];
        for (std::size_t index : tree) {
            const auto &edge = result.pairs[index];
            if (edge.a != added && edge.b != added)
                continue;
            const int parent = edge.a == added ? edge.b : edge.a;
            if (std::find(processed.begin(), processed.end(), parent) == processed.end())
                continue;
            auto &camera = data.cameras[static_cast<std::size_t>(added)];
            const auto &base = data.cameras[static_cast<std::size_t>(parent)];
            camera.beta = camera.fps * (base.beta / base.fps + OrientedOffset(edge, parent));
            break;
        }
        processed.push_back(added);
    }
    const double target_beta =
        input.cameras[static_cast<std::size_t>(options.reference_camera)].beta;
    NormalizeReference(data, options.reference_camera, target_beta);
    Options fixed = options;
    fixed.optimize_offsets = false;
    fixed.optimize_poses = false;
    fixed.optimize_intrinsics = false;
    std::vector<int> active{result.order[0], result.order[1]};
    Solution current = Optimize(data, active, fixed);
    if (!current.usable || !current.order_valid)
        throw std::runtime_error("Initial camera pair optimization failed: " + current.report);

    Options incremental = options;
    incremental.optimize_poses = false;
    incremental.optimize_intrinsics = false;
    for (std::size_t next = 2; next < result.order.size(); ++next) {
        const int added = result.order[next];
        data = current.data;
        // Update the next camera's estimate using the current optimized parent clock.
        for (std::size_t index : tree) {
            const auto &edge = result.pairs[index];
            if (edge.a != added && edge.b != added)
                continue;
            const int parent = edge.a == added ? edge.b : edge.a;
            if (std::find(active.begin(), active.end(), parent) == active.end())
                continue;
            const auto &base = data.cameras[static_cast<std::size_t>(parent)];
            data.cameras[static_cast<std::size_t>(added)].beta =
                data.cameras[static_cast<std::size_t>(added)].fps *
                (base.beta / base.fps + OrientedOffset(edge, parent));
            break;
        }
        const double center = data.cameras[static_cast<std::size_t>(added)].beta;
        const auto boundaries = SlotBoundaries(data, active, added, center - options.search_radius,
                                               center + options.search_radius);
        active.push_back(added);
        Solution best;
        std::size_t best_log = result.trials.size();
        auto trial = [&](double beta, int slot) {
            Dataset candidate = data;
            candidate.cameras[static_cast<std::size_t>(added)].beta = beta;
            NormalizeReference(candidate, options.reference_camera, target_beta);
            Solution solution;
            try {
                solution = Optimize(candidate, active, incremental);
            } catch (const std::runtime_error &error) {
                solution.report = error.what();
            }
            result.trials.push_back({added, slot, beta, solution.cost, false});
            if (solution.usable && solution.order_valid && solution.cost < best.cost) {
                best = std::move(solution);
                best_log = result.trials.size() - 1;
            }
        };
        trial(center, -1); // Include the graph estimate in addition to slot midpoints.
        for (std::size_t i = 1; i < boundaries.size(); ++i) {
            if (boundaries[i] - boundaries[i - 1] >
                2 * options.min_dt * data.cameras[static_cast<std::size_t>(added)].fps)
                trial(0.5 * (boundaries[i] + boundaries[i - 1]), static_cast<int>(i - 1));
        }
        // Equality cases support genuinely synchronous input. Optimize preserves ties.
        for (std::size_t i = 1; i + 1 < boundaries.size(); ++i)
            trial(boundaries[i], -2 - static_cast<int>(i));
        if (!best.usable)
            throw std::runtime_error("All insertion slots failed for camera " +
                                     std::to_string(added));
        result.trials[best_log].accepted = true;
        current = std::move(best);
    }
    result.incremental = current;
    result.global = refine_global ? Optimize(current.data, active, options, &current) : current;
    if (!result.global.usable || !result.global.order_valid)
        throw std::runtime_error("Global optimization failed: " + result.global.report);
    return result;
}

} // namespace stba
