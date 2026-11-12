#include "core/stba.h"
#include "core/beta_bounds.h"
#include "optimization/cost_functions.h"
#include <algorithm>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace stba {
namespace {
class BoundsCallback final : public ceres::IterationCallback {
  public:
    BoundsCallback(const Dataset &data, const std::vector<int> &active, const Options &options)
        : data_(data), active_(active), options_(options) {}
    ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override {
        try {
            CheckBetaBounds(data_, active_, options_, "ceres_iterate", summary.iteration);
        } catch (const std::exception &) {
            return ceres::SOLVER_ABORT;
        }
        return ceres::SOLVER_CONTINUE;
    }
  private:
    const Dataset &data_;
    const std::vector<int> &active_;
    const Options &options_;
};
struct Event {
    int camera, frame;
    double time;
};
std::vector<Event> Events(const Dataset &data, const std::vector<int> &active) {
    std::set<std::pair<int, int>> unique;
    for (const auto &track : data.tracks)
        if (!track.is_static)
            for (const auto &o : track.observations)
                if (std::find(active.begin(), active.end(), o.camera) != active.end())
                    unique.emplace(o.camera, o.frame);
    std::vector<Event> result;
    for (const auto &[c, f] : unique)
        result.push_back({c, f, (f - data.cameras[c].beta) / data.cameras[c].fps});
    std::sort(result.begin(), result.end(), [](const Event &a, const Event &b) {
        if (a.time != b.time)
            return a.time < b.time;
        return std::tie(a.camera, a.frame) < std::tie(b.camera, b.frame);
    });
    return result;
}
bool SameOrder(const Dataset &data, const std::vector<Event> &events) {
    for (std::size_t i = 1; i < events.size(); ++i) {
        const auto &a = events[i - 1];
        const auto &b = events[i];
        double dt = (b.frame - data.cameras[b.camera].beta) / data.cameras[b.camera].fps -
                    (a.frame - data.cameras[a.camera].beta) / data.cameras[a.camera].fps;
        if (b.time - a.time <= 1e-10) {
            if (std::abs(dt) > 1e-9)
                return false;
        } else if (dt <= 0)
            return false;
    }
    return true;
}
double Robust(double squared, double huber) {
    return huber <= 0 || squared <= huber * huber ? squared
                                                  : 2 * huber * std::sqrt(squared) - huber * huber;
}
void Evaluate(Solution &solution, const Options &options) {
    double pixel_sse = 0;
    std::size_t count = 0;
    solution.image_cost = solution.motion_cost = 0;
    for (const auto &trajectory : solution.trajectories) {
        const auto &track = solution.data.tracks.at(trajectory.track);
        for (const auto &node : trajectory.samples)
            for (int index : node.observations) {
                const auto &o = track.observations.at(index);
                const double e =
                    (Project(solution.data.views.at(o.view), node.xyz) - o.uv).squaredNorm();
                pixel_sse += e;
                ++count;
                solution.image_cost += Robust(o.weight * e, options.huber_px);
            }
        if (!track.is_static)
            for (std::size_t j = 1; j < trajectory.samples.size(); ++j) {
                const auto &a = trajectory.samples[j - 1];
                const auto &b = trajectory.samples[j];
                const double dt = Time(solution.data, track.observations.at(b.anchor)) -
                                  Time(solution.data, track.observations.at(a.anchor));
                if (dt <= 0) {
                    solution.usable = false;
                    return;
                }
                const double den = dt + options.epsilon_seconds;
                solution.motion_cost += options.motion_weight * track.motion_weight * 0.5 * dt /
                                        (den * den) * (b.xyz - a.xyz).squaredNorm();
            }
    }
    solution.cost = solution.image_cost + solution.motion_cost;
    solution.reprojection_rmse = count ? std::sqrt(pixel_sse / static_cast<double>(count)) : 0;
    solution.usable = solution.usable && count > 0 && std::isfinite(solution.cost);
}
void WarmStart(Solution &result, const Solution *warm) {
    if (!warm || !warm->usable)
        return;
    std::map<int, const Trajectory *> old;
    for (const auto &t : warm->trajectories)
        old[t.track] = &t;
    for (auto &t : result.trajectories) {
        const auto it = old.find(t.track);
        if (it == old.end())
            continue;
        const auto &previous = *it->second;
        for (auto &node : t.samples) {
            Vec3 sum = Vec3::Zero();
            int count = 0;
            for (int o : node.observations)
                if (o < static_cast<int>(previous.observation_to_sample.size())) {
                    int n = previous.observation_to_sample[o];
                    if (n >= 0) {
                        sum += previous.samples.at(n).xyz;
                        ++count;
                    }
                }
            if (count)
                node.xyz = sum / static_cast<double>(count);
        }
    }
}
} // namespace

Solution Optimize(const Dataset &input, const std::vector<int> &active, const Options &options,
                  const Solution *warm) {
    Validate(input);
    if (active.size() < 2)
        throw std::invalid_argument("Optimize needs at least two cameras");
    std::set<int> distinct;
    for (int c : active)
        if (c < 0 || c >= static_cast<int>(input.cameras.size()) || !distinct.insert(c).second)
            throw std::invalid_argument("Invalid or duplicate active camera");
    if (!(options.motion_weight > 0) || !(options.epsilon_seconds > 0) ||
        options.max_iterations < 1 || !std::isfinite(options.motion_weight) ||
        !std::isfinite(options.epsilon_seconds))
        throw std::invalid_argument("Invalid optimizer options");
    CheckBetaBounds(input, active, options, "optimize_input");
    Solution result;
    result.data = input;
    result.active = active;
    result.trajectories = InitializeTrajectories(result.data, active);
    if (result.trajectories.empty())
        throw std::runtime_error("No triangulatable shared tracks");
    WarmStart(result, warm);
    auto &data = result.data;
    const auto events = Events(data, active);
    const double spacing_guard = std::max(options.min_dt, options.epsilon_seconds);
    // Near-simultaneous samples (dt <= epsilon) must not exploit dt/(dt+epsilon)^2
    // tending to zero. Keep their camera offsets fixed during this local solve.
    // Exact simultaneity defines an equality face. Keep involved camera offsets fixed
    // during this local solve; different sequencing hypotheses belong to outer slots.
    std::set<int> fixed_offsets;
    const int reference =
        distinct.count(options.reference_camera) ? options.reference_camera : active.front();
    fixed_offsets.insert(reference);
    for (std::size_t i = 1; i < events.size(); ++i)
        if (events[i].time - events[i - 1].time <= spacing_guard) {
            fixed_offsets.insert(events[i].camera);
            fixed_offsets.insert(events[i - 1].camera);
        }
    ceres::Problem problem;
    std::set<int> used_views;
    for (int c : active)
        problem.AddParameterBlock(&data.cameras[c].beta, 1);
    for (auto &trajectory : result.trajectories) {
        const auto &track = data.tracks.at(trajectory.track);
        for (auto &node : trajectory.samples)
            for (int index : node.observations) {
                const auto &o = track.observations.at(index);
                auto &v = data.views.at(o.view);
                used_views.insert(o.view);
                problem.AddResidualBlock(
                    ReprojectionCost(o),
                    options.huber_px > 0 ? new ceres::HuberLoss(options.huber_px) : nullptr,
                    node.xyz.data(), v.rotation.data(), v.translation.data(), v.intrinsics.data());
            }
        if (!track.is_static)
            for (std::size_t j = 1; j < trajectory.samples.size(); ++j) {
                auto &a = trajectory.samples[j - 1];
                auto &b = trajectory.samples[j];
                const auto &oa = track.observations.at(a.anchor);
                const auto &ob = track.observations.at(b.anchor);
                const double weight = options.motion_weight * track.motion_weight;
                if (oa.camera == ob.camera || !options.optimize_offsets ||
                    (fixed_offsets.count(oa.camera) && fixed_offsets.count(ob.camera)))
                    problem.AddResidualBlock(MotionCost(Time(data, ob) - Time(data, oa), weight,
                                                        options.epsilon_seconds),
                                             nullptr, a.xyz.data(), b.xyz.data());
                else
                    problem.AddResidualBlock(
                        TimedMotionCost(oa, ob, data, weight, options.epsilon_seconds,
                                        spacing_guard),
                        nullptr, a.xyz.data(), b.xyz.data(), &data.cameras[oa.camera].beta,
                        &data.cameras[ob.camera].beta);
            }
    }
    if (options.optimize_offsets)
        for (std::size_t i = 1; i < events.size(); ++i) {
            const auto &a = events[i - 1];
            const auto &b = events[i];
            if (a.camera != b.camera && b.time - a.time > spacing_guard)
                problem.AddResidualBlock(OrderCost(a.frame, data.cameras[a.camera].fps, b.frame,
                                                   data.cameras[b.camera].fps, spacing_guard),
                                         nullptr, &data.cameras[a.camera].beta,
                                         &data.cameras[b.camera].beta);
        }
    for (int c : active) {
        if (Bounded(options)) {
            problem.SetParameterLowerBound(&data.cameras[c].beta, 0, -options.offset_bound_frames);
            problem.SetParameterUpperBound(&data.cameras[c].beta, 0, options.offset_bound_frames);
        }
        if (!options.optimize_offsets || fixed_offsets.count(c))
            problem.SetParameterBlockConstant(&data.cameras[c].beta);
    }

    int reference_view = -1, scale_view = -1;
    for (int v : used_views)
        if (data.views[v].camera == reference) {
            reference_view = v;
            break;
        }
    if (reference_view < 0)
        throw std::runtime_error("Reference camera has no supported observations");
    double lever = 0;
    int axis = 0;
    const Vec3 center = CameraCenter(data.views[reference_view]);
    for (int id : used_views)
        if (data.views[id].camera != reference) {
            const auto &v = data.views[id];
            double p[3];
            ceres::AngleAxisRotatePoint(v.rotation.data(), center.data(), p);
            for (int k = 0; k < 3; ++k)
                if (std::abs(p[k] + v.translation[k]) > lever) {
                    lever = std::abs(p[k] + v.translation[k]);
                    scale_view = id;
                    axis = k;
                }
        }
    if (options.optimize_poses && (scale_view < 0 || lever < 1e-8))
        throw std::runtime_error("Cannot fix scale: camera baseline is degenerate");
    for (int id : used_views) {
        auto &v = data.views[id];
        if (!options.optimize_poses || id == reference_view) {
            problem.SetParameterBlockConstant(v.rotation.data());
            problem.SetParameterBlockConstant(v.translation.data());
        } else if (id == scale_view)
            problem.SetManifold(v.translation.data(), new ceres::SubsetManifold(3, {axis}));
        if (!options.optimize_intrinsics)
            problem.SetParameterBlockConstant(v.intrinsics.data());
        else {
            // Keep measured principal point; estimate fx/fy. Arbitrary distortion is not modelled.
            problem.SetManifold(v.intrinsics.data(), new ceres::SubsetManifold(4, {2, 3}));
            for (int k = 0; k < 2; ++k) {
                problem.SetParameterLowerBound(v.intrinsics.data(), k, 1);
                problem.SetParameterUpperBound(v.intrinsics.data(), k, 1e5);
            }
        }
    }
    ceres::Solver::Options solver;
    solver.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver.max_num_iterations = options.max_iterations;
    solver.num_threads = 1;
    solver.logging_type = ceres::SILENT;
    solver.function_tolerance = 1e-9;
    solver.parameter_tolerance = 1e-9;
    solver.gradient_tolerance = 1e-10;
    solver.max_num_consecutive_invalid_steps = 30;
    BoundsCallback bounds_callback(data, active, options);
    if (Bounded(options)) {
        solver.update_state_every_iteration = true;
        solver.callbacks.push_back(&bounds_callback);
    }
    ceres::Solver::Summary summary;
    ceres::Solve(solver, &problem, &summary);
    CheckBetaBounds(data, active, options, "optimize_output");
    result.report = summary.BriefReport();
    result.order_valid = SameOrder(data, events);
    result.usable = summary.IsSolutionUsable() && result.order_valid;
    try {
        Evaluate(result, options);
    } catch (const std::exception &error) {
        result.usable = false;
        result.report += "; " + std::string(error.what());
    }
    return result;
}
} // namespace stba
