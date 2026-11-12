#include "core/stba.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace stba;

void Check(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}
void Near(double actual, double expected, double tolerance, const std::string &message) {
    Check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          message + ": actual=" + std::to_string(actual) +
              ", expected=" + std::to_string(expected));
}

Dataset Fixture(double beta = 0.0) {
    Dataset data;
    data.cameras = {{30.0, 0.0}, {30.0, beta}};
    View left, right;
    left.camera = 0;
    left.translation = {1.0, 0.0, 0.0};
    right.camera = 1;
    right.translation = {-1.0, 0.0, 0.0};
    data.views = {left, right};
    for (int id = 0; id < 2; ++id) {
        Track track;
        track.id = id;
        track.is_static = id == 1;
        for (int camera = 0; camera < 2; ++camera) {
            for (int frame = 0; frame < 12; ++frame) {
                const double t =
                    (frame - data.cameras[static_cast<std::size_t>(camera)].beta) / 30.0;
                const Vec3 truth = track.is_static
                                       ? Vec3(0.2, -0.1, 5.0)
                                       : Vec3(0.3 * std::sin(3.0 * t), 0.2 * std::cos(4.0 * t),
                                              5.0 + 0.1 * std::sin(2.0 * t));
                track.observations.push_back(
                    {camera, frame, camera,
                     Project(data.views[static_cast<std::size_t>(camera)], truth), 1.0});
            }
        }
        data.tracks.push_back(std::move(track));
    }
    return data;
}

void TestProjectionAndTriangulation() {
    const Dataset data = Fixture();
    const Vec3 truth(0.25, -0.2, 4.0);
    const Vec2 uv0 = Project(data.views[0], truth);
    Near(uv0.x(), 952.5, 1e-10, "Pinhole projection x");
    Near(uv0.y(), 310.0, 1e-10, "Pinhole projection y");
    Near((Triangulate(data.views[0], uv0, data.views[1], Project(data.views[1], truth)) - truth)
             .norm(),
         0.0, 1e-10, "DLT must recover an exact finite point");
    Near((CameraCenter(data.views[0]) - Vec3(-1, 0, 0)).norm(), 0, 1e-12,
         "Camera centre convention");
    View rolled;
    rolled.rotation = {0, 0, std::acos(-1.0) / 2};
    rolled.translation = {-0.4, -0.7, 0};
    Near((CameraCenter(rolled) - Vec3(0.7, -0.4, 0)).norm(), 0, 1e-12,
         "Rotated camera centre convention");
    Near(Project(rolled, truth).x(), 590, 1e-10, "World-to-camera rotation x");
    Near(Project(rolled, truth).y(), 247.5, 1e-10, "World-to-camera rotation y");
    bool rejected = false;
    try {
        (void)Triangulate(data.views[0], uv0, data.views[0], uv0);
    } catch (const std::exception &) {
        rejected = true;
    }
    Check(rejected, "Zero baseline must be rejected");
}

void TestTimesAndGrouping() {
    Dataset data = Fixture();
    data.cameras = {{30.0, 0.3}, {60.0, 0.6}};
    Observation a;
    a.camera = 0;
    a.frame = 3;
    Observation b;
    b.camera = 1;
    b.frame = 6;
    Near(Time(data, a), 0.09, 1e-14, "Positive beta must mean an earlier exposure");
    Near(Time(data, b), Time(data, a), 1e-14, "Mixed fps time conversion");

    data = Fixture();
    Validate(data);
    const auto trajectories = InitializeTrajectories(data, {0, 1});
    Check(trajectories.size() == 2, "Both fixture tracks must be initialized");
    for (const auto &trajectory : trajectories) {
        const bool is_static = data.tracks[static_cast<std::size_t>(trajectory.track)].is_static;
        Check(trajectory.samples.size() == (is_static ? 1U : 12U),
              "Synchronous samples must merge; a static track must share one point");
        for (const auto &sample : trajectory.samples)
            Check(sample.observations.size() == (is_static ? 24U : 2U),
                  "Merged observation ownership");
        for (int frame = 0; frame < 12; ++frame)
            Check(trajectory.observation_to_sample[static_cast<std::size_t>(frame)] ==
                      trajectory.observation_to_sample[static_cast<std::size_t>(frame + 12)],
                  "Same-time views must reference the same variable");
    }
    const auto asynchronous = InitializeTrajectories(Fixture(0.37), {0, 1});
    Check(asynchronous[0].samples.size() == 24, "Asynchronous samples must stay distinct");
    data.tracks[0].observations.push_back(data.tracks[0].observations.front());
    bool rejected = false;
    try {
        Validate(data);
    } catch (const std::exception &) {
        rejected = true;
    }
    Check(rejected, "Duplicate camera/frame observations must be rejected");
}

void TestObjectiveAccounting() {
    const Dataset data = Fixture(0.37);
    Options options;
    options.optimize_offsets = false;
    options.motion_weight = 100.0;
    const Solution solution = Optimize(data, {0, 1}, options);
    Check(solution.usable && solution.order_valid, "Fixed-offset solver must succeed");
    double image_cost = 0, motion_cost = 0;
    std::size_t count = 0;
    for (const auto &trajectory : solution.trajectories) {
        const auto &track = solution.data.tracks[static_cast<std::size_t>(trajectory.track)];
        for (const auto &sample : trajectory.samples) {
            for (int index : sample.observations) {
                const auto &observation = track.observations[static_cast<std::size_t>(index)];
                image_cost +=
                    observation.weight *
                    (Project(solution.data.views[static_cast<std::size_t>(observation.view)],
                             sample.xyz) -
                     observation.uv)
                        .squaredNorm();
                ++count;
            }
        }
        if (track.is_static)
            continue;
        for (std::size_t i = 1; i < trajectory.samples.size(); ++i) {
            const auto &a = trajectory.samples[i - 1];
            const auto &b = trajectory.samples[i];
            const double dt =
                Time(solution.data, track.observations[static_cast<std::size_t>(b.anchor)]) -
                Time(solution.data, track.observations[static_cast<std::size_t>(a.anchor)]);
            Check(dt > 0, "Motion edges must have positive time steps");
            motion_cost += options.motion_weight * track.motion_weight * 0.5 * dt /
                           std::pow(dt + options.epsilon_seconds, 2) *
                           (b.xyz - a.xyz).squaredNorm();
        }
    }
    Near(solution.image_cost, image_cost, 1e-7 * (1 + image_cost),
         "Independently calculated image cost");
    Near(solution.motion_cost, motion_cost, 1e-7 * (1 + motion_cost), "Eq.5 including dt and 1/2");
    Near(solution.cost, image_cost + motion_cost, 1e-7 * (1 + solution.cost),
         "Reported objective normalization");
    Near(solution.reprojection_rmse, std::sqrt(image_cost / static_cast<double>(count)), 1e-8,
         "RMSE is per 2D point");
}

void TestDisconnectedGraph() {
    Dataset data = Fixture();
    data.cameras.push_back({30, 0});
    PairAlignment edge;
    edge.a = 0;
    edge.b = 1;
    edge.valid = true;
    edge.cost = 1;
    edge.common_tracks = 2;
    edge.baseline = 2;
    std::vector<PairAlignment> edges{edge};
    bool rejected = false;
    try {
        (void)CameraOrder(data, edges);
    } catch (const std::exception &) {
        rejected = true;
    }
    Check(rejected, "A disconnected graph must not silently omit a camera");
}

void TestCycleWeightedCameraOrder() {
    Dataset data = Fixture();
    data.cameras.push_back({30, 0});
    std::vector<PairAlignment> edges;
    auto edge = [&](int a, int b, double offset, double cost) {
        PairAlignment p;
        p.a = a;
        p.b = b;
        p.delta_seconds = offset;
        p.cost = cost;
        p.baseline = 2;
        p.common_tracks = 2;
        p.valid = true;
        edges.push_back(p);
    };
    edge(0, 1, 0.010, 1);
    edge(1, 2, -0.004, 2);
    edge(2, 0, -0.008, 4); // Reverse direction exercises the signed cycle sum.
    const auto order = CameraOrder(data, edges);
    // Every triangle closure has |0.010 - 0.004 - 0.008| = 0.002 seconds.
    for (const auto &p : edges)
        Near(p.edge_weight, p.cost * 0.002 / 4.0, 1e-14, "Cycle-weighted graph edge");
    Check(order == std::vector<int>({0, 1, 2}), "MST must begin with its lowest-cost pair");
}

void TestMixedFpsReference() {
    Dataset data = Fixture();
    data.cameras = {{30.0, 0.17}, {24.0, 0.43}};
    for (auto &track : data.tracks) {
        for (auto &observation : track.observations) {
            const double t = Time(data, observation);
            const Vec3 truth = track.is_static ? Vec3(0.2, -0.1, 5.0)
                                               : Vec3(0.3 * std::sin(3 * t), 0.2 * std::cos(4 * t),
                                                      5 + 0.1 * std::sin(2 * t));
            observation.uv = Project(data.views[static_cast<std::size_t>(observation.view)], truth);
        }
    }
    Options options;
    options.reference_camera = 1;
    options.motion_weight = 100;
    const Solution solution = Optimize(data, {0, 1}, options);
    Check(solution.usable && solution.order_valid,
          "Mixed-fps local optimization must preserve ordering");
    Near(solution.data.cameras[1].beta, 0.43, 1e-14,
         "A nonzero reference camera id must remain fixed");
    for (const auto &trajectory : solution.trajectories) {
        const auto &track = solution.data.tracks[static_cast<std::size_t>(trajectory.track)];
        if (track.is_static)
            continue;
        double previous = -std::numeric_limits<double>::infinity();
        for (const auto &sample : trajectory.samples) {
            const double time =
                Time(solution.data, track.observations[static_cast<std::size_t>(sample.anchor)]);
            Check(time > previous, "Optimized mixed-fps trajectory must be strictly ordered");
            previous = time;
        }
    }
}

void TestNearCoincidenceGuard() {
    // This gap is below epsilon but above the exact-time merge tolerance.
    // Eq.5's dt/(dt+epsilon)^2 decreases towards zero in this region; allowing
    // the clock to slide freely into it could produce an artificial minimum.
    const Dataset data = Fixture(3e-6);
    Options options;
    const Solution solution = Optimize(data, {0, 1}, options);
    Check(solution.usable && solution.order_valid, "Near-coincident data must remain usable");
    Near(solution.data.cameras[1].beta, data.cameras[1].beta, 1e-15,
         "Near-coincident clocks must stay fixed in their local solve");
}

void TestDctEnergyIdentity() {
    // Independent finite-dimensional check: DCT-II diagonalizes D^T D for
    // adjacent first differences with free endpoints. No frequency is removed.
    constexpr int n = 17;
    const double pi = std::acos(-1.0), dt = 0.013, epsilon = 1e-6;
    Eigen::MatrixXd basis(n, n);
    Eigen::VectorXd signal(n);
    for (int i = 0; i < n; ++i) {
        signal[i] = std::sin(0.37 * i) + 0.07 * i;
        for (int k = 0; k < n; ++k)
            basis(i, k) = std::sqrt((k == 0 ? 1.0 : 2.0) / n) * std::cos(pi * k * (i + 0.5) / n);
    }
    Near((basis.transpose() * basis - Eigen::MatrixXd::Identity(n, n)).norm(), 0, 1e-12,
         "DCT orthonormality");
    const Eigen::VectorXd coefficients = basis.transpose() * signal;
    double direct = 0, spectral = 0;
    for (int i = 1; i < n; ++i)
        direct += std::pow(signal[i] - signal[i - 1], 2);
    for (int k = 0; k < n; ++k)
        spectral += 4 * std::pow(std::sin(pi * k / (2 * n)), 2) * coefficients[k] * coefficients[k];
    const double scale = 0.5 * dt / std::pow(dt + epsilon, 2);
    Near(scale * spectral, scale * direct, 1e-10, "DCT spectral and sample-space kinetic energies");
}

} // namespace

int main() {
    try {
        TestProjectionAndTriangulation();
        TestTimesAndGrouping();
        TestObjectiveAccounting();
        TestDisconnectedGraph();
        TestCycleWeightedCameraOrder();
        TestMixedFpsReference();
        TestNearCoincidenceGuard();
        TestDctEnergyIdentity();
        std::cout
            << "PASS: geometry, timestamps, shared variables, objective, graph and DCT identity\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
