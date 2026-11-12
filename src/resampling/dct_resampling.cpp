#include "core/stba.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace stba {
namespace {
constexpr double kPi = 3.14159265358979323846;

// DCT-II at t_i = start + i*dt. Its reflection boundaries are half a sample
// outside the two endpoints. At integer i these vectors are orthonormal.
std::vector<double> Basis(int count, double coordinate) {
    std::vector<double> basis(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k) {
        const double normalization = std::sqrt((k == 0 ? 1.0 : 2.0) / count);
        basis[static_cast<std::size_t>(k)] =
            normalization * std::cos(kPi * k * (coordinate + 0.5) / count);
    }
    return basis;
}

Vec3 Position(const std::vector<double> &basis, const double *coefficients) {
    Vec3 point = Vec3::Zero();
    for (std::size_t k = 0; k < basis.size(); ++k) {
        for (int d = 0; d < 3; ++d) {
            point[d] += basis[k] * coefficients[3 * k + static_cast<std::size_t>(d)];
        }
    }
    return point;
}

// One original image observation depends on all coefficients of its trajectory.
// The analytic Jacobian is simply the projection Jacobian times each basis value.
class DctReprojectionCost final : public ceres::CostFunction {
  public:
    DctReprojectionCost(const View &view, const Observation &observation, std::vector<double> basis)
        : basis_(std::move(basis)), translation_(view.translation.data()), uv_(observation.uv),
          intrinsics_(view.intrinsics), scale_(std::sqrt(observation.weight)) {
        ceres::AngleAxisToRotationMatrix(view.rotation.data(), rotation_.data());
        set_num_residuals(2);
        mutable_parameter_block_sizes()->push_back(static_cast<int>(3 * basis_.size()));
    }

    bool Evaluate(double const *const *parameters, double *residuals,
                  double **jacobians) const override {
        const Vec3 p = rotation_ * Position(basis_, parameters[0]) + translation_;
        if (!p.allFinite() || p.z() <= 1e-8)
            return false;
        const double inverse_z = 1.0 / p.z();
        residuals[0] = scale_ * (intrinsics_[0] * p.x() * inverse_z + intrinsics_[2] - uv_.x());
        residuals[1] = scale_ * (intrinsics_[1] * p.y() * inverse_z + intrinsics_[3] - uv_.y());
        if (jacobians && jacobians[0]) {
            const Eigen::RowVector3d du = scale_ * intrinsics_[0] * inverse_z *
                                          (rotation_.row(0) - p.x() * inverse_z * rotation_.row(2));
            const Eigen::RowVector3d dv = scale_ * intrinsics_[1] * inverse_z *
                                          (rotation_.row(1) - p.y() * inverse_z * rotation_.row(2));
            const std::size_t width = 3 * basis_.size();
            for (std::size_t k = 0; k < basis_.size(); ++k) {
                for (int d = 0; d < 3; ++d) {
                    const std::size_t column = 3 * k + static_cast<std::size_t>(d);
                    jacobians[0][column] = basis_[k] * du[d];
                    jacobians[0][width + column] = basis_[k] * dv[d];
                }
            }
        }
        return true;
    }

  private:
    std::vector<double> basis_;
    Eigen::Matrix3d rotation_;
    Vec3 translation_;
    Vec2 uv_;
    std::array<double, 4> intrinsics_;
    double scale_;
};

// D^T D, for consecutive first differences on an N-sample open chain, has
// eigenvalues 4*sin^2(pi*k/(2N)). The constant coefficient has zero cost.
class DctMotionCost final : public ceres::CostFunction {
  public:
    DctMotionCost(int count, double dt, double weight, double epsilon)
        : scales_(static_cast<std::size_t>(count)) {
        const double base = std::sqrt(weight * dt / 2.0) / (dt + epsilon);
        for (int k = 0; k < count; ++k) {
            scales_[static_cast<std::size_t>(k)] = base * 2.0 * std::sin(kPi * k / (2.0 * count));
        }
        set_num_residuals(3 * count);
        mutable_parameter_block_sizes()->push_back(3 * count);
    }

    bool Evaluate(double const *const *parameters, double *residuals,
                  double **jacobians) const override {
        const std::size_t size = 3 * scales_.size();
        if (jacobians && jacobians[0])
            std::fill(jacobians[0], jacobians[0] + size * size, 0.0);
        for (std::size_t k = 0; k < scales_.size(); ++k) {
            for (std::size_t d = 0; d < 3; ++d) {
                const std::size_t row = 3 * k + d;
                residuals[row] = scales_[k] * parameters[0][row];
                if (jacobians && jacobians[0])
                    jacobians[0][row * size + row] = scales_[k];
            }
        }
        return true;
    }

  private:
    std::vector<double> scales_;
};

Vec3 Interpolate(const std::vector<double> &times, const std::vector<Sample> &samples,
                 double time) {
    if (time <= times.front())
        return samples.front().xyz;
    if (time >= times.back())
        return samples.back().xyz;
    const auto upper = std::upper_bound(times.begin(), times.end(), time);
    const std::size_t b = static_cast<std::size_t>(upper - times.begin());
    const std::size_t a = b - 1;
    const double ratio = (time - times[a]) / (times[b] - times[a]);
    return (1.0 - ratio) * samples[a].xyz + ratio * samples[b].xyz;
}

double StaticImageCost(const Solution &solution, const Trajectory &trajectory, double huber) {
    const Track &track = solution.data.tracks.at(static_cast<std::size_t>(trajectory.track));
    double cost = 0.0;
    for (std::size_t i = 0; i < track.observations.size(); ++i) {
        if (trajectory.observation_to_sample.at(i) < 0)
            continue;
        const Observation &observation = track.observations[i];
        const Vec2 predicted =
            Project(solution.data.views.at(static_cast<std::size_t>(observation.view)),
                    trajectory.samples.front().xyz);
        const double squared = observation.weight * (predicted - observation.uv).squaredNorm();
        cost += huber > 0.0 && squared > huber * huber
                    ? 2.0 * huber * std::sqrt(squared) - huber * huber
                    : squared;
    }
    return cost;
}
} // namespace

std::vector<ResampledTrack> ResampleDCT(const Solution &solution, const Options &options) {
    if (!solution.usable || !solution.order_valid) {
        throw std::invalid_argument("DCT needs a usable, order-valid STBA solution.");
    }
    if (!std::isfinite(options.dct_fps) || options.dct_fps <= 0.0 || options.dct_max_samples < 2 ||
        !std::isfinite(options.motion_weight) || options.motion_weight < 0.0 ||
        !std::isfinite(options.epsilon_seconds) || options.epsilon_seconds <= 0.0 ||
        !std::isfinite(options.huber_px) || options.huber_px < 0.0 || options.max_iterations < 1) {
        throw std::invalid_argument(
            "Invalid DCT resolution, motion weight, epsilon or solver options.");
    }
    std::vector<ResampledTrack> result;
    for (const Trajectory &trajectory : solution.trajectories) {
        if (trajectory.samples.empty())
            continue;
        const Track &track = solution.data.tracks.at(static_cast<std::size_t>(trajectory.track));
        std::vector<double> times;
        for (const Sample &sample : trajectory.samples) {
            times.push_back(Time(solution.data,
                                 track.observations.at(static_cast<std::size_t>(sample.anchor))));
        }
        ResampledTrack output;
        output.track = trajectory.track;
        if (track.is_static || trajectory.samples.size() == 1) {
            output.times = {times.front()};
            output.points = {trajectory.samples.front().xyz};
            output.cost_before = output.cost_after =
                StaticImageCost(solution, trajectory, options.huber_px);
            output.usable = std::isfinite(output.cost_after);
            result.push_back(std::move(output));
            continue;
        }
        for (std::size_t i = 1; i < times.size(); ++i) {
            if (!std::isfinite(times[i]) || times[i] <= times[i - 1]) {
                throw std::invalid_argument(
                    "DCT input trajectory times must be strictly increasing.");
            }
        }
        const double duration = times.back() - times.front();
        const double requested = std::ceil(duration * options.dct_fps) + 1.0;
        if (!std::isfinite(requested) || requested > options.dct_max_samples) {
            throw std::runtime_error("DCT track " + std::to_string(track.id) +
                                     " exceeds dct_max_samples; lower dct_fps, raise the limit, or "
                                     "use a shorter sequence.");
        }
        const int count = std::max(2, static_cast<int>(requested));
        const double dt = duration / (count - 1);
        std::vector<double> coefficients(static_cast<std::size_t>(3 * count), 0.0);
        // Linear interpolation supplies only the starting coefficients. The
        // optimization below uses original image observations, never 3D truth.
        for (int i = 0; i < count; ++i) {
            const double time = times.front() + i * dt;
            const Vec3 initial = Interpolate(times, trajectory.samples, time);
            const auto basis = Basis(count, static_cast<double>(i));
            for (int k = 0; k < count; ++k) {
                for (int d = 0; d < 3; ++d) {
                    coefficients[static_cast<std::size_t>(3 * k + d)] +=
                        basis[static_cast<std::size_t>(k)] * initial[d];
                }
            }
            output.times.push_back(time);
        }

        ceres::Problem problem;
        for (std::size_t i = 0; i < track.observations.size(); ++i) {
            if (trajectory.observation_to_sample.at(i) < 0)
                continue;
            const Observation &observation = track.observations[i];
            const double coordinate = (Time(solution.data, observation) - times.front()) / dt;
            problem.AddResidualBlock(
                new DctReprojectionCost(
                    solution.data.views.at(static_cast<std::size_t>(observation.view)), observation,
                    Basis(count, coordinate)),
                options.huber_px > 0.0 ? new ceres::HuberLoss(options.huber_px) : nullptr,
                coefficients.data());
        }
        problem.AddResidualBlock(new DctMotionCost(count, dt,
                                                   options.motion_weight * track.motion_weight,
                                                   options.epsilon_seconds),
                                 nullptr, coefficients.data());
        ceres::Solver::Options solver_options;
        solver_options.linear_solver_type = ceres::DENSE_QR;
        solver_options.max_num_iterations = options.max_iterations;
        solver_options.num_threads = 1;
        solver_options.function_tolerance = 1e-10;
        solver_options.gradient_tolerance = 1e-10;
        solver_options.parameter_tolerance = 1e-10;
        solver_options.logging_type = ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(solver_options, &problem, &summary);
        output.cost_before = 2.0 * summary.initial_cost;
        output.cost_after = 2.0 * summary.final_cost;
        output.usable =
            summary.IsSolutionUsable() && std::isfinite(output.cost_after) &&
            output.cost_after <= output.cost_before + 1e-8 * (1.0 + std::abs(output.cost_before));
        for (int i = 0; i < count; ++i) {
            const Vec3 point = Position(Basis(count, static_cast<double>(i)), coefficients.data());
            output.points.push_back(point);
            output.usable = output.usable && point.allFinite();
        }
        result.push_back(std::move(output));
    }
    return result;
}
} // namespace stba
