#include "core/stba.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace stba {
namespace {
struct Interpolated {
    View view;
    Vec2 uv;
};
Eigen::Quaterniond Quaternion(const View &view) {
    const Vec3 aa(view.rotation[0], view.rotation[1], view.rotation[2]);
    const double angle = aa.norm();
    if (angle < 1e-15)
        return Eigen::Quaterniond::Identity();
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, aa / angle));
}
Interpolated Interpolate(const Dataset &data, const Track &track, const std::vector<int> &indices,
                         double seconds) {
    const auto &first = track.observations[static_cast<std::size_t>(indices.front())];
    const auto &camera = data.cameras[static_cast<std::size_t>(first.camera)];
    const double frame = seconds * camera.fps + camera.beta;
    if (indices.size() == 1)
        return {data.views[static_cast<std::size_t>(first.view)], first.uv};
    auto upper = std::upper_bound(indices.begin(), indices.end(), frame, [&](double f, int i) {
        return f < track.observations[static_cast<std::size_t>(i)].frame;
    });
    if (upper == indices.begin())
        ++upper;
    else if (upper == indices.end())
        --upper;
    const auto &a = track.observations[static_cast<std::size_t>(*(upper - 1))];
    const auto &b = track.observations[static_cast<std::size_t>(*upper)];
    const double alpha = (frame - a.frame) / static_cast<double>(b.frame - a.frame);
    const View &va = data.views[static_cast<std::size_t>(a.view)];
    const View &vb = data.views[static_cast<std::size_t>(b.view)];
    View view = va;
    if (a.view != b.view) {
        const Eigen::Quaterniond q = Quaternion(va).slerp(alpha, Quaternion(vb));
        const Eigen::AngleAxisd aa(q.normalized());
        const Vec3 rotation = aa.angle() * aa.axis();
        for (std::size_t k = 0; k < 3; ++k) {
            view.rotation[k] = rotation[static_cast<Eigen::Index>(k)];
            view.translation[k] =
                va.translation[k] + alpha * (vb.translation[k] - va.translation[k]);
        }
        for (std::size_t k = 0; k < 4; ++k)
            view.intrinsics[k] = va.intrinsics[k] + alpha * (vb.intrinsics[k] - va.intrinsics[k]);
    }
    // Only an initialization. The optimizer uses the original observations and views.
    return {view, a.uv + alpha * (b.uv - a.uv)};
}
Vec3 InitializeSample(const Dataset &data, const Track &track,
                      const std::vector<std::vector<int>> &views, int anchor) {
    const auto &observation = track.observations[static_cast<std::size_t>(anchor)];
    const View &own = data.views[static_cast<std::size_t>(observation.view)];
    const double seconds = Time(data, observation);
    std::vector<std::pair<double, Interpolated>> candidates;
    for (std::size_t camera = 0; camera < views.size(); ++camera) {
        if (static_cast<int>(camera) == observation.camera || views[camera].empty())
            continue;
        Interpolated other;
        if (track.is_static) {
            const auto &o = track.observations[static_cast<std::size_t>(views[camera].front())];
            other = {data.views[static_cast<std::size_t>(o.view)], o.uv};
        } else
            other = Interpolate(data, track, views[camera], seconds);
        candidates.push_back({(CameraCenter(own) - CameraCenter(other.view)).norm(), other});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    for (const auto &candidate : candidates) {
        try {
            return Triangulate(own, observation.uv, candidate.second.view, candidate.second.uv);
        } catch (const std::runtime_error &) { /* Try the next baseline. */
        }
    }
    throw std::runtime_error("Cannot initialize track " + std::to_string(track.id) +
                             ": no valid stereo geometry");
}
} // namespace

std::vector<Trajectory> InitializeTrajectories(const Dataset &data,
                                               const std::vector<int> &active) {
    std::vector<bool> enabled(data.cameras.size(), false);
    for (int camera : active) {
        if (camera < 0 || static_cast<std::size_t>(camera) >= enabled.size())
            throw std::invalid_argument("Active camera id is out of range");
        if (enabled[static_cast<std::size_t>(camera)])
            throw std::invalid_argument("Repeated active camera");
        enabled[static_cast<std::size_t>(camera)] = true;
    }
    std::vector<Trajectory> output;
    for (std::size_t id = 0; id < data.tracks.size(); ++id) {
        const Track &track = data.tracks[id];
        std::vector<std::vector<int>> views(data.cameras.size());
        std::vector<int> order;
        for (std::size_t i = 0; i < track.observations.size(); ++i) {
            const auto &o = track.observations[i];
            if (enabled[static_cast<std::size_t>(o.camera)]) {
                views[static_cast<std::size_t>(o.camera)].push_back(static_cast<int>(i));
                order.push_back(static_cast<int>(i));
            }
        }
        const auto camera_count =
            std::count_if(views.begin(), views.end(), [](const auto &v) { return !v.empty(); });
        if (camera_count < 2)
            continue;
        for (auto &view : views)
            std::sort(view.begin(), view.end(), [&](int a, int b) {
                return track.observations[static_cast<std::size_t>(a)].frame <
                       track.observations[static_cast<std::size_t>(b)].frame;
            });
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return Time(data, track.observations[static_cast<std::size_t>(a)]) <
                   Time(data, track.observations[static_cast<std::size_t>(b)]);
        });
        Trajectory trajectory;
        trajectory.track = static_cast<int>(id);
        trajectory.observation_to_sample.assign(track.observations.size(), -1);
        double last_time = -std::numeric_limits<double>::infinity();
        for (int observation : order) {
            const double time =
                Time(data, track.observations[static_cast<std::size_t>(observation)]);
            if (trajectory.samples.empty() || (!track.is_static && time - last_time > 1e-10)) {
                Sample sample;
                sample.anchor = observation;
                sample.xyz = InitializeSample(data, track, views, observation);
                trajectory.samples.push_back(std::move(sample));
                last_time = time;
            }
            trajectory.samples.back().observations.push_back(observation);
            trajectory.observation_to_sample[static_cast<std::size_t>(observation)] =
                static_cast<int>(trajectory.samples.size() - 1);
        }
        output.push_back(std::move(trajectory));
    }
    return output;
}
} // namespace stba
