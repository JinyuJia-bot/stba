#pragma once
#include <Eigen/Core>
#include <array>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace stba {
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
// Internal ids are contiguous vector indices. All lengths are metres, times seconds.
struct Camera {
    double fps = 30, beta = 0;
}; // f = fps*t + beta
struct View {
    int camera = 0, frame = -1; // -1: shared static camera; >=0: per-frame pose/intrinsics
    std::array<double, 3> rotation{0, 0, 0}, translation{0, 0, 0}; // angle-axis, world->camera
    std::array<double, 4> intrinsics{1000, 1000, 640, 360};        // fx,fy,cx,cy
};
struct Observation {
    int camera = 0, frame = 0, view = 0;
    Vec2 uv = Vec2::Zero();
    double weight = 1;
};
struct Track {
    int id = 0;
    bool is_static = false;
    double motion_weight = 1;
    std::vector<Observation> observations;
};
struct Dataset {
    std::vector<Camera> cameras;
    std::vector<View> views;
    std::vector<Track> tracks;
};
struct Sample {
    int anchor = 0;
    std::vector<int> observations;
    Vec3 xyz = Vec3::Zero();
};
struct Trajectory {
    int track = 0;
    std::vector<Sample> samples;
    std::vector<int> observation_to_sample;
};
struct Options {
    double motion_weight = 1000, epsilon_seconds = 1e-6, min_dt = 1e-10;
    double search_radius = 0.75, coarse_step = 0.05, fine_step = 0.005;
    // Temporary experiment: absolute residual-frame box after beta_ref=0 normalization.
    // 0 disables bounds and preserves the original code path.
    double offset_bound_frames = 0;
    int max_iterations = 80, reference_camera = 0;
    bool optimize_offsets = true, optimize_poses = false, optimize_intrinsics = false;
    double huber_px = 0; // 0 keeps the paper's quadratic image cost
    double dct_fps = 60;
    int dct_max_samples = 256;
};
struct Solution {
    Dataset data;
    std::vector<int> active;
    std::vector<Trajectory> trajectories;
    bool usable = false, order_valid = false;
    double cost = std::numeric_limits<double>::infinity();
    double image_cost = 0, motion_cost = 0, reprojection_rmse = 0;
    std::string report;
};
struct PairAlignment {
    int a = 0, b = 0;
    double delta_seconds = 0, cost = 0, baseline = 0, edge_weight = 0;
    int common_tracks = 0;
    bool valid = false;
};
struct IncrementLog {
    int camera = 0, slot = 0;
    double beta = 0, cost = 0;
    bool accepted = false;
};
struct PipelineResult {
    Solution incremental, global;
    std::vector<PairAlignment> pairs;
    std::vector<int> order;
    std::vector<IncrementLog> trials;
};
struct ResampledTrack {
    int track = 0;
    std::vector<double> times;
    std::vector<Vec3> points;
    double cost_before = 0, cost_after = 0;
    bool usable = false;
};

double Time(const Dataset &, const Observation &);
Vec3 CameraCenter(const View &);
Vec2 Project(const View &, const Vec3 &);
Vec3 Triangulate(const View &, const Vec2 &, const View &, const Vec2 &);
void Validate(const Dataset &);
Dataset ReadDataset(const std::filesystem::path &);
void WriteDataset(const Dataset &, const std::filesystem::path &);
std::vector<Trajectory> InitializeTrajectories(const Dataset &, const std::vector<int> &active);
// Returned trajectories retain input track and observation indices. Excluded tracks are omitted.
Solution Optimize(const Dataset &, const std::vector<int> &active, const Options &,
                  const Solution *warm_start = nullptr);
PairAlignment AlignPair(const Dataset &, int a, int b, const Options &);
std::vector<int> CameraOrder(const Dataset &, std::vector<PairAlignment> &);
PipelineResult Reconstruct(const Dataset &, const Options &, bool refine_global = true);
std::vector<ResampledTrack> ResampleDCT(const Solution &, const Options &);
void WriteResults(const PipelineResult &, const std::vector<ResampledTrack> &,
                  const std::filesystem::path &);
Options ReadOptions(const std::filesystem::path &);
} // namespace stba
