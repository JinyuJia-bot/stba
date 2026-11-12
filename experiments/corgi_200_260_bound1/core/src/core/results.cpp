#include "core/stba.h"
#include <fstream>
#include <iomanip>
#include <stdexcept>
namespace stba {
namespace {
std::ofstream Open(const std::filesystem::path &path) {
    std::ofstream file(path);
    if (!file)
        throw std::runtime_error("Cannot write " + path.string());
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file << std::setprecision(17);
    return file;
}
void Trajectories(const Solution &solution, const std::filesystem::path &path) {
    auto out = Open(path);
    out << "track_id,camera_id,frame,time,x,y,z\n";
    for (const auto &t : solution.trajectories) {
        const auto &track = solution.data.tracks.at(t.track);
        for (const auto &node : t.samples)
            for (int index : node.observations) {
                const auto &o = track.observations.at(index);
                out << track.id << ',' << o.camera << ',' << o.frame << ','
                    << Time(solution.data, o) << ',' << node.xyz.x() << ',' << node.xyz.y() << ','
                    << node.xyz.z() << '\n';
            }
    }
}
} // namespace
void WriteResults(const PipelineResult &pipeline, const std::vector<ResampledTrack> &dct,
                  const std::filesystem::path &directory) {
    std::filesystem::create_directories(directory);
    WriteDataset(pipeline.global.data, directory);
    Trajectories(pipeline.global, directory / "trajectories.csv");
    Trajectories(pipeline.incremental, directory / "incremental_trajectories.csv");
    auto metrics = Open(directory / "metrics.csv");
    metrics << "stage,cost,image_cost,motion_cost,reprojection_rmse,usable,order_valid\n";
    for (const auto &entry :
         {std::pair<const char *, const Solution *>{"incremental", &pipeline.incremental},
          {"global", &pipeline.global}}) {
        const auto &s = *entry.second;
        metrics << entry.first << ',' << s.cost << ',' << s.image_cost << ',' << s.motion_cost
                << ',' << s.reprojection_rmse << ',' << s.usable << ',' << s.order_valid << '\n';
    }
    auto pairs = Open(directory / "pairwise.csv");
    pairs << "a,b,delta_seconds,cost,baseline,common_tracks,edge_weight,valid\n";
    for (const auto &p : pipeline.pairs)
        pairs << p.a << ',' << p.b << ',' << p.delta_seconds << ',' << p.cost << ',' << p.baseline
              << ',' << p.common_tracks << ',' << p.edge_weight << ',' << p.valid << '\n';
    auto order = Open(directory / "order.csv");
    order << "position,camera_id\n";
    for (std::size_t i = 0; i < pipeline.order.size(); ++i)
        order << i << ',' << pipeline.order[i] << '\n';
    auto trials = Open(directory / "slots.csv");
    trials << "camera_id,slot,beta,cost,accepted\n";
    for (const auto &t : pipeline.trials)
        trials << t.camera << ',' << t.slot << ',' << t.beta << ',' << t.cost << ',' << t.accepted
               << '\n';
    auto samples = Open(directory / "dct.csv");
    samples << "track_id,time,x,y,z\n";
    auto dc = Open(directory / "dct_metrics.csv");
    dc << "track_id,cost_before,cost_after,usable,samples\n";
    for (const auto &t : dct) {
        // ResampledTrack.track is the input track index, CSV writes its external id.
        int id = pipeline.global.data.tracks.at(t.track).id;
        dc << id << ',' << t.cost_before << ',' << t.cost_after << ',' << t.usable << ','
           << t.points.size() << '\n';
        if (t.usable)
            for (std::size_t i = 0; i < t.points.size(); ++i)
                samples << id << ',' << t.times[i] << ',' << t.points[i].x() << ','
                        << t.points[i].y() << ',' << t.points[i].z() << '\n';
    }
    auto report = Open(directory / "solver_report.txt");
    report << "Incremental: " << pipeline.incremental.report
           << "\nGlobal: " << pipeline.global.report << '\n';
}
} // namespace stba
