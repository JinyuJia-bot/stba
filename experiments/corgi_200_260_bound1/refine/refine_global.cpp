// Continue saved XYZ and offsets using precisely the same core objective.
// This does not rerun pair searches or allow event-order transitions.
#include "core/stba.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

int main(int argc, char **argv) {
    try {
        if (argc != 4 && argc != 5)
            throw std::runtime_error(
                "Usage: refine_global INPUT_RESULT OUTPUT_RESULT CONFIG [ITERATIONS=500]");
        const std::filesystem::path input = argv[1], output = argv[2];
        if (std::filesystem::weakly_canonical(input) == std::filesystem::weakly_canonical(output))
            throw std::runtime_error("Input and output must differ");
        if (std::filesystem::exists(output) && !std::filesystem::is_empty(output))
            throw std::runtime_error("Refusing to overwrite an existing result");
        auto options = stba::ReadOptions(argv[3]);
        if (options.optimize_poses || options.optimize_intrinsics)
            throw std::runtime_error("This experiment requires fixed camera calibration");
        options.max_iterations = argc == 5 ? std::stoi(argv[4]) : 500;
        if (options.max_iterations < 1)
            throw std::runtime_error("ITERATIONS must be positive");
        const auto data = stba::ReadDataset(input);
        std::set<int> observed;
        for (const auto &track : data.tracks)
            for (const auto &obs : track.observations)
                observed.insert(obs.camera);
        const std::vector<int> active(observed.begin(), observed.end());
        stba::Solution warm;
        warm.data = data;
        warm.active = active;
        warm.usable = true;
        warm.trajectories = stba::InitializeTrajectories(data, active);

        std::map<std::tuple<int,int,int>,stba::Vec3> points;
        std::ifstream stream(input / "trajectories.csv");
        if (!stream)
            throw std::runtime_error("Cannot read saved trajectories.csv");
        std::string line;
        std::getline(stream,line);
        while (std::getline(stream,line)) {
            if (line.empty()) continue;
            std::stringstream row(line);
            std::string cell;
            std::vector<double> values;
            while (std::getline(row,cell,',')) values.push_back(std::stod(cell));
            if (values.size() != 7)
                throw std::runtime_error("Malformed saved trajectory row");
            const auto key = std::make_tuple(int(values[0]),int(values[1]),int(values[2]));
            if (!points.emplace(key,stba::Vec3(values[4],values[5],values[6])).second)
                throw std::runtime_error("Duplicate trajectory observation");
        }
        std::size_t restored = 0;
        for (auto &trajectory : warm.trajectories)
            for (auto &sample : trajectory.samples) {
                const auto &track = data.tracks.at(trajectory.track);
                const auto &anchor = track.observations.at(sample.anchor);
                sample.xyz = points.at({track.id,anchor.camera,anchor.frame});
                ++restored;
            }

        // Preserve the source metrics alongside its saved trajectory, rather than
        // writing uninitialized/infinite values as the pre-refinement stage.
        std::ifstream old_metrics(input / "metrics.csv");
        std::getline(old_metrics,line);
        bool found_global = false;
        while (std::getline(old_metrics,line)) {
            std::stringstream row(line);
            std::string stage, cell;
            std::getline(row,stage,',');
            if (stage != "global") continue;
            std::vector<double> values;
            while (std::getline(row,cell,',')) values.push_back(std::stod(cell));
            if (values.size() != 6)
                throw std::runtime_error("Malformed saved global metrics");
            warm.cost = values[0];
            warm.image_cost = values[1];
            warm.motion_cost = values[2];
            warm.reprojection_rmse = values[3];
            warm.usable = values[4] != 0;
            warm.order_valid = values[5] != 0;
            found_global = true;
        }
        if (!found_global || !warm.usable || !warm.order_valid)
            throw std::runtime_error("Saved global solution is not usable/order-valid");

        const auto begin = std::chrono::steady_clock::now();
        stba::PipelineResult result;
        result.global = stba::Optimize(data,active,options,&warm);
        result.incremental = warm; // preserve the supplied pre-continuation trajectory
        result.incremental.report = "Supplied saved global solution (warm start); metrics copied from source";
        result.order = active;
        if (!result.global.usable)
            throw std::runtime_error("Refinement unusable: " + result.global.report);
        if (result.global.cost > warm.cost * (1.0 + 1e-8))
            throw std::runtime_error("Continuation increased the objective");
        stba::WriteResults(result,{},output);
        auto samples = stba::ResampleDCT(result.global,options);
        stba::WriteResults(result,samples,output);
        for (const auto &track : samples)
            if (!track.usable)
                throw std::runtime_error("DCT continuation failed");
        std::cout << "restored_samples=" << restored << '\n'
                  << "motion_weight=" << options.motion_weight << '\n'
                  << "huber_px=" << options.huber_px << '\n'
                  << "max_iterations=" << options.max_iterations << '\n'
                  << result.global.report << '\n'
                  << "objective=" << result.global.cost << '\n'
                  << "reprojection_rmse_px=" << result.global.reprojection_rmse << '\n'
                  << "seconds=" << std::chrono::duration<double>(
                         std::chrono::steady_clock::now()-begin).count() << '\n';
        for (std::size_t c = 0; c < result.global.data.cameras.size(); ++c)
            std::cout << "camera " << c << " beta=" << result.global.data.cameras[c].beta << " frames\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
