#include "core/stba.h"
#include <chrono>
#include <iostream>
#include <numeric>
#include <stdexcept>

int main(int argc, char **argv) {
    try {
        std::filesystem::path input, output = "experiments/result", config;
        std::string mode = "full";
        bool poses = false, intrinsics = false, dct = true;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help" || key == "-h") {
                std::cout
                    << "stba --input DATASET --output RESULTS [--config configs/default.yaml]\n"
                    << "     [--mode full|incremental|fixed] [--optimize-poses] "
                       "[--optimize-intrinsics] [--no-dct]\n"
                    << "full: pairwise -> MST -> incremental -> global -> DCT\n"
                    << "incremental: omit final global refinement; fixed: optimize trajectories at "
                       "input offsets\n";
                return 0;
            }
            if (key == "--optimize-poses") {
                poses = true;
                continue;
            }
            if (key == "--optimize-intrinsics") {
                intrinsics = true;
                continue;
            }
            if (key == "--no-dct") {
                dct = false;
                continue;
            }
            if (i + 1 == argc)
                throw std::runtime_error("Missing value for " + key);
            std::string value = argv[++i];
            if (key == "--input")
                input = value;
            else if (key == "--output")
                output = value;
            else if (key == "--config")
                config = value;
            else if (key == "--mode")
                mode = value;
            else
                throw std::runtime_error("Unknown argument: " + key);
        }
        if (input.empty())
            throw std::runtime_error("--input is required; run tools/generate_synthetic.py first");
        if (mode != "full" && mode != "incremental" && mode != "fixed")
            throw std::runtime_error("Unknown mode: " + mode);
        // Output datasets are reusable inputs, but never overwrite the source dataset.
        if (std::filesystem::weakly_canonical(input) == std::filesystem::weakly_canonical(output))
            throw std::runtime_error("Input and output must be different directories");
        auto options = config.empty() ? stba::Options{} : stba::ReadOptions(config);
        options.optimize_poses = options.optimize_poses || poses;
        options.optimize_intrinsics = options.optimize_intrinsics || intrinsics;
        const auto data = stba::ReadDataset(input);
        if (options.reference_camera >= static_cast<int>(data.cameras.size()))
            throw std::runtime_error("Reference camera out of range");
        const auto start = std::chrono::steady_clock::now();
        stba::PipelineResult result;
        if (mode == "fixed") {
            std::vector<int> active(data.cameras.size());
            std::iota(active.begin(), active.end(), 0);
            auto fixed = options;
            fixed.optimize_offsets = false;
            fixed.optimize_poses = false;
            fixed.optimize_intrinsics = false;
            result.global = stba::Optimize(data, active, fixed);
            result.incremental = result.global;
            result.order = active;
        } else {
            result = stba::Reconstruct(data, options, mode == "full");
        }
        if (!result.global.usable)
            throw std::runtime_error("Reconstruction failed: " + result.global.report);
        // Save the reconstruction before a potentially expensive resampling stage.
        stba::WriteResults(result, {}, output);
        if (dct) {
            auto samples = stba::ResampleDCT(result.global, options);
            stba::WriteResults(result, samples, output);
            for (const auto &track : samples)
                if (!track.usable)
                    throw std::runtime_error("DCT track failed; see dct_metrics.csv");
        }
        std::cout << result.global.report << "\nobjective=" << result.global.cost
                  << " reprojection_rmse_px=" << result.global.reprojection_rmse << "\n";
        for (std::size_t c = 0; c < data.cameras.size(); ++c)
            std::cout << "camera " << c << " beta=" << result.global.data.cameras[c].beta
                      << " frames\n";
        std::cout << "seconds="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << "\noutput=" << output.string() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
