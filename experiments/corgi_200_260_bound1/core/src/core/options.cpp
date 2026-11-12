#include "core/stba.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
namespace stba {
namespace {
std::string Trim(std::string s) {
    auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
} // namespace
Options ReadOptions(const std::filesystem::path &path) {
    Options o;
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot read config: " + path.string());
    std::map<std::string, double *> doubles = {{"motion_weight", &o.motion_weight},
                                               {"epsilon_seconds", &o.epsilon_seconds},
                                               {"min_dt", &o.min_dt},
                                               {"search_radius", &o.search_radius},
                                               {"offset_bound_frames", &o.offset_bound_frames},
                                               {"coarse_step", &o.coarse_step},
                                               {"fine_step", &o.fine_step},
                                               {"huber_px", &o.huber_px},
                                               {"dct_fps", &o.dct_fps}};
    std::map<std::string, int *> integers = {{"max_iterations", &o.max_iterations},
                                             {"reference_camera", &o.reference_camera},
                                             {"dct_max_samples", &o.dct_max_samples}};
    std::map<std::string, bool *> booleans = {{"optimize_offsets", &o.optimize_offsets},
                                              {"optimize_poses", &o.optimize_poses},
                                              {"optimize_intrinsics", &o.optimize_intrinsics}};
    std::string line;
    std::set<std::string> seen;
    while (std::getline(file, line)) {
        line = Trim(line.substr(0, line.find('#')));
        if (line.empty())
            continue;
        auto colon = line.find(':');
        if (colon == std::string::npos)
            throw std::runtime_error("Expected key: value in config");
        std::string key = Trim(line.substr(0, colon)), value = Trim(line.substr(colon + 1));
        if (!seen.insert(key).second)
            throw std::runtime_error("Duplicate config key: " + key);
        std::size_t n = 0;
        if (doubles.count(key)) {
            double v = std::stod(value, &n);
            if (n != value.size() || !std::isfinite(v))
                throw std::runtime_error("Invalid value for " + key);
            *doubles.at(key) = v;
        } else if (integers.count(key)) {
            int v = std::stoi(value, &n);
            if (n != value.size())
                throw std::runtime_error("Invalid integer for " + key);
            *integers.at(key) = v;
        } else if (booleans.count(key)) {
            if (value != "true" && value != "false")
                throw std::runtime_error("Expected true/false for " + key);
            *booleans.at(key) = value == "true";
        } else
            throw std::runtime_error("Unknown config key: " + key);
    }
    if (o.motion_weight <= 0 || o.epsilon_seconds <= 0 || o.min_dt < 0 || o.search_radius <= 0 ||
        o.coarse_step <= 0 || o.fine_step <= 0 || o.fine_step > o.coarse_step || o.huber_px < 0 ||
        o.dct_fps <= 0 || o.dct_max_samples < 2 || o.max_iterations < 1 || o.reference_camera < 0 ||
        o.offset_bound_frames < 0)
        throw std::runtime_error("Config contains out-of-range value");
    return o;
}
} // namespace stba
