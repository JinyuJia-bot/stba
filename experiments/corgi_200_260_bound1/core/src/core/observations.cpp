#include "core/stba.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace stba {
namespace {
std::string Trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}
std::vector<std::string> Split(const std::string &line) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = line.find(',', start);
        fields.push_back(
            Trim(line.substr(start, comma == std::string::npos ? comma : comma - start)));
        if (comma == std::string::npos)
            break;
        start = comma + 1;
    }
    return fields;
}
std::vector<std::vector<std::string>> ReadCSV(const std::filesystem::path &file,
                                              const std::string &header) {
    std::ifstream input(file);
    if (!input)
        throw std::runtime_error("Cannot open " + file.string());
    const auto expected = Split(header);
    std::string line;
    bool found_header = false;
    std::vector<std::vector<std::string>> rows;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line_number == 1 && line.compare(0, 3, "\xEF\xBB\xBF") == 0)
            line.erase(0, 3);
        line = Trim(line);
        if (line.empty() || line[0] == '#')
            continue;
        const auto fields = Split(line);
        if (!found_header) {
            if (fields != expected)
                throw std::runtime_error("Unexpected CSV header in " + file.string());
            found_header = true;
        } else {
            if (fields.size() != expected.size())
                throw std::runtime_error("Wrong CSV column count at " + file.string() + ":" +
                                         std::to_string(line_number));
            rows.push_back(fields);
        }
    }
    if (!input.eof())
        throw std::runtime_error("I/O error while reading " + file.string());
    if (!found_header)
        throw std::runtime_error("Missing CSV header in " + file.string());
    return rows;
}
int Integer(const std::string &field) {
    std::size_t used = 0;
    const int value = std::stoi(field, &used);
    if (used != field.size())
        throw std::invalid_argument("Invalid integer field: " + field);
    return value;
}
double Number(const std::string &field) {
    std::size_t used = 0;
    const double value = std::stod(field, &used);
    if (used != field.size() || !std::isfinite(value))
        throw std::invalid_argument("Invalid numeric field: " + field);
    return value;
}
std::ofstream Output(const std::filesystem::path &path) {
    std::ofstream file(path);
    if (!file)
        throw std::runtime_error("Cannot write " + path.string());
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file << std::setprecision(17);
    return file;
}
} // namespace

void Validate(const Dataset &data) {
    if (data.cameras.size() < 2)
        throw std::invalid_argument("Dataset needs at least two cameras");
    if (data.tracks.empty())
        throw std::invalid_argument("Dataset has no tracks");
    for (const auto &camera : data.cameras)
        if (!std::isfinite(camera.fps) || camera.fps <= 0 || !std::isfinite(camera.beta))
            throw std::invalid_argument("Camera fps must be positive and fps/beta finite");
    std::set<std::pair<int, int>> view_keys;
    std::vector<bool> camera_has_view(data.cameras.size(), false);
    for (const auto &view : data.views) {
        if (view.camera < 0 || static_cast<std::size_t>(view.camera) >= data.cameras.size() ||
            view.frame < -1)
            throw std::invalid_argument("View has an invalid camera or frame id");
        if (!view_keys.insert({view.camera, view.frame}).second)
            throw std::invalid_argument("Duplicate camera/frame view");
        camera_has_view[static_cast<std::size_t>(view.camera)] = true;
        for (double v : view.rotation)
            if (!std::isfinite(v))
                throw std::invalid_argument("Non-finite view rotation");
        for (double v : view.translation)
            if (!std::isfinite(v))
                throw std::invalid_argument("Non-finite view translation");
        for (double v : view.intrinsics)
            if (!std::isfinite(v))
                throw std::invalid_argument("Non-finite intrinsics");
        if (view.intrinsics[0] <= 0 || view.intrinsics[1] <= 0)
            throw std::invalid_argument("Focal lengths must be positive");
    }
    if (std::find(camera_has_view.begin(), camera_has_view.end(), false) != camera_has_view.end())
        throw std::invalid_argument("Every camera needs at least one view");
    std::set<int> track_ids;
    for (const auto &track : data.tracks) {
        if (track.id < 0 || !track_ids.insert(track.id).second)
            throw std::invalid_argument("Track ids must be unique nonnegative integers");
        if (track.observations.empty())
            throw std::invalid_argument("Empty track");
        if (!std::isfinite(track.motion_weight) || track.motion_weight <= 0)
            throw std::invalid_argument("Track motion weight must be positive and finite");
        std::set<std::pair<int, int>> observations;
        for (const auto &o : track.observations) {
            if (o.camera < 0 || static_cast<std::size_t>(o.camera) >= data.cameras.size() ||
                o.frame < 0 || o.view < 0 || static_cast<std::size_t>(o.view) >= data.views.size())
                throw std::invalid_argument("Observation id is out of range");
            if (!o.uv.allFinite() || !std::isfinite(o.weight) || o.weight <= 0)
                throw std::invalid_argument(
                    "Observation coordinates must be finite and weight positive");
            const View &view = data.views[static_cast<std::size_t>(o.view)];
            if (view.camera != o.camera || (view.frame != -1 && view.frame != o.frame))
                throw std::invalid_argument("Observation does not match its view");
            if (view.frame == -1 && view_keys.count({o.camera, o.frame}))
                throw std::invalid_argument(
                    "Observation must use the available frame-specific view");
            if (!observations.insert({o.camera, o.frame}).second)
                throw std::invalid_argument("Duplicate camera/frame observation within track " +
                                            std::to_string(track.id));
        }
    }
}

Dataset ReadDataset(const std::filesystem::path &directory) {
    Dataset data;
    for (const auto &row : ReadCSV(directory / "cameras.csv", "camera_id,fps,beta")) {
        if (Integer(row[0]) != static_cast<int>(data.cameras.size()))
            throw std::invalid_argument("Camera rows must have consecutive ids starting at zero");
        data.cameras.push_back({Number(row[1]), Number(row[2])});
    }
    std::map<std::pair<int, int>, int> views;
    for (const auto &row : ReadCSV(directory / "views.csv",
                                   "view_id,camera_id,frame,fx,fy,cx,cy,rx,ry,rz,tx,ty,tz")) {
        const int id = Integer(row[0]);
        if (id != static_cast<int>(data.views.size()))
            throw std::invalid_argument("View rows must have consecutive ids starting at zero");
        View view;
        view.camera = Integer(row[1]);
        view.frame = Integer(row[2]);
        for (std::size_t k = 0; k < 4; ++k)
            view.intrinsics[k] = Number(row[k + 3]);
        for (std::size_t k = 0; k < 3; ++k) {
            view.rotation[k] = Number(row[k + 7]);
            view.translation[k] = Number(row[k + 10]);
        }
        if (!views.emplace(std::make_pair(view.camera, view.frame), id).second)
            throw std::invalid_argument("Repeated camera/frame view");
        data.views.push_back(view);
    }
    std::map<int, std::size_t> tracks;
    for (const auto &row : ReadCSV(directory / "observations.csv",
                                   "track_id,is_static,motion_weight,camera_id,frame,u,v,weight")) {
        const int id = Integer(row[0]);
        const int is_static = Integer(row[1]);
        if (is_static != 0 && is_static != 1)
            throw std::invalid_argument("is_static must be 0 or 1");
        const double motion_weight = Number(row[2]);
        if (!tracks.count(id)) {
            tracks[id] = data.tracks.size();
            Track track;
            track.id = id;
            track.is_static = is_static != 0;
            track.motion_weight = motion_weight;
            data.tracks.push_back(track);
        }
        Track &track = data.tracks[tracks.at(id)];
        if (track.is_static != (is_static != 0) || track.motion_weight != motion_weight)
            throw std::invalid_argument("Inconsistent metadata within track " + std::to_string(id));
        Observation o;
        o.camera = Integer(row[3]);
        o.frame = Integer(row[4]);
        o.uv = Vec2(Number(row[5]), Number(row[6]));
        o.weight = Number(row[7]);
        auto view = views.find({o.camera, o.frame});
        if (view == views.end())
            view = views.find({o.camera, -1});
        if (view == views.end())
            throw std::invalid_argument("No view for observation camera/frame");
        o.view = view->second;
        track.observations.push_back(o);
    }
    Validate(data);
    return data;
}

void WriteDataset(const Dataset &data, const std::filesystem::path &directory) {
    Validate(data);
    std::filesystem::create_directories(directory);
    auto cameras = Output(directory / "cameras.csv");
    cameras << "camera_id,fps,beta\n";
    for (std::size_t i = 0; i < data.cameras.size(); ++i)
        cameras << i << ',' << data.cameras[i].fps << ',' << data.cameras[i].beta << '\n';
    auto views = Output(directory / "views.csv");
    views << "view_id,camera_id,frame,fx,fy,cx,cy,rx,ry,rz,tx,ty,tz\n";
    for (std::size_t i = 0; i < data.views.size(); ++i) {
        const View &view = data.views[i];
        views << i << ',' << view.camera << ',' << view.frame;
        for (double x : view.intrinsics)
            views << ',' << x;
        for (double x : view.rotation)
            views << ',' << x;
        for (double x : view.translation)
            views << ',' << x;
        views << '\n';
    }
    auto observations = Output(directory / "observations.csv");
    observations << "track_id,is_static,motion_weight,camera_id,frame,u,v,weight\n";
    for (const auto &track : data.tracks)
        for (const auto &o : track.observations)
            observations << track.id << ',' << track.is_static << ',' << track.motion_weight << ','
                         << o.camera << ',' << o.frame << ',' << o.uv.x() << ',' << o.uv.y() << ','
                         << o.weight << '\n';
}
} // namespace stba
