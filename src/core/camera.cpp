#include "core/stba.h"

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <cmath>
#include <stdexcept>

namespace stba {
namespace {
Eigen::Matrix3d Rotation(const View &view) {
    const Vec3 aa(view.rotation[0], view.rotation[1], view.rotation[2]);
    const double angle = aa.norm();
    if (angle < 1e-15)
        return Eigen::Matrix3d::Identity();
    return Eigen::AngleAxisd(angle, aa / angle).toRotationMatrix();
}
Eigen::Matrix<double, 3, 4> ProjectionMatrix(const View &view) {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = view.intrinsics[0];
    K(1, 1) = view.intrinsics[1];
    K(0, 2) = view.intrinsics[2];
    K(1, 2) = view.intrinsics[3];
    Eigen::Matrix<double, 3, 4> Rt;
    Rt.leftCols<3>() = Rotation(view);
    Rt.col(3) = Vec3(view.translation[0], view.translation[1], view.translation[2]);
    return K * Rt;
}
} // namespace

double Time(const Dataset &data, const Observation &observation) {
    const Camera &camera = data.cameras.at(static_cast<std::size_t>(observation.camera));
    return (static_cast<double>(observation.frame) - camera.beta) / camera.fps;
}
Vec3 CameraCenter(const View &view) {
    return -Rotation(view).transpose() *
           Vec3(view.translation[0], view.translation[1], view.translation[2]);
}
Vec2 Project(const View &view, const Vec3 &world) {
    const Vec3 p = Rotation(view) * world +
                   Vec3(view.translation[0], view.translation[1], view.translation[2]);
    if (!p.allFinite() || p.z() <= 1e-8)
        throw std::runtime_error("Projection: point is behind the camera or non-finite");
    return {view.intrinsics[0] * p.x() / p.z() + view.intrinsics[2],
            view.intrinsics[1] * p.y() / p.z() + view.intrinsics[3]};
}
Vec3 Triangulate(const View &a, const Vec2 &uv_a, const View &b, const Vec2 &uv_b) {
    if ((CameraCenter(a) - CameraCenter(b)).norm() < 1e-8)
        throw std::runtime_error("Triangulation: zero camera baseline");
    const auto Pa = ProjectionMatrix(a), Pb = ProjectionMatrix(b);
    Eigen::Matrix4d A;
    A.row(0) = uv_a.x() * Pa.row(2) - Pa.row(0);
    A.row(1) = uv_a.y() * Pa.row(2) - Pa.row(1);
    A.row(2) = uv_b.x() * Pb.row(2) - Pb.row(0);
    A.row(3) = uv_b.y() * Pb.row(2) - Pb.row(1);
    const Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    const Eigen::Vector4d h = svd.matrixV().col(3);
    if (std::abs(h.w()) < 1e-12 || svd.singularValues()[2] < 1e-10 * svd.singularValues()[0])
        throw std::runtime_error("Triangulation: degenerate rays or point at infinity");
    const Vec3 world = h.head<3>() / h.w();
    Project(a, world);
    Project(b, world); // finite coordinates and positive depth
    return world;
}
} // namespace stba
