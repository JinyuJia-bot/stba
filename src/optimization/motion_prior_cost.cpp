#include "optimization/cost_functions.h"
#include <ceres/ceres.h>
#include <cmath>
namespace stba {
namespace {
struct FixedMotion {
    double scale;
    template <class T> bool operator()(const T *a, const T *b, T *r) const {
        for (int j = 0; j < 3; ++j)
            r[j] = T(scale) * (b[j] - a[j]);
        return true;
    }
};
struct TimedMotion {
    int fa, fb;
    double fps_a, fps_b, weight, epsilon, min_dt;
    template <class T>
    bool operator()(const T *a, const T *b, const T *beta_a, const T *beta_b, T *r) const {
        const T dt = (T(fb) - beta_b[0]) / T(fps_b) - (T(fa) - beta_a[0]) / T(fps_a);
        if (dt <= T(min_dt))
            return false; // A trial step must stay in this ordering cell.
        using std::sqrt;
        const T scale = sqrt(T(weight * 0.5) * dt) / (dt + T(epsilon));
        for (int j = 0; j < 3; ++j)
            r[j] = scale * (b[j] - a[j]);
        return true;
    }
};
struct Order {
    int fa, fb;
    double fps_a, fps_b, min_dt;
    template <class T> bool operator()(const T *a, const T *b, T *r) const {
        r[0] = T(0);
        return (T(fb) - b[0]) / T(fps_b) - (T(fa) - a[0]) / T(fps_a) > T(min_dt);
    }
};
} // namespace
ceres::CostFunction *MotionCost(double dt, double weight, double epsilon) {
    return new ceres::AutoDiffCostFunction<FixedMotion, 3, 3, 3>(
        new FixedMotion{std::sqrt(weight * dt / 2) / (dt + epsilon)});
}
ceres::CostFunction *TimedMotionCost(const Observation &a, const Observation &b, const Dataset &d,
                                     double weight, double epsilon, double min_dt) {
    return new ceres::AutoDiffCostFunction<TimedMotion, 3, 3, 3, 1, 1>(
        new TimedMotion{a.frame, b.frame, d.cameras.at(a.camera).fps, d.cameras.at(b.camera).fps,
                        weight, epsilon, min_dt});
}
ceres::CostFunction *OrderCost(int fa, double fpsa, int fb, double fpsb, double min_dt) {
    return new ceres::AutoDiffCostFunction<Order, 1, 1, 1>(new Order{fa, fb, fpsa, fpsb, min_dt});
}
} // namespace stba
