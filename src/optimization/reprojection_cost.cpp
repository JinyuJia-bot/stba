#include "optimization/cost_functions.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
namespace stba {
namespace {
struct ImageResidual {
    Observation observation;
    template <class T>
    bool operator()(const T *x, const T *rotation, const T *translation, const T *k, T *r) const {
        T p[3];
        ceres::AngleAxisRotatePoint(rotation, x, p);
        for (int j = 0; j < 3; ++j)
            p[j] += translation[j];
        if (p[2] <= T(1e-8))
            return false;
        const T w = T(std::sqrt(observation.weight));
        r[0] = w * (k[0] * p[0] / p[2] + k[2] - T(observation.uv.x()));
        r[1] = w * (k[1] * p[1] / p[2] + k[3] - T(observation.uv.y()));
        return true;
    }
};
} // namespace
ceres::CostFunction *ReprojectionCost(const Observation &observation) {
    return new ceres::AutoDiffCostFunction<ImageResidual, 2, 3, 3, 3, 4>(
        new ImageResidual{observation});
}
} // namespace stba
