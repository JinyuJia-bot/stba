#pragma once
#include "core/stba.h"
namespace ceres {
class CostFunction;
}
namespace stba {
ceres::CostFunction *ReprojectionCost(const Observation &);
ceres::CostFunction *MotionCost(double dt, double weight, double epsilon);
ceres::CostFunction *TimedMotionCost(const Observation &, const Observation &, const Dataset &,
                                     double weight, double epsilon, double min_dt);
ceres::CostFunction *OrderCost(int frame_a, double fps_a, int frame_b, double fps_b, double min_dt);
} // namespace stba
