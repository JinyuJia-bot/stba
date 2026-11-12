cmake_minimum_required(VERSION 3.16)
project(stba_reproduction VERSION 1.0 LANGUAGES CXX)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres 2.2 REQUIRED)
add_library(stba_core
  src/core/camera.cpp src/core/trajectory.cpp src/core/observations.cpp
  src/core/options.cpp src/core/results.cpp
  src/optimization/reprojection_cost.cpp src/optimization/motion_prior_cost.cpp
  src/optimization/pairwise_alignment.cpp src/optimization/incremental_stba.cpp
  src/optimization/global_stba.cpp src/resampling/dct_resampling.cpp)
target_include_directories(stba_core PUBLIC src)
target_compile_features(stba_core PUBLIC cxx_std_17)
target_link_libraries(stba_core PUBLIC Eigen3::Eigen Ceres::ceres)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(stba_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_executable(stba src/main.cpp)
target_link_libraries(stba PRIVATE stba_core)
include(CTest)
if(BUILD_TESTING)
  add_executable(test_core tests/test_core.cpp)
  target_link_libraries(test_core PRIVATE stba_core)
  add_test(NAME core_geometry_objective_dct COMMAND test_core)
endif()
