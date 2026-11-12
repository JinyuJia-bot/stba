cmake_minimum_required(VERSION 3.16)
project(corgi_window_refine LANGUAGES CXX)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres 2.2 REQUIRED)
get_filename_component(STBA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_library(stba_core_cached STATIC IMPORTED)
set_target_properties(stba_core_cached PROPERTIES
    IMPORTED_LOCATION "${STBA_ROOT}/build/libstba_core.a"
    INTERFACE_INCLUDE_DIRECTORIES "${STBA_ROOT}/src")
target_link_libraries(stba_core_cached INTERFACE Eigen3::Eigen Ceres::ceres)
add_executable(refine_global refine_global.cpp)
target_compile_features(refine_global PRIVATE cxx_std_17)
target_compile_options(refine_global PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(refine_global PRIVATE stba_core_cached)
