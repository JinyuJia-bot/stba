# STBA Reproduction

复现 CVPR 2016 论文 *Spatiotemporal Bundle Adjustment for Dynamic 3D Reconstruction*。

论文：https://openaccess.thecvf.com/content_cvpr_2016/html/Vo_Spatiotemporal_Bundle_Adjustment_CVPR_2016_paper.html

## 技术栈

- C++：Ceres Solver、Eigen、OpenCV；后期使用 COLMAP 初始化相机与静态点。
- Python：NumPy、Matplotlib，用于数据生成、评估和绘图。
- 工程：CMake、GoogleTest；计划用 yaml-cpp 读取配置。

## 目录

- `include/stba/`：公共头文件。
- `src/core/`：相机、轨迹、观测与时间模型。
- `src/initialization/`：三角化初始化。
- `src/optimization/`：重投影与运动先验、两两对齐、相机排序、增量与全局 STBA。
- `src/resampling/`：DCT 轨迹重采样。
- `src/io/`：数据集与 COLMAP 数据读写。
- `apps/`：实验程序入口。
- `configs/`：实验配置。
- `tools/`：合成数据、误差评估与绘图脚本。
- `tests/`：数值与算法测试。
- `experiments/`：合成数据与 Mocap 实验。

## 实现计划

1. 双相机、固定标定、已知对应关系的合成实验。
2. 多相机排序与增量时间对齐。
3. 相机、静态点、动态轨迹与时间偏移的联合优化，以及 DCT 重采样。
4. Mocap 实验与真实数据接入。

当前仅创建项目骨架，源码、配置、测试与 CMakeLists.txt 均为空文件，暂不可构建或运行。
