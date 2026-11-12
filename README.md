# STBA 复现项目

复现论文 **Spatiotemporal Bundle Adjustment for Dynamic 3D Reconstruction**（Minh Vo、Srinivasa G. Narasimhan、Yaser Sheikh）的主要优化流程。

这是一个可编译、可运行、可做对照实验的 C++17 项目。输入是相机初值和已经关联身份的二维轨迹，输出是亚帧时间偏移、三维轨迹、可选的相机参数优化结果和 DCT 重采样轨迹。固定机位与逐帧移动相机使用同一套数据格式。

工程保持一层数据结构和普通函数的组织方式：Eigen 负责几何，Ceres 负责非线性最小二乘；没有插件系统、继承层级、通用图框架或自建优化器。Python 只负责生成、转换和评价数据。

## 1. 先跑通

建议 Ubuntu 22.04/24.04 或兼容环境，C++17、CMake ≥ 3.16、Eigen ≥ 3.3、Ceres ≥ 2.1（含稀疏线性求解支持）。本次验证使用 GCC 13.3、Eigen 3.4、Ceres 2.2。

```bash
sudo apt install build-essential cmake libeigen3-dev libceres-dev
python3 -m pip install -r requirements.txt

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# 压缩包内已有三相机输入和参考输出；下面可直接重新求解
./build/stba --input examples/three_cameras/input --output experiments/my_first_run

python3 tools/evaluate_alignment.py \
  --input examples/three_cameras/input --result experiments/my_first_run
python3 tools/plot_trajectories.py \
  --input examples/three_cameras/input --result experiments/my_first_run --track 0
```

也可以自己生成输入：

```bash
python3 tools/generate_synthetic.py --out experiments/synthetic/input \
  --cameras 3 --frames 24 --tracks 4 --static-points 8 \
  --beta 0,0.31,-0.27 --noise 0.1 --seed 42
./build/stba --input experiments/synthetic/input --output experiments/synthetic/result
```

命令从项目根目录执行。输入目录与输出目录必须不同；多次实验请使用不同输出目录。生成器和求解器的同名输出文件会被更新，批量实验工具则要求输出目录为空。

## 2. 已实现的流程

| 环节 | 实际行为 | 主要文件 |
| --- | --- | --- |
| 数据与相机 | 严格 CSV 检查、针孔投影、角轴旋转、逐帧位姿、静态/动态轨迹 | `src/core/` |
| 初始三维点 | DLT 三角化；其他视角二维插值仅用于建立初值 | `trajectory.cpp` |
| 成对时间对齐 | 固定偏移的粗细网格搜索，每个候选优化三维轨迹 | `pairwise_alignment.cpp` |
| 相机排序 | 三角环一致性、轨迹数、基线与代价构成边权，Kruskal 最小生成树 | `incremental_stba.cpp` |
| 增量 STBA | 枚举新增相机的时间顺序区间，在区间内优化偏移和轨迹 | `incremental_stba.cpp` |
| 全局 STBA | 联合优化全部轨迹与偏移，可选优化位姿和焦距 | `global_stba.cpp` |
| 残差 | 加权重投影、最小动能运动先验、时间顺序约束 | 两个 `*_cost.cpp` |
| DCT 重采样 | 完整 DCT-II 系数、原始二维重投影、频域运动代价 | `dct_resampling.cpp` |
| 数据工具 | 合成/MoCap 输入、COLMAP 导入、ArUco 轨迹提取、评价、绘图、批量实验 | `tools/` |

`src/main.cpp` 负责参数和串联，`src/core/stba.h` 集中定义共享数据及函数接口。输出、参数解析单独放在 `results.cpp`、`options.cpp`，避免算法文件混入文件操作。

## 3. 三个运行模式

```bash
# 对照：采用输入时间偏移，只优化三维轨迹，仍有运动先验
./build/stba --input DATA --output RESULT_FIXED --mode fixed

# 成对对齐、相机排序、增量优化，跳过最后一次全局优化
./build/stba --input DATA --output RESULT_INCREMENTAL --mode incremental

# 默认：完整时间/轨迹流程，保留输入相机标定
./build/stba --input DATA --output RESULT_FULL --mode full

# 在最后的全局阶段联合优化位姿、fx/fy
./build/stba --input DATA --output RESULT_JOINT --mode full \
  --optimize-poses --optimize-intrinsics
```

默认固定相机标定，方便先验证时间和轨迹部分。使用联合优化时需提供可靠的相机初值和足够的静态点/多视角约束；主点 cx/cy 始终固定，未建模畸变。移动相机用每帧独立视图，不添加额外相机平滑先验。

**实测发现自由位姿+焦距优化会发生标定漂移：残差降低但三维精度恶化。** 因此先使用默认固定标定；需要微调时先只加 `--optimize-poses`。这项限制和未改善的测试结果都保留在 VALIDATION.md，不把“求解成功”当成“标定准确”。

三个模式默认都会做 DCT；添加 `--no-dct` 可跳过。两个相机的 `incremental` 结果只经过成对网格搜索，`full` 才继续连续细化。

## 4. 时间、目标函数与规范

时间模型：

```text
frame = fps * t + beta
t = (frame - beta) / fps
```

beta 的单位是本相机的帧，fps 已知且固定。正 beta 表示同一帧号对应更早的世界时间。不同帧率的相机比较偏移时，要先用 beta/fps 转成秒。

默认目标是：

```text
S = sum_observations weight * ||project(X) - uv||²
    + lambda * sum_dynamic_tracks motion_weight
      * sum_adjacent_samples dt / (2*(dt+epsilon)²) * ||X_next-X||²
```

Ceres 内部使用全局 1/2；输出的各项代价已换回上式。重投影 RMSE 是未加权的每个二维观测欧氏像素误差；它与加权目标值不同。静态轨迹共用一个三维点，不加运动先验。动态轨迹同一时刻的观测共用一个三维位置。

固定一个相机的 beta 消除全局时间平移自由度。位姿联合优化时固定参考相机的一个视图位姿，并固定另一视图的一项平移分量约束尺度。米制尺度来自输入标定；仅靠无尺度的多视角图像无法恢复绝对米制尺度。

原始观测不被插值替换，输入的真值文件不参与求解。只在当前至少两台活动相机可见的轨迹上重建；评价同时报告覆盖率。

## 5. 配置与结果

`configs/default.yaml` 是严格的平面 `key: value` 配置，不是完整 YAML 解析器。未知项、重复项、非法值会报错：

```bash
./build/stba --input DATA --output RESULT --config configs/default.yaml
```

主要可调参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| motion_weight | 1000 | 全局运动代价权重；与长度单位和噪声相关 |
| epsilon_seconds | 1e-6 | 运动项时间分母稳定量，单位秒 |
| search_radius | 0.75 | 网格/插入搜索半径，单位为被搜索相机的帧 |
| coarse_step / fine_step | 0.05 / 0.005 | 成对搜索步长，单位帧 |
| max_iterations | 80 | 每个局部 Ceres 求解的迭代上限 |
| reference_camera | 0 | 时间基准相机 |
| huber_px | 0 | 0 表示平方损失；正数启用加权重投影残差上的 Huber |
| dct_fps / dct_max_samples | 60 / 256 | 重采样目标频率与每条轨迹样本上限 |

默认权重用于自带米制合成数据，没有保证适合所有场景。显著不同的速度、尺度或像素噪声需做权重敏感性实验。若全部相机间偏移相差数帧，先完成整数帧粗对齐，再将初值写入 cameras.csv。

| 输出 | 含义 |
| --- | --- |
| cameras.csv / views.csv / observations.csv | 优化后可再次读取的数据 |
| trajectories.csv | 全局结果按原始观测展开的三维点与估计时刻 |
| incremental_trajectories.csv | 最后一次增量结果 |
| pairwise.csv | 成对偏移、代价、基线、共同轨迹数和边权 |
| order.csv / slots.csv | 相机顺序、插入候选及被选候选 |
| metrics.csv / solver_report.txt | 阶段代价、像素误差与 Ceres 状态 |
| dct.csv / dct_metrics.csv | 均匀时间采样及 DCT 优化前后代价 |

`usable=true` 表示 Ceres 认为当前结果可用且顺序检查通过，不代表全局最优。DCT 代价在自己的均匀网格目标中比较；不应直接与非均匀 STBA 代价相减。若 DCT 超出样本上限，程序报错并保留已经完成的 STBA 结果；可以缩短片段、提高上限或使用 `--no-dct`。

## 6. 接真实数据与 MoCap

- 数据格式、单位、缺失观测及逐帧视图：见 [DATA_FORMAT.md](DATA_FORMAT.md)。
- COLMAP 文本模型导入、视频 ArUco 角点提取、评价和批处理：见 [tools/TOOLS.md](tools/TOOLS.md)。
- MoCap CSV/C3D 转换和异步投影实验：见 [experiments/mocap/README.md](experiments/mocap/README.md)。
- DCT 网格、完整基函数、权重推导：见 [DCT_NOTES.md](DCT_NOTES.md)。

**COLMAP 是可选的数据准备工具，C++ 求解器不链接 COLMAP。** 导入器读取已注册图像的位姿、针孔内参和静态点关联；动态轨迹需要自己提供，或使用本项目的 ArUco 提取工具。项目没有自动完成任意自然场景动态物体的跨相机匹配。

## 7. 阅读和修改顺序

先看 `stba.h` 的数据，再看 `camera.cpp`、两个残差文件和 `global_stba.cpp`，最后看成对、增量与 DCT。第一遍可以暂时跳过 CSV 和结果输出细节。

需要改运动先验时，修改 `motion_prior_cost.cpp`，同步更新 `global_stba.cpp` 中的代价统计和 DCT 权重；需要改相机模型时，修改投影与重投影残差，随后更新输入格式。不要只改残差而留下不一致的报告或评价。

## 8. 验证与复现范围

```bash
ctest --test-dir build --output-on-failure
python3 tests/test_tools.py
python3 tests/test_mocap_import.py
python3 tests/run_integration.py --binary build/stba --work experiments/checks --extended
```

测试记录、数值结果和环境见 [VALIDATION.md](VALIDATION.md)。包内示例保留输入、独立真值和参考输出，便于检查自己环境的结果；数值优化不承诺跨版本逐字节一致。

本工程实现主要算法链和实验工具，不宣称已经复现论文的全部数据集、表格或作者内部实现。特别说明：

1. 论文中的观测噪声、深度/运动相关权重没有完整自动估计；通过观测 weight、轨迹 motion_weight 和全局 lambda 显式设置，默认均匀权重。
2. 无三角环的稀疏相机图使用一帧时长作为环误差回退量；这是让双相机及稀疏图可运行的明确选择。Nij 按共同轨迹数计算。
3. 时间顺序在优化步中限制，并在结束后复核。恰好同时或间隔不大于 epsilon 的事件会固定涉及相机的偏移，防止接近零间隔钻运动代价的数值空子。这比仅保持观测顺序更保守，无法让一个同步相机子组作为整体自由移动。
4. DCT 使用完整 DCT-II 与开放链一阶差分的精确离散权重。论文正文未展开所有边界和网格细节，因此这是一种明确、可检验的离散实现选择。
5. 不估计帧率漂移、滚动快门、曝光时间或镜头畸变；不处理严重错误的轨迹身份；不自动补出未注册帧的相机位姿。
6. 成对全搜索、逐候选优化和完整 DCT 优先保证可理解性，面向短片段复现实验，没有做大规模系统性能优化。
