# 数据准备与结果分析

下面的命令从项目根目录执行。求解器使用 CSV 输入；Python 工具负责准备数据和分析结果。

## 合成实验

```bash
python3 tools/generate_synthetic.py --out experiments/synthetic/demo --cameras 4 --noise 0.1 --seed 42
./build/stba --input experiments/synthetic/demo --output experiments/synthetic/demo_result
python3 tools/evaluate_alignment.py --input experiments/synthetic/demo --result experiments/synthetic/demo_result
python3 tools/plot_trajectories.py --input experiments/synthetic/demo --result experiments/synthetic/demo_result --track 0
```

生成器与求解器参数详见各自 `--help`。其中 `truth.csv`、`true_cameras.csv`、`true_views.csv` 仅用于评价，不能作为求解器初值来源。

批量实验保留失败项和对应日志，不会把失败结果从报告中悄悄删除：

```bash
python3 tools/run_experiments.py --binary build/stba --out experiments/batch \
  --cameras 3,5 --noise 0,0.5 --seeds 1,2,3 --modes fixed,incremental,full
```

输出目录必须为空。`fixed` 是使用输入偏移、优化轨迹的对照，不应描述为“不含运动先验的纯几何法”。每个案例共用同一份输入和真值，汇总表包含重建覆盖率、相对时间偏移误差以及静态/动态点的三维误差。

## 导入 COLMAP 相机初值和静态点关联

先通过 COLMAP 导出文本模型。支持 `PINHOLE`、`SIMPLE_PINHOLE`，其余模型会明确报错。带畸变的图片与观测坐标需要先去畸变，并使用与这些坐标一致的针孔内参；不能只把模型名称改成 PINHOLE。

准备映射表 `image_mapping.csv`：

```csv
image_name,camera_id,frame,fps,beta
cam0/000000.png,0,0,30,0
cam0/000001.png,0,1,30,0
cam1/000000.png,1,0,30,0
cam1/000001.png,1,1,30,0
```

`image_name` 必须与 COLMAP `images.txt` 中的图像名称完全一致。这里的 `camera_id` 表示一条视频流，从 0 连续编号；它与 COLMAP 用来共享内参的 CAMERA_ID 不是同一概念。`frame` 是对应视频的原始帧号；`fps` 与初始 `beta` 在同一视频流内保持一致。

```bash
python3 tools/import_colmap.py --model data/colmap_text --mapping image_mapping.csv --out data/imported
```

COLMAP 的尺度通常不能直接视为米。用已知基线或标尺确定“每个 COLMAP 长度单位对应多少米”，通过 `--scale` 传入；默认 `1` 表示输入已经使用米。程序按该比例缩放平移量，三角化得到的点也相应使用米。尺度会影响运动先验与像素代价的相对权重。

转换结果包含逐帧世界到相机位姿、内参、按 POINT3D_ID 关联的静态二维观测，以及原始静态点 ID 映射。三维点的 COLMAP 坐标不被当成轨迹真值。用于动态对象的二维轨迹需要另行提供：

```bash
python3 tools/import_colmap.py --model data/colmap_text --mapping image_mapping.csv \
  --dynamic-observations dynamic_observations.csv --out data/with_dynamic_tracks
```

已有动态轨迹 ID 会保留，导入的静态轨迹从新的 ID 开始，避免碰撞。`--no-static-points` 只导出相机/视图与显式提供的动态观测。

该工具读取已注册图像的位姿，不负责运行 COLMAP、注册缺失帧或识别动态匹配。每个参与 STBA 的观测都必须有对应视图；静态相机也可以在 `views.csv` 中使用 `frame=-1` 的共享视图。

## 从标定视频提取 ArUco 角点轨迹

可选依赖：

```bash
python3 -m pip install 'opencv-python-headless>=4.7'
```

准备 `videos.csv`，相对路径以该 CSV 的目录为基准：

```csv
camera_id,path,frame_offset
0,cam0.mp4,0
1,cam1.mp4,0
```

相机标定目录包含 `cameras.csv`、`views.csv`，可选包含静态观测 `observations.csv`。

```bash
python3 tools/extract_markers.py --calibration data/calibration --videos videos.csv \
  --dictionary DICT_4X4_50 --out data/marker_tracks
./build/stba --input data/marker_tracks --output results/marker_tracks
```

每个角点以“marker ID + 规范角点序号”建立跨视角身份。已存在的观测会保留，新动态轨迹采用新的 ID。视频中的每个物理标签必须具有独一无二、全程不变的 ID，并使用同一字典；同一帧发现重复标签 ID 时会报错。

`--every 2` 读取每隔一帧的画面，保留原始帧号和 fps。`frame_offset` 用于表示一个视频片段对应完整序列的起始帧号。若需要制造异步降采样序列，应同时正确更新输出帧号、fps 和初始偏移；单纯改时间标签不等于改变曝光时刻。

视频应已经去畸变，且视频分辨率、坐标与内参相匹配。静态相机使用共享视图；运动相机需要对应帧的位姿。脚本检查解码 fps 与标定文件的一致性。提取结果只使用标签提供的二维身份，**没有把标签尺寸、角点间距或刚体约束加入 STBA**。

## 指标含义

`evaluate_alignment.py` 没有真值时只报告求解器的残差与重建数量。提供真值时：

- 三维误差按 `(track_id,camera_id,frame)` 匹配同一次物理观测，报告米为单位的 RMSE。
- 默认直接在输入世界坐标系中评价。相机联合优化可能改变空间尺度规范，因此也支持 `--align-similarity`，额外报告用全部匹配点拟合一个全局 Sim(3) 后的误差。原始误差始终保留，两种指标会明确标记。该选项需要 NumPy。
- DCT 误差在每条原始观测的估计时刻对导出的均匀网格进行线性插值，再匹配该次物理观测的三维真值；该指标包含网格插值误差，并报告匹配数量。它是原始世界坐标误差，不自动使用相似变换对齐。
- 时间偏移先转换成 `beta/fps` 秒，再相对最小公共 camera_id 消除全局时间原点差异；每台相机的帧误差使用自身真实 fps 换算。
- 重建覆盖率和误差一起报告，避免通过丢弃困难轨迹得到虚假的低误差。
- 图表默认展示未经空间对齐的坐标；空间规范不同时应先理解原始误差与对齐误差的区别。

## 重建可视化

有独立真值时，可同时查看真实轨迹、STBA 三维采样点、DCT 输出、原始二维观测和逐点误差：

```bash
python3 tools/plot_reconstruction.py \
  --input experiments/mocap/02_01_10cams_41tracks/input \
  --result experiments/mocap/02_01_10cams_41tracks/result \
  --source experiments/mocap/02_01_10cams_41tracks/trajectory.csv \
  --tracks 0,15,39 \
  --out experiments/mocap/02_01_10cams_41tracks/plots
```

`--input` 需有 `truth.csv` 和 `true_cameras.csv`。可选 `--source` 是生成实验时使用的 MoCap CSV，必须保持同一空间坐标系，并由输入的 `generation.json` 提供源时间原点及轨迹 ID 映射；不传时用观测真值点连接成参考轨迹。省略 `--tracks` 时，自动为 RMSE 最大、中位排名和最小的轨迹生成详细图。

输出包括 `overview.png`、选中轨迹的详细 PNG、覆盖全部动态轨迹的 `all_tracks.pdf` 和 `per_track_metrics.csv`。详细图左上是三维轨迹叠加，右上是最差观测的局部放大，左下是实际二维输入与重投影，右下是按同一次物理观测匹配的三维误差。三维蓝点是重建结果；实际求解输入是二维像素点。没有做空间对齐或放大误差向量；DCT 曲线及其评价使用导出网格的线性插值。

两条曲线过于重合时，可添加 `--curve-separation-mm 40`，将三维图中的 GT 沿 Z 方向下移 20 mm、DCT 上移 20 mm。图例会标注显示偏移；原始数据、局部误差放大图、二维观测和误差统计不使用该偏移。设为 `0` 恢复原坐标叠加。建议用另一个 `--out` 目录保留两种视图。

## 工具验证

```bash
python3 tests/test_tools.py
```

测试包括非连续 COLMAP ID、四元数转轴角、静态/动态轨迹 ID 合并、畸变模型拒绝、已有输出保护，以及时间规范消除和可选全局相似变换评价。安装 OpenCV 后，还会生成两段移动 ArUco 图案视频，检查四个角点在两路视频中的稳定身份和帧号。该检测测试已在 OpenCV 4.12.0 上运行通过；真实相机数据效果仍取决于标定、清晰度、遮挡和标签可见性。

格式及 API 参考：

- COLMAP 官方格式：<https://colmap.github.io/format.html>
- OpenCV 官方 ArUco 检测教程：<https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html>
