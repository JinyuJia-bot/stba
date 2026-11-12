# 数据格式

求解器读取一个目录中的三个 CSV：`cameras.csv`、`views.csv`、`observations.csv`。
所有列均为数字，首行列名和顺序必须与下方一致。允许空行、以 `#` 开头的整行注释和 UTF-8 BOM；不支持带引号或逗号的字符串字段。缺失观测直接省略该行。

- 相机和 view 的 id 必须从 0 开始连续，CSV 按 id 排列。
- `track_id` 可不连续，但必须是非负整数；同一 id 代表同一物理点。
- 相机坐标满足 `X_camera = R(angle_axis) * X_world + translation`，不是 camera-to-world。
- 长度单位米，时间单位秒，旋转 angle-axis 单位弧度。
- 时间模型为 `frame = fps * t + beta`；`beta` 单位是各相机自己的帧间隔。
- 帧号是非负整数，从 0 开始。已知整数帧粗对齐可通过输入 `beta` 表达。
- 输入像素应已去畸变，并与输入针孔内参一致。当前四个内参为 `fx,fy,cx,cy`。

## cameras.csv

```csv
camera_id,fps,beta
0,30,0
1,30,0
```

这里的 `beta` 是初始估计，不是评价真值。各相机 fps 可以不同，且在求解中视为已知常量。

## views.csv

```csv
view_id,camera_id,frame,fx,fy,cx,cy,rx,ry,rz,tx,ty,tz
0,0,-1,1000,1000,640,360,0,0,0,0,0,5
1,1,-1,1000,1000,640,360,0,0,0,-1,0,5
```

`frame=-1` 表示该相机所有帧共享这组相机参数，适合固定机位。手持/移动相机使用每帧一行，frame 为实际帧号。观察某一帧时，优先匹配该帧的 view，找不到才使用 `frame=-1` 的共享 view。每个 `(camera_id,frame)` 最多一个 view。

旋转、平移、内参可以逐帧变化。本项目的优化不会自动补出没有提供的相机位姿；输入每条二维观测都必须能找到匹配 view。

## observations.csv

```csv
track_id,is_static,motion_weight,camera_id,frame,u,v,weight
0,0,1,0,0,640,360,1
0,0,1,1,0,440,360,1
1,1,1,0,0,660,360,1
1,1,1,1,0,460,360,1
```

- `is_static=1`：整个轨迹共用一个三维点；`is_static=0`：不同观测时间拥有不同的三维位置。
- `weight` 是该观测对重投影平方误差的乘数，必须为正数。它不是标准差；若使用像素标准差 σ，可自行设置成 `1/σ²`。
- `motion_weight` 是轨迹运动先验的相对乘数，必须为正数，再与全局配置的运动权重相乘。静态点不使用运动残差。
- 同一轨迹的 `is_static` 和 `motion_weight` 必须一致，同一 `(track_id,camera_id,frame)` 不得重复。

优化活动相机集合中，至少有两台相机观测到的轨迹才加入此次重建。这些轨迹在活动相机中的全部原始观测都会进入求解。三维初值可通过二维插值/外推建立；插值仅用于初始化，不作为新的优化观测。动态观测时间恰好相同时共享一个三维变量，静态轨迹始终共享一个三维变量。

若一条满足上述条件的轨迹仍无法找到有效的正深度三角化初值，初始化会报错，不会默默删除它来降低候选偏移的代价。

## 合成数据与评价真值

从项目根目录运行：

```bash
python3 tools/generate_synthetic.py --out experiments/synthetic/input
python3 tools/generate_synthetic.py --cameras 3 --frames 24 --beta 0,0.31,-0.27 --noise 0.1 --drop-rate 0.1 --out experiments/synthetic/missing
python3 tools/generate_synthetic.py --moving --pose-noise 0.005 --out experiments/synthetic/moving
```

生成器仅依赖 NumPy。可用 `--fps-list 30,24,20` 指定每台相机不同的已知帧率；否则所有相机使用 `--fps`。`--pose-noise` 同时控制角轴扰动（弧度）和平移扰动（米）的标准差；`--intrinsics-noise` 控制焦距的对数扰动标准差。为提供受控空间基准，这些标定扰动仅施加于 camera_id≥2，前两台相机保持准确。

默认有 4 台相机、每台 30 帧、6 条动态轨迹、20 个静态点，30 fps，像素坐标噪声标准差 0.1 px。默认时间真值依次为 `[0,0.37,-0.23,0.11]` frame，输入相机的初始 beta 全为 0。通过真正不同的采样时刻计算三维位置后再投影，生成异步观测。

生成器额外写入以下评价文件，求解器不读取：

- `truth.csv`：`track_id,camera_id,frame,x,y,z`，每条输入观测对应的真实三维位置。
- `true_cameras.csv`：`camera_id,fps,beta`，真实时间偏移。
- `true_views.csv`：列名与 `views.csv` 相同，真实相机参数。
- `generation.json`：生成实验所用参数和随机种子。

`WriteDataset()` 只输出三个求解输入 CSV，便于导出优化后的相机参数和重新运行；不复制任何真值文件。

## MoCap 轨迹输入

MoCap 用于生成带真实人体运动的异步二维观测；输入的是三维标记点轨迹，不是从视频自动提取的人体骨架。可选 C3D 读取器使用 [ezc3d 官方 Python API](https://github.com/pyomeca/ezc3d)：

```bash
python3 -m pip install ezc3d
python3 tools/import_mocap.py data/recording.c3d --out data/trajectory.csv --center
python3 tools/generate_synthetic.py --trajectory-csv data/trajectory.csv --tracks 12 --frames 30 --out experiments/mocap/input
```

也可直接转换通用长表 CSV：

```csv
track_id,frame,x,y,z
4,0,200,100,300
4,1,205,101,300
```

```bash
python3 tools/import_mocap.py data/markers.csv --fps 120 --units mm --center --out data/trajectory.csv
```

CSV 可以用秒单位的 `time` 列替代 `frame`，此时不需要 `--fps`。CSV 坐标默认是米，也可用 `--units mm` 或 `--units cm` 指明。C3D 默认读取 `POINT:UNITS` 和 `POINT:RATE`，缺失/未知单位要求显式指定，残差为负的无效标记点被省略。空白或非有限 CSV 坐标也被省略。ASF/AMC 是骨架与关节角数据，本项目不直接解析；请先使用对应骨架的前向运动学工具导出 XYZ 轨迹。

导入器输出标准的 `track_id,time,x,y,z`，XYZ 单位始终为米。整体时间原点移到首个有效样本，`--center` 仅施加一次固定的全局平移，使记录包围盒中心位于原点；不会逐帧去掉人体平移。标签、原单位、固定平移和时间原点保存在同名 `.metadata.json`。只有至少含两个有效样本的轨迹被保留，可用 `--max-tracks` 限制数量。

生成器的 `--tracks` 取按源 id 排序的前 N 条有效轨迹，输出动态 id 从 0 重新编号；源 id 对应关系记录在 `generation.json` 的 `source_track_ids`。生成器将全部相机的 `(frame-beta)/fps` 平移到所选轨迹的共同有效区间，在相邻源样本之间线性插值。默认不跨越大于该轨迹中位采样间隔 2.01 倍的缺口，这些位置作为缺失二维观测；可用 `--max-gap 秒数` 显式指定。不会在记录首尾外推或重复端点；时长不足会报错，请减小 `--frames`。`--start-time` 可指定相机 0 第 0 帧对应的源时间。

MoCap 模式仍然只将初始相机参数和二维观测交给求解器，所有输入 beta 为 0。`truth.csv` 来自源轨迹插值，仅用于评价；相机真值和真实时间偏移也只写入评价文件。这里复用了真实运动轨迹和合成相机，并不等于复现论文的全部原始数据集或实验设置。
