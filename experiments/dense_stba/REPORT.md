# 固定相机 RoMa Dense STBA：实际运行报告

日期：2026-11-11。代码、输入、日志和输出均在 `experiments/dense_stba/`。本轮新增独立 Python 原型，复用 `experiments/roma_lsm/` 已验证的 RoMa 解码、相机投影和 LSM；没有修改原有 C++ STBA 或上一轮实验缓存。

已完成连续异步视频的 **RoMa 跨视角关联 → 单相机时间跟踪 → 时间偏移粗估计 → 连续三维样条初始化 → 全局共享 beta 与轨迹控制点联合优化**。已实际跑过合成真值实验、corgi 两种输入、LSM/运动先验消融，以及 ROI 内 stride=1 的逐像素优化。

合成动态场景通过了端到端验证：四个有充分运动的场景，最大时间偏移误差为 **0.00536 帧（30 FPS 下 0.179 ms）**。真实 corgi 能输出上万条连续三维轨迹，但估计时间偏移偏离数据集公布的同步参考，整数补偿组 cam10 触及边界，狗本体的可靠轨迹覆盖不足。因此，当前证据支持“原型的异步估计链路有效”，尚不支持“已得到准确的 corgi 狗表面或可靠真实同步”。

## 1. 先看结果

| 查看内容 | 文件 |
|---|---|
| 原始帧／整数补偿的逐像素对比 | [results_density1/comparison.png](results_density1/comparison.png) |
| stride4 与 stride1 四组共同观测对比 | [real_comparison/comparison.png](real_comparison/comparison.png) |
| 原始帧动态点云 | [pointcloud.gif](results_density1/corgi_raw/figures/pointcloud.gif) |
| 整数补偿动态点云 | [pointcloud.gif](results_density1/corgi_integer_aligned/figures/pointcloud.gif) |
| 原始帧三维快照 | [pointcloud_snapshots.png](results_density1/corgi_raw/figures/pointcloud_snapshots.png) |
| 原始帧二维跟踪／三维轨迹 | [tracks_2d.png](results_density1/corgi_raw/figures/tracks_2d.png)、[trajectories_3d.png](results_density1/corgi_raw/figures/trajectories_3d.png) |
| 合成真值与估计三维轨迹 | [trajectories_3d.png](results/synthetic_fractional/figures/trajectories_3d.png) |
| 机器可读统计 | 各结果目录 `metrics.json`、`metrics.csv`、`B3.json`；[共同观测比较](real_comparison/comparison.json) |

真实 stride1 图使用参考图像 RGB 和小点，每幅最多展示确定抽样的 3,500 条轨迹；轨迹线图仅显示 12 条细线。这是绘图抽样，实际优化和指标使用全部有效轨迹。图中的空白、断裂和缺帧没有填补。真实数据坐标采用标定原生世界单位，没有独立核实其米制比例。

## 2. corgi 数据与时间约定

数据来自 `/data/datasets/selfcap/corgi-release`，使用 07、09、10 三个固定相机。检查了原始连续视频（3840×2160、60 FPS、3,500 帧）及对应的 `videos/camXX/images_2` 图像序列，工作分辨率为 480×270。没有混用 30 FPS 视频目录。相机 K、畸变 D、R、t 均来自数据集固定标定，图像按该标定去畸变和缩放；相机参数不进入优化变量。输入保存源文件 SHA256，并核对了与上一轮 frame230 的图像一致性。

本实现统一使用 **`physical_time = nominal_timestamp + beta_seconds`**，参考相机 07 的 beta 固定为 0。它与旧 C++ 实验中的减号约定不同，不能直接比较带符号数值。

数据集同步定义换算到此约定后，原始帧的总偏移参考为：

| 相机 | 公布的总 beta / 帧 | 整数取帧偏移 k / 帧 | 整数补偿后的参考余量 / 帧 |
|---|---:|---:|---:|
| 07 | 0 | 0 | 0 |
| 09 | +0.7453377172 | −1 | −0.2546622828 |
| 10 | +2.9637638479 | −3 | −0.0362361521 |

两种输入的实际原始帧号：

| 输入情况 | cam07 | cam09 | cam10 | 估计变量的硬边界 |
|---|---|---|---|---|
| raw | [200,260) | [200,260) | [200,260) | 总 beta ∈ [−6,+6] 帧 |
| integer_aligned | [200,260) | [199,259) | [197,257) | 残余 beta ∈ [−1,+1] 帧 |

整数补偿只移动取帧索引，没有合成或插值图像。局部名义时间均为 `n/60`。换算关系为 `beta_total = beta_residual − k/60`（单位秒）；以帧表示时直接减 k。完整 `sync.json` 小数值只作为独立文件中的外部评估参考；raw 估计器不读取它，integer 估计器只接收按整数 k 取好的序列。这里的 ±1 是相对基准相机的边界，并不普遍保证任意两台相机偏移跨度 ≤1 帧。

公布同步信息可用作外部参考，但本实验没有独立测量其误差，也没有真实逐点三维 GT。

## 3. corgi 两种情况：真正执行的 stride1 结果

查询 ROI 为参考帧230中的 `x∈[190,315), y∈[50,242)`，大小125×192，共 **24,000 个原始像素**。它覆盖人和狗附近的区域，是手工矩形，不是 GT 分割。这里的“逐像素”限定在此 ROI，不能解读为整幅480×270图像全部重建。

| 指标 | raw | integer_aligned |
|---|---:|---:|
| ROI 查询像素 | 24,000 | 24,000 |
| 参考视角成功时间跟踪种子 | 23,691 | 23,691 |
| 多视角关联并实际优化的三维轨迹 | **12,908** | **12,777** |
| 三视角关联种子 | 7,315 | 7,186 |
| 三维轨迹数 / 查询数 | 53.783% | 53.238% |
| 有观测支持的三维时刻样本 | 758,855 | 752,714 |
| cam09 优化变量 / 帧 | +0.492330 | −0.498261（余量） |
| cam10 优化变量 / 帧 | +1.619326 | **−1.000000（余量，下界）** |
| cam09 换算回总 beta / 帧 | **+0.492330** | **+0.501739** |
| cam10 换算回总 beta / 帧 | **+1.619326** | **+2.000000** |
| 各自全部相机留出观测 RMSE / px | 0.201427 | 0.193074 |
| **相同 cam07 留出观测 RMSE / px** | **0.143318** | **0.151633** |

最后一行使用相同的 **150,503 条** `(camera07, point_id, raw_frame_id)` 留出观测；原始二维坐标逐字节相同，两个方法均有有限预测，没有按残差大小筛选。这些样本占两组共同原始留出目标的55.968%。前一行的观测集合不同（397,037／392,276条），不能据此作配对优劣判断。

所以这次实验中，整数补偿没有在共同 cam07 留出目标上改善预测。两组结果还同时受到目标相机取帧区间、跨视角关联和 beta 搜索边界变化的影响，不能将差异完全归因于整数预处理。

raw 两个非参考相机满足局部曲率与收敛检查，但“局部可观测”仅表示当前观测和模型附近存在可辨别的局部最优，不代表真实同步准确。integer 的 cam09 被标记为 `iteration_limit_or_no_convergence`，cam10 为 `at_beta_bound`；不能把 cam10 的 −1.000000 当作高精度估计。

**狗本体覆盖仍然很差。** 在用于诊断的近似狗矩形 `(207,145)–(241,213)` 内，stride1共有2,312个查询像素，但 raw 仅47个两视角候选、8个三视角候选；integer仅40／5个。人附近的13,392个查询像素则有7,391／7,261个两视角候选。当前结果主要来自人和衣服等容易匹配的部分，点云中存在缺失和碎片，不能将总轨迹数当作狗表面的覆盖率。该矩形统计只是图像区域诊断，不是语义分割精度评估。

stride4也完整保留：1,488个查询，raw／integer分别恢复831／823条轨迹，支持48,851／48,471个三维时刻样本，平均有效时间范围0.9631／0.9649秒。在各自共同的 B0/B1/B3 留出目标上，RMSE分别为 raw `0.227450/0.223090/0.222947 px`，integer `0.226453/0.214966/0.214904 px`。联合优化相对较强的粗初始化只有小幅变化。

## 4. 合成动态真值验证

合成序列包含3个固定相机、192×144图像、32帧、30 FPS。带连续纹理的平面进行非匀速三维平移、三轴旋转与深度变化；每台相机在独立物理时刻光线投射成像，使用2×2像素积分。另有噪声与真实遮挡、近静止场景。参考像素的物理材料坐标、连续三维位置、可见性和 beta 存放在独立 GT 文件中。

正式 B0/B1/B3/B4 只读取图像、名义时间与固定标定；GT不参与筛选、初始化、优化或调回被拒绝的轨迹。仅 B2 显式读取 beta GT，是 oracle 对照。三维误差直接在原始世界坐标计算，没有事后刚体或相似变换对齐。

各方法定义：

| 方法 | 实际执行内容 |
|---|---|
| B0 | 假设 beta=0，按名义同步的多视图三角化 |
| B1 | 自动粗估 beta 后固定 beta，优化连续轨迹 |
| B2 | **oracle-beta**：固定真值 beta，使用自动前端优化轨迹 |
| B3 | 自动粗估后联合优化全局 beta 与轨迹 |
| B4 | 对 RoMa 种子做可选 LSM 后重新跟踪、关联和联合优化 |

以下是每个场景各方法**共同有效真值时刻集合**上的位置 RMSE，单位mm。fractional包含B4的交集，其他行包含B0–B3的交集；不同场景的集合不同。

| 场景 | 共同样本数 | B0 | B1 | B2 oracle | B3 | B4 |
|---|---:|---:|---:|---:|---:|---:|
| fractional：beta=(0,+0.3,−0.5) | 26,728 | 181.430 | 11.451 | 11.436 | **11.428** | **6.817** |
| wide：beta=(0,+1.2,−0.5) | 26,693 | 417.966 | 9.449 | 9.402 | 9.468 | — |
| zero：beta=(0,0,0) | 26,499 | 12.570 | 12.490 | 12.397 | 12.468 | — |
| noisy_occlusion | 22,350 | 175.832 | 21.394 | 21.238 | 21.214 | — |
| near_static（退化测试） | 45,048 | 8.235 | 8.188 | 8.220 | 8.188 | — |

| 场景 | B3估计beta，帧（参考=0） | 最大非参考偏移误差 / 帧 | B3有效轨迹 / 1,408查询 | 独立GT时刻覆盖率 |
|---|---|---:|---:|---:|
| fractional | (0,+0.299238,−0.497273) | 0.002727 | 1,100 | 63.062% |
| wide | (0,+1.205361,−0.498014) | 0.005361 | 1,098 | 62.985% |
| zero | (0,+0.002639,−0.000210) | 0.002639 | 1,098 | 62.520% |
| noisy_occlusion | (0,+0.295039,−0.501983) | 0.004961 | 944 | 53.853% |
| near_static | (0,0,0)，`insufficient_motion` | **不宣称同步成功** | 1,408 | 100% |

覆盖率分母独立于估计结果：全部查询材料点中，在拍摄物理时间范围内至少被两台GT相机看到的时刻样本。轨迹数比例与时刻覆盖率是不同指标。每个 `metrics.json` 还保存 MAE、P50/P90、速度/加速度误差、5/10/20/50 mm正确覆盖率、前端关联误差代理、逐时刻覆盖、训练与留出重投影误差。

可以据此回答：

1. **亚帧偏移可恢复。** 在有足够运动的四组合成视频中成立；近静止明确失败，没有返回伪可靠同步。
2. **主要收益来自处理异步。** B0→B1/B3提升很大；B1→B3收益小，wide甚至略变差，当前实验不证明联合优化总能优于良好的先同步再拟合。
3. **残余误差主要不在 beta。** B3与正确 beta 的B2已接近；二维测量、关联、图像采样和样条模型仍限制几何精度。
4. **覆盖率存在代价。** 移动场景只有约54–63%的独立可评估时刻被有效恢复，部分尾部时刻因跟踪门控大量失效，不能只展示成功区域。
5. **LSM效果依场景变化。** fractional的真实3D误差由11.428降到6.817 mm，但真实数据没有3D GT，二维诊断也没有显示普遍改善。

额外GT诊断发现：fractional参考视角中1,731个被5帧循环门控拒绝、且GT仍可见的提案，其相邻帧累积轨迹误差全部小于1px，平均0.109px。说明重复纹理下直接跳5帧的匹配也可能出错，严格循环门控会错杀原本准确的短步跟踪。诊断结果在 [cycle_rejection_diagnostic.json](results/synthetic_fractional/cycle_rejection_diagnostic.json)；这些点没有根据GT被重新接纳。

## 5. 算法实现与估计边界

**前端。** 环境中未找到可直接使用的本地 CoTracker3 实现及权重，因此采用已有 OpenCV DIS medium 稠密光流。所有参考点均来自规则网格或像素，而非关键点检测。相邻帧正反流在当前亚像素位置传播，保留物理轨迹ID；使用纹理、正反一致性和5帧直接/复合流一致性检查。失败后停止该分支，没有重新识别或填洞。

每个真实相机对在localframe15、30、45运行原始RoMa v1，共6次网络匹配；合成使用相应中间及前后锚帧。保留完整560×1120 warp/certainty缓存，未调用稀疏 `sample()` 代替稠密场。跨相机关联需要 RoMa置信度≥0.15、正反误差≤2px、目标种子冲突排除，以及**两个非种子锚帧分别**与时间轨迹一致（误差≤1.5px）。种子自身的零误差不计为验证支持。异步动态候选不使用同帧普通极线阈值筛除。

**时间初始化。** 从训练观测中选运动跨度≥2px、速度≥0.02px/frame的轨迹；至少8条动态轨迹，每条至少6个共同时间样本。对候选 beta 插值目标二维轨迹，按固定相机极线几何评分，候选之间使用相同支持集合，每轨迹等额聚合稳健误差。0.1帧粗搜索后连续精化所有局部谷值，检查分离多峰、曲率和边界。插值不外推，端点间距不能超过2.1帧。这里二维插值仅用于初始化和支持检查，不作为额外测量重复计入优化。

**连续轨迹与联合优化。** 每条轨迹使用三次B-spline，节点间隔4帧，覆盖允许的时间范围。两视角以上三角化提供初值；最少8个合格三维样本才能拟合。未知量只有轨迹控制点和两个全局共享 beta。`stba_optimizer.py` 对每轨迹累积局部正规块，通过Schur消元汇成一个共同 beta 系统，并回代所有轨迹。batch=128只改变计算分块，不产生分块独立的时钟。使用float64、LM与回溯，最多50次迭代。

目标为真实二维观测的置信度加权Huber重投影损失，加 `lambda_motion * sum integral ||X''(t)||² dt`，默认lambda=1e−4；积分使用样条导数与精确分段二点Gauss求积。不加入空间刚性先验。置信度是经验权重，并非经过标定的测量逆方差。运动不足的轨迹只帮助估计自身几何，不参与beta信息累积。运动项尺度随世界单位及秒制变化，迁移数据需重新检查。

最终有效三维时刻必须有至少两视角的有效观测端点支持、满足支持时间范围、正深度及最终观测内点检查（≤4px）。单视角情况下的样条延续另存 `model_only_mask`，不计入有效多视图覆盖。B1/B3共享相同初始控制点、观测和粗beta。没有无限外推，没有用模型延续填补稠密洞。

**留出协议。** localframe中每5帧、phase=2的原始二维观测不参与时间评分、三维初始化或优化。前端仍使用完整图像序列构建跟踪，因此留出误差衡量的是对跟踪测量的预测一致性，不能等同独立三维真值或盲图像测试。跨raw/integer的目标视角物理帧不同，公平配对仅使用同一cam07的同一原始像点。

## 6. 消融与稳定性

fractional B3使用相同自动前端和粗初始化，仅改优化项：

| 设置 | 自身有效样本三维RMSE / mm | beta09 / 帧 | beta10 / 帧 |
|---|---:|---:|---:|
| 默认lambda=1e−4 | 11.428 | +0.299238 | −0.497273 |
| 无运动先验 | 11.887 | +0.299327 | −0.497471 |
| lambda×10 | 11.540 | +0.298915 | −0.496703 |
| 优化观测等权 | 11.452 | +0.299642 | −0.497605 |

这个单场景消融显示适量平滑略有帮助，尚不足以证明一种权重或正则强度普遍最好。

真实B4的各自留出RMSE为raw0.241199／integer0.234942px。为了避免更换LSM测量后“换尺子”，额外在B0原始二维测量的共同目标上比较：raw B3/B4为0.221890/0.280016px，integer为0.212695/0.277058px。真实LSM在此诊断中变差；没有真实三维GT，不能推断绝对几何误差。合成fractional恰好出现相反情况：B4真实3D更好，但相对原始RoMa二维测量更差，进一步说明重投影一致性与真实3D不是同一指标。

为检查时间稳定性，stride4另外独立拟合local窗口[0,40)和[20,60)：

| 情况 | 前窗口总beta09/10，帧 | 后窗口总beta09/10，帧 |
|---|---|---|
| raw | +0.335780 / +1.482525 | +0.543127 / +1.797406 |
| integer | +0.432351 / +2.000000 | +0.502446 / +2.000000 |

raw跨窗口变化0.20735／0.31488帧。integer的cam10两个窗口都在余量−1边界，零变化不能解释为可靠稳定。这些是独立诊断拟合，不是把不同beta的窗口拼成一个重建结果；主实验使用整个60帧和唯一共享beta。

还保持integer输入不变，将余量界限放宽至±6帧：估计余量变为 `(0,−0.621436,−1.689063)`，换算总偏移 `(0,+0.378564,+1.310937)`，仍偏离公布参考。这说明±1硬界限制了结果，同时“放宽界限”也没有自动解决真实关联或模型误差。正式请求中的integer结果仍保留±1约束。

## 7. 计算量与扩展性

实际使用 `roma_corr` 环境、PyTorch 2.5.1+cu121、NumPy2.2.6、OpenCV4.13；优化在一张RTX4090上运行。完整环境和本地RoMa依赖见 [environment.json](environment.json)。

| stride1阶段 | raw | integer |
|---|---:|---:|
| 时间粗估计 | 131.48s | 34.65s |
| B3拟合＋优化＋保存 | 59.33s | 57.90s |
| 其中：联合优化器 | 23.65s | 36.19s |
| 其中：NPZ压缩保存 | 6.32s | 6.26s |
| 优化器峰值CUDA allocated | 203.48MiB | 198.23MiB |
| 真实训练二维观测 | 1,587,794 | 1,568,827 |
| 每轨迹控制点 | 21 | 19 |
| 共享非参考beta变量 | 2 | 2 |

以上均不包含RoMa模型加载、初次匹配、光流和图像准备。stride1重用完整RoMa/光流场，重新对每个ROI像素构建轨迹，前端缓存运行约13.57／11.95s；没有插值stride4三维结果。显存为优化器PyTorch分配峰值，不含RoMa网络和CUDA上下文，也不是端到端进程总显存。

raw从831到12,908条轨迹，优化耗时由约1.56s到23.65s，与点数近似线性增长，没有建立全像素平方规模Hessian。当前主要可优化开销在CPU粗时间搜索：每个候选重建二维插值端点和公共支持，raw±6区间比integer±1昂贵。可以缓存端点索引、在支持检查中避免分配二维坐标，并只处理合格动态候选；本轮未为加速而改变已经报告的算法和结果。

## 8. 输出格式与审计

`observations.npz` 按查询网格存储，而非大量Python对象：

| 字段 | 说明 |
|---|---|
| `point_ids`, `seed_uv`, `query_mask`, `seed_roi` | 原图网格ID=`y*width+x`、参考像素坐标和查询域 |
| `uv[N,C,T,2]`, `visible`, `weight`, `status` | 原始二维轨迹、有效性、经验权重和失败状态 |
| `timestamps[C,T]` | 每相机名义时间，秒 |
| `edge_status`, `roma_confidence`, `roma_fb_error` | 跨视角候选状态与RoMa质量 |
| `anchor_error`, `anchor_support`, `temporal_fb_error` | 多锚帧与时间跟踪一致性 |
| `temporal_success`, `multiview_seed`, `all_three_seed` | 跟踪／两视角／三视角阶段的有效性 |

`B3.npz` 继续保留全部查询ID，失败轨迹以mask/NaN表示：

| 字段 | 说明 |
|---|---|
| `beta[C]`, `initial_beta` | 秒；帧数在 `B3.json` |
| `control[N,K,3]`, `knots` | 连续三次样条，可调用 `evaluate_spline` 在支持区间内取样 |
| `trajectory_times[T]`, `xyz[N,T,3]` | 参考时轴与原始世界坐标，无显示偏移 |
| `velocity`, `acceleration` | 对秒求导，世界单位/秒、世界单位/秒² |
| `valid`, `trajectory_valid`, `view_support` | 时刻有效性、轨迹有效性与视角支持数 |
| `support_interval`, `model_only_mask` | 实际支持范围及不能视为多视图测量的模型延续 |
| `observed_uv`, `predicted_uv` | 原始测量与在实际相机时间计算的投影 |
| `train_visible`, `holdout_visible`, `observation_inlier` | 训练、留出和最终内点mask |
| `confidence`, `motion_mask`, `beta_information` | 经验轨迹质量、时钟运动资格、局部Schur信息 |

`B3.json`保存配置、迭代目标/步长、初始与最终beta、收敛/可观测状态、轨迹数、时间和显存。`meta.json`记录固定标定、时间约定、输入路径及SHA256。每个结果目录冻结自己的观测NPZ；评估前校验哈希和像素身份，避免后续可复用缓存变化污染既有结果。

最终运行了 **42个单元/回归测试，全部通过**，涵盖渲染与GT分离、时间符号、跟踪传播、样条导数、全局共享beta、退化/多峰、留出隔离、覆盖统计、真实共同观测对齐。见 [unit_tests.log](unit_tests.log)。

独立回放审计覆盖 **17个结果目录、39个方法输出，0失败**，见 [results_audit.json](results_audit.json)、[results_audit.log](results_audit.log)。检查包括固定标定源哈希、冻结观测、规范自由度、硬边界、训练/留出/窗口隔离、最终内点支持及正深度。按控制点重放XYZ/速度/加速度最大差为0，重投影差约1.14e−13px。该审计证明实现/结果自洽，不代替真实几何精度验证。

## 9. 复现

从仓库根目录执行。现有环境包含本地RoMa实现与权重；数据集路径和标定记录在manifest，另机运行需相应迁移数据和依赖。脚本没有隐式安装包或改写标定。

```bash
export DENSE_PY=/data/tools/miniconda3/envs/roma_corr/bin/python
export CUDA_VISIBLE_DEVICES=5  # 换成空闲GPU
export MPLCONFIGDIR=/tmp/dense_stba_matplotlib

# 输入生成、测试
"$DENSE_PY" experiments/dense_stba/synthetic_dynamic.py
"$DENSE_PY" experiments/dense_stba/data_loader.py --start 200 --stop 260
"$DENSE_PY" -m unittest discover -s experiments/dense_stba/tests -v

# stride4端到端；real自动省略没有GT的B2
"$DENSE_PY" experiments/dense_stba/run_experiment.py --selection synthetic --evaluate
"$DENSE_PY" experiments/dense_stba/run_experiment.py --selection real --evaluate

# 可选LSM消融
"$DENSE_PY" experiments/dense_stba/run_experiment.py --selection fractional --skip-frontend --methods B4 --evaluate
"$DENSE_PY" experiments/dense_stba/run_experiment.py --selection real --skip-frontend --methods B4 --evaluate

# ROI逐像素：复用的是完整网络场与光流，不是三维结果
"$DENSE_PY" experiments/dense_stba/prepare_density.py
"$DENSE_PY" experiments/dense_stba/run_experiment.py --selection real --skip-frontend --methods B3 \
  --config experiments/dense_stba/configs/density1.json \
  --cache-root experiments/dense_stba/cache_density1 \
  --results-root experiments/dense_stba/results_density1 --evaluate

# 绘制一组；另一组将corgi_raw换为corgi_integer_aligned
"$DENSE_PY" experiments/dense_stba/visualize.py experiments/dense_stba/results_density1/corgi_raw \
  --max-cloud-points 3500 --cloud-rgb --max-trajectory-lines 12
"$DENSE_PY" experiments/dense_stba/compare_real.py
"$DENSE_PY" experiments/dense_stba/audit_results.py --roots \
  experiments/dense_stba/results experiments/dense_stba/results_density1 \
  experiments/dense_stba/stability experiments/dense_stba/ablations
```

完整分阶段入口（包括独立窗口、消融和全部绘图）见 [reproduce.sh](reproduce.sh)：

```bash
bash experiments/dense_stba/reproduce.sh --help
CUDA_VISIBLE_DEVICES=5 bash experiments/dense_stba/reproduce.sh full
# 已有结果时，也可只重评估、绘图和校验：
bash experiments/dense_stba/reproduce.sh evaluate visualize compare audit tests
```

该脚本已通过 `bash -n` 和帮助入口检查；本次实际实验由各阶段命令运行，未为验证入口再次重复全量实验。复跑会重写所选阶段对应的实验输出；需要保留新旧实验时，用 `run_experiment.py --results-root` 指定新目录。关键实际日志为 `experiment_synthetic.log`、`experiment_real.log`、`experiment_lsm.log`、`experiment_density1.log`、`stability.log`、`ablation.log`、`density1_visualization.log`。

## 10. 结论与下一步

本轮已经实现并运行真正的全局时间偏移与连续稠密轨迹联合估计，满足三台固定相机、异步动态图像、自动RoMa关联、正式估计不使用GT、独立合成真值评估的验证条件。实际stride1也完成了12,000余条轨迹优化。

真实corgi的覆盖统计直接暴露出动态表面身份关联不足。当前人/衣服贡献了大部分轨迹，狗身上的多视角身份很少；优化可以在较小二维残差下给出偏离外部同步参考的解，边界和窗口变化进一步暴露了限制。同步差异还可能混入固定标定和轨迹模型误差，当前实验没有独立分离这些因素。

建议下一步优先对狗区域建立更可靠的长时间跟踪和遮挡处理，检查毛发、重复纹理及视角变化下的跨相机身份，用人工核验的一小组动态对应诊断固定标定误差、二维漂移与时间偏移的混淆；再扩展有足够运动的时间窗口。应同时报告精度、独立覆盖率和失败状态。暂不把这些有洞的轨迹直接连成网格并据此声称获得准确稠密表面。
