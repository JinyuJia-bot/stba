# 可直接运行的三相机示例

input 是 24 帧、4 条动态轨迹、8 个静态点、0.1 px 噪声的合成输入，时间真值为 [0, 0.31, -0.27] 帧，输入偏移全为 0。truth 和 true_* 只用于评价。

expected 是本次编译得到的 full 参考结果，包含增量日志和 DCT。新运行请写到其他目录，不覆盖参考结果：

```bash
./build/stba --input examples/three_cameras/input --output experiments/example_run
python3 tools/evaluate_alignment.py --input examples/three_cameras/input --result experiments/example_run
```

预期最大偏移误差约 0.0033 帧，动态三维 RMSE 约 4.52 mm。跨 Ceres/线性代数库版本允许小幅数值差异。
