# MoCap 实验

将 C3D 或 XYZ 长表转换成米制点轨迹，再由生成器投影成多路异步二维观测。

```bash
python3 -m pip install ezc3d
python3 tools/import_mocap.py data/recording.c3d --out data/trajectory.csv --center
python3 tools/generate_synthetic.py --trajectory-csv data/trajectory.csv \
  --tracks 12 --frames 30 --out experiments/mocap/input
./build/stba --input experiments/mocap/input --output experiments/mocap/result
python3 tools/evaluate_alignment.py --input experiments/mocap/input --result experiments/mocap/result
```

命令从项目根目录执行。默认采用源轨迹共同有效时段，长缺口当作缺失观测；时长不足时减小 frames。
详见 [DATA_FORMAT.md](../../DATA_FORMAT.md) 的 MoCap 一节。ASF/AMC 需先做骨架前向运动学导出 XYZ；压缩包不附带第三方人体运动数据。

转换和采样验证：`python3 tests/test_mocap_import.py`。C3D 测试仅在安装 ezc3d 时执行。
