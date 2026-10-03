# 2026-10-03 跨批次快速迭代

目标：沿用当前四折划分，使四折测试 Macro-F1 算术平均超过 0.60。

## 已独立核验的达标模型

模型是 **RBF landmark 核特征 + 线性光谱特征的岭分类器**，不是 PatchTST。

实验目录：

```text
star-bio/outputs/search_kernel_20261003/rbf_raw_snv_ch0_linearTrue/26-10-03@10-00-41+commit-f87d022/
```

| 测试批次 | 验证批次 | Accuracy | Balanced Accuracy | Macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 2 | 0.660678 | 0.658987 | 0.617601 |
| 2 | 3 | 0.622053 | 0.631811 | 0.637986 |
| 3 | 4 | 0.756815 | 0.708706 | 0.700397 |
| 4 | 1 | 0.597694 | 0.618381 | 0.571808 |
| AVG | | **0.659310** | **0.654471** | **0.631948** |

不是每一折均超过 0.60；达到的是目标指定的四折 AVG。此前最佳 PatchTST `p64_bs64` 的 AVG 为 0.495820，本模型高 0.136128（13.61 个百分点）。

## 方法与选择规则

- 原始 CSV 的有效列 `[220:1260]` 插值到现有公共网格；每条光谱独立做 9 点二阶 Savitzky–Golay 平滑和 SNV。获胜模型只使用第一通道，不使用额外导数通道、不截断负值、不做原来的 101 点背景扣除。
- 每折只从训练集合按类别/批次权重选取 1024 个 landmark。训练均值、landmark 距离尺度、分类器系数只由训练数据得到。
- 特征为 RBF 与线性光谱特征拼接；使用加权岭回归拟合 one-hot 标签。
- gamma 与 ridge alpha 候选由验证批次 Macro-F1 选择，各折最后只对选定模型执行测试。没有把验证或测试样本加入训练。
- 公共插值网格继承已有缓存的文件头交集；所有强度预处理均逐样本完成，没有拟合测试批次分布。
- 所有模型与每折参数见 `best.pt`、`history.json`、`spec.json`；test_metrics.json、summary.json 是原始测试汇总。
- `independent_verification.json` 记录从原始 CSV 重新构建特征和 checkpoint 复算的结果。四折混淆矩阵逐项相同，测试数与原 splits 完全相同。

本轮对原有四批数据进行了多配置探索，报告为开发/探索结果。若要作最终泛化结论，需要新采集独立批次或严格嵌套验证；不能将当前 63.19% 解释为从未用于研究选择的新数据性能保证。

## 复算与单细胞推理

```bash
conda activate nnunet_seg
cd /home/wjx/CodeData/code/tsai-main
python star-bio/predict_kernel.py --verify-run \
  'star-bio/outputs/search_kernel_20261003/rbf_raw_snv_ch0_linearTrue/26-10-03@10-00-41+commit-f87d022' \
  --device cuda:0

python star-bio/predict_kernel.py --checkpoint \
  'star-bio/outputs/search_kernel_20261003/rbf_raw_snv_ch0_linearTrue/26-10-03@10-00-41+commit-f87d022/fold1/best.pt' \
  --csv '/home/wjx/CodeData/data/Star-Com/7class-4patch/acb-1/Cell_data.csv' --row 1
```

模型输出判别分数，不是校准概率。复算脚本读取 feature_path 同名 JSON 中的网格/预处理模式；请同时保留 `outputs/search_features/*.json`、原始 CSV 和 cache_v2.h5。

## 实验代码与调度

- search_train.py：GPU 常驻数据训练，避免逐细胞 HDF5 读取开销，使用原四折 train/val/test 协议。
- search_models.py：本地 tsai PatchTST 的 LayerNorm/BatchNorm、池化对照，以及 MLP/CNN 基线。
- search_features.py：逐光谱预处理对照，保持缓存样本顺序并校验文件名、数量和 fingerprint。
- search_linear.py：验证集选正则化的 ridge/LDA 对照。
- search_kernel.py：验证集选带宽与正则化的 RBF landmark 分类器。
- search_schedule.py：可动态扩展的 GPU 任务队列；按显存占用低于 10% 优先，其次按可用显存排序；剩余显存至少 12 GiB 才接任务。同一个调度器不会重复占用自身活动任务的卡；不同队列可在有余量的 GPU 上并发。完成后自动接下一项。

每项任务都是独立进程；日志与 spec 均保存在该实验时间戳目录内。`scheduler.json` 保存 PID、GPU、完成状态，失败任务不会冒充完成结果。正式四折结果追加到 `results.csv`，不修改原来的历史实验。代码每轮提交，队列/输出均在 Git 忽略目录中。

已启动的队列位于 outputs/search_20261003、search_raw_20261003、search_linear_20261003、search_kernel_20261003。各队列的 scheduler.json 和 launcher.log 是当前运行状态的依据。
