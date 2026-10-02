# star-bio: Star-Com PatchTST

本目录将 `/home/wjx/CodeData/code/tsai-main` 中的 PatchTST 适配到 Star-Com 细菌单细胞光谱数据。默认数据根目录为：

```text
/home/wjx/CodeData/data/Star-Com/7class-4patch
```

目录格式是 `<类别>-<批次>/Cell_data.csv`，例如 `acb-1/Cell_data.csv`。缓存、类别映射、波数网格和批次划分沿用 Cell 工程的自描述 `cache_v2.h5` 协议，但 PatchTST 模型和输出全部保存在本目录。

## 环境

使用 `nnunet_seg` 环境：

```bash
conda activate nnunet_seg
cd /home/wjx/CodeData/code/tsai-main
```

## 实验模式、时间戳和 Git 记录

每次运行都会创建唯一实验目录，目录名以 `YY-MM-DD@HH-MM-SS` 开头。`results.csv` 中的 `experiment_id` 使用 `--output-root` 的一级目录名，`timestamp` 和 `commit` 单独记录本次运行：例如 `compare_runs/aug_strong/26-09-30@14-39-46+commit-7b0dbaa/` 会记录为 `experiment_id=aug_strong`、`timestamp=26-09-30@14-39-46`、`commit=7b0dbaa`。

```text
debug/YY-MM-DD@HH-MM-SS+debug/       # 调试实验，不 git commit
outputs/YY-MM-DD@HH-MM-SS+commit-HASH/ # 正式实验，先提交 star-bio 代码
```

调试时加 `--debug`，结果写到 `star-bio/debug/`，不会运行 git 命令：

```bash
python star-bio/train_patchtst.py --debug --fold 1 --epochs 1 \
  --limit-per-class 1 --device cpu
```

正式模式是默认模式。启动时只提交 `star-bio/` 下的代码，commit message 记录实验时间；随后实验输出写入 `star-bio/outputs/`，不会加入该 commit。正式模式需要当前 git 仓库已配置用户名/邮箱且允许提交，否则程序会在训练前报错。

正式 `--all-folds` 实验的四折测试完成后，会将本次各项测试指标追加到 `/home/wjx/CodeData/code/tsai-main/star-bio/results.csv`。每项指标一行，含实验 ID、时间戳、commit、各实际 fold 列、四折简单算术平均 `AVG` 和标准差 `STD`。记录包括 ACC、Balanced Accuracy、宏/加权 Precision、Recall、F1、测试 loss，以及各类别的 Precision、Recall、F1 和 support。单折实验和 `--debug` 实验不追加汇总。CSV 更新采用进程锁和原子替换，可避免多个单卡实验同时结束时互相覆盖。此前已有的 `patchtst_lobo_v1` 会以 `legacy-patchtst_lobo_v1` 名称补录；更早结果没有测试 loss 时该指标留空。

TensorBoard event 文件放在实验根目录下，以同一个时间戳和 commit 信息命名，例如：

```text
outputs/YY-MM-DD@HH-MM-SS+commit-HASH/tensorboard/YY-MM-DD@HH-MM-SS+HASH/fold1/events.out.tfevents...
```

这样多个实验可一起加载到 TensorBoard 时按时间戳区分曲线。可指定 `--output-root` 更换 `debug/` 或 `outputs/` 的父目录；实验时间戳目录仍会自动创建。

依赖检查：

```bash
python - <<'PY'
import sys
sys.path.insert(0, '/home/wjx/CodeData/code/tsai-main')
from tsai.models.PatchTST import PatchTST
print(PatchTST)
PY
```

如果缺少 tsai 运行依赖：

```bash
python -m pip install -r star-bio/requirements.txt
```

## 缓存

新数据目录已有 `cache_v2.h5`。如果原始 CSV 有变化，使用本目录的脚本重建：

```bash
python star-bio/prepare_data.py \
  --data-root /home/wjx/CodeData/data/Star-Com/7class-4patch \
  --cache /home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5 \
  --force
```

## CPU 冒烟测试

```bash
python star-bio/train_patchtst.py \
  --fold 1 --epochs 1 --batch-size 2 --workers 0 \
  --limit-per-class 1 --device cpu \
  --d-model 32 --d-ff 64 --layers 1 --heads 4 \
  --output-root star-bio/outputs/smoke
```

## 单卡 GPU 训练

```bash
python star-bio/train_patchtst.py \
  --fold 1 --epochs 30 --workers 4 \
  --gpu-id 0 --device cuda \
  --output-root star-bio/outputs/fold1
```

## 8 张 GPU 轮流训练

```bash
python star-bio/train_patchtst.py \
  --all-folds --epochs 300 --workers 4 \
  --gpu-id 0 --device cuda \
  --patch-len 32 --stride 16 --layers 3 --heads 8 \
  --d-model 128 --d-ff 256 \
  --output-root star-bio/outputs/patchtst_lobo_v2
```

每次训练只使用一张 GPU。`--gpu-id 0` 指定 GPU 0；省略 `--gpu-id` 时自动选择当前空闲显存最多的 GPU。省略 `--batch-size` 时，脚本用真实前向和反向逐步探测 batch size，默认目标显存占用约 92%，并细化到最后一个可行整数 batch。`--memory-target 0.95` 可提高目标占用，`--auto-batch-max 4096` 控制上限。8 张 GPU 可以分别启动 8 个独立实验进程，但不要使用 `torchrun`。

TensorBoard 默认开启，每个实验写入 `<实验目录>/tensorboard/<时间戳+commit>/fold*/`：

```bash
tensorboard --logdir star-bio/outputs --port 6006
```

如果希望在 8 张 GPU 上同时跑 8 个相互独立的 fold 或实验，可分别启动 8 个普通 Python 进程，并给每个进程传递不同的 `--gpu-id` 和输出目录；不要使用 `torchrun`。

默认最大训练轮数为 300，默认早停耐心为 40。每折每个 epoch 会记录训练 loss、验证 loss、Accuracy、Balanced Accuracy、Macro-F1、宏平均 precision/recall、每个类别的 precision/recall/F1、学习率和混淆矩阵。`history.json` 仍是原有的 JSON 数组格式；同时生成 `history.csv`（扁平化的全部指标）和 `curves.png`（loss、学习率、宏指标、逐类 F1 曲线）。终端日志由 Loguru 输出，只在 rank 0 显示，避免多卡重复刷屏。

## 八卡并行对比实验（batch、归一化与池化）

启动器 `star-bio/launch_8_compare.sh` 为每个实验固定一张 GPU，并顺序完成四折。所有组都使用 Patch64 / stride32、d_model128 / d_ff256、3 层 / 8 heads、joint 采样、学习率 2e-4、seed3407。前四组只改变 batch，后四组以 GPU 2 为参考。这里显式指定 batch，不使用自动显存探测；占满显存不是本轮性能对照的目标。

| GPU | 实验名 | Batch | RevIN | 池化 | 批次漂移增强 |
|---:|---|---:|---|---|---|
| 0 | p64_bs4096 | 4096 | 开 | mean | 关 |
| 1 | p64_bs1024 | 1024 | 开 | mean | 关 |
| 2 | p64_bs256 | 256 | 开 | mean | 关 |
| 3 | p64_bs64 | 64 | 开 | mean | 关 |
| 4 | p64_bs256_norevin | 256 | 关 | mean | 关 |
| 5 | p64_bs256_seg4 | 256 | 开 | 4 段分区 | 关 |
| 6 | p64_bs256_attn | 256 | 开 | attention | 关 |
| 7 | p64_bs256_drift | 256 | 开 | mean | 开，强度 0.25 |

仅预览命令（不会创建实验目录、提交 Git 或启动进程）：

```bash
conda activate nnunet_seg
cd /home/wjx/CodeData/code/tsai-main
DRY_RUN=1 MODE=formal bash star-bio/launch_8_compare.sh
```

实际启动由用户执行：

```bash
# debug：不提交 Git，也不写入正式 results.csv
bash star-bio/launch_8_compare.sh
# 正式：共享一次代码提交，完整四折结束后汇总 results.csv
MODE=formal EPOCHS=300 bash star-bio/launch_8_compare.sh
```

`SEED=3408`、`LR=0.0001`、`WORKERS=2` 可以统一覆盖八组参数；不要在组间随意改变学习率或种子。默认 300 是最大轮数，仍有 patience=40 的早停。同样 epoch 数下小 batch 的更新次数更多，结论应同时比较优化器步数和训练耗时；严格等更新预算的复验需另行安排。

输出在 `star-bio/compare_runs/<实验名>/<时间戳+commit>/`，启动日志仍在 `star-bio/compare_runs/<实验名>/launcher.log`。每次重新启动同名实验时 launcher.log 会覆盖，请先保存需要保留的旧日志。查看所有曲线：

```bash
tensorboard --logdir star-bio/compare_runs --port 6006
```

### 新增参数与记录

- `--revin / --no-revin`：默认开；关闭时直接把缓存光谱送入 patch 编码器。
- `--pooling mean|segments|attention`：默认 mean；`--pool-segments 4` 控制分区数。
- `--seed`：默认 3407，每折实际训练种子为 seed + 测试批次编号。
- `--batch-shift / --no-batch-shift` 与 `--augmentation-strength`：增强只应用于训练数据。增强后从第一通道重新生成导数和 SNV，使用与缓存相同的 Savitzky–Golay 参数和总体标准差。导数按采样点计算，修正了原来的 slope 尺度错误；配置记录 `augmentation_version=drift-recompute-v2`。

保留 history.json 原有字段和 JSON 数组格式，新增以下字段并同步到 history.csv / TensorBoard：

| 字段 | 含义 |
|---|---|
| optimizer_steps | 累计实际优化器更新次数，扣除 AMP 非有限梯度导致的跳步 |
| epoch_optimizer_steps | 本轮实际更新次数 |
| batches_seen | 累计读入训练 batch 数 |
| train_seconds | 本轮训练循环耗时，含数据加载，CUDA 计时前后同步 |
| total_train_seconds | 累计训练循环耗时 |
| epoch_seconds | 本轮训练 + 验证耗时，不含绘图和落盘 |
| elapsed_seconds | 当前 fold 已经过的时间（不含 batch 探测） |

test_metrics.json 还保存 best_epoch、best_optimizer_steps 和全程更新次数/耗时。TensorBoard 在早停轮也会写入数据；curves.png、history.json、results.csv 的原有用途保持一致。

## 评估和推理

```bash
python star-bio/evaluate_patchtst.py \
  --checkpoint star-bio/outputs/patchtst_lobo_v1/fold1/best.pt

python star-bio/predict_patchtst.py \
  --checkpoint star-bio/outputs/patchtst_lobo_v1/fold1/best.pt \
  --csv /home/wjx/CodeData/data/Star-Com/7class-4patch/acb-1/Cell_data.csv \
  --row 1
```

`best.pt` 保存类别、波数、预处理、缓存指纹和 PatchTST 结构参数。评估时会检查缓存指纹，防止混用不同版本数据。

## 模型说明

分类器直接使用本地 tsai 的 patch 编码器，移除预测头和输出反归一化，输入为归一化光谱、一阶导数、SNV 三通道。encoder 输出 `[N, 3, d_model, patches]`：mean/attention 池化产生 `3 × d_model` 维特征，4 段分区池化保留有序的四个光谱区域，产生 `12 × d_model` 维特征，再接 LayerNorm + MLP。分区池化的分类头参数更多，比较结果时应考虑容量差异。

config.json 与 checkpoint 保存 revin、pooling、pool_segments；评估和推理会自动按这些参数构建模型。已有 encoder + mean 池化 checkpoint 未含新字段时默认 RevIN 开启、mean 池化，仍可加载；更早的三标量 forecasting-head checkpoint 不适用于这个 encoder 架构。

`--limit-per-class` 仅用于冒烟测试，正式实验不能使用。正式结果应报告每个测试批次的 Accuracy、Balanced Accuracy、Macro-F1 和所有折的均值/标准差。
