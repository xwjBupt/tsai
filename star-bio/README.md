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

每次运行都会创建唯一实验目录，目录名以 `YY-MM-DD@HH-MM-SS` 开头：

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
  --output star-bio/outputs/smoke
```

## 单卡 GPU 训练

```bash
python star-bio/train_patchtst.py \
  --fold 1 --epochs 30 --workers 4 \
  --gpu-id 0 --device cuda \
  --output star-bio/outputs/fold1
```

## 8 张 GPU 轮流训练

```bash
python star-bio/train_patchtst.py \
  --all-folds --epochs 300 --workers 4 \
  --gpu-id 0 --device cuda \
  --patch-len 32 --stride 16 --layers 3 --heads 8 \
  --d-model 128 --d-ff 256 \
  --output star-bio/outputs/patchtst_lobo_v2
```

每次训练只使用一张 GPU。`--gpu-id 0` 指定 GPU 0；省略 `--gpu-id` 时自动选择当前空闲显存最多的 GPU。省略 `--batch-size` 时，脚本用真实前向和反向逐步探测 batch size，默认目标显存占用约 92%，并细化到最后一个可行整数 batch。`--memory-target 0.95` 可提高目标占用，`--auto-batch-max 4096` 控制上限。8 张 GPU 可以分别启动 8 个独立实验进程，但不要使用 `torchrun`。

TensorBoard 默认开启，每个 fold 写入 `fold*/tensorboard/`：

```bash
tensorboard --logdir star-bio/outputs/patchtst_lobo_v2/fold1/tensorboard --port 6006
```

如果希望在 8 张 GPU 上同时跑 8 个相互独立的 fold 或实验，可分别启动 8 个普通 Python 进程，并给每个进程传递不同的 `--gpu-id` 和输出目录；不要使用 `torchrun`。

默认最大训练轮数为 300，默认早停耐心为 40。每折每个 epoch 会记录训练 loss、验证 loss、Accuracy、Balanced Accuracy、Macro-F1、宏平均 precision/recall、每个类别的 precision/recall/F1、学习率和混淆矩阵。`history.json` 仍是原有的 JSON 数组格式；同时生成 `history.csv`（扁平化的全部指标）和 `curves.png`（loss、学习率、宏指标、逐类 F1 曲线）。终端日志由 Loguru 输出，只在 rank 0 显示，避免多卡重复刷屏。

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

PatchTST 原生是预测模型，输入 `[batch, channels, sequence]`，输出预测序列。本适配设置 `pred_dim=1`，对 3 个光谱通道分别产生一个 PatchTST 表征，再接 LayerNorm + MLP 分类头。输入仍是三个预处理通道：归一化光谱、一阶导数和 SNV。

`--limit-per-class` 仅用于冒烟测试，正式实验不能使用。正式结果应报告每个测试批次的 Accuracy、Balanced Accuracy、Macro-F1 和所有折的均值/标准差。
