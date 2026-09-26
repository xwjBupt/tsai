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
python -m pip install fastai fastcore psutil pyts imbalanced-learn
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

## 单折 GPU 测试

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  star-bio/train_patchtst.py \
  --fold 1 --epochs 30 --batch-size 32 --workers 4 \
  --device cuda --backend nccl \
  --output star-bio/outputs/fold1
```

## 8 卡正式实验

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  star-bio/train_patchtst.py \
  --all-folds --epochs 100 --batch-size 32 --workers 4 \
  --device cuda --backend nccl \
  --patch-len 32 --stride 16 --layers 3 --heads 8 \
  --d-model 128 --d-ff 256 \
  --output star-bio/outputs/patchtst_lobo_v1
```

`--batch-size` 是每张 GPU 的 batch size，总 batch size 约为 `batch-size × GPU 数`。省略它时，脚本按每张 GPU 自动探测。`--all-folds` 会轮流将每个实际批次作为测试批次；验证批次默认使用排序后的下一个批次。

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
