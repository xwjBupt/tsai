from dataclasses import dataclass, asdict
import json

@dataclass
class Config:
    data_root: str = "/home/wjx/CodeData/data/Star-Com/7class-4patch"
    cache_path: str = "/home/wjx/CodeData/data/Star-Com/7class-4patch/cache_v2.h5"
    output_root: str = "/home/wjx/CodeData/code/tsai-main/star-bio/outputs"
    n_points: int = 0
    n_classes: int = 0
    n_batches: int = 0
    labels: list[str] | None = None
    batch_values: list[int] | None = None
    batch_size: int = 64
    workers: int = 4
    epochs: int = 150
    lr: float = 2e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 8
    base_channels: int = 32
    transformer_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.10
    drop_path: float = 0.15
    label_smoothing: float = 0.05
    contrastive_weight: float = 0.10
    domain_weight: float = 0.0
    temperature: float = 0.10
    seed: int = 3407
    amp: bool = True
    ema: bool = True
    patience: int = 25

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))
