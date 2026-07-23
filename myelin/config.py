"""Run configuration: one dataclass tree, JSON in/out."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

from myelin.allocator import AllocatorConfig
from myelin.model import ModelConfig


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    alloc: AllocatorConfig = field(default_factory=AllocatorConfig)

    #: connectivity | structural | flow | random | random_churn | kquant |
    #: uniform | fp32
    strategy: str = "connectivity"

    # data: either a prepared data_dir (train.bin/val.bin) or synthetic
    data_dir: str = ""
    synthetic: bool = False
    synthetic_vocab: int = 512
    synthetic_length: int = 500_000

    steps: int = 10_000
    batch_size: int = 32
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    lr_warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    seed: int = 1337
    threads: int = 0  # 0 = leave torch default
    eval_interval: int = 500
    eval_iters: int = 40
    log_interval: int = 50
    out_dir: str = "runs/dev"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @staticmethod
    def from_dict(d: dict) -> "TrainConfig":
        d = dict(d)
        model = ModelConfig(**d.pop("model", {}))
        alloc = AllocatorConfig(**d.pop("alloc", {}))
        return TrainConfig(model=model, alloc=alloc, **d)

    @staticmethod
    def from_json_file(path: str) -> "TrainConfig":
        with open(path) as f:
            return TrainConfig.from_dict(json.load(f))
