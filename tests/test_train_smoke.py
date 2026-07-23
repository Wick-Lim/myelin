import json
import os

import pytest

from myelin.allocator import AllocatorConfig
from myelin.config import TrainConfig
from myelin.model import ModelConfig
from myelin.train import Trainer


def _cfg(tmp_path, strategy, steps=60):
    return TrainConfig(
        model=ModelConfig(
            vocab_size=64, block_size=32, n_layer=1, n_head=2, d_model=16, d_ff=32
        ),
        alloc=AllocatorConfig(
            avg_bits=4.0, min_bits=2, max_bits=8,
            signal_warmup_steps=10, period=10, k_start=0.3,
        ),
        strategy=strategy,
        synthetic=True,
        synthetic_vocab=64,
        synthetic_length=20_000,
        steps=steps,
        batch_size=8,
        lr=1e-3,
        lr_warmup_steps=5,
        eval_interval=30,
        eval_iters=4,
        log_interval=10,
        seed=7,
        out_dir=str(tmp_path / strategy),
    )


@pytest.mark.parametrize("strategy", ["connectivity", "random", "fp32"])
def test_smoke_train(tmp_path, strategy):
    cfg = _cfg(tmp_path, strategy)
    summary = Trainer(cfg).run()

    assert summary["val_loss"] > 0
    out = cfg.out_dir
    assert os.path.exists(os.path.join(out, "summary.json"))
    assert os.path.exists(os.path.join(out, "ckpt.pt"))

    metrics = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
    train_rows = [m for m in metrics if "train_loss" in m]
    assert train_rows[-1]["train_loss"] < train_rows[0]["train_loss"] + 0.5

    if strategy != "fp32":
        alloc = [json.loads(l) for l in open(os.path.join(out, "alloc.jsonl"))]
        assert alloc and alloc[0]["kind"] == "init"
        # budget respected exactly after every event
        for e in alloc:
            n_ch = sum(len(l["bits"]) for l in e["layers"])
            total = sum(sum(l["bits"]) for l in e["layers"])
            assert total == round(cfg.alloc.avg_bits * n_ch)
        assert abs(summary["mean_bits"] - cfg.alloc.avg_bits) < 0.01
        assert "val_loss_fp" in summary


def test_same_seed_same_data_order_across_strategies(tmp_path):
    """Paired-comparison validity: batches must be identical across strategies."""
    import torch

    from myelin.data import get_batch, markov_corpus

    corpus = markov_corpus(64, 5000, seed=7)
    g1 = torch.Generator().manual_seed(8)
    g2 = torch.Generator().manual_seed(8)
    x1, _ = get_batch(corpus, 4, 32, g1)
    x2, _ = get_batch(corpus, 4, 32, g2)
    assert torch.equal(x1, x2)
