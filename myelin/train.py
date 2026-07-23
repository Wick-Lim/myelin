"""Training loop with periodic connectivity-driven bit reallocation."""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from myelin.allocator import BitAllocator
from myelin.config import TrainConfig
from myelin.data import get_batch, load_split, markov_corpus
from myelin.model import MiniGPT
from myelin.signals import QuantIndex, make_signal


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        if cfg.threads > 0:
            torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)

        # --- data ------------------------------------------------------ #
        if cfg.synthetic:
            corpus = markov_corpus(
                cfg.synthetic_vocab, cfg.synthetic_length, seed=cfg.seed
            )
            n_val = max(cfg.model.block_size + 2, len(corpus) // 10)
            self.train_data = corpus[:-n_val]
            self.val_data = corpus[-n_val:]
            cfg.model.vocab_size = cfg.synthetic_vocab
        else:
            if not cfg.data_dir:
                raise ValueError("set data_dir or synthetic=True")
            meta_path = os.path.join(cfg.data_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    cfg.model.vocab_size = json.load(f)["vocab_size"]
            self.train_data = load_split(cfg.data_dir, "train")
            self.val_data = load_split(cfg.data_dir, "val")

        # --- model / strategy ------------------------------------------ #
        self.model = MiniGPT(cfg.model)
        self.index = QuantIndex(self.model)
        self.fp32 = cfg.strategy == "fp32"
        self.needs_grad = False
        if self.fp32:
            self.model.set_quant_enabled(False)
            self.model.set_tracking(False)
            self.allocator = None
        else:
            signal = make_signal(cfg.strategy, seed=cfg.seed + 2)
            self.allocator = BitAllocator(
                self.index, cfg.alloc, signal, total_steps=cfg.steps
            )
            self.needs_grad = signal.needs_grad

        self.opt = self.model.configure_optimizer(cfg.lr, cfg.weight_decay)
        self.g_data = torch.Generator().manual_seed(cfg.seed + 1)

        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
            f.write(cfg.to_json())
        self.metrics_f = open(os.path.join(cfg.out_dir, "metrics.jsonl"), "w")
        self.alloc_f = open(os.path.join(cfg.out_dir, "alloc.jsonl"), "w")

    # ------------------------------------------------------------------ #

    def lr_at(self, step: int) -> float:
        cfg = self.cfg
        if step < cfg.lr_warmup_steps:
            return cfg.lr * (step + 1) / cfg.lr_warmup_steps
        t = (step - cfg.lr_warmup_steps) / max(1, cfg.steps - cfg.lr_warmup_steps)
        min_lr = cfg.lr * cfg.min_lr_frac
        return min_lr + 0.5 * (cfg.lr - min_lr) * (1 + math.cos(math.pi * t))

    @torch.no_grad()
    def evaluate(self) -> dict:
        cfg = self.cfg
        self.model.eval()
        out = {}
        modes = [("val_loss", True)]
        if not self.fp32:
            modes.append(("val_loss_fp", False))
        for key, quant in modes:
            if not self.fp32:
                self.model.set_quant_enabled(quant)
            g = torch.Generator().manual_seed(cfg.seed + 3)
            losses = []
            for _ in range(cfg.eval_iters):
                x, y = get_batch(
                    self.val_data, cfg.batch_size, cfg.model.block_size, g
                )
                _, loss = self.model(x, y)
                losses.append(loss.item())
            out[key] = float(np.mean(losses))
        if not self.fp32:
            self.model.set_quant_enabled(True)
        self.model.train()
        return out

    def _log(self, f, row: dict) -> None:
        f.write(json.dumps(row) + "\n")
        f.flush()

    def _bit_stats(self) -> tuple[float, float]:
        """(mean bits per weight, weight bytes streamed per forward).

        fp32 runs report the true 32-bit numbers, not the untouched init_bits
        buffers, so anchor artifacts can't be mistaken for a 2-bit run.
        """
        if self.fp32:
            n_weights = sum(
                m.in_features * m.out_features for m in self.index.layers
            )
            return 32.0, n_weights * 4.0
        bits = self.index.flat_bits()
        return float(bits.float().mean()), self.model.weight_traffic_bytes()

    def run(self) -> dict:
        cfg = self.cfg
        self.model.train()
        t0 = time.time()
        train_loss_ema = None
        last_eval: dict = {}

        for step in range(cfg.steps):
            lr = self.lr_at(step)
            for group in self.opt.param_groups:
                group["lr"] = lr

            x, y = get_batch(
                self.train_data, cfg.batch_size, cfg.model.block_size, self.g_data
            )
            _, loss = self.model(x, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                )
            if self.allocator is not None and self.needs_grad:
                for m in self.index.layers:
                    m.observe_grad()
            self.opt.step()

            done = step + 1
            if self.allocator is not None:
                event = self.allocator.maybe_update(done)
                if event is not None:
                    self._log(self.alloc_f, event)

            l = loss.item()
            train_loss_ema = l if train_loss_ema is None else (
                0.9 * train_loss_ema + 0.1 * l
            )
            if done % cfg.log_interval == 0 or done == cfg.steps:
                self._log(
                    self.metrics_f,
                    {
                        "step": done,
                        "train_loss": round(train_loss_ema, 5),
                        "lr": lr,
                        "elapsed_s": round(time.time() - t0, 1),
                    },
                )
            if done % cfg.eval_interval == 0 or done == cfg.steps:
                last_eval = self.evaluate()
                mean_bits, traffic = self._bit_stats()
                self._log(
                    self.metrics_f,
                    {
                        "step": done,
                        **{k: round(v, 5) for k, v in last_eval.items()},
                        "mean_bits": mean_bits,
                        "weight_traffic_mb": round(traffic / 1e6, 3),
                    },
                )

        mean_bits, traffic = self._bit_stats()
        summary = {
            "strategy": cfg.strategy,
            "seed": cfg.seed,
            "avg_bits_budget": cfg.alloc.avg_bits,
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "params": self.model.num_params(),
            "final_train_loss": train_loss_ema,
            **last_eval,
            "mean_bits": mean_bits,
            "weight_traffic_mb": round(traffic / 1e6, 3),
            "elapsed_s": round(time.time() - t0, 1),
        }
        with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        torch.save(
            {"model": self.model.state_dict(), "config": cfg.to_json()},
            os.path.join(cfg.out_dir, "ckpt.pt"),
        )
        self.metrics_f.close()
        self.alloc_f.close()
        return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Myelin trainer")
    p.add_argument("--config", type=str, default="", help="JSON config path")
    p.add_argument("--strategy", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--budget", type=float, default=None, help="avg bits")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_argparser().parse_args(argv)
    cfg = (
        TrainConfig.from_json_file(args.config) if args.config else TrainConfig()
    )
    if args.strategy is not None:
        cfg.strategy = args.strategy
    if args.seed is not None:
        cfg.seed = args.seed
    if args.steps is not None:
        cfg.steps = args.steps
    if args.out is not None:
        cfg.out_dir = args.out
    if args.budget is not None:
        cfg.alloc.avg_bits = args.budget
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.synthetic:
        cfg.synthetic = True
    if args.threads is not None:
        cfg.threads = args.threads
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    summary = Trainer(cfg).run()
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
