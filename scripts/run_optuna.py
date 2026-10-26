#!/usr/bin/env python3
"""Hyper-parameter search for the U-Net with Optuna.

    python scripts/run_optuna.py [--trials 10] [--iters 400]

The search uses its own split of recording a: frames up to a55 for training,
a70 to a90 for validation, a60 and a65 left out as a buffer. The test folds
of scripts/run_cv.py were run with fixed settings chosen before this search,
so the search does not leak into those numbers. The best settings are only
used for the final model (scripts/train_final.py).
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from dropseg.data import load_frames  # noqa: E402
from dropseg.metrics import pooled_scores  # noqa: E402
from dropseg.train import TrainConfig, predict_proba, train  # noqa: E402

OUT = ROOT / "results" / "optuna.json"


def main():
    import optuna
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()
    A = [f for f in load_frames(sessions=("A",))]
    tr = [f for f in A if f.minute <= 27.5]
    va = [f for f in A if f.minute >= 34.5]
    print("train", [f.id for f in tr], "validation", [f.id for f in va])

    def objective(trial):
        arch = trial.suggest_categorical("arch", ["plain", "resnet18"])
        cfg = TrainConfig(arch=arch,
                          lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                          dice_weight=trial.suggest_float("dice_weight", 0.0, 2.0),
                          water_bias=trial.suggest_float("water_bias", 0.2, 0.9),
                          dropout=trial.suggest_float("dropout", 0.0, 0.3),
                          crop=trial.suggest_categorical("crop", [160, 192, 256]),
                          iters=args.iters, batch=4, threads=8, seed=trial.number)
        model, _ = train(tr, cfg, log=lambda *a: None)
        d = pooled_scores([(predict_proba(model, f.image) >= 0.5, f.mask) for f in va])["dice"]
        print("trial %2d  %s  dice %.3f" % (trial.number, trial.params, d), flush=True)
        return d

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial({"arch": "resnet18", "lr": 3e-4, "dice_weight": 1.0, "water_bias": 0.6,
                         "dropout": 0.1, "crop": 192})   # the settings used in the cross-validation
    study.optimize(objective, n_trials=args.trials)
    trials = [{"number": t.number, "params": t.params, "dice": t.value} for t in study.trials]
    OUT.write_text(json.dumps({"train": [f.id for f in tr], "validation": [f.id for f in va],
                               "iters": args.iters, "best": study.best_params,
                               "best_dice": study.best_value, "trials": trials}, indent=1))
    print("best", study.best_params, "%.3f" % study.best_value)


if __name__ == "__main__":
    main()
