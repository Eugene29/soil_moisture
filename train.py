import argparse
import glob
import itertools
import json
import os
import pickle
import random
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from terratorch.models.backbones.prithvi_mae import PrithviViT
from terratorch.tasks import PixelwiseRegressionTask

from utils import *
from model.models import *
from data_loader.data_loader import *


def build_data(cfg):
    """Read inputs, build temporal windows, and wrap them in datamodules."""
    cfg_data = cfg["data"]
    T_HLS = cfg["T_HLS"]
    T_MERRA = cfg["T_MERRA"]
    test_year = cfg["test_year"]
    train_batch_size = cfg["training"]["train_batch_size"]
    test_batch_size = cfg["testing"]["test_batch_size"]

    print("TEST YEAR", test_year)
    means = np.array(cfg_data[f"means_for{test_year}test"])
    stds = np.array(cfg_data[f"stds_for{test_year}test"])
    merra_means = np.array(cfg_data[f"merra_means_for{test_year}test"])
    merra_stds = np.array(cfg_data[f"merra_stds_for{test_year}test"])
    gpp_means = np.array(cfg_data[f"gpp_means_for{test_year}test"])
    gpp_stds = np.array(cfg_data[f"gpp_stds_for{test_year}test"])

    # read merra, gpp inputs
    df = pd.read_csv("/home/yjean234/Azad/Prithvi-EO-2.0/examples/carbon_flux/data_train_hls_37sites_v0_1.csv")

    # get train_test splits, then build temporal windows per (site, year).
    test_df = df[df["year"] == test_year]
    train_df = df[df["year"] != test_year]

    train_chips, merra_train, train_target, train_dates = build_windows(
        train_df, cfg_data["chips"], T_HLS, T_MERRA
    )
    test_chips, merra_test, test_target, test_dates = build_windows(
        test_df, cfg_data["test_chips"], T_HLS, T_MERRA
    )
    print(
        f"T_HLS={T_HLS}  T_MERRA={T_MERRA}  train windows={len(train_chips)}  test windows={len(test_chips)}"
    )

    # Each sample's first arg is a list of T_HLS chip paths (oldest -> newest).
    flux_dataset_train = flux_dataset(
        train_chips,
        means,
        stds,
        merra_train,
        merra_means,
        merra_stds,
        gpp_means,
        gpp_stds,
        train_target,
        dates=train_dates,
    )
    flux_dataset_test = flux_dataset(
        test_chips,
        means,
        stds,
        merra_test,
        merra_means,
        merra_stds,
        gpp_means,
        gpp_stds,
        test_target,
        dates=test_dates,
    )

    datamodule = flux_dataloader(
        flux_dataset_train, flux_dataset_test, train_batch_size, test_batch_size, cfg
    )
    # datamodule_ serves the train set as its "test" loader, so we can score the
    # train split through the same predict path used for the test split.
    datamodule_ = flux_dataloader(
        flux_dataset_train,
        flux_dataset_train,
        train_batch_size,
        test_batch_size,
        cfg,
    )
    return (
        datamodule,
        datamodule_,
        flux_dataset_train,
        flux_dataset_test,
        gpp_means,
        gpp_stds,
    )


def build_model(cfg, wt_file, use_TL_encoding, manually_parse_weights):
    """Assemble the frozen Prithvi encoder + regression head into a Lightning task."""
    cfg_model = cfg["model"]
    T_HLS = cfg["T_HLS"]
    T_MERRA = cfg["T_MERRA"]

    coords_encoding = ["time", "location"] if use_TL_encoding else []

    prithvi_instance = PrithviViT(
        patch_size=cfg_model["patch_size"],
        num_frames=T_HLS,
        in_chans=cfg_model["n_channel"],
        embed_dim=cfg_model["embed_dim"],
        num_heads=cfg_model["num_heads"],
        mlp_ratio=cfg_model["mlp_ratio"],
        head_dropout=cfg_model["head_dropout"],
        backbone_input_size=[T_HLS, 50, 50],
        encoder_only=False,
        padding=True,
        depth=cfg_model["depth"],
        coords_encoding=coords_encoding,
    )
    prithvi_model = prithvi_terratorch(
        wt_file,
        prithvi_instance,
        manually_parse_weights=manually_parse_weights,
        use_TL_encoding=use_TL_encoding,
    )
    prithvi_model.freeze_encoder()

    # Encoder emits (patches_per_frame * T_HLS + 1) tokens; padded 50x50 / patch 16 -> 3x3 = 9 patches/frame.
    n_tokens = 9 * T_HLS + 1
    model_comb = RegressionModel_flux(prithvi_model, n_tokens=n_tokens, T_MERRA=T_MERRA)
    task = PixelwiseRegressionTask(
        None, None, model=model_comb, loss="mse", optimizer="AdamW"
    )
    return task


def build_trainer(cfg, task, output_dir):
    """Construct the Lightning Trainer with the run's callbacks and logger."""
    checkpoint_callback = ModelCheckpoint(
        monitor=task.monitor, save_top_k=1, save_last=True
    )
    # Logger created for its side effect of fixing the log dir layout, as in the
    # original script (Trainer itself logs to default_root_dir).
    TensorBoardLogger(save_dir=str(output_dir), name="carbon_flux")

    trainer = Trainer(
        accelerator="cuda",
        callbacks=[
            RichProgressBar(),
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        max_epochs=cfg["n_iteration"],
        default_root_dir=str(output_dir),
        log_every_n_steps=1,
        check_val_every_n_epoch=200,
    )
    return trainer


def evaluate_split(
    trainer, task, datamodule, eval_dataset, gpp_means, gpp_stds, label,
    model_name, test_year, output_dir,
):
    """Predict + score one split and save its scatter plot. Returns the score dict."""
    scores = predict_and_score(
        trainer, task, datamodule, eval_dataset, gpp_means, gpp_stds
    )
    save_scatter(
        scores["targ_unnorm"],
        scores["pred_unnorm"],
        scores["r2_unnorm"],
        f"{model_name} {test_year} {label}",
        output_dir / f"{label}_scatter.png",
    )
    return scores


def save_metrics(zs, test, train, cfg, output_dir):
    """Collect per-split scores into one metrics dict and write it as JSON."""
    learning_rate = float(cfg["training"]["optimizer"]["params"]["lr"])
    metrics = {
        "zeroshot_r2_norm": zs["r2_norm"],
        "zeroshot_r2_unnorm": zs["r2_unnorm"],
        "test_r2_norm": test["r2_norm"],
        "test_r2_unnorm": test["r2_unnorm"],
        "test_mse_unnorm": test["mse_unnorm"],
        "test_mae_unnorm": test["mae_unnorm"],
        "train_r2_norm": train["r2_norm"],
        "train_r2_unnorm": train["r2_unnorm"],
        "train_mse_unnorm": train["mse_unnorm"],
        "train_mae_unnorm": train["mae_unnorm"],
        "num_epochs": cfg["n_iteration"],
        "learning_rate": learning_rate,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

def read_and_save_config(cfg_fname, T_HLS, T_MERRA, test_year, output_dir, seed=0):
    ### Read model configs from YAML and stamp in this run's resolved values.
    with open(cfg_fname, "r") as file:
        cfg = yaml.safe_load(file)
    cfg["T_HLS"] = T_HLS
    cfg["T_MERRA"] = T_MERRA
    cfg["seed"] = seed
    cfg["test_year"] = test_year

    # Snapshot the resolved config before any in-place numpy conversions below,
    # so the dumped file stays plain YAML (no numpy/Path tags).
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    
    return cfg


def run(
    model_name,
    wt_file,
    use_TL_encoding,
    output_dir,
    manually_parse_weights,
    cfg,
):
    set_seed(cfg["seed"])

    (datamodule, datamodule_train, flux_dataset_train, flux_dataset_test,
     gpp_means, gpp_stds) = build_data(cfg)

    task = build_model(cfg, wt_file, use_TL_encoding, manually_parse_weights)
    trainer = build_trainer(cfg, task, output_dir)

    # zeroshot eval -> fit -> post-training eval on the test and train splits.
    zs = evaluate_split(
        trainer, task, datamodule, flux_dataset_test, gpp_means, gpp_stds,
        "zeroshot", model_name, cfg["test_year"], output_dir,
    )
    trainer.fit(model=task, datamodule=datamodule)
    test = evaluate_split(
        trainer, task, datamodule, flux_dataset_test, gpp_means, gpp_stds,
        "test", model_name, cfg["test_year"], output_dir,
    )
    train = evaluate_split(
        trainer, task, datamodule_train, flux_dataset_train, gpp_means, gpp_stds,
        "train", model_name, cfg["test_year"], output_dir,
    )

    metrics = save_metrics(zs, test, train, cfg, output_dir)
    print(f'  test R2={test["r2_unnorm"]:.4f}  train R2={train["r2_unnorm"]:.4f}')
    return metrics


class ModelConfig(NamedTuple):
    model_name: str
    config_fname: str
    wt_file: str
    use_TL_encoding: bool
    T_HLS: int = 1
    T_MERRA: int = 1
    seed: int = 0
    # years: list[int]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="1")
    parser.add_argument("--t-hls", type=int, default=1)
    parser.add_argument("--t-merra", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tl-encoding", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.chdir(Path(__file__).absolute().parent)

    # only use one gpu for now as I'm seeing distributed sampling issue.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev

    # Single config driven by CLI args; the grid + GPU scheduling lives in the
    # launcher (launch_sweep.sh), which fans these out one-per-GPU.
    model_name = "Prithvi-EO-2.0-300M-TL" if args.tl_encoding else "Prithvi-EO-2.0-300M"
    model_configs = [
        # (model_name, config_fname, wt_file, use_TL_encoding, T_HLS, T_MERRA, seed)
        ModelConfig(
            model_name,
            "fluxconfig_trainer.yaml",
            "/home/yjean234/Azad/Prithvi-EO-2.0/examples/carbon_flux/Prithvi_EO_V2_300M_TL.pt",
            args.tl_encoding,
            args.t_hls,
            args.t_merra,
            args.seed,
        )
    ]
    years = [2020]

    # --- Toggle: original T_HLS x T_MERRA sweep (9 runs, single seed) ---
    # model_configs = [
    #     # (model_name, config_fname, wt_file, use_TL_encoding, T_HLS, T_MERRA)
    #     ModelConfig(
    #         "Prithvi-EO-2.0-300M",
    #         "fluxconfig_trainer.yaml",
    #         "Prithvi_EO_V2_300M_TL.pt",
    #         False,
    #         x,
    #         y,
    #     ) for x, y, in itertools.product([1, 4, 7], [1, 7, 14])
    # ]
    # model_configs = [
    #     # (model_name, config_fname, wt_file, use_TL_encoding)
    #     ('Prithvi-EO-2.0-300M',    'fluxconfig_trainer.yaml',       'Prithvi_EO_V2_300M_TL.pt', False),
    #     ('Prithvi-EO-2.0-300M-TL', 'fluxconfig_trainer.yaml',       'Prithvi_EO_V2_300M_TL.pt', True),
    #     ('Prithvi-EO-2.0-600M',    'fluxconfig_trainer_large.yaml', 'Prithvi_EO_V2_600M_TL.pt', False),
    #     ('Prithvi-EO-2.0-600M-TL', 'fluxconfig_trainer_large.yaml', 'Prithvi_EO_V2_600M_TL.pt', True),
    # ]
    # years = [2018, 2019, 2020, 2021]

    output_root = Path("outputs")
    manually_parse_weights = True

    output_root.mkdir(exist_ok=True)
    # Per-process summary so parallel one-per-GPU launches don't clobber each other.
    tl_tag = "_tl" if args.tl_encoding else ""
    summary_path = (
        output_root
        / f"summary_thls{args.t_hls}_tmerra{args.t_merra}_seed{args.seed}{tl_tag}.csv"
    )

    summary_rows = []
    for (
        model_name,
        config_fname,
        wt_file,
        use_TL_encoding,
        T_HLS,
        T_MERRA,
        seed,
    ) in model_configs:
        for test_year in years:
            print(
                f"\n=== {model_name} | test year {test_year} | T_HLS {T_HLS} | seed {seed} ==="
            )
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = (
                output_root
                / model_name
                / f"year_{test_year}"
                / f"thls_{T_HLS}_tmerra_{T_MERRA}_seed_{seed}"
                / ts
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg = read_and_save_config(
                config_fname, T_HLS, T_MERRA, test_year, run_dir, seed
            )

            metrics = run(
                model_name=model_name,
                wt_file=wt_file,
                use_TL_encoding=use_TL_encoding,
                output_dir=run_dir,
                manually_parse_weights=manually_parse_weights,
                cfg=cfg,
            )
            summary_rows.append({"model": model_name, "year": test_year, **metrics})
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

            print(f"Run output dir at {run_dir.absolute()}")

    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
