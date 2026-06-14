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
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from terratorch.models.backbones.prithvi_mae import PrithviViT
from terratorch.tasks import PixelwiseRegressionTask

from utils import *
from model.models import *
from data_loader.data_loader import *
from preprocess.compute_data_stats import hls_stats, col_stats


def load_merra_frame(stations, merra_root):
    """Load + merge MERRA for a set of stations into one MultiIndexed DataFrame.

    For each (network, station) in `stations`, reads the LND and SLV CSVs, parses
    `time`, and merges them on `time` (LND/SLV share identical hourly timestamps)
    so each row carries all MERRA_COLS. The concatenated frame is indexed on
    (network, station) for fast per-station lookup. MERRA filenames sanitize
    '#' -> '_' (HLS folders keep '#').
    """
    frames = []
    for network, station in stations:
        st = station.replace("#", "_")
        def _read(product):
            hits = sorted(glob.glob(
                os.path.join(merra_root, f"{network}_{st}_{product}_*.csv")))
            if not hits:
                raise FileNotFoundError(
                    f"No MERRA {product} file for {network}_{st} in {merra_root}")
            d = pd.read_csv(hits[0])
            d["time"] = pd.to_datetime(d["time"])
            return d
        lnd = _read("M2T1NXLND")[["time"] + MERRA_LND_COLS]
        slv = _read("M2T1NXSLV")[["time"] + MERRA_SLV_COLS]
        merged = lnd.merge(slv, on="time", how="inner")
        merged["network"] = network
        merged["station"] = station
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    return out.set_index(["network", "station"]).sort_index()


def dummy_sm_stats():
    """Hardcoded normalization stats matching compute_sm_stats' return shapes."""
    n_hls = 6
    n_merra = len(MERRA_COLS)  # 10
    return (
        np.zeros(n_hls, dtype=np.float32),       # means
        np.ones(n_hls, dtype=np.float32),        # stds
        np.zeros(n_merra, dtype=np.float32),     # merra_means
        np.ones(n_merra, dtype=np.float32),      # merra_stds
        np.zeros(1, dtype=np.float32),           # sm_mean
        np.ones(1, dtype=np.float32),            # sm_std
    )


def compute_sm_stats(train_df, chips_root, merra_train, output_dir):
    """Compute train-split-only normalization stats for the SM task and save them.

    HLS stats are per-band over one resolved chip per train row. We seed the
    global numpy RNG first so the chip resolution here is reproducible. MERRA
    stats are over the per-row overpass-time vectors; SM stats over the
    soil_moisture column. The same stats normalize train AND test, so the held-out
    split never leaks into normalization. Stats are written to the run's output
    dir (they are bound to this split/seed), and also returned in-memory.
    """
    # HLS + MERRA: resolve one chip per train row (its overpass time drives the
    # MERRA lookup), accumulate chip paths and MERRA vectors together.
    chip_paths = []
    merra_rows = []
    for _, row in train_df.iterrows():
        network, station = str(row["network"]), str(row["station"])
        day = datetime.fromisoformat(str(row["date"])).date()
        chip = resolve_chip(network, station, day, chips_root)
        chip_paths.append(chip)
        merra_rows.append(
            merra_vector(merra_train, network, station, overpass_datetime(chip))
        )
    # chips_dir="" so hls_stats uses the absolute paths as-is.
    means, stds = hls_stats(chip_paths, chips_dir="")

    merra_rows = np.stack(merra_rows)
    merra_means = merra_rows.mean(axis=0)
    merra_stds = merra_rows.std(axis=0, ddof=1)

    # SM target: mean/std of the soil_moisture column (ddof=1).
    sm_mean, sm_std = col_stats(train_df, ["soil_moisture"])

    stats = {
        "provenance": {
            "n_train_rows": int(len(train_df)),
            "n_train_stations": int(train_df.groupby(["network", "station"]).ngroups),
            "merra_cols": MERRA_COLS,
            "computed_at": datetime.now().isoformat(timespec="seconds"),
        },
        "hls": {"means": means.tolist(), "stds": stds.tolist()},
        "merra": {"means": merra_means.tolist(), "stds": merra_stds.tolist()},
        "sm": {"mean": sm_mean.tolist(), "std": sm_std.tolist()},
    }
    with open(Path(output_dir) / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return (means, stds, merra_means, merra_stds, sm_mean, sm_std)


def build_data(cfg, output_dir):
    """Build the SM train/test datasets from the master CSV and wrap in datamodules."""
    print("building data...")
    cfg_data = cfg["data"]
    train_batch_size = cfg["training"]["train_batch_size"]
    test_batch_size = cfg["testing"]["test_batch_size"]
    chips_root = cfg_data["chips_root"]
    merra_root = cfg_data["merra_root"]

    df = pd.read_csv(cfg_data["master_csv"])
    df = df[df["has_HLS"] == 1].reset_index(drop=True)

    # Spatial (by-station) split -- leakage-safe.
    train_df, test_df = spatial_split(df, test_fraction=cfg_data.get("test_fraction", 0.2))
    print(f"SM split: train rows={len(train_df)}  test rows={len(test_df)}")

    # Load MERRA once per split (read + LND/SLV merge), indexed by station.
    train_stations = list(train_df[["network", "station"]].drop_duplicates().itertuples(index=False, name=None))
    test_stations = list(test_df[["network", "station"]].drop_duplicates().itertuples(index=False, name=None))
    merra_train = load_merra_frame(train_stations, merra_root)
    merra_test = load_merra_frame(test_stations, merra_root)

    if cfg.get("debug", False):
        print("debug=True -- skipping compute_sm_stats, using dummy stats")
        means, stds, merra_means, merra_stds, sm_mean, sm_std = dummy_sm_stats()
    else:
        means, stds, merra_means, merra_stds, sm_mean, sm_std = compute_sm_stats(
            train_df, chips_root, merra_train, output_dir
        )

    sm_dataset_train = sm_dataset(
        train_df, chips_root, merra_train,
        means, stds, merra_means, merra_stds, sm_mean, sm_std,
    )
    sm_dataset_test = sm_dataset(
        test_df, chips_root, merra_test,
        means, stds, merra_means, merra_stds, sm_mean, sm_std,
    )

    datamodule = sm_dataloader(
        sm_dataset_train, sm_dataset_test, train_batch_size, test_batch_size, cfg
    )
    # datamodule_ serves the train set as its "test" loader, so we can score the
    # train split through the same predict path used for the test split.
    datamodule_ = sm_dataloader(
        sm_dataset_train, sm_dataset_train, train_batch_size, test_batch_size, cfg
    )
    return (
        datamodule,
        datamodule_,
        sm_dataset_train,
        sm_dataset_test,
        sm_mean,
        sm_std,
    )


def build_model(cfg, wt_file, use_TL_encoding, manually_parse_weights):
    """Assemble the frozen Prithvi encoder + regression head into a Lightning task."""
    print("building a model...")
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
    model_comb = RegressionModelSM(prithvi_model, n_tokens=n_tokens, T_MERRA=T_MERRA)
    task = PixelwiseRegressionTask(
        None, None, model=model_comb, loss="mse", optimizer="AdamW"
    )
    return task


def build_trainer(cfg, task, output_dir, model_name, ts):
    """Construct the Lightning Trainer with the run's callbacks and logger."""
    print("building a trainer...")
    checkpoint_callback = ModelCheckpoint(
        monitor=task.monitor, save_top_k=1, save_last=True
    )
    # Logger created for its side effect of fixing the log dir layout, as in the
    # original script (Trainer itself logs to default_root_dir).
    TensorBoardLogger(save_dir=str(output_dir), name="carbon_flux")
    # Descriptive run name (with the run timestamp) so re-runs of the same
    # (T_HLS, T_MERRA, seed) combo stay distinguishable in the dashboard, and it
    # matches the output folder. log_model=False -> no checkpoint artifacts.
    run_name = (
        f"{model_name}_thls{cfg['T_HLS']}_tmerra{cfg['T_MERRA']}"
        f"_{ts}"
    )
    wandb_logger = WandbLogger(
        project="soil_moisture",
        name=run_name,
        log_model=False,
        save_dir=str(output_dir),
    )
    wandb_logger.experiment.config.update(cfg) 

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
        check_val_every_n_epoch=1,
        logger=wandb_logger,
    )
    return trainer, wandb_logger


def evaluate_split(
    trainer, task, datamodule, eval_dataset, sm_mean, sm_std, label,
    model_name, output_dir,
):
    """Predict + score one split and save its scatter plot. Returns the score dict."""
    print("evaluating a split...")
    scores = predict_and_score(
        trainer, task, datamodule, eval_dataset, sm_mean, sm_std
    )
    save_scatter(
        scores["targ_unnorm"],
        scores["pred_unnorm"],
        scores["r2_unnorm"],
        f"{model_name} {label}",
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

def read_and_save_config(cfg_fname, T_HLS, T_MERRA, output_dir, seed=0):
    ### Read model configs from YAML and stamp in this run's resolved values.
    with open(cfg_fname, "r") as file:
        cfg = yaml.safe_load(file)
    cfg["T_HLS"] = T_HLS
    cfg["T_MERRA"] = T_MERRA
    cfg["seed"] = seed

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
    ts,
):
    set_seed(cfg["seed"])

    (datamodule, datamodule_train, sm_dataset_train, sm_dataset_test,
     sm_mean, sm_std) = build_data(cfg, output_dir)

    task = build_model(cfg, wt_file, use_TL_encoding, manually_parse_weights)
    trainer, wandb_logger = build_trainer(cfg, task, output_dir, model_name, ts)

    # zeroshot eval -> fit -> post-training eval on the test and train splits.
    zs = evaluate_split(
        trainer, task, datamodule, sm_dataset_test, sm_mean, sm_std,
        "zeroshot", model_name, output_dir,
    )
    trainer.fit(model=task, datamodule=datamodule)
    test = evaluate_split(
        trainer, task, datamodule, sm_dataset_test, sm_mean, sm_std,
        "test", model_name, output_dir,
    )
    train = evaluate_split(
        trainer, task, datamodule_train, sm_dataset_train, sm_mean, sm_std,
        "train", model_name, output_dir,
    )

    metrics = save_metrics(zs, test, train, cfg, output_dir)
    wandb_logger.log_metrics(metrics)
    wandb_logger.experiment.finish()
    print(f'test R2={test["r2_unnorm"]:.4f}  train R2={train["r2_unnorm"]:.4f}')
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
    parser.add_argument("--debug", action="store_true")
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
        print(
            f"\n=== {model_name} | T_HLS {T_HLS} | T_MERRA {T_MERRA} | seed {seed} ==="
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            output_root
            / model_name
            / f"thls_{T_HLS}_tmerra_{T_MERRA}"
            / ts
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = read_and_save_config(config_fname, T_HLS, T_MERRA, run_dir, seed)
        cfg["debug"] = args.debug

        metrics = run(
            model_name=model_name,
            wt_file=wt_file,
            use_TL_encoding=use_TL_encoding,
            output_dir=run_dir,
            manually_parse_weights=manually_parse_weights,
            cfg=cfg,
            ts=ts,
        )
        summary_rows.append({"model": model_name, **metrics})
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

        print(f"Run output dir at {run_dir.absolute()}")

    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
