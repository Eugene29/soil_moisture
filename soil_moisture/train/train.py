import json
import os
import time
import warnings
from pathlib import Path

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from terratorch.tasks import PixelwiseRegressionTask

from soil_moisture.train.utils import set_seed, save_scatter, predict_and_score
from soil_moisture.model.models import prithvi_terratorch, RegressionModelSM, PrithviViT
from soil_moisture.data.data_loader import sm_dataloader, sm_dataset
from soil_moisture.data.data_utils import spatial_split


def build_data(cfg, output_dir):
    """Build the SM train/test datasets from the master CSV and wrap in datamodules."""
    print("building data...")

    df = pd.read_csv(cfg.data.master_csv)
    df = df[df["has_HLS"] == 1].reset_index(drop=True)
    # TODO: for a pure MERRA-only experiment this has_HLS filter can be relaxed
    #       to use all SM rows; kept on for now so the HLS/MERRA/both ablation
    #       compares the three on the identical sample.

    train_df, test_df = spatial_split(df, test_fraction=cfg.data.test_fraction)
    print(f"SM split: train rows={len(train_df)}  test rows={len(test_df)}")

    modality_factories = [
        hydra.utils.instantiate(cfg.data.modalities[m]) for m in cfg.modalities
    ]
    train_modalities = [make(train_df) for make in modality_factories]
    train_stats = [(m.mean, m.std) for m in train_modalities]
    test_modalities  = [
        make(test_df, stats=stats) 
        for make, stats in zip(modality_factories, train_stats)
    ]
    
    sm_dataset_train = sm_dataset(train_df, train_modalities)
    sm_mean, sm_std = sm_dataset_train.sm_mean, sm_dataset_train.sm_std
    sm_dataset_test = sm_dataset(test_df, test_modalities, sm_mean, sm_std)

    # datamodule_ serves as a dataloader to predict on the train split.
    datamodule = sm_dataloader(sm_dataset_train, sm_dataset_test, cfg)
    datamodule_ = sm_dataloader(sm_dataset_train, sm_dataset_train, cfg)

    return (
        datamodule,
        datamodule_,
        sm_dataset_train,
        sm_dataset_test,
        sm_mean,
        sm_std,
    )


def build_model(cfg, wt_file, use_TL_encoding):
    """Assemble the active-modality model into a Lightning task.

    The frozen Prithvi encoder is built only when HLS is used; merra-only skips
    it entirely (no weight load, no encoder).
    """
    print("building a model...")
    T_HLS = cfg.T_HLS
    T_MERRA = cfg.T_MERRA
    use_hls = "hls" in cfg.modalities
    use_merra = "merra" in cfg.modalities
    modality = "both" if use_hls and use_merra else ("hls" if use_hls else "merra")

    prithvi_model = None
    # Encoder emits (patches_per_frame * T_HLS + 1) tokens; padded 50x50 / patch 16 -> 3x3 = 9 patches/frame.
    n_tokens = 9 * T_HLS + 1
    if use_hls:
        coords_encoding = ["time", "location"] if use_TL_encoding else []

        prithvi_instance = PrithviViT(
            patch_size=cfg.model.patch_size,
            num_frames=T_HLS,
            in_chans=cfg.model.n_channel,
            embed_dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            head_dropout=cfg.model.head_dropout,
            backbone_input_size=[T_HLS, 50, 50],
            encoder_only=False,
            padding=True,
            depth=cfg.model.depth,
            coords_encoding=coords_encoding,
        )
        prithvi_model = prithvi_terratorch(
            wt_file,
            prithvi_instance,
            use_TL_encoding=use_TL_encoding,
        )
        prithvi_model.freeze_encoder()

    model_comb = RegressionModelSM(
        prithvi_model, n_tokens=n_tokens, T_MERRA=T_MERRA, modality=modality
    )
    task = PixelwiseRegressionTask(
        None, None, model=model_comb, loss="mse",
        optimizer=cfg.training.optimizer.name,
        lr=cfg.training.optimizer.lr,
        optimizer_hparams=cfg.training.optimizer.params,
    )
    return task


def build_trainer(cfg, task, output_dir, model_name, ts):
    """Construct the Lightning Trainer with the run's callbacks and logger."""
    print("building a trainer...")
    modality = "-".join(cfg.modalities)
    run_name = (
        f"{model_name}_{modality}_{ts}"
    )
    wandb_logger = WandbLogger(
        project="soil_moisture",
        name=run_name,
        log_model=False,
        save_dir=str(output_dir),
    )

    wandb_logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True)
    )

    trainer = Trainer(
        accelerator="cuda",
        callbacks=[
            RichProgressBar(),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        max_epochs=cfg.n_iteration,
        default_root_dir=str(output_dir),
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        enable_checkpointing=False,
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
    learning_rate = float(cfg.training.optimizer.lr)
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
        "num_epochs": cfg.n_iteration,
        "learning_rate": learning_rate,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

def run(
    model_name,
    wt_file,
    use_TL_encoding,
    output_dir,
    cfg,
    ts,
):
    set_seed(cfg.seed)

    (datamodule, datamodule_train, sm_dataset_train, sm_dataset_test,
     sm_mean, sm_std) = build_data(cfg, output_dir)

    task = build_model(cfg, wt_file, use_TL_encoding)
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


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    strt = time.time()
    
    warnings.filterwarnings("ignore")
    # only use one gpu for now as I'm seeing distributed sampling issue.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.dev)

    # TODO: add flags to change model variation.
    wt_file = cfg.wt_file
    model_name = (
        "Prithvi-EO-2.0-300M-TL" if cfg.tl_encoding else "Prithvi-EO-2.0-300M"
    )
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ts = run_dir.name  # the timestamp folder; used in the wandb run name.

    print(
        f"\n=== {model_name} | modalities {list(cfg.modalities)} "
        f"| T_HLS {cfg.T_HLS} | T_MERRA {cfg.T_MERRA} | seed {cfg.seed} ==="
    )
    metrics = run(
        model_name=model_name,
        wt_file=wt_file,
        use_TL_encoding=cfg.tl_encoding,
        output_dir=run_dir,
        cfg=cfg,
        ts=ts,
    )

    # Per-run summary, written into this run's dir so parallel launches never
    # clobber each other.
    pd.DataFrame([{"model": model_name, "modalities": list(cfg.modalities), **metrics}]).to_csv(
        run_dir / "summary.csv", index=False
    )
    print(f"Run output dir at {run_dir.absolute()}")

    dur = time.time() - strt
    print(f"total time taken: {dur}")

if __name__ == "__main__":
    main()