import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import r2_score
from torcheval.metrics import R2Score


### Evaluation metrics to log while training
def get_mse(pred, targ):
    criterion = nn.MSELoss()
    mse_loss = criterion(pred, targ)
    return mse_loss.item()


def get_mae(pred, targ):
    criterion = nn.L1Loss()
    mae_loss = criterion(pred, targ)
    return mae_loss.item()


# can use sklearn R2 instead -- as used to evalute whole dataset
def get_r_sq(pred, targ):
    metric = R2Score()
    metric.update(pred, targ)
    r2 = metric.compute()
    # Extracting the first element if it's a tensor with multiple elements
    if isinstance(r2, torch.Tensor) and r2.numel() > 1:
        return r2[0].item()
    else:
        return r2.item()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_scatter(targ, pred, r2, title, path):
    fig, ax = plt.subplots()
    ax.scatter(targ, pred, alpha=0.6)
    lo, hi = float(min(targ)), float(max(targ))
    ax.plot([lo, hi], [lo, hi], color="red", lw=2, label="Perfect fit")
    ax.set_xlabel("True GPP", fontsize=14)
    ax.set_ylabel("Predicted GPP", fontsize=14)
    ax.grid(True)
    ax.set_title(f"{title} — R2: {r2:.4f} - size: {len(targ)}")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def predict_and_score(
    trainer, task, datamodule, flux_dataset_eval, gpp_means, gpp_stds
):
    results = trainer.predict(
        model=task, datamodule=datamodule, return_predictions=True
    )
    pred = np.concatenate([i[0] for i in results], axis=0)
    targ = np.concatenate([j["mask"] for j in flux_dataset_eval], axis=0)[:, None]

    r2_norm = r2_score(targ, pred)

    mean_gpp = gpp_means.reshape(-1, 1, 1)
    stds_gpp = gpp_stds.reshape(-1, 1, 1)
    pred_unnorm = (pred * stds_gpp + mean_gpp).flatten()[:, None]
    targ_unnorm = (targ * stds_gpp + mean_gpp).flatten()[:, None]
    r2_unnorm = r2_score(targ_unnorm, pred_unnorm)
    mse_unnorm = float(np.mean((targ_unnorm - pred_unnorm) ** 2))
    mae_unnorm = float(np.mean(np.abs(targ_unnorm - pred_unnorm)))

    return {
        "r2_norm": float(r2_norm),
        "r2_unnorm": float(r2_unnorm),
        "mse_unnorm": mse_unnorm,
        "mae_unnorm": mae_unnorm,
        "pred_unnorm": pred_unnorm,
        "targ_unnorm": targ_unnorm,
    }