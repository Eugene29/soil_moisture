
from datetime import date

from lightning import LightningModule
import numpy as np
import torch
from torchgeo.datasets import NonGeoDataset
from torch.utils.data import DataLoader

from soil_moisture.data.data_utils import (
    doy_to_date,
    resolve_chip,
)

class sm_dataset(NonGeoDataset):
    """Single-frame soil-moisture dataset.

    One master-CSV row -> one HLS chip -> one MERRA day -> one SM target.
    Stats (HLS means/stds, MERRA means/stds, SM mean/std) are train-derived and
    passed in; the same stats normalize train and test.
    """

    def __init__(self, df, modalities, sm_mean=None, sm_std=None):
        # rows: a DataFrame of filtered master-CSV rows (has_HLS==1).
        # merra_df: per-split MERRA frame, MultiIndexed on (network, station),
        #           with a `time` column and all MERRA_COLS (LND+SLV merged).
        self.rows = df.reset_index(drop=True)
        self.modalities = modalities
        
        if sm_mean is None:
            self.sm_mean, self.sm_std = self.sm_stats(df=df, cols=["soil_moisture"])
        else:
            self.sm_mean, self.sm_std = sm_mean, sm_std
        
    def sm_stats(self, df, cols, ddof=1):
        """Computes the mean and std of a dataframe using ddof=1."""
        arr = df[cols].to_numpy()
        return arr.mean(axis=0), arr.std(axis=0, ddof=ddof)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sample = {}
        
        row = self.rows.iloc[idx]

        sm = (float(row["soil_moisture"]) - self.sm_mean) / self.sm_std
        sm = torch.from_numpy(np.asarray(sm, dtype=np.float32).reshape(1))

        for modality in self.modalities:
            sample.update(modality[idx])
        # mask is the label
        sample.update({"mask": sm.to(torch.float)})
        # PixelwiseRegressionTask.predict_step requires filename
        sample.update({"filename": ""})

        return sample


# ---------------------------------------------------------------------------
# Dormant path: temporal (T_HLS/T_MERRA) machinery, kept for later development.
# NOT exercised by the current task -- gated behind an assert below.
# ---------------------------------------------------------------------------
# def is_contiguous_window(years: pd.Series):
#     """Checks if years only have one year or two consecutive years."""
#     unique_years = years.unique()
#     if len(unique_years) == 1:
#         return True
#     elif len(unique_years) == 2:
#         return unique_years.min() == unique_years.max() - 1
#     else:
#         return False


# def build_windows(df_split, chips_dir, T_HLS, T_MERRA):
#     """Build temporal windows per site with independent HLS / MERRA lengths.

#     DORMANT: this temporal codepath has not been adapted to the SM master-CSV
#     schema (it still references flux columns SITE_ID/Chip/GPP/year/doy) and is
#     kept only as a starting point for future multi-frame work.
#     """
#     assert False, "Temporal (T_HLS/T_MERRA) codepath hasn't been explored yet."

#     chips, merras, targets, dates = [], [], [], []
#     df_sorted = df_split.sort_values(["year", "doy"])
#     n_window = max(T_HLS, T_MERRA)  # need enough history to fill the longer sequence
#     for _, group in df_sorted.groupby(["SITE_ID"]):
#         rows = group.reset_index(drop=True)
#         for i in range(n_window - 1, len(rows)):
#             hls_window = rows.iloc[i - T_HLS + 1 : i + 1]
#             merra_window = rows.iloc[i - T_MERRA + 1 : i + 1]
#             full_window = rows.iloc[i - n_window + 1 : i + 1]
#             if not is_contiguous_window(full_window["year"]):
#                 continue
#             target = hls_window.iloc[-1]
#             chips.append(
#                 [chips_dir + "/" + str(c) for c in hls_window["Chip"].tolist()]
#             )
#             merras.append(merra_window[MERRA_COLS].values.astype(float).tolist())
#             targets.append(float(target["GPP"]))
#             dates.append(
#                 list(zip(hls_window["year"].astype(int), hls_window["doy"].astype(int)))
#             )
#     return chips, merras, targets, dates


# class flux_dataset(NonGeoDataset):
#     """DORMANT multi-frame dataset (temporal windows). Kept for later; gated."""

#     def __init__(self, path, means, stds, merras_data, merra_means, merra_stds,
#                  gpp_mean, gpp_std, target, dates=None):
#         assert False, "Temporal flux_dataset codepath hasn't been explored yet."
#         self.data_dir = path
#         self.means = means
#         self.stds = stds
#         self.merras = merras_data
#         self.merra_means = merra_means
#         self.merra_stds = merra_stds
#         self.gpp_means = gpp_mean
#         self.gpp_stds = gpp_std
#         self.target = target
#         self.dates = dates

#     def __len__(self):
#         return len(self.data_dir)

#     def __getitem__(self, idx):
#         image_paths = self.data_dir[idx]
#         if_image = 1
#         frames = []
#         center_lonlat = None
#         for frame_path in image_paths:
#             image, center_lonlat = load_raster(frame_path, if_image, crop=(50, 50), return_center_lonlat=True)
#             frames.append(preprocess_image(image, self.means, self.stds))
#         final_image = torch.stack(frames, dim=1)  # (C, T, H, W)

#         merra_arr = np.array(self.merras[idx], dtype=np.float32)  # (T_MERRA, n_vars)
#         merra_vars = torch.from_numpy(merra_arr.T.copy())         # (n_vars, T_MERRA)
#         mean_merra = self.merra_means.reshape(-1, 1)
#         stds_merra = self.merra_stds.reshape(-1, 1)
#         merra_vars_norm = (merra_vars - mean_merra) / stds_merra

#         mean_gpp = self.gpp_means.reshape(-1, 1, 1)
#         stds_gpp = self.gpp_stds.reshape(-1, 1, 1)
#         gpp_vars_norm = (self.target[idx] - mean_gpp) / stds_gpp
#         gpp_vars_norm = torch.from_numpy(np.array(gpp_vars_norm).reshape(1))

#         temporal_coords = torch.tensor(
#             [[float(year), float(doy) - 1.0] for year, doy in self.dates[idx]],
#             dtype=torch.float32,
#         )
#         lon, lat = center_lonlat
#         location_coords = torch.tensor([float(lat), float(lon)], dtype=torch.float32)

#         return {
#             "image": final_image.to(torch.float),
#             "pt1d": merra_vars_norm.to(torch.float),
#             "mask": gpp_vars_norm.to(torch.float),
#             "filename": image_paths[-1],
#             "temporal_coords": temporal_coords,
#             "location_coords": location_coords,
#         }


# ---------------------------------------------------------------------------
# Lightning datamodule wrapper
# ---------------------------------------------------------------------------
class sm_dataloader(LightningModule):

    def __init__(self, dataset_train=None, dataset_test=None, config=None):
        super().__init__()
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.train_batch_size = config["training"]["train_batch_size"]
        self.test_batch_size = config["testing"]["test_batch_size"]
        self.config = config

    def setup(self, stage: str = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.train_batch_size,
                          shuffle=self.config["training"]["shuffle"])

    # val returning val loss and for wandb logging
    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.test_batch_size,
                          shuffle=self.config["testing"]["shuffle"])

    # predict is used for storing output per sample
    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.test_batch_size,
                          shuffle=False)


if __name__ == "__main__":
    date1 = doy_to_date(2020, 10)
    print(f"date: {date1}", flush=True)