"""Data loading for the soil-moisture (SM) downstream task.

Inputs are resolved ON THE FLY from the master SM table
(`master_sm_hls_tx_2020.csv`, columns: network, station, longitude, latitude,
depth_from, depth_to, date, soil_moisture, has_HLS). For each row the loader:

  1. resolves the HLS chip by globbing the station folder and matching the row's
     calendar date to the filename's day-of-year token, picking ONE at random
     when several tiles/sensors cover the same (station, date); and
  2. reads that date's MERRA values from the per-station hourly LND/SLV CSVs,
     reducing hourly -> daily.

Normalization uses train-split-only stats passed in by the caller (see
`preprocess/compute_data_stats.py`); the SAME stats are applied to train and
test so the held-out split never leaks into normalization.

Two codepaths live here:
  * `sm_dataset`  -- the active, single-frame path (one chip -> one SM value).
  * `flux_dataset` + `build_windows` -- the dormant temporal (T_HLS/T_MERRA)
    machinery, kept for later development. It is gated behind an assert and is
    NOT exercised by the current task.
"""

import os
import re
import glob
from datetime import date, datetime, timedelta

from lightning import LightningModule
import numpy as np
import pandas as pd
import torch
from torchgeo.datasets import NonGeoDataset
import rasterio

from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from torch.utils.data import DataLoader


NO_DATA_FLOAT = 0.0001

# --- MERRA variable set for the SM task ------------------------------------
# Chosen as physically relevant to surface soil moisture. The hourly MERRA2
# data is split across two products: LND (land surface) and SLV (single level).
# TODO: revisit MERRA variable selection AND the hourly->daily reduction
#       (currently daily-mean for all; precip-as-sum and T2MIN/MAX may be added).
MERRA_LND_COLS = [
    "PRECTOTLAND",  # total precipitation over land -- primary SM driver
    "GWETTOP",      # surface soil wetness -- closest proxy to the target
    "GWETROOT",     # root-zone soil wetness
    "SFMC",         # surface soil moisture content
    "RZMC",         # root-zone soil moisture content
    "EVLAND",       # evaporation over land -- moisture loss
    "LHLAND",       # latent heat flux -- couples to ET / drying
    "RUNOFF",       # runoff -- water leaving the column
]
MERRA_SLV_COLS = [
    "T2M",          # 2m air temperature -- evaporative demand
    "T2MDEW",       # 2m dewpoint -- humidity / drying potential
]
MERRA_COLS = MERRA_LND_COLS + MERRA_SLV_COLS  # fixed output order, len == 10

# HLS filename date token, e.g. HLS.S30.T14RQU.2020032T170659.v2.0.merged.subset.tif
#                                              ^^^^ ^^^  = YYYY DDD (day-of-year)
_HLS_DATE_TOKEN = re.compile(r"\.(\d{4})(\d{3})T\d{6}\.")


def doy_to_date(year: int, doy: int) -> date:
    """Convert a (year, day-of-year) pair to a calendar date (2020 is a leap year)."""
    return date(year, 1, 1) + timedelta(days=doy - 1)


# ---------------------------------------------------------------------------
# On-the-fly resolution helpers
# ---------------------------------------------------------------------------
def resolve_chip(network, station, day, chips_root):
    """Find one HLS merged-chip path for a (station, date) on the fly.

    Globs `<chips_root>/<network>_<station>/*.merged.subset.tif`, keeps files
    whose day-of-year token maps to `day` (a `datetime.date`). Since a
    station/date can be covered by several tiles or sensors (S30/L30) -- picks
    ONE at random via `rng`. Random selection happens per access, so over many
    epochs the model sees the different valid tiles (a mild augmentation).

    Raises if no merged chip matches (callers should only pass has_HLS==1 rows).
    """
    folder = os.path.join(chips_root, f"{network}_{station}")
    matches = []
    for path in glob.glob(os.path.join(folder, "*.merged.subset.tif")):
        m = _HLS_DATE_TOKEN.search(os.path.basename(path))
        if not m:
            continue
        if doy_to_date(int(m.group(1)), int(m.group(2))) == day:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No merged HLS chip for {network}_{station} on {day} in {folder}"
        )
    return matches[np.random.randint(len(matches))]


def overpass_datetime(chip_path):
    """Parse the HLS acquisition datetime (UTC) from a chip filename.

    Filename token is `.<YYYY><DDD>T<HHMMSS>.`, e.g. 2020012T172032 -> 2020-01-12
    17:20:32. Returns a `datetime.datetime`.
    """
    name = os.path.basename(chip_path)
    m = re.search(r"\.(\d{4})(\d{3})T(\d{2})(\d{2})(\d{2})\.", name)
    if not m:
        raise ValueError(f"No HLS date/time token in {chip_path}")
    d = doy_to_date(int(m.group(1)), int(m.group(2)))
    return datetime(d.year, d.month, d.day,
                    int(m.group(3)), int(m.group(4)), int(m.group(5)))


def merra_vector(merra_df, network, station, when):
    """MERRA feature vector for the row nearest the HLS overpass time `when`.

    `merra_df` is a per-split frame MultiIndexed on (network, station), with a
    `time` column (datetime) and all MERRA_COLS columns (LND+SLV already merged
    on time). Returns a float32 array of shape (len(MERRA_COLS),) for the single
    hourly row closest to `when` (a datetime) for that station.

    TODO: verify the MERRA `time` column is UTC (HLS overpass times are UTC). A
          timezone mismatch would shift the selected hour.
    """
    station_rows = merra_df.loc[(network, station)]
    pos = (station_rows["time"] - pd.Timestamp(when)).abs().to_numpy().argmin()
    return station_rows[MERRA_COLS].to_numpy(dtype=np.float32)[pos]


def load_raster(path, crop=None, return_center_lonlat=False):

        with rasterio.open(path) as src:
            img = src.read()

            # load selected 6 bands (HLS surface reflectance)
            img = img[0:6, :, :]

            # Remap the NO_DATA sentinel and NaN gaps (some SM chips have NaN
            # pixels) to NO_DATA_FLOAT so they don't propagate through z-scoring.
            img = np.where(np.isnan(img), NO_DATA_FLOAT, img)

            if crop:
                img = img[:, -crop[0]:, -crop[1]:]

            center_lonlat = None
            if return_center_lonlat:
                # Compute chip-center coords in the file's CRS, then reproject to EPSG:4326.
                h, w = src.height, src.width
                cx, cy = src.transform * (w / 2.0, h / 2.0)
                lon, lat = warp_transform(src.crs, "EPSG:4326", [cx], [cy])
                center_lonlat = (float(lon[0]), float(lat[0]))

        if return_center_lonlat:
            return img, center_lonlat
        return img


def load_pixel(path, lon, lat):
    """Read the 6 HLS bands at the single (lon, lat) pixel of a chip.

    Reprojects the station's EPSG:4326 (lon, lat) into the chip CRS, maps it to a
    (row, col) with rasterio's affine index, and reads a 1x1 window -- so only the
    one pixel covering the ISMN location is touched (cheap, no full-chip read).
    Returns a float32 array of shape (6,). NaN gaps are remapped to NO_DATA_FLOAT
    (matching load_raster). Raises if the location falls outside the raster.
    """
    with rasterio.open(path) as src:
        x, y = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = src.index(x[0], y[0])
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise IndexError(
                f"({lon}, {lat}) -> (row={row}, col={col}) outside "
                f"{src.width}x{src.height} raster {os.path.basename(path)}"
            )
        img = src.read(window=Window(col, row, 1, 1))[0:6, 0, 0]  # (6,)
    img = np.where(np.isnan(img), NO_DATA_FLOAT, img)
    return img.astype(np.float32)


def preprocess_image(image, means, stds):
    """Per-band z-score a single (C, H, W) frame; returns a float32 tensor."""
    means1 = means.reshape(-1, 1, 1)  # per-channel mean over H, W
    stds1 = stds.reshape(-1, 1, 1)    # per-channel std over H, W
    normalized = (image - means1) / stds1
    return torch.from_numpy(normalized).to(torch.float32)


def spatial_split(df, test_fraction=0.2):
    """Split master-CSV rows into (train_df, test_df) BY STATION.

    Splitting by station (not by row) keeps the split leakage-safe: a station's
    rows fall entirely on one side, so no station appears in both train and test.
    """
    pairs = (
        df[["network", "station"]]
        .drop_duplicates()
        .sort_values(["network", "station"])
        .itertuples(index=False, name=None)
    )
    pairs = [(str(n), str(s)) for n, s in pairs]

    n_test = max(1, int(round(len(pairs) * test_fraction)))
    test_pairs = set(pairs[:n_test])

    in_test = np.array(
        [(str(n), str(s)) in test_pairs
         for n, s in zip(df["network"], df["station"])]
    )
    test_df = df[in_test].reset_index(drop=True)
    train_df = df[~in_test].reset_index(drop=True)
    return train_df, test_df


class sm_dataset(NonGeoDataset):
    """Single-frame soil-moisture dataset.

    One master-CSV row -> one HLS chip -> one MERRA day -> one SM target.
    Stats (HLS means/stds, MERRA means/stds, SM mean/std) are train-derived and
    passed in; the same stats normalize train and test.
    """

    def __init__(self, rows, chips_root, merra_df,
                 means, stds, merra_means, merra_stds, sm_mean, sm_std,
                 modality="both", hls_mode="chip"):
        # rows: a DataFrame of filtered master-CSV rows (has_HLS==1).
        # merra_df: per-split MERRA frame, MultiIndexed on (network, station),
        #           with a `time` column and all MERRA_COLS (LND+SLV merged).
        self.rows = rows.reset_index(drop=True)
        self.chips_root = chips_root
        self.merra_df = merra_df
        self.means = np.asarray(means)
        self.stds = np.asarray(stds)
        self.merra_means = np.asarray(merra_means)
        self.merra_stds = np.asarray(merra_stds)
        self.sm_mean = np.asarray(sm_mean)
        self.sm_std = np.asarray(sm_std)
        # Random tile selection uses the global numpy RNG (seeded by set_seed).
        assert modality in ("hls", "merra", "both"), f"bad modality {modality}"
        assert hls_mode in ("chip", "pixel"), f"bad hls_mode {hls_mode}"
        self.modality = modality
        self.hls_mode = hls_mode
        self.use_hls = modality in ("hls", "both")
        self.use_merra = modality in ("merra", "both")
        # When HLS is on, hls_mode picks the spatial extent: the full 50x50 chip
        # (chip -> Prithvi) or the single ISMN-location pixel (pixel -> MLP).
        self.use_chip = self.use_hls and hls_mode == "chip"
        self.use_pixel = self.use_hls and hls_mode == "pixel"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        # TODO: need to pipeline the multi-temporal codepath here
        
        row = self.rows.iloc[idx]
        network, station = str(row["network"]), str(row["station"])
        day = date.fromisoformat(str(row["date"]))

        # Resolve the chip path in any modality: HLS reads its pixels, and MERRA
        # is keyed off the chip's overpass time, so the path is needed even in
        # merra-only (the has_HLS==1 filter guarantees a chip exists).
        # TODO: use the actual soil-moisture measurement time when available;
        #       currently unobtainable, so MERRA is keyed off the HLS overpass
        #       time (from the chip filename) to stay comparable with `both`.
        chip_path = resolve_chip(network, station, day, self.chips_root)

        # --- HLS chip: read + per-band z-score (only when hls_mode=chip) ---
        if self.use_chip:
            image, center_lonlat = load_raster(
                chip_path, crop=(50, 50), return_center_lonlat=True
            )
            image = preprocess_image(image, self.means, self.stds)  # (C, H, W)
            image = image.to(torch.float)
        else:
            # terratorch's PixelwiseRegressionTask hard-requires batch["image"];
            # emit a tiny dummy placeholder. Both the merra-only model and the
            # pixel-HLS model ignore it (pixel HLS rides in via "hls_pixel").
            image = torch.zeros(1, dtype=torch.float)

        # --- SM target: z-score (always) ---
        sm = (float(row["soil_moisture"]) - self.sm_mean) / self.sm_std
        sm = torch.from_numpy(np.asarray(sm, dtype=np.float32).reshape(1))

        sample = {
            "image": image,
            "mask": sm.to(torch.float),
            "filename": chip_path,
        }

        # --- HLS pixel: 6 bands at the ISMN location, per-band z-score (only
        if self.use_pixel:
            pix = load_pixel(chip_path, float(row["longitude"]), float(row["latitude"]))
            pix_norm = (pix - self.means) / self.stds
            sample["hls_pixel"] = torch.from_numpy(pix_norm.astype(np.float32)).to(torch.float)

        # --- MERRA: vector nearest the HLS overpass time, per-variable z-score ---
        if self.use_merra:
            when = overpass_datetime(chip_path)
            merra = merra_vector(self.merra_df, network, station, when)  # (n_vars,)
            merra_norm = (merra - self.merra_means) / self.merra_stds
            merra_norm = torch.from_numpy(merra_norm.astype(np.float32))
            sample["pt1d"] = merra_norm.to(torch.float)

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

    def __init__(self, dataset_train=None, dataset_test=None,
                 train_batch_size=None, test_batch_size=None, config=None):
        super().__init__()
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
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