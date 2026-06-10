from lightning import LightningModule
import numpy as np
import pandas as pd
import torch
from torchgeo.datasets import NonGeoDataset
import rasterio

from rasterio.warp import transform as warp_transform
from torch.utils.data import DataLoader


NO_DATA = -0.9999
NO_DATA_FLOAT = 0.0001
MERRA_COLS = [
    "T2MIN",
    "T2MAX",
    "T2MEAN",
    "TSMDEWMEAN",
    "GWETROOT",
    "LHLAND",
    "SHLAND",
    "SWLAND",
    "PARDFLAND",
    "PRECTOTLAND",
]


def load_raster(path,if_img,crop=None,return_center_lonlat=False):

        with rasterio.open(path) as src:
            img = src.read()

            # load  selected 6 bands for Sentinnel 2 (S2)
            if if_img==1:
                bands=[0,1,2,3,4,5]
                img = img[bands,:,:]

            img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)# update our NO_DATA with -0.9999 -- chips are already scaled
            #print("img size",img.shape)

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


def preprocess_image(image,means,stds):
    """
    Returns: C H W
    """
    # normalize image
    means1 = means.reshape(-1,1,1)  # Mean across height and width, for each channel
    stds1 = stds.reshape(-1,1,1)    # Std deviation across height and width, for each channel
    normalized = ((image - means1) / stds1)

    # Return a single frame as (C, H, W). The temporal axis is added by stacking
    # frames in flux_dataset.__getitem__.
    return torch.from_numpy(normalized).to(torch.float32)

def is_contiguous_window(years: pd.Series):
    """Checks if years only have one year or two consecutive years"""
    unique_years = years.unique()
    if len(unique_years) == 1:
        return True
    elif len(unique_years) == 2:
        return unique_years.min() == unique_years.max() - 1
    else:
        return False


def build_windows(df_split, chips_dir, T_HLS, T_MERRA):
    """Build temporal windows per site with independent HLS / MERRA lengths.

    For each target row, emit one sample with prior T_HLS HLS frames and T_MERRA MERRA frames. Windows can cross a year boundary but without any train/test split leakage. Rows without a full window (for the larger of
    the two lengths) are dropped. The GPP target comes from the last (target) frame only.
    """
    chips, merras, targets, dates = [], [], [], []
    df_sorted = df_split.sort_values(["year", "doy"])
    n_window = max(T_HLS, T_MERRA)  # need enough history to fill the longer sequence
    for _, group in df_sorted.groupby(["SITE_ID"]):
        rows = group.reset_index(drop=True)
        for i in range(n_window - 1, len(rows)):
            hls_window = rows.iloc[i - T_HLS + 1 : i + 1]
            merra_window = rows.iloc[i - T_MERRA + 1 : i + 1]
            # Contiguity must hold across the full span actually used, so neither
            # sequence silently jumps a non-consecutive year gap.
            full_window = rows.iloc[i - n_window + 1 : i + 1]
            if not is_contiguous_window(full_window["year"]):
                continue
            target = hls_window.iloc[-1]
            chips.append(
                [chips_dir + "/" + str(c) for c in hls_window["Chip"].tolist()]
            )
            # MERRA for every frame in its own window -> (T_MERRA, n_vars), oldest -> newest.
            merras.append(merra_window[MERRA_COLS].values.astype(float).tolist())
            targets.append(float(target["GPP"]))
            dates.append(
                list(zip(hls_window["year"].astype(int), hls_window["doy"].astype(int)))
            )
    return chips, merras, targets, dates

class flux_dataset(NonGeoDataset):

    def __init__(self,path,means,stds, merras_data, merra_means, merra_stds, gpp_mean, gpp_std, target, dates=None):
        self.data_dir=path
        self.means=means
        self.stds=stds
        self.merras=merras_data
        self.merra_means=merra_means
        self.merra_stds=merra_stds
        self.gpp_means=gpp_mean
        self.gpp_stds=gpp_std
        self.target=target
        # dates: list of (year, doy) pairs aligned with `path`. Required for TL encodings.
        self.dates=dates
        

    def __len__(self):
        return len(self.data_dir)
    
    
    def __getitem__(self,idx):
        # data_dir[idx] and dates[idx] are lists of T_HLS entries, ordered oldest -> newest.
        # The last frame is the target frame (its GPP/MERRA are the regression target).
        image_paths = self.data_dir[idx]
        if_image = 1
        frames = []
        center_lonlat = None
        for frame_path in image_paths:
            image, center_lonlat = load_raster(frame_path, if_image, crop=(50, 50), return_center_lonlat=True)
            frames.append(preprocess_image(image, self.means, self.stds))
        # Each frame is (C, H, W); stack along a new temporal axis -> (C, T, H, W).
        final_image = torch.stack(frames, dim=1)
        # center_lonlat is from the last (target) frame.

        # merras[idx] is (T_MERRA, n_vars), oldest -> newest. Transpose to (n_vars, T_MERRA)
        # so the MERRA branch can run a Conv1d over the temporal axis ([C, T] layout).
        merra_arr = np.array(self.merras[idx], dtype=np.float32)  # (T_MERRA, n_vars)
        merra_vars = torch.from_numpy(merra_arr.T.copy())         # (n_vars, T_MERRA)

        # Per-variable z-score; reshape to (n_vars, 1) broadcasts over the temporal axis.
        mean_merra = self.merra_means.reshape(-1, 1)
        stds_merra = self.merra_stds.reshape(-1, 1)
        merra_vars_norm = (merra_vars - mean_merra) / stds_merra  # (n_vars, T_MERRA)

        mean_gpp = self.gpp_means.reshape(-1,1,1)  # Mean across height and width, for each channel
        stds_gpp = self.gpp_stds.reshape(-1,1,1)    # Std deviation across height and width, for each channel
        #print('mean, std gpp', mean_gpp, stds_gpp)
        gpp_vars_norm=(self.target[idx]-mean_gpp)/(stds_gpp)
        gpp_vars_norm=torch.from_numpy(np.array(gpp_vars_norm).reshape(1))
        #print('gpp is', gpp.shape)

        # Build TL coords. temporal_coords: (T, 2) -> (year, doy0) per frame. location_coords: (2,) -> (lat, lon).
        # The DataLoader collates these into (B, T, 2) and (B, 2) which is what PrithviViT expects.
        temporal_coords = torch.tensor(
            [[float(year), float(doy) - 1.0] for year, doy in self.dates[idx]],
            dtype=torch.float32,
        )
        lon, lat = center_lonlat
        location_coords = torch.tensor([float(lat), float(lon)], dtype=torch.float32)

        output = {
            "image": final_image.to(torch.float),
            "pt1d": merra_vars_norm.to(torch.float),
            "mask": gpp_vars_norm.to(torch.float),
            "filename": image_paths[-1],
            "temporal_coords": temporal_coords,
            "location_coords": location_coords,
        }

        return output #final_image, merra_vars_norm, gpp_vars_norm

class flux_dataloader(LightningModule):

    def __init__(self, dataset_train=None, dataset_test=None, train_batch_size=None, test_batch_size=None, config=None):
        super().__init__()
        self.flux_dataset_train = dataset_train
        self.flux_dataset_test = dataset_test
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.config = config 

    def setup(self, stage:str=None):

        pass

    def train_dataloader(self):
        data_loader_flux_train = DataLoader(self.flux_dataset_train, batch_size=self.train_batch_size, shuffle=self.config["training"]["shuffle"])
        return data_loader_flux_train        

    def test_dataloader(self):
        data_loader_flux_test = DataLoader(self.flux_dataset_test, batch_size=self.test_batch_size, shuffle=self.config["testing"]["shuffle"])
        return data_loader_flux_test

    def predict_dataloader(self):
        data_loader_flux_test = DataLoader(self.flux_dataset_test, batch_size=self.test_batch_size, shuffle=self.config["testing"]["shuffle"])
        return data_loader_flux_test