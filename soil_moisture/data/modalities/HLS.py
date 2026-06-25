# HLS filename date token, e.g. HLS.S30.T14RQU.2020032T170659.v2.0.merged.subset.tif
#                                              ^^^^ ^^^  = YYYY DDD (day-of-year)
from datetime import date, datetime
import re

import numpy as np
import rasterio
import torch
from rasterio.warp import transform as warp_transform

from soil_moisture.data.data_utils import resolve_chip


_HLS_DATE_TOKEN = re.compile(r"\.(\d{4})(\d{3})T\d{6}\.")
NO_DATA_FLOAT = 0.0001

class HLSModality():
    def __init__(self, df, chips_root, stats=None):
        self.df = df
        self.chips_root = chips_root
        
        if stats is None:
            self.mean, self.std = self.compute_stats()
        else:
            self.mean, self.std = stats

    def preprocess_image(self, image, means, stds):
        """Per-band z-score a single (C, H, W) frame; returns a float32 tensor."""
        means1 = means.reshape(-1, 1, 1)  # per-channel mean over H, W
        stds1 = stds.reshape(-1, 1, 1)    # per-channel std over H, W
        normalized = (image - means1) / stds1
        return torch.from_numpy(normalized).to(torch.float32)

    def load_raster(self, path, crop=None, return_center_lonlat=False):
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
    
    def compute_stats(self,):
        """Compute train-split-only normalization stats for the active modality."""
        
        # Resolve one chip per train row; its overpass time drives the MERRA lookup.
        n = np.zeros(6)   # num valid pixels per band
        s = np.zeros(6)   # sum of values per band
        ss = np.zeros(6)  # sum of squares per band
        
        chip_paths = []
        for _, row in self.df.iterrows():
            network, station = str(row["network"]), str(row["station"])
            day = datetime.fromisoformat(str(row["date"])).date()
            chip = resolve_chip(self.chips_root, network, station, day)
            chip_paths.append(chip)

        for path in chip_paths:
            with rasterio.open(path) as src:
                img = src.read()
            img = np.where(np.isnan(img), NO_DATA_FLOAT, img)
            img = img[0:6, -50:, -50:]
            n += img.shape[1] * img.shape[2]
            s += img.sum(axis=(1,2))
            ss += (img**2).sum(axis=(1,2))

        mean = s / n
        std = np.sqrt(ss/n - mean**2)  # ddof = 0

        return mean, std

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        network, station = str(row["network"]), str(row["station"])
        day = date.fromisoformat(str(row["date"]))

        # Resolve the chip path in any modality: HLS reads its pixels, and MERRA
        # is keyed off the chip's overpass time, so the path is needed even in
        # merra-only (the has_HLS==1 filter guarantees a chip exists).
        # TODO: use the actual soil-moisture measurement time when available;
        #       currently unobtainable, so MERRA is keyed off the HLS overpass
        #       time (from the chip filename) to stay comparable with `both`.
        chip_path = resolve_chip(self.chips_root, network, station, day)

        # --- HLS: read + per-band z-score ---
        image, center_lonlat = self.load_raster(
            chip_path, crop=(50, 50), return_center_lonlat=True
        )
        image = self.preprocess_image(image, self.mean, self.std)  # (C, H, W)
        image = image.to(torch.float)

        return {"image": image}