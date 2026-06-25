import os
import re
import glob
from datetime import date, datetime

import numpy as np
import pandas as pd
import torch

from soil_moisture.data.data_utils import doy_to_date, resolve_chip

# --- MERRA variable set for the SM task ------------------------------------
# Chosen as physically relevant to surface soil moisture.
# The hourly MERRA2 data is split across two products: LND (land surface) and SLV 
# (single level).
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


class MERRA2Modality():
    def __init__(self, df, chips_root, merra_root, stats=None):
        self.rows = df.reset_index(drop=True)
        stations = list(self.rows[["network", "station"]].drop_duplicates().itertuples(index=False, name=None))

        self.merra_root = merra_root
        self.chips_root = chips_root
        self.df = self.load_merra_frame(stations)

        if stats is None:
            self.mean, self.std = self.compute_stats()
        else:
            self.mean, self.std = stats

    def load_merra_frame(self, stations,):
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
                    os.path.join(self.merra_root, f"{network}_{st}_{product}_*.csv")))
                if not hits:
                    raise FileNotFoundError(
                        f"No MERRA {product} file for {network}_{st} in {self.merra_root}")
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

    def compute_stats(self):
        # Resolve one chip per train row; its overpass time drives the MERRA lookup.
        merra_rows = []
        for row in self.rows.itertuples():
            network, station = str(row.network), str(row.station)
            day = datetime.fromisoformat(str(row.date)).date()
            chip = resolve_chip(self.chips_root, network, station, day)
            merra_rows.append(
                merra_vector(self.df, network, station, overpass_datetime(chip))
            )

        merra_rows = np.stack(merra_rows)
        means = merra_rows.mean(axis=0)
        stds = merra_rows.std(axis=0, ddof=1)

        return means, stds

    def __getitem__(self, idx):
        # TODO: need to pipeline the multi-temporal codepath here
        row = self.rows.iloc[idx]
        network, station = str(row["network"]), str(row["station"])
        day = date.fromisoformat(str(row["date"]))

        # Resolve the chip path: MERRA is keyed off the chip's overpass time.
        # TODO: use the actual soil-moisture measurement time when available;
        #       currently unobtainable, so MERRA is keyed off the HLS overpass
        #       time (from the chip filename) to stay comparable with `both`.
        chip_path = resolve_chip(self.chips_root, network, station, day)

        # --- MERRA: vector nearest the HLS overpass time, per-variable z-score ---
        when = overpass_datetime(chip_path)
        merra = merra_vector(self.df, network, station, when)  # (n_vars,)
        merra_norm = (merra - self.mean) / self.std
        merra_norm = torch.from_numpy(merra_norm.astype(np.float32))

        return {"pt1d": merra_norm.to(torch.float)}


def overpass_datetime(chip_path):
    """Parse the HLS acquisition datetime (UTC) into a datetime.

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