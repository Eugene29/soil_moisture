#!/usr/bin/env python3
"""Build a master table from ISMN_daily_qc, flagged with presence of each modality.

Base rows come from ISMN_daily_qc.parquet. Each row gets
has_HLS / has_SENTINEL / has_SMAPL3 / has_SMAPL4 / has_MERRA columns: 1 if that
source has data for the row's (network, station, date), else 0.
"""

import os
import re
import glob
from datetime import date, timedelta

import pandas as pd

DATA = "/net/arch-lauprs2.arch.tamu.edu/tank/mercury/eku/data/soil-moisture"
# ISMN = f"{DATA}/ISMN/siyus/ISMN_daily_qc.parquet"
ISMN = f"{DATA}/ISMN/azads/raw_sm_daily_surface.csv"
HLS_ROOT = f"{DATA}/HLS/tx_ismn_2015_2025"
SENTINEL = f"{DATA}/SENTINEL-1/sentinel_1_sar.geoparquet"
SMAPL3 = f"{DATA}/SMAP/smapl3.geoparquet"
SMAPL4 = f"{DATA}/SMAP/smapl4.geoparquet"
MERRA_ROOT = f"{DATA}/MERRA2/merra2_tx_2015_2025"
# OUT_CSV = f"{DATA}/ISMN/master.csv"
OUT_CSV = f"/tmp/master.csv"

DATE_TOKEN = re.compile(r"\.(\d{4})(\d{3})T\d{6}\.")


def doy_to_date(year, doy):
    return date(year, 1, 1) + timedelta(days=doy - 1)


def prefix_lookup(pairs, sanitize=None):
    lookup = {}
    for network, station in pairs:
        disk_station = sanitize(station) if sanitize else station
        lookup[f"{network}_{disk_station}"] = (network, station)
    return lookup

def hls_keys(root, pairs):
    lookup = prefix_lookup(pairs)
    keys = set()
    for folder in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(folder):
            continue
        pair = lookup.get(os.path.basename(folder))
        if pair is None:
            continue
        network, station = pair
        for fname in os.listdir(folder):
            m = DATE_TOKEN.search(fname)
            if m:
                d = doy_to_date(int(m.group(1)), int(m.group(2))).isoformat()
                keys.add((network, station, d))
    return keys

def merra_keys(root, pairs):
    lookup = prefix_lookup(pairs, sanitize=lambda s: s.replace("#", "_"))
    keys = set()
    for path in glob.glob(os.path.join(root, "*_M2T1NXLND_*.csv")):
        base = os.path.basename(path)
        prefix = base.split("_M2T1NXLND_")[0]
        pair = lookup.get(prefix)
        if pair is None:
            continue
        network, station = pair
        dates = pd.read_csv(path, usecols=["time"])["time"].astype(str).str[:10]
        keys.update((network, station, d) for d in dates)
    return keys

def parquet_keys(path):
    df = pd.read_parquet(path, columns=["network", "station", "date"])
    df["date"] = df["date"].astype(str)
    return set(zip(df["network"], df["station"], df["date"]))


def flag(df, keys):
    triples = zip(df["network"], df["station"], df["date"].astype(str))
    return pd.Series([t in keys for t in triples], index=df.index).astype("int8")


def main():
    # df = pd.read_parquet(ISMN)
    df = pd.read_csv(ISMN)
    df["network"] = df["network"].astype(str)
    df["station"] = df["station"].astype(str)

    pairs = set(zip(df["network"], df["station"]))

    sources = {
        "has_HLS": hls_keys(HLS_ROOT, pairs),
        "has_SENTINEL": parquet_keys(SENTINEL),
        "has_SMAPL3": parquet_keys(SMAPL3),
        "has_SMAPL4": parquet_keys(SMAPL4),
        "has_MERRA": merra_keys(MERRA_ROOT, pairs),
    }
    for col, keys in sources.items():
        df[col] = flag(df, keys)
        print(f"{col}: {len(keys)} keys, {int(df[col].sum())} matched rows")

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()