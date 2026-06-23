#!/usr/bin/env python3
"""Build a TX + 2015-2025 subset of the daily surface SM table, flagged with has_HLS.

Output keeps ONLY rows that are both (a) a Texas station listed in
hls_stations_tx.csv and (b) dated within [YEAR_MIN, YEAR_MAX]. Each kept row gets
a has_HLS column: 1 if at least one HLS .tif exists for that station on that
calendar date, else 0. MERRA2 is intentionally ignored (it exists every day).

Performance: HLS coverage is reduced once to a small in-memory set of
(network, station, date) keys parsed from filenames. The 3M-row SM CSV is
streamed in chunks, filtered to the year range + TX stations first (which drops
the vast majority of rows), then flagged with a vectorized O(1) set lookup.
"""

import os
import re
import glob
from datetime import date, timedelta

import pandas as pd

DATA = "/net/arch-lauprs2.arch.tamu.edu/tank/mercury/eku/data/soil-moisture"
# SM tables (raw + the master output) live under ISMN/; HLS chips + station list
# live under HLS/.
SM_CSV = f"{DATA}/ISMN/raw_sm_daily_surface.csv"
HLS_ROOT = f"{DATA}/HLS/tx_ismn_2015_2025"
TX_STATIONS_CSV = f"{DATA}/HLS/hls_stations_tx.csv"
OUT_CSV = f"{DATA}/ISMN/master_sm_hls_tx_2015_2025.csv"

YEAR_MIN = 2015
YEAR_MAX = 2025
CHUNKSIZE = 500_000

# HLS filename token, e.g. HLS.S30.T14RQU.2020032T170659.v2.0.merged.subset.tif
#                                          ^^^^ ^^^  = YYYY DDD (day-of-year)
DATE_TOKEN = re.compile(r"\.(\d{4})(\d{3})T\d{6}\.")


def doy_to_date(year: int, doy: int) -> date:
    return date(year, 1, 1) + timedelta(days=doy - 1)


def load_tx_stations(path: str) -> set:
    """Allow-set of (network, station) for Texas stations."""
    df = pd.read_csv(path, dtype={"network": str, "station": str})
    return set(zip(df["network"], df["station"]))


def build_hls_index(root: str) -> set:
    """Set of (network, station, 'YYYY-MM-DD') that have >=1 HLS file.

    Folder name is '<network>_<station>' (e.g. 'SCAN_Bushland#1'); split on the
    FIRST underscore so station names containing '_' stay intact.
    """
    index = set()
    for folder in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(folder):
            continue
        network, _, station = os.path.basename(folder).partition("_")
        if not station:
            continue  # malformed folder name, skip
        for fname in os.listdir(folder):
            m = DATE_TOKEN.search(fname)
            if not m:
                continue
            d = doy_to_date(int(m.group(1)), int(m.group(2))).isoformat()
            index.add((network, station, d))
    return index


def main() -> None:
    tx_stations = load_tx_stations(TX_STATIONS_CSV)
    hls_index = build_hls_index(HLS_ROOT)
    print(f"TX stations (allow-list): {len(tx_stations)}")
    print(f"HLS (network, station, date) keys: {len(hls_index)}")

    # Sanity check: which TX stations actually have HLS folders/files?
    hls_stations = {(n, s) for (n, s, _) in hls_index}
    missing = sorted(tx_stations - hls_stations)
    if missing:
        print(f"NOTE: {len(missing)} TX station(s) have NO HLS files "
              f"(they can still appear with has_HLS=0): {missing}")

    first = True
    total_out = 0
    total_hls = 0
    for chunk in pd.read_csv(SM_CSV, chunksize=CHUNKSIZE,
                             dtype={"network": str, "station": str}):
        # Filter to the year range first (cheap, drops most rows). Dates are
        # ISO 'YYYY-MM-DD', so the 4-char year prefix compares lexically as int.
        year = chunk["date"].astype(str).str.slice(0, 4)
        chunk = chunk[(year >= str(YEAR_MIN)) & (year <= str(YEAR_MAX))]
        if chunk.empty:
            continue

        # Filter to TX stations.
        keep = [k in tx_stations for k in zip(chunk["network"], chunk["station"])]
        chunk = chunk[pd.Series(keep, index=chunk.index)]
        if chunk.empty:
            continue

        # Vectorized has_HLS lookup on (network, station, date).
        keys = zip(chunk["network"], chunk["station"], chunk["date"].astype(str))
        flag = pd.Series([k in hls_index for k in keys], index=chunk.index).astype("int8")
        chunk = chunk.assign(has_HLS=flag)

        chunk.to_csv(OUT_CSV, mode="w" if first else "a", header=first, index=False)
        first = False
        total_out += len(chunk)
        total_hls += int(flag.sum())

    if first:
        print("WARNING: no rows matched TX stations + year range. No output written.")
        return

    print(f"Wrote {total_out} rows -> {OUT_CSV}")
    print(f"  has_HLS=1: {total_hls}  ({100*total_hls/total_out:.1f}%)")
    print(f"  has_HLS=0: {total_out - total_hls}")


if __name__ == "__main__":
    main()
