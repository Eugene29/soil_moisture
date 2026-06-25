import glob
import os
import re
from datetime import date, timedelta

import numpy as np


def doy_to_date(year: int, doy: int) -> date:
    """Convert a (year, day-of-year) pair to a calendar date (2020 is a leap year)."""
    return date(year, 1, 1) + timedelta(days=doy - 1)

def spatial_split(df, test_fraction=0.2):
    """Split master-CSV rows into (train_df, test_df) BY STATION."""
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

def resolve_chip(chips_root, network, station, day):
    """Find one HLS merged-chip path for a (station, date) on the fly.

    Globs `<chips_root>/<network>_<station>/*.merged.subset.tif`, keeps files
    whose day-of-year token maps to `day` (a `datetime.date`). Since a
    station/date can be covered by several tiles or sensors (S30/L30) -- picks
    ONE at random via `rng`. Random selection happens per access, so over many
    epochs the model sees the different valid tiles (a mild augmentation).

    Raises if no merged chip matches (callers should only pass has_HLS==1 rows).
    """
    # TODO: move it inside HLS class once MERRA2 dependency is gone. 
    _HLS_DATE_TOKEN = re.compile(r"\.(\d{4})(\d{3})T\d{6}\.")

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