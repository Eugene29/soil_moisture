"""
Utility file for computing the mean and std of the HLS and dataframes. 
"""
import os, json, datetime
import numpy as np
import pandas as pd
import rasterio

NO_DATA = -0.9999
NO_DATA_FLOAT = 0.0001
MERRA_FOR_GPP_COLS = ["T2MIN","T2MAX","T2MEAN","TSMDEWMEAN","GWETROOT",
              "LHLAND","SHLAND","SWLAND","PARDFLAND","PRECTOTLAND"]

def hls_stats(chip_paths, chips_dir):
    """Compute the mean and std of all chip_paths"""
    n = np.zeros(6)  # num pixels per band
    s = np.zeros(6)  # sum values per band
    ss = np.zeros(6)
    
    for name in chip_paths:
        with rasterio.open(os.path.join(chips_dir, str(name))) as src:
            img = src.read()
        img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
        img = img[:, -50:, -50:]
        n += img.shape[1] * img.shape[2]
        s += img.sum(axis=(1,2))
        ss += (img**2).sum(axis=(1,2))
    mean = s / n
    std = np.sqrt(ss/n - mean**2)  # ddof = 0
    return mean, std

def col_stats(df, cols, ddof=1):
    """Computes the mean and std of a dataframe using ddof=1."""
    arr = df[cols].to_numpy()
    return arr.mean(axis=0), arr.std(axis=0, ddof=ddof)

if __name__ == "__main__":
    # validation of data stats pipeline
    CSV = "/home/yjean234/Azad/Prithvi-EO-2.0/examples/carbon_flux/data_train_hls_37sites_v0_1.csv"
    CHIPS_DIR = "/home/yjean234/Azad/Prithvi-EO-2.0/examples/carbon_flux/all/images"
    test_year = 2018

    df = pd.read_csv(CSV)
    train_df = df[df["year"] != test_year]
    test_df = df[df["year"] == test_year]
    hm, hs = hls_stats(train_df["Chip"].tolist(), CHIPS_DIR)
    mm, ms = col_stats(train_df, MERRA_FOR_GPP_COLS)
    gm, gs = col_stats(train_df, ["GPP"])

    # print(f"hm, hs: {hm, hs}", flush=True)
    # print(f"mm, ms: {mm, ms}", flush=True)
    # print(f"gm, gs: {gm, gs}", flush=True)

    # Check diff with the given statistics from the reference notebook. 
    print(hm - [0.07286696773903256, 0.10036772476940378, 0.11363777043869523, 0.2720510638470194, 0.2201167122609674, 0.1484162876040495])
    print(hs - [0.13271414936598172, 0.13268933338964875, 0.1384673725283858, 0.12089142598551804, 0.10977084890500641, 0.0978705241034744])
    print(mm - [282.011721, 295.823746,288.291530, 278.243071,0.552373,55.363476, 48.984387, 202.461732, 22.907336,0.000004])
    print(ms - [9.141752,11.374619,10.224494,7.912334,0.178115,50.069111,48.238661,74.897672,9.277971,0.000014])
    print(gm - [3.455948])
    print(gs - [3.754123])