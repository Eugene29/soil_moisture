import time
import rasterio

# warm up
strt = time.time()
fname = "/home/yjean234/Azad/Prithvi-EO-2.0/examples/carbon_flux/all/images/HLS.L30.T06VWR.2020118T210620.v2.0.US-xDJ_merged.50x50pixels.tiff"
src = rasterio.open(fname)
img = src.read()
dur = time.time() - strt

strt = time.time()
fname = "/net/arch-lauprs2.arch.tamu.edu/tank/mercury/eku/data/HLS/tx_ismn_2020/COSMOS_Bushland/HLS.S30.T13SGV.2020286T173259.v2.0.merged.subset.tif"
src = rasterio.open(fname)
img = src.read()
dur = time.time() - strt
print(f"network storage io (sec): {dur}", flush=True)

strt = time.time()
fname = "/home/yjean234/Azad/Prithvi-EO-2.0/examples/carbon_flux/all/images/HLS.S30.T17SQD.2021348T160651.v2.0.US-xBL_merged.50x50pixels.tiff"
src = rasterio.open(fname)
img = src.read()
dur = time.time() - strt
print(f"home storage io (sec): {dur}", flush=True)