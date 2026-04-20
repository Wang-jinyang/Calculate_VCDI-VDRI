import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

try:
    from private_core import process_pixel_timeseries
except ImportError:
    def process_pixel_timeseries(ts_data):
        if np.isnan(ts_data).any():
            return np.full(ts_data.shape[0], np.nan, dtype=np.float32)

        T = ts_data.shape[0]

        return np.zeros(T, dtype=np.float32)

path_xSPI = "./data/input/xSPI"
path_xSRI = "./data/input/xSRI"
path_SWI = "./data/input/SWI"
path_GGDI = "./data/input/GGDI"

input_paths = [path_xSPI, path_xSRI, path_SWI, path_GGDI]
var_names = ["xSPI", "xSRI", "SWI", "GGDI"]

output_dir = "./data/output/VCDI"
os.makedirs(output_dir, exist_ok=True)

years = range(2000, 2023)
months = range(1, 13)

file_matrix = {name: [] for name in var_names}
time_labels = []

for year in years:
    for month in months:
        month_str = f"{year}_{month:02d}"
        time_labels.append(month_str)
        for p, name in zip(input_paths, var_names):
            file_matrix[name].append(os.path.join(p, f"{name}_{month_str}.tif"))

T_total = len(time_labels)

template_file = file_matrix[var_names[0]][0]
with rasterio.open(template_file) as src:
    meta = src.meta.copy()
    height, width = src.height, src.width
    meta.update(dtype="float32", count=1, nodata=np.nan)

output_files = []
for t_label in time_labels:
    out_path = os.path.join(output_dir, f"VCDI_{t_label}.tif")
    output_files.append(out_path)
    if not os.path.exists(out_path):
        with rasterio.open(out_path, "w", **meta) as dst:
            empty_arr = np.full((height, width), np.nan, dtype=np.float32)
            dst.write(empty_arr, 1)

BLOCK_SIZE = 256

for row0 in tqdm(range(0, height, BLOCK_SIZE)):
    for col0 in range(0, width, BLOCK_SIZE):
        row1 = min(row0 + BLOCK_SIZE, height)
        col1 = min(col0 + BLOCK_SIZE, width)
        window = Window.from_slices((row0, row1), (col0, col1))

        block_h = row1 - row0
        block_w = col1 - col0
        n_pixels = block_h * block_w

        block_data = np.full((len(var_names), T_total, block_h, block_w), np.nan, dtype=np.float32)

        skip_block = False
        for v_idx, name in enumerate(var_names):
            for t_idx, fpath in enumerate(file_matrix[name]):
                if not os.path.exists(fpath):
                    skip_block = True
                    break
                with rasterio.open(fpath) as src:
                    block_data[v_idx, t_idx, :, :] = src.read(1, window=window)
            if skip_block:
                break

        if skip_block:
            continue

        ts_pixels = block_data.transpose(2, 3, 1, 0).reshape(n_pixels, T_total, 4)

        valid_mask = ~np.isnan(ts_pixels).any(axis=(1, 2))
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        results = Parallel(n_jobs=-1)(
            delayed(process_pixel_timeseries)(ts_pixels[idx])
            for idx in valid_indices
        )

        out_block = np.full((n_pixels, T_total), np.nan, dtype=np.float32)
        out_block[valid_indices] = results

        out_block = out_block.reshape(block_h, block_w, T_total).transpose(2, 0, 1)

        for t_idx, out_path in enumerate(output_files):
            with rasterio.open(out_path, "r+") as dst:
                dst.write(out_block[t_idx, :, :], 1, window=window)