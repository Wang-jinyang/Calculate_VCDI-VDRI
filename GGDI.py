import os
import re

import numpy as np
import rasterio
from tqdm import tqdm

input_folder = r""
output_folder = r""
output_nodata = -9999.0
file_pattern = re.compile(r"GWSA_(\d{4})_(\d{2})\.tif$", re.IGNORECASE)


def list_monthly_files():
    monthly = {month: [] for month in range(1, 13)}
    for file_name in os.listdir(input_folder):
        match = file_pattern.match(file_name)
        if match:
            monthly[int(match.group(2))].append(
                (int(match.group(1)), os.path.join(input_folder, file_name))
            )
    for month in monthly:
        monthly[month].sort()
    return monthly


def read_stack(files):
    arrays = []
    profile = None
    for _, file_path in files:
        with rasterio.open(file_path) as source:
            arrays.append(
                source.read(1, masked=True).filled(np.nan).astype(np.float64)
            )
            if profile is None:
                profile = source.profile.copy()
    return np.stack(arrays), profile


def write_layer(file_path, data, profile):
    profile = profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=output_nodata,
        compress="deflate",
        predictor=3,
        tiled=True,
    )
    with rasterio.open(file_path, "w", **profile) as destination:
        destination.write(
            np.where(np.isfinite(data), data, output_nodata).astype(np.float32), 1
        )


def main():
    if not input_folder or not output_folder:
        raise ValueError("Set input_folder and output_folder before running the script.")
    gsd_folder = os.path.join(output_folder, "GSD")
    ggdi_folder = os.path.join(output_folder, "GGDI")
    os.makedirs(gsd_folder, exist_ok=True)
    os.makedirs(ggdi_folder, exist_ok=True)
    monthly_files = list_monthly_files()
    for month in tqdm(range(1, 13), desc="GGDI"):
        files = monthly_files[month]
        if not files:
            continue
        gwsa, profile = read_stack(files)
        climatology = np.nanmean(gwsa, axis=0)
        gsd = gwsa - climatology
        gsd_mean = np.nanmean(gsd, axis=0)
        gsd_std = np.nanstd(gsd, axis=0)
        ggdi = np.divide(
            gsd - gsd_mean,
            gsd_std,
            out=np.full(gsd.shape, np.nan, dtype=np.float64),
            where=np.isfinite(gsd_std) & (gsd_std > 0),
        )
        for position, (year, _) in enumerate(files):
            write_layer(
                os.path.join(gsd_folder, f"GSD_{year}_{month:02d}.tif"),
                gsd[position],
                profile,
            )
            write_layer(
                os.path.join(ggdi_folder, f"GGDI_{year}_{month:02d}.tif"),
                ggdi[position],
                profile,
            )


if __name__ == "__main__":
    main()
