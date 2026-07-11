from pathlib import Path
import re

import numpy as np
import rasterio


input_folder = r""
output_folder = r""
start_year = 2000
end_year = 2022
t_days = 30.0
delta_t_days = 30.0
epsilon = 1e-10
nodata = -9999.0


def monthly_files(folder, first_year, last_year):
    pattern = re.compile(r"SM_(\d{4})_(\d{1,2})\.tif$")
    files = []
    for path in folder.glob("SM_*.tif"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        year, month = map(int, match.groups())
        if first_year <= year <= last_year and 1 <= month <= 12:
            files.append((year, month, path))
    files.sort()
    expected = (last_year - first_year + 1) * 12
    if len(files) != expected:
        raise ValueError(f"Expected {expected} monthly soil-moisture files, found {len(files)}.")
    return files


def read_raster(path):
    with rasterio.open(path) as source:
        data = source.read(1, masked=True).filled(np.nan).astype(np.float64)
        profile = source.profile.copy()
    return data, profile


def write_raster(path, data, profile):
    output_profile = profile.copy()
    output_profile.update(dtype="float32", count=1, nodata=nodata, compress="lzw")
    output = np.where(np.isfinite(data), data, nodata).astype(np.float32)
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(output, 1)


def update_extrema(current_min, current_max, values):
    valid = np.isfinite(values)
    current_min = np.where(valid & (np.isnan(current_min) | (values < current_min)), values, current_min)
    current_max = np.where(valid & (np.isnan(current_max) | (values > current_max)), values, current_max)
    return current_min, current_max


def recursive_gain(length, characteristic_time, interval):
    gains = np.ones(length, dtype=np.float64)
    decay = np.exp(-interval / characteristic_time)
    for index in range(1, length):
        gains[index] = gains[index - 1] / (gains[index - 1] + decay)
    return gains


def main():
    if not input_folder or not output_folder:
        raise ValueError("Set input_folder and output_folder before running the script.")

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    records = monthly_files(input_path, start_year, end_year)
    output_path.mkdir(parents=True, exist_ok=True)

    first_sm, profile = read_raster(records[0][2])
    sm_min = np.full(first_sm.shape, np.nan, dtype=np.float64)
    sm_max = np.full(first_sm.shape, np.nan, dtype=np.float64)

    for _, _, path in records:
        sm, _ = read_raster(path)
        if sm.shape != first_sm.shape:
            raise ValueError(f"Grid shape differs from the first input: {path.name}")
        sm_min, sm_max = update_extrema(sm_min, sm_max, sm)

    write_raster(output_path / "SM_min.tif", sm_min, profile)
    write_raster(output_path / "SM_max.tif", sm_max, profile)

    gains = recursive_gain(len(records), t_days, delta_t_days)
    swi_previous = None

    for index, (year, month, path) in enumerate(records):
        sm, _ = read_raster(path)
        normalized = np.full(sm.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(sm) & np.isfinite(sm_min) & np.isfinite(sm_max)
        normalized[valid] = (sm[valid] - sm_min[valid]) / (sm_max[valid] - sm_min[valid] + epsilon)

        if swi_previous is None:
            swi = normalized
        else:
            swi = np.full(sm.shape, np.nan, dtype=np.float64)
            valid = np.isfinite(normalized) & np.isfinite(swi_previous)
            swi[valid] = swi_previous[valid] + gains[index] * (normalized[valid] - swi_previous[valid])

        write_raster(output_path / f"SWI_{year}_{month:02d}.tif", swi, profile)
        swi_previous = swi


if __name__ == "__main__":
    main()
