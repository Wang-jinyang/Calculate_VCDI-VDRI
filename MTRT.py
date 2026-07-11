import os
import re

import numpy as np
import rasterio
from tqdm import tqdm

input_folder = r""
output_folder = r""
start_year = 2000
end_year = 2022
drought_threshold = -1.0
severe_threshold = -2.0
recovery_threshold = 0.0
maximum_gap = 1
float_nodata = -9999.0
integer_nodata = -9999
date_pattern = re.compile(r".*?(\d{4})[_-]?(\d{2}).*\.tif$", re.IGNORECASE)


def identify_events(series):
    drought_months = np.flatnonzero(series < drought_threshold)
    if drought_months.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(drought_months) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, drought_months.size - 1]
    events = [
        [int(drought_months[start]), int(drought_months[end])]
        for start, end in zip(starts, ends)
    ]
    events = [
        event
        for event in events
        if event[1] > event[0] or series[event[0]] <= severe_threshold
    ]
    if not events:
        return []
    merged = [events[0]]
    for event in events[1:]:
        previous = merged[-1]
        gap_start = previous[1] + 1
        gap_end = event[0]
        gap = series[gap_start:gap_end]
        can_merge = (
            0 < gap.size <= maximum_gap
            and np.all(gap >= drought_threshold)
            and np.all(gap < recovery_threshold)
        )
        if can_merge:
            previous[1] = event[1]
        else:
            merged.append(event)
    return merged


def list_rasters():
    rasters = []
    for file_name in os.listdir(input_folder):
        match = date_pattern.match(file_name)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            if start_year <= year <= end_year and 1 <= month <= 12:
                rasters.append((year, month, os.path.join(input_folder, file_name)))
    return sorted(rasters)


def read_stack(rasters):
    arrays = []
    profile = None
    for _, _, file_path in tqdm(rasters, desc="Reading monthly rasters"):
        with rasterio.open(file_path) as source:
            arrays.append(
                source.read(1, masked=True).filled(np.nan).astype(np.float32)
            )
            if profile is None:
                profile = source.profile.copy()
    return np.stack(arrays), profile


def calculate_characteristics(data):
    rows, cols = data.shape[1:]
    duration = np.zeros((rows, cols), dtype=np.int32)
    severity = np.zeros((rows, cols), dtype=np.float32)
    intensity = np.zeros((rows, cols), dtype=np.float32)
    count = np.zeros((rows, cols), dtype=np.int32)
    invalid = np.isnan(data).any(axis=0)
    for row in tqdm(range(rows), desc="MTRT"):
        for col in range(cols):
            if invalid[row, col]:
                continue
            series = data[:, row, col]
            events = identify_events(series)
            count[row, col] = len(events)
            for start, end in events:
                duration[row, col] += end - start + 1
                event = series[start:end + 1]
                severity[row, col] += np.abs(event[event < drought_threshold]).sum()
            if duration[row, col] > 0:
                intensity[row, col] = severity[row, col] / duration[row, col]
    duration[invalid] = integer_nodata
    count[invalid] = integer_nodata
    severity[invalid] = float_nodata
    intensity[invalid] = float_nodata
    return duration, severity, intensity, count


def write_layer(file_path, data, profile, dtype, nodata):
    profile = profile.copy()
    profile.update(
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="deflate",
        tiled=True,
    )
    with rasterio.open(file_path, "w", **profile) as destination:
        destination.write(data.astype(dtype), 1)


def main():
    if not input_folder or not output_folder:
        raise ValueError("Set input_folder and output_folder before running the script.")
    os.makedirs(output_folder, exist_ok=True)
    rasters = list_rasters()
    expected = (end_year - start_year + 1) * 12
    if len(rasters) != expected:
        raise ValueError(f"Expected {expected} monthly rasters, found {len(rasters)}.")
    data, profile = read_stack(rasters)
    duration, severity, intensity, count = calculate_characteristics(data)
    write_layer(
        os.path.join(output_folder, "drought_duration.tif"),
        duration,
        profile,
        "int32",
        integer_nodata,
    )
    write_layer(
        os.path.join(output_folder, "drought_severity.tif"),
        severity,
        profile,
        "float32",
        float_nodata,
    )
    write_layer(
        os.path.join(output_folder, "drought_intensity.tif"),
        intensity,
        profile,
        "float32",
        float_nodata,
    )
    write_layer(
        os.path.join(output_folder, "drought_count.tif"),
        count,
        profile,
        "int32",
        integer_nodata,
    )


if __name__ == "__main__":
    main()
