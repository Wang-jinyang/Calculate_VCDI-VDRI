import os
from contextlib import ExitStack

import numpy as np
import pyvinecopulib as pv
import rasterio
from rasterio.windows import Window
from scipy.stats import norm, rankdata
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

path_xSPI = r""
path_xSRI = r""
path_SWI = r""
path_GGDI = r""
output_dir = r""

input_paths = [path_xSPI, path_xSRI, path_SWI, path_GGDI]
variable_names = ["xSPI", "xSRI", "SWI", "GGDI"]
years = range(2000, 2023)
months = range(1, 13)
block_size = 32
probability_epsilon = 1e-6
output_nodata = -9999.0


def validate_configuration():
    if any(not path for path in input_paths):
        raise ValueError("Set the four input directories before running the script.")
    if not output_dir:
        raise ValueError("Set the output directory before running the script.")


def build_time_index():
    return [(year, month) for year in years for month in months]


def build_input_files(time_index):
    files = []
    missing = []
    for year, month in time_index:
        monthly_files = []
        for path, name in zip(input_paths, variable_names):
            file_path = os.path.join(path, f"{name}_{year}_{month:02d}.tif")
            monthly_files.append(file_path)
            if not os.path.exists(file_path):
                missing.append(file_path)
        files.append(monthly_files)
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))
    return files


def build_output_files(time_index):
    os.makedirs(output_dir, exist_ok=True)
    return [
        os.path.join(output_dir, f"VCDI_{year}_{month:02d}.tif")
        for year, month in time_index
    ]


def iter_windows(width, height):
    for row_off in range(0, height, block_size):
        window_height = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            window_width = min(block_size, width - col_off)
            yield Window(col_off, row_off, window_width, window_height)


def pseudo_observations(values):
    n = values.shape[0]
    probabilities = np.empty(values.shape, dtype=np.float64)
    for column in range(values.shape[1]):
        probabilities[:, column] = (
            rankdata(values[:, column], method="average") - 0.5
        ) / n
    return np.clip(probabilities, probability_epsilon, 1.0 - probability_epsilon)


def fit_pixel_vine(probabilities, controls):
    vine = pv.Vinecop(d=probabilities.shape[1])
    vine.select(probabilities, controls=controls)
    return vine


def calculate_pixel_vcdi(pixel_series, controls):
    valid = np.isfinite(pixel_series).all(axis=1)
    if np.count_nonzero(valid) <= pixel_series.shape[1]:
        return np.full(pixel_series.shape[0], np.nan, dtype=np.float32)
    probabilities = pseudo_observations(pixel_series[valid])
    vine = fit_pixel_vine(probabilities, controls)
    joint_probabilities = np.asarray(vine.cdf(probabilities), dtype=np.float64)
    joint_probabilities = np.clip(
        joint_probabilities,
        probability_epsilon,
        1.0 - probability_epsilon,
    )
    vcdi = np.full(pixel_series.shape[0], np.nan, dtype=np.float32)
    vcdi[valid] = norm.ppf(joint_probabilities).astype(np.float32)
    return vcdi


def create_controls():
    return pv.FitControlsVinecop(
        family_set=[
            pv.BicopFamily.gaussian,
            pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel,
            pv.BicopFamily.frank,
        ],
        parametric_method="itau",
        trunc_lvl=2,
        tree_criterion="tau",
        threshold=0.01,
        selection_criterion="bic",
    )


def check_grid(dataset, template):
    return (
        dataset.width == template.width
        and dataset.height == template.height
        and dataset.crs == template.crs
        and dataset.transform.almost_equals(template.transform)
    )


def main():
    validate_configuration()
    time_index = build_time_index()
    input_files = build_input_files(time_index)
    output_files = build_output_files(time_index)
    controls = create_controls()

    with ExitStack() as stack:
        input_datasets = [
            [stack.enter_context(rasterio.open(file_path)) for file_path in monthly_files]
            for monthly_files in input_files
        ]
        template = input_datasets[0][0]
        for monthly_datasets in input_datasets:
            for dataset in monthly_datasets:
                if not check_grid(dataset, template):
                    raise ValueError(f"Grid mismatch: {dataset.name}")

        profile = template.profile.copy()
        profile.update(
            dtype="float32",
            count=1,
            nodata=output_nodata,
            compress="deflate",
            predictor=3,
            tiled=True,
        )
        output_datasets = [
            stack.enter_context(rasterio.open(file_path, "w", **profile))
            for file_path in output_files
        ]

        windows = list(iter_windows(template.width, template.height))
        for window in tqdm(windows, desc="Pixel-wise R-Vine VCDI"):
            window_height = int(window.height)
            window_width = int(window.width)
            pixel_count = window_height * window_width
            time_count = len(time_index)
            data = np.full(
                (time_count, len(variable_names), pixel_count),
                np.nan,
                dtype=np.float64,
            )

            for time_position, monthly_datasets in enumerate(input_datasets):
                for variable_position, dataset in enumerate(monthly_datasets):
                    layer = dataset.read(1, window=window, masked=True)
                    data[time_position, variable_position] = np.asarray(
                        layer.filled(np.nan), dtype=np.float64
                    ).reshape(-1)

            output = np.full((time_count, pixel_count), np.nan, dtype=np.float32)
            for pixel_position in range(pixel_count):
                pixel_series = data[:, :, pixel_position]
                try:
                    output[:, pixel_position] = calculate_pixel_vcdi(
                        pixel_series, controls
                    )
                except Exception:
                    continue

            for time_position, dataset in enumerate(output_datasets):
                layer = output[time_position].reshape(window_height, window_width)
                dataset.write(
                    np.where(np.isfinite(layer), layer, output_nodata).astype(np.float32),
                    1,
                    window=window,
                )


if __name__ == "__main__":
    main()
