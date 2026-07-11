"""
Construct xSPI from multi-scale SPI layers using monthly PCA.

This script implements the procedure described in the manuscript:
1. For each monthly time step, stack SPI-1, SPI-3, SPI-6, and SPI-12.
2. Use complete-case valid pixels as samples and the four time scales as variables.
3. Apply PCA with mean centering and without additional variance scaling.
4. Retain PC1 as xSPI.
5. Correct the arbitrary PC1 sign so that higher xSPI indicates wetter conditions.

The same workflow can be reused for xSRI by replacing the four input folders and
the output prefix.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import rasterio
from rasterio.profiles import Profile
from sklearn.decomposition import PCA
from tqdm import tqdm


DEFAULT_SPI01_DIR = r""
DEFAULT_SPI03_DIR = r""
DEFAULT_SPI06_DIR = r""
DEFAULT_SPI12_DIR = r""
DEFAULT_OUTPUT_DIR = r""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct monthly xSPI rasters from SPI-1, SPI-3, SPI-6, and SPI-12 layers."
    )
    parser.add_argument("--spi01-dir", default=DEFAULT_SPI01_DIR, help="Folder containing SPI-1 GeoTIFF files.")
    parser.add_argument("--spi03-dir", default=DEFAULT_SPI03_DIR, help="Folder containing SPI-3 GeoTIFF files.")
    parser.add_argument("--spi06-dir", default=DEFAULT_SPI06_DIR, help="Folder containing SPI-6 GeoTIFF files.")
    parser.add_argument("--spi12-dir", default=DEFAULT_SPI12_DIR, help="Folder containing SPI-12 GeoTIFF files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder for output xSPI GeoTIFF files.")
    parser.add_argument("--output-prefix", default="xSPI", help="Prefix used in output filenames.")
    parser.add_argument("--start-year", type=int, default=2000, help="First year retained in the output.")
    parser.add_argument("--end-year", type=int, default=2022, help="Last year retained in the output.")
    parser.add_argument("--nodata", type=float, default=-9999.0, help="Output NoData value.")
    return parser.parse_args()


def extract_time_key(filepath: str) -> Tuple[int, int] | None:
    """Extract a (year, month) key from a filename.

    The function searches for a 4-digit year followed later by a 1- or 2-digit month.
    This supports names such as SPI01_2000_01.tif and SPEI_2000_1.tif.
    """
    stem = Path(filepath).stem
    matches = re.findall(r"(?<!\d)(19\d{2}|20\d{2})\D+(\d{1,2})(?!\d)", stem)
    if not matches:
        return None

    year_text, month_text = matches[-1]
    year = int(year_text)
    month = int(month_text)
    if 1 <= month <= 12:
        return year, month
    return None


def build_time_index(folder: str) -> Dict[Tuple[int, int], str]:
    files = sorted(glob(os.path.join(folder, "*.tif")))
    if not files:
        raise FileNotFoundError(f"No GeoTIFF files were found in: {folder}")

    indexed_files: Dict[Tuple[int, int], str] = {}
    for filepath in files:
        key = extract_time_key(filepath)
        if key is None:
            continue
        indexed_files[key] = filepath

    if not indexed_files:
        raise ValueError(f"No valid year-month keys could be parsed from files in: {folder}")
    return indexed_files


def common_time_keys(
    indices: Iterable[Dict[Tuple[int, int], str]],
    start_year: int,
    end_year: int,
) -> List[Tuple[int, int]]:
    iterator = iter(indices)
    common = set(next(iterator))
    for index in iterator:
        common &= set(index)

    retained = [key for key in common if start_year <= key[0] <= end_year]
    if not retained:
        raise ValueError("No common monthly time steps were found within the requested output period.")
    return sorted(retained)


def read_single_band(path: str) -> Tuple[np.ndarray, Profile, float | None]:
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
    return array, profile, nodata


def validate_profiles(reference: Profile, current: Profile, path: str) -> None:
    keys = ("height", "width", "crs", "transform")
    for key in keys:
        if reference.get(key) != current.get(key):
            raise ValueError(f"Spatial metadata mismatch for {path}: {key} differs from the reference raster.")


def valid_pixel_mask(arrays: List[np.ndarray], nodata_values: List[float | None]) -> np.ndarray:
    flat_arrays = [array.ravel() for array in arrays]
    mask = np.ones(flat_arrays[0].shape, dtype=bool)

    for flat_array, nodata in zip(flat_arrays, nodata_values):
        mask &= np.isfinite(flat_array)
        if nodata is not None and np.isfinite(nodata):
            mask &= flat_array != nodata

    return mask


def orient_pc1_to_wet_direction(pc1_scores: np.ndarray, valid_data: np.ndarray) -> np.ndarray:
    """Orient PC1 so that higher scores indicate wetter conditions.

    SPI values are positive under wetter conditions and negative under drier
    conditions. The sign of PCA scores is arbitrary, so PC1 is compared with the
    pixel-wise mean of the four SPI inputs. A negative Pearson correlation means
    that PC1 is pointing in the opposite direction and should be reversed.
    """
    input_mean = valid_data.mean(axis=1)

    if np.nanstd(pc1_scores) == 0 or np.nanstd(input_mean) == 0:
        return pc1_scores

    correlation = np.corrcoef(pc1_scores, input_mean)[0, 1]
    if np.isfinite(correlation) and correlation < 0:
        return -pc1_scores
    return pc1_scores


def construct_xspi_for_month(
    input_paths: List[str],
    reference_profile: Profile,
    output_nodata: float,
) -> np.ndarray:
    arrays: List[np.ndarray] = []
    nodata_values: List[float | None] = []

    for path in input_paths:
        array, profile, nodata = read_single_band(path)
        validate_profiles(reference_profile, profile, path)
        arrays.append(array)
        nodata_values.append(nodata)

    flat_data = np.stack([array.ravel() for array in arrays], axis=1)
    mask = valid_pixel_mask(arrays, nodata_values)
    valid_data = flat_data[mask]

    output_flat = np.full(flat_data.shape[0], output_nodata, dtype=np.float32)
    if valid_data.shape[0] > 1:
        pca = PCA(n_components=1)
        pc1_scores = pca.fit_transform(valid_data).ravel()
        pc1_scores = orient_pc1_to_wet_direction(pc1_scores, valid_data)
        output_flat[mask] = pc1_scores.astype(np.float32)

    height = reference_profile["height"]
    width = reference_profile["width"]
    return output_flat.reshape(height, width)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_indices = [
        build_time_index(args.spi01_dir),
        build_time_index(args.spi03_dir),
        build_time_index(args.spi06_dir),
        build_time_index(args.spi12_dir),
    ]
    time_keys = common_time_keys(scale_indices, args.start_year, args.end_year)

    reference_path = scale_indices[0][time_keys[0]]
    _, reference_profile, _ = read_single_band(reference_path)
    output_profile = reference_profile.copy()
    output_profile.update(dtype=rasterio.float32, count=1, nodata=args.nodata, compress="lzw")

    start_time = time.time()
    for year, month in tqdm(time_keys, desc=f"Constructing {args.output_prefix}"):
        input_paths = [index[(year, month)] for index in scale_indices]
        output_array = construct_xspi_for_month(input_paths, reference_profile, args.nodata)

        output_name = f"{args.output_prefix}_{year}_{month:02d}.tif"
        output_path = output_dir / output_name
        with rasterio.open(output_path, "w", **output_profile) as dst:
            dst.write(output_array, 1)

    elapsed_min = (time.time() - start_time) / 60
    print(f"Finished {len(time_keys)} monthly {args.output_prefix} rasters in {elapsed_min:.2f} min.")


if __name__ == "__main__":
    main()
