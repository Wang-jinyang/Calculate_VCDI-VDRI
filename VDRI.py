import os

import numpy as np
import rasterio
from tqdm import tqdm

kndvi_folder = r""
vcdi_folder = r""
lst_folder = r""
output_folder = r""

kndvi_prefix = "NorkNDVI"
vcdi_prefix = "NorVCDI"
lst_prefix = "NorLST"
output_prefix = "VDRI"
output_nodata = -9999.0


def read_layer(file_path):
    with rasterio.open(file_path) as source:
        data = source.read(1, masked=True).filled(np.nan).astype(np.float64)
        return data, source.profile.copy(), source.crs, source.transform


def calculate_vdri(kndvi, vcdi, lst):
    valid = np.isfinite(kndvi) & np.isfinite(vcdi) & np.isfinite(lst)
    result = np.full(kndvi.shape, np.nan, dtype=np.float32)
    result[valid] = np.sqrt(
        vcdi[valid] ** 2 + (1.0 - lst[valid]) ** 2 + kndvi[valid] ** 2
    )
    return result


def main():
    folders = [kndvi_folder, vcdi_folder, lst_folder, output_folder]
    if any(not folder for folder in folders):
        raise ValueError("Set all input and output directories before running the script.")
    os.makedirs(output_folder, exist_ok=True)
    files = sorted(
        file_name
        for file_name in os.listdir(kndvi_folder)
        if file_name.lower().endswith(".tif")
    )
    for file_name in tqdm(files, desc="VDRI"):
        kndvi_path = os.path.join(kndvi_folder, file_name)
        vcdi_name = file_name.replace(kndvi_prefix, vcdi_prefix, 1)
        lst_name = file_name.replace(kndvi_prefix, lst_prefix, 1)
        vcdi_path = os.path.join(vcdi_folder, vcdi_name)
        lst_path = os.path.join(lst_folder, lst_name)
        if not os.path.exists(vcdi_path) or not os.path.exists(lst_path):
            continue
        kndvi, profile, crs, transform = read_layer(kndvi_path)
        vcdi, _, vcdi_crs, vcdi_transform = read_layer(vcdi_path)
        lst, _, lst_crs, lst_transform = read_layer(lst_path)
        if kndvi.shape != vcdi.shape or kndvi.shape != lst.shape:
            raise ValueError(f"Shape mismatch for {file_name}")
        if crs != vcdi_crs or crs != lst_crs:
            raise ValueError(f"CRS mismatch for {file_name}")
        if not transform.almost_equals(vcdi_transform) or not transform.almost_equals(lst_transform):
            raise ValueError(f"Transform mismatch for {file_name}")
        result = calculate_vdri(kndvi, vcdi, lst)
        profile.update(
            dtype="float32",
            count=1,
            nodata=output_nodata,
            compress="deflate",
            predictor=3,
            tiled=True,
        )
        output_name = file_name.replace(kndvi_prefix, output_prefix, 1)
        output_path = os.path.join(output_folder, output_name)
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(
                np.where(np.isfinite(result), result, output_nodata), 1
            )


if __name__ == "__main__":
    main()
