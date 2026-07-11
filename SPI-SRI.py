import os

import numpy as np
from climate_indices import compute, indices
from osgeo import gdal
from tqdm import tqdm

gdal.UseExceptions()

input_file = r""
output_dir = r""
index_name = "SPI"
data_start_year = 1998
data_end_year = 2023
output_start_year = 2000
output_end_year = 2022
scales = (1, 3, 6, 12)
output_nodata = -99.0


def read_stack(file_path):
    dataset = gdal.Open(file_path)
    data = dataset.ReadAsArray().astype(np.float64)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    nodata = dataset.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, dataset.GetGeoTransform(), dataset.GetProjection()


def calculate_standardized_index(data, scale):
    result = np.full(data.shape, np.nan, dtype=np.float32)
    for row in tqdm(range(data.shape[1]), desc=f"{index_name}-{scale}"):
        for col in range(data.shape[2]):
            series = data[:, row, col]
            if not np.isfinite(series).any():
                continue
            series = np.where(np.isfinite(series) & (series < 0), 0, series)
            try:
                result[:, row, col] = indices.spi(
                    values=series,
                    scale=scale,
                    distribution=indices.Distribution.gamma,
                    periodicity=compute.Periodicity.monthly,
                    data_start_year=data_start_year,
                    calibration_year_initial=data_start_year,
                    calibration_year_final=data_end_year,
                )
            except Exception:
                continue
    return result


def write_layer(file_path, data, geotransform, projection):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        file_path,
        data.shape[1],
        data.shape[0],
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES"],
    )
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(output_nodata)
    band.WriteArray(np.where(np.isfinite(data), data, output_nodata).astype(np.float32))
    dataset.FlushCache()
    dataset = None


def write_monthly_layers(data, scale, geotransform, projection):
    scale_dir = os.path.join(output_dir, f"{index_name}{scale:02d}")
    os.makedirs(scale_dir, exist_ok=True)
    for year in range(output_start_year, output_end_year + 1):
        for month in range(1, 13):
            position = (year - data_start_year) * 12 + month - 1
            file_path = os.path.join(
                scale_dir,
                f"{index_name}{scale:02d}_{year}_{month:02d}.tif",
            )
            write_layer(file_path, data[position], geotransform, projection)


def main():
    if index_name not in {"SPI", "SRI"}:
        raise ValueError("index_name must be 'SPI' or 'SRI'.")
    if not input_file or not output_dir:
        raise ValueError("Set input_file and output_dir before running the script.")
    data, geotransform, projection = read_stack(input_file)
    expected_months = (data_end_year - data_start_year + 1) * 12
    if data.shape[0] != expected_months:
        raise ValueError(f"Expected {expected_months} monthly bands, found {data.shape[0]}.")
    for scale in scales:
        standardized = calculate_standardized_index(data, scale)
        write_monthly_layers(standardized, scale, geotransform, projection)


if __name__ == "__main__":
    main()
