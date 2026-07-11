# VCDI and VDRI workflow

This repository provides executable Python scripts for constructing the monthly Vertical Composite Drought Index (VCDI) and Vegetation Drought Resistance Index (VDRI) datasets for China from 2000 to 2022. Paths are intentionally left blank in the scripts and must be configured before execution.

## Requirements

Use Python 3.11 or later with the following packages:

```bash
pip install numpy rasterio scipy scikit-learn tqdm climate_indices pyvinecopulib
```

`SPI-SRI.py` also requires GDAL with Python bindings (`osgeo.gdal`). All raster inputs must use a common coordinate reference system and be spatially aligned before they are passed to a given script.

## Workflow

1. Calculate SPI and SRI at 1-, 3-, 6-, and 12-month accumulation windows with `SPI-SRI.py`.
2. Use `xSPI_PCA.py` to integrate the four SPI layers into xSPI. Run the same script with SRI folders and `--output-prefix xSRI` to create xSRI.
3. Calculate SWI from monthly soil-moisture layers with `SWI.py`.
4. Derive groundwater storage deviation (GSD) and the Groundwater Drought Index (GGDI) from monthly GWSA layers with `GGDI.py`.
5. Harmonize xSPI, xSRI, SWI, and GGDI to the 1 km xSPI grid, then calculate pixel-wise monthly VCDI with `R-vine_VCDI.py`.
6. Normalize VCDI, LST, and kNDVI using fixed minimum and maximum values from the full study period, resample VCDI and LST to the 250 m kNDVI grid, and calculate VDRI with `VDRI.py`.
7. Use `MTRT.py` to identify VCDI-based drought events and calculate drought duration, severity, intensity, and event count.

## Input preparation

All scripts use monthly GeoTIFF files. The study period is 2000-2022 unless otherwise specified.

| Variable                    | Required filename pattern                                                             | Notes                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Soil moisture               | `SM_YYYY_MM.tif`                                                                    | Used by`SWI.py`.                                                                       |
| Groundwater storage anomaly | `GWSA_YYYY_MM.tif`                                                                  | Used by`GGDI.py`.                                                                  |
| xSPI, xSRI, SWI, GGDI       | `xSPI_YYYY_MM.tif`, `xSRI_YYYY_MM.tif`, `SWI_YYYY_MM.tif`, `GGDI_YYYY_MM.tif` | Used by`R-vine_VCDI.py`.                                                               |
| VCDI                        | `VCDI_YYYY_MM.tif`                                                                  | Produced by`R-vine_VCDI.py`; used for event analysis after any required normalization. |

The monthly precipitation and runoff time series used for SPI/SRI must cover 1998-2023. This period is used to fit the Gamma distribution and calculate standardized indices; only outputs from 2000-2022 are retained.

Before final VDRI calculation, use the full 2000-2022 record to normalize VCDI, LST, and kNDVI. The three normalized layers must be aligned to the native 250 m kNDVI grid. The supplied `VDRI.py` expects matching names whose prefixes are `NorkNDVI`, `NorVCDI`, and `NorLST`; these prefixes can be changed at the top of the script to match local filenames.

Monthly NDVI can be converted to kNDVI as:

```text
kNDVI = tanh(NDVI^2)
```

## Scripts

### `SPI-SRI.py`

Calculates pixel-wise SPI or SRI from a multiband monthly GeoTIFF time series. Set `input_file`, `output_dir`, and `index_name` (`"SPI"` or `"SRI"`) at the top of the script. Gamma distributions are fitted independently for every valid pixel. Negative input values are set to zero before fitting. The script writes 1-, 3-, 6-, and 12-month layers in separate output folders.

### `xSPI_PCA.py`

Constructs xSPI or xSRI from four standardized-index time-scale folders. PCA is applied independently for each month, using valid pixels as samples and the four time-scale layers as variables. Inputs are mean-centred without additional variance scaling, and PC1 is retained. The PC1 sign is corrected using its Pearson correlation with the pixel-wise mean of the four inputs, so higher values consistently indicate wetter precipitation or runoff conditions. Pixels lacking any of the four inputs are written as NoData.

Example for xSPI:

```bash
python xSPI_PCA.py \
  --spi01-dir "path/to/SPI01" \
  --spi03-dir "path/to/SPI03" \
  --spi06-dir "path/to/SPI06" \
  --spi12-dir "path/to/SPI12" \
  --output-dir "path/to/xSPI" \
  --output-prefix xSPI
```

Use the same command for xSRI after replacing the four input folders and setting `--output-prefix xSRI`.

### `SWI.py`

Calculates SWI independently for each valid pixel. Monthly soil-moisture values are normalized using the pixel-specific minimum and maximum over 2000-2022 with `epsilon = 1e-10`. The normalized series is then filtered recursively using an exponential filter with `T = 30 days`, `delta_t = 30 days`, and initial condition `SWI(t1) = ms(t1)`. The script also writes `SM_min.tif` and `SM_max.tif`.

### `GGDI.py`

Calculates monthly GSD and GGDI from monthly GWSA GeoTIFF files. For each calendar month, the script subtracts the monthly climatological GWSA mean and standardizes the resulting GSD series. The GWSA input must already be derived as:

```text
GWSA = TWSA - SMSA - SWEA - CWSA - SWSA
```

where TWSA is obtained from GTWS-MLrec and SMSA, SWEA, CWSA, and SWSA are anomalies derived from the NASA GLDAS Noah Land Surface Model output.

### `R-vine_VCDI.py`

Calculates VCDI independently for each valid pixel using the 2000-2022 monthly time series of xSPI, xSRI, SWI, and GGDI. The script converts each pixel time series to rank-based pseudo-observations and fits an R-vine Copula using:

| Setting               | Value                                 |
| --------------------- | ------------------------------------- |
| Pair-copula families  | Gaussian, Clayton, Gumbel, Frank      |
| Structure criterion   | Absolute Kendall's tau                |
| Pair-family selection | BIC                                   |
| Parameter estimation  | Inversion of Kendall's tau (`itau`) |
| Vine truncation level | 2                                     |
| Dependence threshold  | 0.01                                  |

For each valid month, the fitted joint cumulative probability is transformed to the standard normal distribution. Lower VCDI values indicate drier integrated hydroclimatic conditions. The script processes the raster in 32 x 32 pixel blocks; change `block_size` if needed for available memory.

### `VDRI.py`

Calculates VDRI from normalized and aligned 250 m layers:

```text
VDRI = sqrt(nVCDI^2 + (1 - nLST)^2 + nkNDVI^2)
```

The reference state is low VCDI, high LST, and low kNDVI. Higher VDRI values therefore indicate more favourable joint hydroclimatic, thermal, and vegetation-greenness conditions. VDRI is a resistance-related indicator and is not a direct physiological measurement of plant drought resistance.

### `MTRT.py`

Identifies pixel-wise VCDI drought events using Multi-Threshold Run Theory and writes drought duration, severity, intensity, and event count. The implemented settings are:

| Parameter                | Value                                                       |
| ------------------------ | ----------------------------------------------------------- |
| Drought-run threshold    | `D1 = -1`                                                 |
| Severe-drought threshold | `D2 = -2`                                                 |
| Recovery threshold       | `D0 = 0`                                                  |
| One-month event rule     | Exclude unless`Dt <= D2`                                  |
| Adjacent-event rule      | Merge when one intervening month satisfies`D1 <= Dt < D0` |

## Flux-site metadata

`Flux_site_information.csv` contains the site identifier, name, coordinates, vegetation type, data period, and source information for the 22 flux-tower sites used in validation.

## Reproducibility notes

The scripts implement the dataset-generation algorithms and retain only the parameters required to reproduce the published workflow. They do not include proprietary local paths or the original input data. Input data must be obtained from their cited providers and used according to their respective licences and terms of use.
