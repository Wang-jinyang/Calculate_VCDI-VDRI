# VCDI and VDRI Dataset Construction

[![Python Version](https://img.shields.io/badge/python-3.11.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains the source code used to construct the monthly **VCDI** and **VDRI** datasets. These indices are designed to evaluate multidimensional drought conditions and ecosystem resilience, specifically focusing on dynamics from 2000 to 2022.

The source code used to construct the monthly VCDI and VDRI datasets is hosted on this GitHub repository under the MIT License. Future updates, bug fixes, and maintenance will be managed through this repository.

## Dependencies and Environment
All processing scripts were developed using **Python version 3.11.13**. 

Due to the distinct requirements of spatial data science packages, the processing workflow utilizes two types of environments:
1. **Open-Source Scientific Computing:** Requires `numpy`, `scipy`, `rasterio`, `pyvinecopulib`, `joblib`, and `tqdm`.
2. **ArcGIS Spatial Analysis:** Requires the `arcpy` module (typically bundled with ArcGIS Pro configured with Python 3).

### Recommended Installation (Conda)
```bash
# Create a dedicated environment for VCDI calculation
conda create -n vdri_env python=3.11.13
conda activate vdri_env

# Install required spatial and statistical packages
pip install numpy scipy rasterio pyvinecopulib joblib tqdm