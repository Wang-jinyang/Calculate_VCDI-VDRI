import arcpy
import os
import numpy as np
from tqdm import tqdm

kndvi_folder = "./data/input/NorkNDVI"
vcdi_folder = "./data/input/NorVCDI"
lst_folder = "./data/input/NorLST"
output_folder = "./data/output/VDRI"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

kndvi_files = sorted([f for f in os.listdir(kndvi_folder) if f.endswith('.tif')])

for file_name in tqdm(kndvi_files):
    try:
        kndvi_path = os.path.join(kndvi_folder, file_name)
        vcdi_path = os.path.join(vcdi_folder, file_name.replace('NorkNDVI', 'NorVCDI'))
        lst_path = os.path.join(lst_folder, file_name.replace('NorkNDVI', 'NorLST'))

        if not (os.path.exists(kndvi_path) and os.path.exists(vcdi_path) and os.path.exists(lst_path)):
            continue

        kndvi_raster = arcpy.Raster(kndvi_path)
        vcdi_raster = arcpy.Raster(vcdi_path)
        lst_raster = arcpy.Raster(lst_path)

        kndvi_arr = arcpy.RasterToNumPyArray(kndvi_path).astype(np.float64)
        vcdi_arr = arcpy.RasterToNumPyArray(vcdi_path).astype(np.float64)
        lst_arr = arcpy.RasterToNumPyArray(lst_path).astype(np.float64)

        vdri_arr = np.sqrt(kndvi_arr**2 + vcdi_arr**2 + (1 - lst_arr)**2)

        ref_raster = arcpy.Raster(kndvi_path)
        lower_left = arcpy.Point(arcpy.Describe(kndvi_path).extent.XMin, arcpy.Describe(kndvi_path).extent.YMin)
        cell_size = arcpy.Describe(kndvi_path).meanCellWidth

        vdri_raster = arcpy.NumPyArrayToRaster(vdri_arr, lower_left_corner=lower_left, x_cell_size=cell_size, y_cell_size=cell_size)

        spatial_ref = arcpy.Describe(kndvi_path).spatialReference
        arcpy.DefineProjection_management(vdri_raster, spatial_ref)

        out_name = file_name.replace('NorkNDVI', 'VDRI')
        vdri_raster.save(os.path.join(output_folder, out_name))

    except Exception:
        continue