import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
import os

grid_file = "data/Europe_Shape.geojson"
ghspop_file = "data/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.tif"
output_file = "data/Europe_Grid_Population_2025.csv"
if not os.path.exists(grid_file):
    exit()
if not os.path.exists(ghspop_file):
    exit()
grid = gpd.read_file(grid_file)
results = []
with rasterio.open(ghspop_file) as src:
    if grid.crs != src.crs:
        grid_reprojected = grid.to_crs(src.crs)
    else:
        grid_reprojected = grid.copy()
    total = len(grid_reprojected)
    for idx, cell in grid_reprojected.iterrows():
        if (idx + 1) % 500 == 0:
            pass
        try:
            grid_id = cell["grid_id"]
            lon_idx = cell.get("lon_idx", None)
            lat_idx = cell.get("lat_idx", None)
            center_lon = cell.get("center_lon", cell.geometry.centroid.x)
            center_lat = cell.get("center_lat", cell.geometry.centroid.y)
            try:
                out_image, out_transform = mask(
                    src, [cell.geometry], crop=True, nodata=-200
                )
                pixel_values = out_image[0]
                valid_pixels = pixel_values[
                    (pixel_values != -200) & ~np.isnan(pixel_values)
                ]
                if len(valid_pixels) > 0:
                    population = float(np.sum(valid_pixels))
                    population = max(0, population)
                    method = "raster_sum"
                else:
                    population = 0.0
                    method = "no_data"
            except Exception:
                try:
                    centroid = cell.geometry.centroid
                    coords = [(centroid.x, centroid.y)]
                    values = list(src.sample(coords))
                    if len(values) > 0 and values[0][0] != -200:
                        pixel_pop = float(values[0][0])
                        population = max(0, pixel_pop)
                        method = "centroid_sample"
                    else:
                        population = 0.0
                        method = "no_data"
                except:
                    population = 0.0
                    method = "failed"
            results.append(
                {
                    "grid_id": grid_id,
                    "lon_idx": lon_idx,
                    "lat_idx": lat_idx,
                    "center_lon": center_lon,
                    "center_lat": center_lat,
                    "population_2025": population,
                    "method": method,
                }
            )
        except Exception:
            results.append(
                {
                    "grid_id": cell.get("grid_id", idx),
                    "lon_idx": cell.get("lon_idx", None),
                    "lat_idx": cell.get("lat_idx", None),
                    "center_lon": cell.get("center_lon", None),
                    "center_lat": cell.get("center_lat", None),
                    "population_2025": 0.0,
                    "method": "error",
                }
            )
pop_df = pd.DataFrame(results)
pop_df.to_csv(output_file, index=False)
pop_cells = (pop_df["population_2025"] > 0).sum()
empty_cells = (pop_df["population_2025"] == 0).sum()
method_counts = pop_df["method"].value_counts()
for method, count in method_counts.items():
    pass
top10 = pop_df.nlargest(10, "population_2025")[
    ["grid_id", "center_lon", "center_lat", "population_2025"]
]
total_extracted = pop_df["population_2025"].sum()
expected_europe_2025 = 740000000
if 0.8 < total_extracted / expected_europe_2025 < 1.2:
    pass
else:
    pass
