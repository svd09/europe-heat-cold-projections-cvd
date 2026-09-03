import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

base_path = "data/"
grid_file = base_path + "Europe_Shape.geojson"
ssp_folders = {
    "SSP2": "data/ssp_population/ssp2_1km/",
    "SSP5": "data/ssp_population/ssp5_1km/",
}
decadal_years = [2020, 2030, 2040, 2050, 2060, 2070, 2080]
all_years = list(range(2020, 2081))
grid_gdf = gpd.read_file(grid_file)
if grid_gdf.crs != "EPSG:4326":
    grid_gdf = grid_gdf.to_crs("EPSG:4326")
required_cols = ["grid_id", "center_lon", "center_lat"]
missing = [col for col in required_cols if col not in grid_gdf.columns]
if missing:
    exit()


def extract_population_for_year(ssp_folder, ssp_name, year, grid_gdf):
    ssp_num = ssp_name.lower()
    pop_file = Path(ssp_folder) / f"{ssp_num}_total_{year}.nc4"
    if not pop_file.exists():
        return None
    ds = xr.open_dataset(pop_file)
    pop_var = None
    for var in ds.data_vars:
        if var != "crs":
            pop_var = var
            break
    if pop_var is None:
        ds.close()
        return None
    grid_pops = []
    for idx, cell_row in grid_gdf.iterrows():
        geometry = cell_row["geometry"]
        minx, miny, maxx, maxy = geometry.bounds
        try:
            if "lat" in ds.coords and "lon" in ds.coords:
                cell_data = ds[pop_var].sel(
                    lat=slice(miny, maxy), lon=slice(minx, maxx)
                )
            elif "y" in ds.coords and "x" in ds.coords:
                cell_data = ds[pop_var].sel(y=slice(miny, maxy), x=slice(minx, maxx))
            else:
                grid_pops.append(0)
                continue
            total_pop = float(cell_data.sum().values)
            if np.isnan(total_pop) or total_pop < 0:
                total_pop = 0
            grid_pops.append(total_pop)
        except Exception:
            grid_pops.append(0)
        if (idx + 1) % 2000 == 0:
            pass
    ds.close()
    sum(grid_pops)
    return grid_pops


decadal_results = {"grid_id": grid_gdf["grid_id"].tolist()}
for ssp_name, ssp_folder in ssp_folders.items():
    for year in decadal_years:
        pops = extract_population_for_year(ssp_folder, ssp_name, year, grid_gdf)
        if pops is not None:
            col_name = f"{ssp_name.lower()}_{year}"
            decadal_results[col_name] = pops
        else:
            pass
decadal_df = pd.DataFrame(decadal_results)


def interpolate_annual(df, ssp_name, decadal_years, all_years):
    ssp = ssp_name.lower()
    available_years = [y for y in decadal_years if f"{ssp}_{y}" in df.columns]
    if len(available_years) < 2:
        return None
    annual_data = {"grid_id": df["grid_id"].tolist()}
    for year in all_years:
        if f"{ssp}_{year}" in df.columns:
            annual_data[f"{ssp}_{year}"] = df[f"{ssp}_{year}"].tolist()
        else:
            lower_year = max([y for y in available_years if y < year])
            upper_year = min([y for y in available_years if y > year])
            lower_pop = df[f"{ssp}_{lower_year}"].values
            upper_pop = df[f"{ssp}_{upper_year}"].values
            fraction = (year - lower_year) / (upper_year - lower_year)
            interpolated_pop = lower_pop + fraction * (upper_pop - lower_pop)
            annual_data[f"{ssp}_{year}"] = interpolated_pop.tolist()
        if year % 10 == 0:
            sum(annual_data[f"{ssp}_{year}"])
    return pd.DataFrame(annual_data)


ssp2_annual = interpolate_annual(decadal_df, "SSP2", decadal_years, all_years)
ssp5_annual = interpolate_annual(decadal_df, "SSP5", decadal_years, all_years)
base_cols = ["grid_id", "lon_idx", "lat_idx", "center_lon", "center_lat"]
available_cols = [col for col in base_cols if col in grid_gdf.columns]
grid_base = grid_gdf[available_cols].copy()
if isinstance(grid_base, gpd.GeoDataFrame):
    grid_base = pd.DataFrame(grid_base.drop(columns="geometry"))
if ssp2_annual is not None:
    ssp2_full = grid_base.merge(ssp2_annual, on="grid_id", how="left")
else:
    ssp2_full = None
if ssp5_annual is not None:
    ssp5_full = grid_base.merge(ssp5_annual, on="grid_id", how="left")
else:
    ssp5_full = None
for ssp_name, ssp_df in [("SSP2", ssp2_full), ("SSP5", ssp5_full)]:
    if ssp_df is None:
        continue
    ssp = ssp_name.lower()
    sample_years = [2020, 2030, 2040, 2050, 2060, 2070, 2080]
    for year in sample_years:
        col = f"{ssp}_{year}"
        if col in ssp_df.columns:
            total = ssp_df[col].sum()
            if year == 2020:
                pass
            else:
                baseline = ssp_df[f"{ssp}_2020"].sum()
                change = total - baseline
                pct = change / baseline * 100
if ssp2_full is not None:
    output_file_ssp2 = base_path + "Europe_Grid_SSP2_Population_Annual_2020_2080.csv"
    ssp2_full.to_csv(output_file_ssp2, index=False)
if ssp5_full is not None:
    output_file_ssp5 = base_path + "Europe_Grid_SSP5_Population_Annual_2020_2080.csv"
    ssp5_full.to_csv(output_file_ssp5, index=False)
