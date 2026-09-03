import ee
import time

PROJECT_ID = "YOUR_GEE_PROJECT_ID"
try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    exit()
GRID_ASSET = "projects/YOUR_GEE_PROJECT_ID/assets/Ukraine_Belarus"
try:
    grid = ee.FeatureCollection(GRID_ASSET)
    num_cells = grid.size().getInfo()
except Exception:
    exit()
ERA5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]


def extract_air_temp(date):
    date = ee.Date(date)
    date_str = date.format("YYYY-MM-dd")
    era5_day = ERA5.filterDate(date, date.advance(1, "day")).first()
    era5_exists = ee.Algorithms.If(era5_day, 1, 0)
    air_temp = ee.Image(
        ee.Algorithms.If(
            era5_exists,
            era5_day.select("temperature_2m").subtract(273.15),
            ee.Image.constant(-999),
        )
    )
    grid_temps = air_temp.reduceRegions(
        collection=grid, reducer=ee.Reducer.mean(), scale=11132
    )

    def rename_and_add_date(feature):
        return feature.set({"date": date_str, "temp_mean": feature.get("mean")}).select(
            [
                "grid_id",
                "lon_idx",
                "lat_idx",
                "center_lon",
                "center_lat",
                "date",
                "temp_mean",
            ]
        )

    return grid_temps.map(rename_and_add_date)


tasks = []
for year in YEARS:
    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"
    dates = ee.List.sequence(
        ee.Date(start_date).millis(),
        ee.Date(end_date).advance(-1, "day").millis(),
        1000 * 60 * 60 * 24,
    ).map(lambda d: ee.Date(d))
    num_days = dates.size().getInfo()
    all_grid_temps = ee.FeatureCollection(dates.map(extract_air_temp)).flatten()
    task_name = f"Ukraine_Belarus_ERA5Land_Baseline_{year}"
    selectors = [
        "grid_id",
        "lon_idx",
        "lat_idx",
        "center_lon",
        "center_lat",
        "date",
        "temp_mean",
    ]
    task = ee.batch.Export.table.toDrive(
        collection=all_grid_temps,
        description=task_name,
        selectors=selectors,
        fileFormat="CSV",
        folder="Ukraine_Belarus_ERA5Land_Baseline",
    )
    task.start()
    tasks.append((year, task_name))
    time.sleep(2)
for year, task_name in tasks:
    pass
