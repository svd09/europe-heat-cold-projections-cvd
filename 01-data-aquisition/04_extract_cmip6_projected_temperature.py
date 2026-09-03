import ee
import time

PROJECT_ID = "YOUR_GEE_PROJECT_ID"
try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    exit()
GRID_ASSET = "projects/YOUR_GEE_PROJECT_ID/assets/Europe"
try:
    grid = ee.FeatureCollection(GRID_ASSET)
    num_cells = grid.size().getInfo()
except Exception:
    exit()
MODELS = {
    "UKESM1-0-LL": "UK Earth System Model",
    "GFDL-ESM4": "NOAA Geophysical Fluid Dynamics Lab",
    "MIROC6": "Japanese climate model",
    "CNRM-ESM2-1": "French climate model",
    "NorESM2-MM": "Norwegian Earth System Model",
}
SCENARIOS = ["ssp245", "ssp585"]
START_YEAR = 2020
END_YEAR = 2080
NUM_YEARS = END_YEAR - START_YEAR + 1
for i, (model, description) in enumerate(MODELS.items(), 1):
    pass


def extract_grid_temps(image):
    date = image.date().format("YYYY-MM-dd")
    temp_mean = image.select("tas")
    temp_image = ee.Image.cat([temp_mean.rename("temp_mean")])
    grid_temps = temp_image.reduceRegions(
        collection=grid, reducer=ee.Reducer.mean(), scale=25000
    )

    def add_date(feature):
        return feature.set("date", date)

    return grid_temps.map(add_date)


tasks = []
for model_name in MODELS.keys():
    for scenario in SCENARIOS:
        start_date = ee.Date(f"{START_YEAR}-01-01")
        end_date = ee.Date(f"{END_YEAR}-12-31")
        cmip6 = (
            ee.ImageCollection("NASA/GDDP-CMIP6")
            .filter(ee.Filter.eq("model", model_name))
            .filter(ee.Filter.eq("scenario", scenario))
            .filterDate(start_date, end_date)
            .select(["tas"])
        )
        try:
            num_images = cmip6.size().getInfo()
            expected_images = NUM_YEARS * 365
            if num_images < expected_images * 0.95:
                pass
        except:
            num_images = NUM_YEARS * 365
        if num_images == 0:
            continue
        all_grid_temps = cmip6.map(extract_grid_temps).flatten()
        scenario_label = scenario.upper().replace("SSP", "SSP")
        task_name = f"Europe_{model_name}_{scenario_label}_2020-2080"
        selectors = [
            "grid_id",
            "lon_idx",
            "lat_idx",
            "center_lon",
            "center_lat",
            "date",
            "mean",
        ]
        task = ee.batch.Export.table.toDrive(
            collection=all_grid_temps,
            description=task_name,
            selectors=selectors,
            fileFormat="CSV",
            folder="Europe_CMIP6_Projections",
        )
        task.start()
        tasks.append((model_name, scenario, task_name))
        time.sleep(3)
for i, (model, scenario, task_name) in enumerate(tasks, 1):
    pass
expected_rows = num_cells * NUM_YEARS * 365
expected_size_gb = expected_rows * 7 * 10 / 1024**3
for i, (model, desc) in enumerate(MODELS.items(), 1):
    pass
