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
MODELS = {
    "UKESM1-0-LL": "UK Earth System Model",
    "GFDL-ESM4": "NOAA Geophysical Fluid Dynamics Lab",
    "MIROC6": "Japanese climate model",
    "CNRM-ESM2-1": "French climate model",
    "NorESM2-MM": "Norwegian Earth System Model",
}
MODEL_NAMES = list(MODELS.keys())
SCENARIOS = ["ssp245", "ssp585"]
PERIODS = [
    ("2020-2023", 2020, 2023),
    ("2046-2055", 2046, 2055),
    ("2071-2080", 2071, 2080),
]
for i, (model, description) in enumerate(MODELS.items(), 1):
    pass
for label, start_year, end_year in PERIODS:
    pass
CMIP6_BASE = (
    ee.ImageCollection("NASA/GDDP-CMIP6")
    .filter(ee.Filter.inList("model", MODEL_NAMES))
    .select(["tas"])
)


def make_extract_fn(scenario):

    def extract_ensemble_mean(date):
        date = ee.Date(date)
        date_str = date.format("YYYY-MM-dd")
        daily_models = CMIP6_BASE.filter(ee.Filter.eq("scenario", scenario)).filterDate(
            date, date.advance(1, "day")
        )
        model_count = daily_models.size()
        ensemble_mean = ee.Image(
            ee.Algorithms.If(
                model_count.gt(0),
                daily_models.mean().rename("temp_mean"),
                ee.Image.constant(-999).rename("temp_mean"),
            )
        )
        grid_temps = ensemble_mean.reduceRegions(
            collection=grid, reducer=ee.Reducer.mean(), scale=25000
        )

        def add_date_and_qa(feature):
            return feature.set({"date": date_str, "num_models": model_count}).select(
                [
                    "grid_id",
                    "lon_idx",
                    "lat_idx",
                    "center_lon",
                    "center_lat",
                    "date",
                    "mean",
                    "num_models",
                ]
            )

        return grid_temps.map(add_date_and_qa)

    return extract_ensemble_mean


tasks = []
for scenario in SCENARIOS:
    scenario_label = scenario.upper()
    extract_fn = make_extract_fn(scenario)
    for period_label, start_year, end_year in PERIODS:
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year + 1}-01-01"
        dates = ee.List.sequence(
            ee.Date(start_date).millis(),
            ee.Date(end_date).advance(-1, "day").millis(),
            1000 * 60 * 60 * 24,
        ).map(lambda d: ee.Date(d))
        num_days = dates.size().getInfo()
        all_grid_temps = ee.FeatureCollection(dates.map(extract_fn)).flatten()
        task_name = f"Ukraine_Belarus_CMIP6_Ensemble_{scenario_label}_{period_label}"
        selectors = [
            "grid_id",
            "lon_idx",
            "lat_idx",
            "center_lon",
            "center_lat",
            "date",
            "mean",
            "num_models",
        ]
        task = ee.batch.Export.table.toDrive(
            collection=all_grid_temps,
            description=task_name,
            selectors=selectors,
            fileFormat="CSV",
            folder="Ukraine_Belarus_CMIP6_Ensemble",
        )
        task.start()
        tasks.append((scenario_label, period_label, task_name, num_days))
        expected_rows = num_cells * num_days
        expected_size_mb = expected_rows * 8 * 10 / 1024**2
        time.sleep(3)
for i, (scenario_label, period_label, task_name, num_days) in enumerate(tasks, 1):
    pass
for i, (model, desc) in enumerate(MODELS.items(), 1):
    pass
for label, start_year, end_year in PERIODS:
    pass
