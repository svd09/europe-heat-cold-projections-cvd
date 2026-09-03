import pandas as pd
import numpy as np
from pathlib import Path
import time

desktop_path = Path("data")
model_dir = Path("data/raw_cmip6_unprocessed/bias_corrected_models")
output_dir = desktop_path / "Model_Excess_Results"
output_dir.mkdir(exist_ok=True)
baseline_path = Path("data/baseline_analysis_bundle")
tmrel_file = baseline_path / "Europe_Grid_Heat_Excess_2016-2023.csv"
models = {
    "CNRM-ESM2-1": ["SSP245", "SSP585"],
    "GFDL-ESM4": ["SSP245", "SSP585"],
    "MIROC6": ["SSP245", "SSP585"],
    "NorESM2-MM": ["SSP245", "SSP585"],
    "UKESM1-0-LL": ["SSP245", "SSP585"],
}
target_years = list(range(2046, 2056)) + list(range(2071, 2081))
if tmrel_file.exists():
    pass
else:
    exit(1)
missing_files = []
for model_name, scenarios in models.items():
    for scenario in scenarios:
        filepath = model_dir / f"{model_name}_{scenario}_2046-2080_BiasCorrect.csv"
        if filepath.exists():
            size_gb = filepath.stat().st_size / 1024**3
        else:
            missing_files.append(str(filepath))
if missing_files:
    for f in missing_files:
        pass
    exit(1)
tmrel_df = pd.read_csv(tmrel_file)
tmrel_lookup = tmrel_df[["grid_id", "TMREL"]].set_index("grid_id")
overall_start = time.time()
files_processed = 0
for model_name, scenarios in models.items():
    for scenario in scenarios:
        files_processed += 1
        input_file = model_dir / f"{model_name}_{scenario}_2046-2080_BiasCorrect.csv"
        output_file = output_dir / f"{model_name}_{scenario}_HeatCold_Excess.csv"
        model_start = time.time()
        all_results = []
        for year in target_years:
            year_start = time.time()
            year_data = []
            chunk_num = 0
            for chunk in pd.read_csv(input_file, chunksize=5000000):
                chunk_num += 1
                chunk["date"] = pd.to_datetime(chunk["date"])
                chunk["year"] = chunk["date"].dt.year
                year_chunk = chunk[chunk["year"] == year]
                if len(year_chunk) > 0:
                    year_data.append(year_chunk)
            if len(year_data) == 0:
                continue
            year_df = pd.concat(year_data, ignore_index=True)
            grid_results = []
            unique_grids = year_df["grid_id"].unique()
            for grid_id in unique_grids:
                if grid_id not in tmrel_lookup.index:
                    continue
                tmrel = tmrel_lookup.loc[grid_id, "TMREL"]
                grid_temps = year_df[year_df["grid_id"] == grid_id]["mean"].values
                n_days = len(grid_temps)
                heat_excess_daily = np.maximum(grid_temps - tmrel, 0)
                total_heat_excess = heat_excess_daily.sum()
                avg_daily_heat_excess = total_heat_excess / n_days if n_days > 0 else 0
                days_above_tmrel = (grid_temps > tmrel).sum()
                cold_excess_daily = np.maximum(tmrel - grid_temps, 0)
                total_cold_excess = cold_excess_daily.sum()
                avg_daily_cold_excess = total_cold_excess / n_days if n_days > 0 else 0
                days_below_tmrel = (grid_temps < tmrel).sum()
                grid_results.append(
                    {
                        "grid_id": grid_id,
                        "year": year,
                        "model": model_name,
                        "scenario": scenario,
                        "tmrel": tmrel,
                        "avg_daily_heat_excess": avg_daily_heat_excess,
                        "days_above_tmrel": days_above_tmrel,
                        "avg_daily_cold_excess": avg_daily_cold_excess,
                        "days_below_tmrel": days_below_tmrel,
                        "n_days": n_days,
                        "mean_temp": grid_temps.mean(),
                        "min_temp": grid_temps.min(),
                        "max_temp": grid_temps.max(),
                    }
                )
            all_results.extend(grid_results)
            year_time = time.time() - year_start
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_file, index=False)
        model_time = time.time() - model_start
total_time = time.time() - overall_start
for model_name, scenarios in models.items():
    for scenario in scenarios:
        output_file = output_dir / f"{model_name}_{scenario}_HeatCold_Excess.csv"
        if output_file.exists():
            size_mb = output_file.stat().st_size / 1024**2
