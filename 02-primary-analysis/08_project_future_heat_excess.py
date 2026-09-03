import pandas as pd
import numpy as np
from pathlib import Path
import time

desktop_path = Path("data")
data_dir = Path("data/raw_cmip6_unprocessed")
baseline_path = Path("data/baseline_analysis_bundle")
tmrel_file = baseline_path / "Europe_Grid_Heat_Excess_2016-2023.csv"
scenarios = {
    "SSP245": data_dir / "ENSEMBLE_SSP245_2020-80_BiasCorrect.csv",
    "SSP585": data_dir / "ENSEMBLE_SSP585_2020-80_BiasCorrect.csv",
}
try:
    tmrel_df = pd.read_csv(tmrel_file)
except FileNotFoundError:
    exit()
tmrel_lookup = tmrel_df[["grid_id", "TMREL"]].copy()
tmrel_lookup = tmrel_lookup.set_index("grid_id")
for scenario_name, scenario_file in scenarios.items():
    if not scenario_file.exists():
        continue
    all_results = []
    years_to_process = range(2025, 2081)
    overall_start = time.time()
    for year in years_to_process:
        year_start = time.time()
        year_data = []
        chunksize = 5000000
        chunk_num = 0
        for chunk in pd.read_csv(scenario_file, chunksize=chunksize):
            chunk_num += 1
            chunk["date"] = pd.to_datetime(chunk["date"])
            chunk["year"] = chunk["date"].dt.year
            year_chunk = chunk[chunk["year"] == year].copy()
            if len(year_chunk) > 0:
                year_data.append(year_chunk)
        if len(year_data) == 0:
            continue
        year_df = pd.concat(year_data, ignore_index=True)
        grid_results = []
        unique_grids = year_df["grid_id"].unique()
        for idx, grid_id in enumerate(unique_grids):
            if (idx + 1) % 2000 == 0:
                pass
            if grid_id not in tmrel_lookup.index:
                continue
            tmrel = tmrel_lookup.loc[grid_id, "TMREL"]
            grid_temps = year_df[year_df["grid_id"] == grid_id]["mean"].values
            heat_excess_daily = np.maximum(grid_temps - tmrel, 0)
            total_heat_excess = heat_excess_daily.sum()
            n_days = len(grid_temps)
            avg_daily_heat_excess = total_heat_excess / n_days if n_days > 0 else 0
            days_above_tmrel = (grid_temps > tmrel).sum()
            grid_results.append(
                {
                    "grid_id": grid_id,
                    "year": year,
                    "scenario": scenario_name,
                    "tmrel": tmrel,
                    "total_heat_excess": total_heat_excess,
                    "avg_daily_heat_excess": avg_daily_heat_excess,
                    "days_above_tmrel": days_above_tmrel,
                    "n_days": n_days,
                    "mean_temp": grid_temps.mean(),
                    "max_temp": grid_temps.max(),
                }
            )
        all_results.extend(grid_results)
        year_time = time.time() - year_start
    results_df = pd.DataFrame(all_results)
    output_file = desktop_path / f"Future_Heat_Excess_{scenario_name}_2025-2080.csv"
    results_df.to_csv(output_file, index=False)
    overall_time = time.time() - overall_start
    annual_avg = results_df.groupby("year")["avg_daily_heat_excess"].mean()
for scenario_name in scenarios.keys():
    output_file = desktop_path / f"Future_Heat_Excess_{scenario_name}_2025-2080.csv"
    if output_file.exists():
        pass
