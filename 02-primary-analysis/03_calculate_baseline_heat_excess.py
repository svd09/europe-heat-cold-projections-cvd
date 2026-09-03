import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import time

desktop_path = Path("data")
input_file = desktop_path / "Europe_ERA5_2016-23.csv"
start_time = time.time()
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    exit()
required_cols = ["grid_id", "date", "temp_mean"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    exit()
if "center_lon" not in df.columns or "center_lat" not in df.columns:
    exit()
df["date"] = pd.to_datetime(df["date"])
df["temp_rounded"] = df["temp_mean"].round(1)
grid_coords = df[["grid_id", "center_lon", "center_lat"]].drop_duplicates()
coords_array = grid_coords[["center_lon", "center_lat"]].values
tree = cKDTree(coords_array)
grid_id_list = grid_coords["grid_id"].tolist()
missing_before = df["temp_rounded"].isna().sum()
if missing_before > 0:
    impute_start = time.time()
    df_pivot = df.pivot_table(
        index="date", columns="grid_id", values="temp_rounded", aggfunc="first"
    )
    grid_neighbors = {}
    for i, grid_id in enumerate(grid_id_list):
        if (i + 1) % 1000 == 0:
            pass
        current_coords = coords_array[i]
        distances, indices = tree.query(current_coords, k=11)
        neighbor_ids = [grid_id_list[idx] for idx in indices[1:]]
        grid_neighbors[grid_id] = neighbor_ids
    grids_with_missing = df[df["temp_rounded"].isna()]["grid_id"].unique()
    imputed_count = 0
    for i, grid_id in enumerate(grids_with_missing):
        if (i + 1) % 50 == 0 or i + 1 == len(grids_with_missing):
            pass
        neighbors = grid_neighbors.get(grid_id, [])
        if len(neighbors) == 0:
            continue
        grid_missing_mask = (df["grid_id"] == grid_id) & df["temp_rounded"].isna()
        missing_dates = df.loc[grid_missing_mask, "date"].values
        if len(missing_dates) == 0:
            continue
        for neighbor_id in neighbors:
            still_missing = df.loc[grid_missing_mask, "temp_rounded"].isna()
            if not still_missing.any():
                break
            if neighbor_id not in df_pivot.columns:
                continue
            neighbor_data = df_pivot[neighbor_id].reindex(missing_dates)
            still_missing_idx = df[grid_missing_mask & df["temp_rounded"].isna()].index
            for idx, date in zip(still_missing_idx, df.loc[still_missing_idx, "date"]):
                if date in neighbor_data.index:
                    neighbor_val = neighbor_data.loc[date]
                    if pd.notna(neighbor_val):
                        df.at[idx, "temp_rounded"] = neighbor_val
                        imputed_count += 1
    impute_time = time.time() - impute_start
    missing_after = df["temp_rounded"].isna().sum()
    if missing_after > 0:
        pass
else:
    pass
start_date = df["date"].min()
end_date = df["date"].max()
STUDY_PERIOD_DAYS = (end_date - start_date).days + 1
grid_stats = []
unique_grids = df["grid_id"].unique()
calc_start = time.time()
for idx, grid_id in enumerate(unique_grids):
    if (idx + 1) % 1000 == 0:
        elapsed = time.time() - calc_start
        rate = (idx + 1) / elapsed
        remaining = (len(unique_grids) - idx - 1) / rate
    grid_data = df[df["grid_id"] == grid_id].copy()
    center_lon = grid_data["center_lon"].iloc[0]
    center_lat = grid_data["center_lat"].iloc[0]
    grid_data_clean = grid_data.dropna(subset=["temp_rounded"])
    if len(grid_data_clean) == 0:
        continue
    p54 = np.percentile(grid_data_clean["temp_rounded"], 54)
    p92 = np.percentile(grid_data_clean["temp_rounded"], 92)
    percentile_filtered = grid_data_clean[
        (grid_data_clean["temp_rounded"] >= p54)
        & (grid_data_clean["temp_rounded"] <= p92)
    ].copy()
    mode_series = percentile_filtered["temp_rounded"].mode()
    if len(mode_series) > 0:
        tmrel = mode_series.iloc[0]
        mode_count = (percentile_filtered["temp_rounded"] == tmrel).sum()
        if mode_count == 1:
            percentile_filtered["temp_whole"] = percentile_filtered[
                "temp_rounded"
            ].round(0)
            mode_series_whole = percentile_filtered["temp_whole"].mode()
            if len(mode_series_whole) > 0:
                tmrel = mode_series_whole.iloc[0]
                mode_count = (percentile_filtered["temp_whole"] == tmrel).sum()
            else:
                tmrel = percentile_filtered["temp_rounded"].median()
                mode_count = 0
    else:
        tmrel = percentile_filtered["temp_rounded"].median()
        mode_count = 0
    total_days = len(grid_data_clean)
    grid_data_clean["heat_excess"] = grid_data_clean["temp_rounded"] - tmrel
    grid_data_clean.loc[grid_data_clean["heat_excess"] <= 0, "heat_excess"] = 0
    total_heat_excess = grid_data_clean["heat_excess"].sum()
    avg_daily_heat_excess = total_heat_excess / STUDY_PERIOD_DAYS
    days_above_tmrel = (grid_data_clean["temp_rounded"] > tmrel).sum()
    pct_days_above_tmrel = days_above_tmrel / total_days * 100 if total_days > 0 else 0
    max_heat_excess = grid_data_clean["heat_excess"].max()
    mean_temp = grid_data_clean["temp_rounded"].mean()
    min_temp = grid_data_clean["temp_rounded"].min()
    max_temp = grid_data_clean["temp_rounded"].max()
    missing_days = STUDY_PERIOD_DAYS - total_days
    grid_stats.append(
        {
            "grid_id": grid_id,
            "center_lon": center_lon,
            "center_lat": center_lat,
            "TMREL": round(tmrel, 1),
            "TMREL_Frequency": mode_count,
            "Total_Days": total_days,
            "Missing_Days": missing_days,
            "Total_Heat_Excess_Celsius": round(total_heat_excess, 2),
            "Avg_Daily_Heat_Excess": round(avg_daily_heat_excess, 4),
            "Days_Above_TMREL": days_above_tmrel,
            "Percent_Days_Above_TMREL": round(pct_days_above_tmrel, 2),
            "Max_Heat_Excess": round(max_heat_excess, 2),
            "Mean_Temperature": round(mean_temp, 2),
            "Min_Temperature": round(min_temp, 2),
            "Max_Temperature": round(max_temp, 2),
            "P54_Temperature": round(p54, 2),
            "P92_Temperature": round(p92, 2),
        }
    )
calc_time = time.time() - calc_start
results_df = pd.DataFrame(grid_stats)
output_file = desktop_path / "Europe_Grid_Heat_Excess_2016-2023.csv"
results_df.to_csv(output_file, index=False)
total_time = time.time() - start_time
