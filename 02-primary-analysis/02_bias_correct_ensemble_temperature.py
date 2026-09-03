import pandas as pd
import numpy as np
from pathlib import Path
from scipy import interpolate
from scipy.spatial import cKDTree
import warnings
import gc
import pickle
import time

warnings.filterwarnings("ignore")
obs_dir = Path("data/raw_era5")
model_dir = Path("data/raw_cmip6_unprocessed")
output_dir = model_dir
obs_files = [
    "Europe_ERA5Land_Baseline_2020.csv",
    "Europe_ERA5Land_Baseline_2021.csv",
    "Europe_ERA5Land_Baseline_2022.csv",
    "Europe_ERA5Land_Baseline_2023.csv",
]
ensemble_files = ["ENSEMBLE_SSP245_2020-80.csv", "ENSEMBLE_SSP585_2020-80.csv"]
scenarios = ["SSP245", "SSP585"]


def detect_temp_column(df):
    if "mean" in df.columns:
        return "mean"
    elif "temp_mean" in df.columns:
        return "temp_mean"
    else:
        raise ValueError(f"No temperature column found. Columns: {list(df.columns)}")


def load_observed_data_chunked(obs_directory, file_list, chunksize=500000):
    all_data = []
    temp_col = None
    for i, filename in enumerate(file_list, 1):
        filepath = obs_directory / filename
        if not filepath.exists():
            continue
        filepath.stat().st_size / 1024**3
        year_data = []
        chunk_num = 0
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            chunk_num += 1
            if chunk_num % 10 == 0:
                pass
            if temp_col is None:
                temp_col = detect_temp_column(chunk)
            if temp_col != "mean":
                chunk = chunk.rename(columns={temp_col: "mean"})
            chunk["mean"] = chunk["mean"].round(1)
            if chunk_num == 1:
                pass
            year_data.append(chunk)
        year_df = pd.concat(year_data, ignore_index=True)
        all_data.append(year_df)
        del year_data, year_df
        gc.collect()
    obs_data = pd.concat(all_data, ignore_index=True)
    obs_data["date"] = pd.to_datetime(obs_data["date"])
    missing_before = obs_data["mean"].isna().sum()
    if missing_before > 0:
        impute_start = time.time()
        grid_coords = obs_data[
            ["grid_id", "center_lon", "center_lat"]
        ].drop_duplicates()
        coords_array = grid_coords[["center_lon", "center_lat"]].values
        tree = cKDTree(coords_array)
        grid_id_list = grid_coords["grid_id"].tolist()
        df_pivot = obs_data.pivot_table(
            index="date", columns="grid_id", values="mean", aggfunc="first"
        )
        grid_neighbors = {}
        for i, grid_id in enumerate(grid_id_list):
            if (i + 1) % 1000 == 0:
                pass
            current_coords = coords_array[i]
            distances, indices = tree.query(current_coords, k=11)
            neighbor_ids = [grid_id_list[idx] for idx in indices[1:]]
            grid_neighbors[grid_id] = neighbor_ids
        grids_with_missing = obs_data[obs_data["mean"].isna()]["grid_id"].unique()
        imputed_count = 0
        for i, grid_id in enumerate(grids_with_missing):
            if (i + 1) % 10 == 0 or i + 1 == len(grids_with_missing):
                pass
            neighbors = grid_neighbors.get(grid_id, [])
            if len(neighbors) == 0:
                continue
            grid_missing_mask = (obs_data["grid_id"] == grid_id) & obs_data[
                "mean"
            ].isna()
            missing_dates = obs_data.loc[grid_missing_mask, "date"].values
            if len(missing_dates) == 0:
                continue
            for neighbor_id in neighbors:
                still_missing = obs_data.loc[grid_missing_mask, "mean"].isna()
                if not still_missing.any():
                    break
                if neighbor_id not in df_pivot.columns:
                    continue
                neighbor_data = df_pivot[neighbor_id].reindex(missing_dates)
                still_missing_idx = obs_data[
                    grid_missing_mask & obs_data["mean"].isna()
                ].index
                for idx, date in zip(
                    still_missing_idx, obs_data.loc[still_missing_idx, "date"]
                ):
                    if date in neighbor_data.index:
                        neighbor_val = neighbor_data.loc[date]
                        if pd.notna(neighbor_val):
                            obs_data.at[idx, "mean"] = neighbor_val
                            imputed_count += 1
        time.time() - impute_start
        missing_after = obs_data["mean"].isna().sum()
        if missing_after > 0:
            pass
    return obs_data


def load_model_overlap_period(model_file, overlap_start, overlap_end, chunksize=500000):
    overlap_data = []
    chunk_num = 0
    temp_col = None
    for chunk in pd.read_csv(model_file, chunksize=chunksize):
        chunk_num += 1
        if chunk_num % 10 == 0:
            pass
        if temp_col is None:
            temp_col = detect_temp_column(chunk)
            if temp_col != "mean":
                pass
        if temp_col != "mean":
            chunk = chunk.rename(columns={temp_col: "mean"})
        if chunk["mean"].mean() > 100:
            chunk["mean"] = chunk["mean"] - 273.15
            if chunk_num == 1:
                pass
        chunk["date"] = pd.to_datetime(chunk["date"])
        mask = (chunk["date"] >= overlap_start) & (chunk["date"] <= overlap_end)
        chunk_overlap = chunk[mask]
        if len(chunk_overlap) > 0:
            overlap_data.append(chunk_overlap)
    model_overlap = pd.concat(overlap_data, ignore_index=True)
    return model_overlap


def compute_quantile_mapping_by_grid(obs_data, model_data, n_quantiles=100):
    quantile_maps = {}
    quantiles = np.linspace(0, 1, n_quantiles)
    obs_grids = set(obs_data["grid_id"].unique())
    model_grids = set(model_data["grid_id"].unique())
    common_grids = sorted(list(obs_grids & model_grids))
    import sys

    for i, grid_id in enumerate(common_grids):
        if (i + 1) % 500 == 0:
            (i + 1) / len(common_grids) * 100
            sys.stdout.flush()
        obs_temps = obs_data[obs_data["grid_id"] == grid_id]["mean"].values
        model_temps = model_data[model_data["grid_id"] == grid_id]["mean"].values
        obs_temps = obs_temps[~np.isnan(obs_temps)]
        model_temps = model_temps[~np.isnan(model_temps)]
        if len(obs_temps) < 10 or len(model_temps) < 10:
            continue
        obs_quantiles = np.quantile(obs_temps, quantiles)
        model_quantiles = np.quantile(model_temps, quantiles)
        try:
            correction_func = interpolate.interp1d(
                model_quantiles,
                obs_quantiles,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            quantile_maps[grid_id] = correction_func
        except:
            continue
    return quantile_maps


def apply_bias_correction_chunked(
    input_file, output_file, quantile_maps, chunksize=500000
):
    chunk_num = 0
    total_rows = 0
    corrected_rows = 0
    first_chunk = True
    temp_col = None
    import sys

    start_time = time.time()
    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        chunk_num += 1
        if temp_col is None:
            temp_col = detect_temp_column(chunk)
            if temp_col != "mean":
                pass
        if temp_col != "mean":
            chunk = chunk.rename(columns={temp_col: "mean"})
        if chunk["mean"].mean() > 100:
            chunk["mean"] = chunk["mean"] - 273.15
            if chunk_num == 1:
                pass
        chunk["mean_corrected"] = chunk.apply(
            lambda row: (
                quantile_maps[row["grid_id"]](row["mean"])
                if row["grid_id"] in quantile_maps
                else row["mean"]
            ),
            axis=1,
        )
        chunk["mean"] = chunk["mean_corrected"]
        chunk = chunk.drop("mean_corrected", axis=1)
        if temp_col != "mean":
            chunk = chunk.rename(columns={"mean": temp_col})
        if first_chunk:
            chunk.to_csv(output_file, index=False, mode="w")
            first_chunk = False
        else:
            chunk.to_csv(output_file, index=False, mode="a", header=False)
        total_rows += len(chunk)
        corrected_rows += chunk["grid_id"].isin(quantile_maps.keys()).sum()
        if chunk_num % 5 == 0:
            elapsed = time.time() - start_time
            rate = total_rows / elapsed if elapsed > 0 else 0
            est_total = (
                total_rows * (450 / chunk_num) if chunk_num < 450 else total_rows
            )
            remaining_rows = max(0, est_total - total_rows)
            eta_seconds = remaining_rows / rate if rate > 0 else 0
            eta_seconds / 60
            sys.stdout.flush()
        del chunk
        gc.collect()
    time.time() - start_time
    return (total_rows, corrected_rows)


def compute_summary_statistics(
    bias_corrected_file, quantile_maps, output_file, chunksize=500000
):
    chunk_num = 0
    temp_col = None
    grid_year_data = {}
    for chunk in pd.read_csv(bias_corrected_file, chunksize=chunksize):
        chunk_num += 1
        if chunk_num % 10 == 0:
            pass
        if temp_col is None:
            temp_col = detect_temp_column(chunk)
        if temp_col != "mean":
            chunk = chunk.rename(columns={temp_col: "mean"})
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk["year"] = chunk["date"].dt.year
        for (grid_id, year), group in chunk.groupby(["grid_id", "year"]):
            key = (grid_id, year)
            temps = group["mean"].values
            n = len(temps)
            if key not in grid_year_data:
                grid_year_data[key] = {
                    "count": n,
                    "sum": np.sum(temps),
                    "sum_sq": np.sum(temps**2),
                    "min": np.min(temps),
                    "max": np.max(temps),
                    "sorted_temps": np.sort(temps).tolist(),
                }
            else:
                stats = grid_year_data[key]
                stats["count"] += n
                stats["sum"] += np.sum(temps)
                stats["sum_sq"] += np.sum(temps**2)
                stats["min"] = min(stats["min"], np.min(temps))
                stats["max"] = max(stats["max"], np.max(temps))
                stats["sorted_temps"] = list(
                    np.sort(np.concatenate([stats["sorted_temps"], temps]))
                )
        if chunk_num % 100 == 0:
            gc.collect()
        del chunk
        gc.collect()
    summary_rows = []
    for i, ((grid_id, year), stats) in enumerate(grid_year_data.items(), 1):
        if i % 1000 == 0:
            pass
        n = stats["count"]
        mean = stats["sum"] / n
        variance = stats["sum_sq"] / n - mean**2
        std = np.sqrt(max(0, variance))
        sorted_array = np.array(stats["sorted_temps"])
        median = np.median(sorted_array)
        p05 = np.percentile(sorted_array, 5)
        p95 = np.percentile(sorted_array, 95)
        summary_rows.append(
            {
                "grid_id": grid_id,
                "year": year,
                "count": n,
                "mean": mean,
                "median": median,
                "std": std,
                "min": stats["min"],
                "max": stats["max"],
                "p05": p05,
                "p95": p95,
                "was_corrected": grid_id in quantile_maps,
            }
        )
        del sorted_array
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["grid_id", "year"])
    summary_df.to_csv(output_file, index=False)
    return summary_df


def generate_overall_summary_report(summary_df, scenario_name, output_file):
    with open(output_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"BIAS CORRECTION SUMMARY REPORT: {scenario_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total grids: {summary_df['grid_id'].nunique():,}\n")
        f.write(f"Total grid-years: {len(summary_df):,}\n")
        f.write(
            f"Years covered: {summary_df['year'].min()} - {summary_df['year'].max()}\n"
        )
        f.write(f"Grids with bias correction: {summary_df['was_corrected'].sum():,}\n")
        f.write("\n")
        f.write("TEMPERATURE STATISTICS (Kelvin)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Overall mean: {summary_df['mean'].mean():.2f} K\n")
        f.write(f"Overall median: {summary_df['median'].median():.2f} K\n")
        f.write(f"Overall std: {summary_df['std'].mean():.2f} K\n")
        f.write(f"Global min: {summary_df['min'].min():.2f} K\n")
        f.write(f"Global max: {summary_df['max'].max():.2f} K\n")
        f.write("\n")
        f.write("STATISTICS BY YEAR\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Year':<8} {'Mean (K)':<12} {'Median (K)':<12} {'Std (K)':<12}\n")
        f.write("-" * 70 + "\n")
        for year in sorted(summary_df["year"].unique()):
            year_data = summary_df[summary_df["year"] == year]
            f.write(f"{year:<8} {year_data['mean'].mean():<12.2f} ")
            f.write(f"{year_data['median'].median():<12.2f} ")
            f.write(f"{year_data['std'].mean():<12.2f}\n")
        f.write("\n")
        f.write("TEMPERATURE EXTREMES (Average across all years)\n")
        f.write("-" * 70 + "\n")
        grid_avg = summary_df.groupby("grid_id")["mean"].mean().sort_values()
        f.write("\nColdest 10 grids:\n")
        for grid_id, temp in grid_avg.head(10).items():
            f.write(f"  Grid {grid_id}: {temp:.2f} K ({temp - 273.15:.2f} °C)\n")
        f.write("\nWarmest 10 grids:\n")
        for grid_id, temp in grid_avg.tail(10).items():
            f.write(f"  Grid {grid_id}: {temp:.2f} K ({temp - 273.15:.2f} °C)\n")
        f.write("\n" + "=" * 70 + "\n")


def main():
    CHUNKSIZE = 500000
    obs_data = load_observed_data_chunked(obs_dir, obs_files, CHUNKSIZE)
    overlap_start = obs_data["date"].min()
    overlap_end = obs_data["date"].max()
    for scenario, ensemble_file in zip(scenarios, ensemble_files):
        model_file = model_dir / ensemble_file
        if not model_file.exists():
            continue
        model_data_overlap = load_model_overlap_period(
            model_file, overlap_start, overlap_end, CHUNKSIZE
        )
        quantile_maps = compute_quantile_mapping_by_grid(
            obs_data, model_data_overlap, n_quantiles=100
        )
        maps_file = output_dir / f"quantile_maps_{scenario}.pkl"
        with open(maps_file, "wb") as f:
            pickle.dump(quantile_maps, f)
        del model_data_overlap
        gc.collect()
        output_file = output_dir / f"ENSEMBLE_{scenario}_2020-80_BiasCorrect.csv"
        total_rows, corrected_rows = apply_bias_correction_chunked(
            model_file, output_file, quantile_maps, CHUNKSIZE
        )
        del quantile_maps
        gc.collect()
    for scenario in scenarios:
        pass


if __name__ == "__main__":
    main()
