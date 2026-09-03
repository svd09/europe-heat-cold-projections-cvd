import pandas as pd
import numpy as np
from pathlib import Path
from scipy import interpolate
import warnings
import gc
import time

warnings.filterwarnings("ignore")
obs_dir = Path("data/raw_era5")
model_dir = Path("data/raw_cmip6_unprocessed")
output_dir = model_dir / "Bias_Corrected_Models"
output_dir.mkdir(exist_ok=True)
obs_files = [
    "Europe_ERA5Land_Baseline_2020.csv",
    "Europe_ERA5Land_Baseline_2021.csv",
    "Europe_ERA5Land_Baseline_2022.csv",
    "Europe_ERA5Land_Baseline_2023.csv",
]
models = {
    "CNRM-ESM2-1": {
        "SSP245": "CNRM-ESM2-1_SSP245_2020-80.csv",
        "SSP585": "CNRM-ESM2-1_SSP585_2020-80.csv",
    },
    "GFDL-ESM4": {
        "SSP245": "GFDL-ESM4_SSP245_2020-80.csv",
        "SSP585": "GFDL-ESM4_SSP585_2020-80.csv",
    },
    "MIROC6": {
        "SSP245": "MIROC6_SSP245_2020-80.csv",
        "SSP585": "MIROC6_SSP585_2020-80.csv",
    },
    "NorESM2-MM": {
        "SSP245": "NorESM2-MM_SSP245_2020-80.csv",
        "SSP585": "NorESM2-MM_SSP585_2020-80.csv",
    },
    "UKESM1-0-LL": {
        "SSP245": "UKESM1-0-LL_SSP245_2020-80.csv",
        "SSP585": "UKESM1-0-LL_SSP585_2020-80.csv",
    },
}
periods = {"mid_century": (2046, 2055), "late_century": (2071, 2080)}
overlap_years = [2020, 2021, 2022, 2023]


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
            missing_count = chunk["mean"].isna().sum()
            if missing_count > 0:
                chunk = chunk.dropna(subset=["mean"])
            year_data.append(chunk[["grid_id", "date", "mean"]])
        year_df = pd.concat(year_data, ignore_index=True)
        all_data.append(year_df)
        del year_data, year_df
        gc.collect()
    obs_data = pd.concat(all_data, ignore_index=True)
    obs_data["date"] = pd.to_datetime(obs_data["date"])
    return obs_data


def build_quantile_maps(obs_data, model_data_overlap, n_quantiles=100):
    grids = sorted(obs_data["grid_id"].unique())
    quantile_maps = {}
    grids_processed = 0
    grids_skipped = 0
    for i, grid_id in enumerate(grids, 1):
        if i % 500 == 0:
            pass
        obs_grid = obs_data[obs_data["grid_id"] == grid_id]["mean"].values
        if grid_id not in model_data_overlap["grid_id"].values:
            grids_skipped += 1
            continue
        model_grid = model_data_overlap[model_data_overlap["grid_id"] == grid_id][
            "mean"
        ].values
        if len(obs_grid) < 100 or len(model_grid) < 100:
            grids_skipped += 1
            continue
        probs = np.linspace(0, 1, n_quantiles)
        obs_quantiles = np.quantile(obs_grid, probs)
        model_quantiles = np.quantile(model_grid, probs)
        interp_func = interpolate.interp1d(
            model_quantiles,
            obs_quantiles,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        quantile_maps[grid_id] = {
            "interp_func": interp_func,
            "obs_min": obs_grid.min(),
            "obs_max": obs_grid.max(),
            "model_min": model_grid.min(),
            "model_max": model_grid.max(),
        }
        grids_processed += 1
    return quantile_maps


def main():
    missing_files = []
    for obs_file in obs_files:
        filepath = obs_dir / obs_file
        if filepath.exists():
            filepath.stat().st_size / 1024**3
        else:
            missing_files.append(str(filepath))
    for model_name, model_files in models.items():
        for scenario in ["SSP245", "SSP585"]:
            filepath = model_dir / model_files[scenario]
            if filepath.exists():
                filepath.stat().st_size / 1024**3
            else:
                missing_files.append(str(filepath))
    if output_dir.exists():
        pass
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    if missing_files:
        for f in missing_files:
            pass
        return
    obs_data = load_observed_data_chunked(obs_dir, obs_files)
    for model_name, model_files in models.items():
        for scenario in ["SSP245", "SSP585"]:
            model_file = model_dir / model_files[scenario]
            if not model_file.exists():
                continue
            model_overlap_data = []
            temp_col = None
            for chunk in pd.read_csv(model_file, chunksize=500000):
                if temp_col is None:
                    temp_col = detect_temp_column(chunk)
                if temp_col != "mean":
                    chunk = chunk.rename(columns={temp_col: "mean"})
                chunk["date"] = pd.to_datetime(chunk["date"])
                chunk["year"] = chunk["date"].dt.year
                chunk = chunk[chunk["year"].isin(overlap_years)]
                if len(chunk) > 0:
                    model_overlap_data.append(chunk[["grid_id", "date", "mean"]])
                del chunk
                gc.collect()
            if len(model_overlap_data) == 0:
                continue
            model_overlap = pd.concat(model_overlap_data, ignore_index=True)
            quantile_maps = build_quantile_maps(obs_data, model_overlap)
            del model_overlap, model_overlap_data
            gc.collect()
            output_file = (
                output_dir / f"{model_name}_{scenario}_2046-2080_BiasCorrect.csv"
            )
            first_write = True
            total_all = 0
            corrected_all = 0
            for period_name, (start_year, end_year) in periods.items():
                start_time = time.time()
                temp_col = None
                total_rows = 0
                corrected_rows = 0
                chunk_num = 0
                for chunk in pd.read_csv(model_file, chunksize=500000):
                    chunk_num += 1
                    if chunk_num % 10 == 0:
                        pass
                    if temp_col is None:
                        temp_col = detect_temp_column(chunk)
                    if temp_col != "mean":
                        chunk = chunk.rename(columns={temp_col: "mean"})
                    chunk["date"] = pd.to_datetime(chunk["date"])
                    chunk["year"] = chunk["date"].dt.year
                    chunk = chunk[
                        (chunk["year"] >= start_year) & (chunk["year"] <= end_year)
                    ]
                    if len(chunk) == 0:
                        continue
                    total_rows += len(chunk)
                    for grid_id in chunk["grid_id"].unique():
                        if grid_id not in quantile_maps:
                            continue
                        mask = chunk["grid_id"] == grid_id
                        model_temps = chunk.loc[mask, "mean"].values
                        interp_func = quantile_maps[grid_id]["interp_func"]
                        corrected_temps = interp_func(model_temps)
                        chunk.loc[mask, "mean"] = corrected_temps
                        corrected_rows += mask.sum()
                    chunk_output = chunk[["grid_id", "date", "mean"]].copy()
                    if first_write:
                        chunk_output.to_csv(output_file, index=False, mode="w")
                        first_write = False
                    else:
                        chunk_output.to_csv(
                            output_file, index=False, mode="a", header=False
                        )
                    del chunk, chunk_output
                    gc.collect()
                time.time() - start_time
                total_all += total_rows
                corrected_all += corrected_rows
            del quantile_maps
            gc.collect()


if __name__ == "__main__":
    main()
