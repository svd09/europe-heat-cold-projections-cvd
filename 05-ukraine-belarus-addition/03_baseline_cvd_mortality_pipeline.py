import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

N_SIMULATIONS = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
DATA_DIR = Path("data")
era5_file = DATA_DIR / "Ukraine_Belarus_ERA5Land_Baseline_2016_2023.csv"
zones_file = DATA_DIR / "Ukraine_Belarus_Zones.csv"
cvd_file = DATA_DIR / "Ukraine_Belarus_Grid_CVD_Deaths_by_AgeBracket.csv"
rr_coeff_file = DATA_DIR / "RR_Coeff.csv"
output_excess = DATA_DIR / "Ukraine_Belarus_Grid_Heat_Cold_Excess_2016_2023.csv"
output_grid_deaths = DATA_DIR / "Ukraine_Belarus_Grid_Attributable_Deaths.csv"
output_country_deaths = DATA_DIR / "Ukraine_Belarus_Country_Attributable_Deaths.csv"
output_grid_rates = DATA_DIR / "Ukraine_Belarus_Grid_Mortality_Rates_per_100k.csv"
output_country_rates = DATA_DIR / "Ukraine_Belarus_Country_Mortality_Rates_per_100k.csv"
AGE_BRACKETS = ["<20", "20-54", "55-64", "65-74", "75+"]
start_time = time.time()
df = pd.read_csv(era5_file)
df["date"] = pd.to_datetime(df["date"])
df["temp_rounded"] = df["temp_mean"].round(1)
grid_coords = df[["grid_id", "center_lon", "center_lat"]].drop_duplicates()
coords_array = grid_coords[["center_lon", "center_lat"]].values
tree = cKDTree(coords_array)
grid_id_list = grid_coords["grid_id"].tolist()
missing_before = df["temp_rounded"].isna().sum()
imputed_count = 0
if missing_before > 0:
    impute_start = time.time()
    df_pivot = df.pivot_table(
        index="date", columns="grid_id", values="temp_rounded", aggfunc="first"
    )
    grid_neighbors = {}
    for i, grid_id in enumerate(grid_id_list):
        distances, indices = tree.query(coords_array[i], k=min(11, len(grid_id_list)))
        neighbor_ids = [grid_id_list[idx] for idx in np.atleast_1d(indices)[1:]]
        grid_neighbors[grid_id] = neighbor_ids
    grids_with_missing = df.loc[df["temp_rounded"].isna(), "grid_id"].unique()
    for grid_id in grids_with_missing:
        grid_missing_mask = (df["grid_id"] == grid_id) & df["temp_rounded"].isna()
        missing_dates = df.loc[grid_missing_mask, "date"]
        for neighbor_id in grid_neighbors.get(grid_id, []):
            still_missing = df.loc[grid_missing_mask, "temp_rounded"].isna()
            if not still_missing.any():
                break
            if neighbor_id not in df_pivot.columns:
                continue
            neighbor_vals = df_pivot[neighbor_id].reindex(missing_dates).values
            fillable = df.loc[grid_missing_mask].index[
                still_missing.values & pd.notna(neighbor_vals)
            ]
            if len(fillable) == 0:
                continue
            df.loc[fillable, "temp_rounded"] = (
                df_pivot[neighbor_id].reindex(df.loc[fillable, "date"]).values
            )
            imputed_count += len(fillable)
    impute_time = time.time() - impute_start
    missing_after = df["temp_rounded"].isna().sum()
else:
    pass
study_start = df["date"].min()
study_end = df["date"].max()
STUDY_PERIOD_DAYS = (study_end - study_start).days + 1
excess_rows = []
unique_grids = df["grid_id"].unique()
calc_start = time.time()
for i, grid_id in enumerate(unique_grids):
    if (i + 1) % 500 == 0:
        pass
    grid_data = df[df["grid_id"] == grid_id]
    center_lon = grid_data["center_lon"].iloc[0]
    center_lat = grid_data["center_lat"].iloc[0]
    grid_clean = grid_data.dropna(subset=["temp_rounded"])
    if len(grid_clean) == 0:
        continue
    p54 = np.percentile(grid_clean["temp_rounded"], 54)
    p92 = np.percentile(grid_clean["temp_rounded"], 92)
    pf = grid_clean[
        (grid_clean["temp_rounded"] >= p54) & (grid_clean["temp_rounded"] <= p92)
    ].copy()
    mode_series = pf["temp_rounded"].mode()
    if len(mode_series) > 0:
        tmrel = mode_series.iloc[0]
        mode_count = (pf["temp_rounded"] == tmrel).sum()
        if mode_count == 1:
            pf["temp_whole"] = pf["temp_rounded"].round(0)
            mode_whole = pf["temp_whole"].mode()
            if len(mode_whole) > 0:
                tmrel = mode_whole.iloc[0]
            else:
                tmrel = pf["temp_rounded"].median()
    else:
        tmrel = pf["temp_rounded"].median()
    total_days = len(grid_clean)
    heat_excess = (grid_clean["temp_rounded"] - tmrel).clip(lower=0)
    cold_excess = (tmrel - grid_clean["temp_rounded"]).clip(lower=0)
    avg_daily_heat_excess = heat_excess.sum() / STUDY_PERIOD_DAYS
    avg_daily_cold_excess = cold_excess.sum() / STUDY_PERIOD_DAYS
    excess_rows.append(
        {
            "grid_id": grid_id,
            "center_lon": center_lon,
            "center_lat": center_lat,
            "TMREL": round(tmrel, 1),
            "Total_Days": total_days,
            "Avg_Daily_Heat_Excess": avg_daily_heat_excess,
            "Avg_Daily_Cold_Excess": avg_daily_cold_excess,
        }
    )
calc_time = time.time() - calc_start
excess_df = pd.DataFrame(excess_rows)
excess_df.to_csv(output_excess, index=False)
zones = pd.read_csv(zones_file)
n_dupes = zones.duplicated("grid_id").sum()
if n_dupes:
    zones = zones.drop_duplicates(subset="grid_id")
zones = zones[["grid_id", "climate_zone"]]
cvd = pd.read_csv(cvd_file)
cvd["population_2020"] = cvd[[f"{b}_2020" for b in AGE_BRACKETS]].sum(axis=1)
rr_coeff = pd.read_csv(rr_coeff_file)
data = cvd.merge(
    excess_df[["grid_id", "Avg_Daily_Heat_Excess", "Avg_Daily_Cold_Excess"]],
    on="grid_id",
    how="inner",
)
data = data.merge(zones, on="grid_id", how="inner")
rr_lookup = rr_coeff.set_index("climate_zone")
sim_start = time.time()
grid_results = []
for i, row in data.iterrows():
    if (i + 1) % 500 == 0:
        pass
    zone = row["climate_zone"]
    if zone not in rr_lookup.index:
        continue
    rr = rr_lookup.loc[zone]
    beta_heat_se = (rr["beta_heat_upper"] - rr["beta_heat_lower"]) / (2 * 1.96)
    beta_cold_se = (rr["beta_cold_upper"] - rr["beta_cold_lower"]) / (2 * 1.96)
    beta_heat_samples = np.random.normal(rr["beta_heat"], beta_heat_se, N_SIMULATIONS)
    beta_cold_samples = np.random.normal(rr["beta_cold"], beta_cold_se, N_SIMULATIONS)
    RR_heat = np.exp(beta_heat_samples * row["Avg_Daily_Heat_Excess"])
    RR_cold = np.exp(beta_cold_samples * row["Avg_Daily_Cold_Excess"])
    PAF_heat = np.where(RR_heat > 0, (RR_heat - 1) / RR_heat, 0)
    PAF_cold = np.where(RR_cold > 0, (RR_cold - 1) / RR_cold, 0)
    result_row = {
        "grid_id": row["grid_id"],
        "country": row["country"],
        "climate_zone": zone,
        "heat_excess": row["Avg_Daily_Heat_Excess"],
        "cold_excess": row["Avg_Daily_Cold_Excess"],
    }
    total_heat_sims = np.zeros(N_SIMULATIONS)
    total_cold_sims = np.zeros(N_SIMULATIONS)
    total_net_sims = np.zeros(N_SIMULATIONS)
    for bracket in AGE_BRACKETS:
        cvd_mean = row[f"{bracket}_deaths"]
        cvd_max = row[f"{bracket}_deaths_upper"]
        cvd_min = row[f"{bracket}_deaths_lower"]
        if cvd_max == cvd_min:
            cvd_samples = np.full(N_SIMULATIONS, cvd_mean)
        else:
            cvd_samples = np.random.triangular(
                cvd_min, cvd_mean, cvd_max, N_SIMULATIONS
            )
        heat_deaths = cvd_samples * PAF_heat
        cold_deaths = cvd_samples * PAF_cold
        net_deaths = heat_deaths + cold_deaths
        result_row[f"heat_deaths_{bracket}_mean"] = heat_deaths.mean()
        result_row[f"heat_deaths_{bracket}_lower"] = np.percentile(heat_deaths, 2.5)
        result_row[f"heat_deaths_{bracket}_upper"] = np.percentile(heat_deaths, 97.5)
        result_row[f"cold_deaths_{bracket}_mean"] = cold_deaths.mean()
        result_row[f"cold_deaths_{bracket}_lower"] = np.percentile(cold_deaths, 2.5)
        result_row[f"cold_deaths_{bracket}_upper"] = np.percentile(cold_deaths, 97.5)
        result_row[f"net_deaths_{bracket}_mean"] = net_deaths.mean()
        result_row[f"net_deaths_{bracket}_lower"] = np.percentile(net_deaths, 2.5)
        result_row[f"net_deaths_{bracket}_upper"] = np.percentile(net_deaths, 97.5)
        total_heat_sims += heat_deaths
        total_cold_sims += cold_deaths
        total_net_sims += net_deaths
    result_row["heat_deaths_total_mean"] = total_heat_sims.mean()
    result_row["heat_deaths_total_lower"] = np.percentile(total_heat_sims, 2.5)
    result_row["heat_deaths_total_upper"] = np.percentile(total_heat_sims, 97.5)
    result_row["cold_deaths_total_mean"] = total_cold_sims.mean()
    result_row["cold_deaths_total_lower"] = np.percentile(total_cold_sims, 2.5)
    result_row["cold_deaths_total_upper"] = np.percentile(total_cold_sims, 97.5)
    result_row["net_deaths_total_mean"] = total_net_sims.mean()
    result_row["net_deaths_total_lower"] = np.percentile(total_net_sims, 2.5)
    result_row["net_deaths_total_upper"] = np.percentile(total_net_sims, 97.5)
    grid_results.append(result_row)
sim_time = time.time() - sim_start
results_df = pd.DataFrame(grid_results)
total_heat = results_df["heat_deaths_total_mean"].sum()
total_cold = results_df["cold_deaths_total_mean"].sum()
results_df.to_csv(output_grid_deaths, index=False)
sum_cols = [
    c
    for c in results_df.columns
    if c.startswith(("heat_deaths_", "cold_deaths_", "net_deaths_"))
]
country_deaths = results_df.groupby("country")[sum_cols].sum().reset_index()
country_deaths.insert(1, "N_Grids", results_df.groupby("country").size().values)
total_row = {"country": "TOTAL", "N_Grids": len(results_df)}
for c in sum_cols:
    total_row[c] = results_df[c].sum()
country_deaths = pd.concat(
    [country_deaths, pd.DataFrame([total_row])], ignore_index=True
)
country_deaths.to_csv(output_country_deaths, index=False)
pop_cols = {b: f"{b}_2020" for b in AGE_BRACKETS}
grid_pop = cvd[["grid_id", "country", "population_2020"] + list(pop_cols.values())]
grid_rates = results_df.merge(grid_pop, on=["grid_id", "country"], how="inner")
for bracket, pop_col in pop_cols.items():
    for metric in ["heat", "cold", "net"]:
        for stat in ["mean", "lower", "upper"]:
            grid_rates[f"{metric}_rate_{bracket}_{stat}"] = (
                grid_rates[f"{metric}_deaths_{bracket}_{stat}"]
                / grid_rates[pop_col]
                * 100000
            )
for metric in ["heat", "cold", "net"]:
    for stat in ["mean", "lower", "upper"]:
        grid_rates[f"{metric}_rate_total_{stat}"] = (
            grid_rates[f"{metric}_deaths_total_{stat}"]
            / grid_rates["population_2020"]
            * 100000
        )
rate_cols = ["grid_id", "country", "climate_zone", "population_2020"]
rate_cols += [
    c
    for c in grid_rates.columns
    if c.startswith(("heat_rate_", "cold_rate_", "net_rate_"))
]
grid_rates_output = grid_rates[rate_cols]
grid_rates_output.to_csv(output_grid_rates, index=False)
country_pop = (
    cvd.groupby("country")[["population_2020"] + list(pop_cols.values())]
    .sum()
    .reset_index()
)
country_merged = country_deaths.merge(country_pop, on="country", how="left")
total_mask = country_merged["country"] == "TOTAL"
if total_mask.any():
    country_merged.loc[total_mask, "population_2020"] = country_pop[
        "population_2020"
    ].sum()
    for pop_col in pop_cols.values():
        country_merged.loc[total_mask, pop_col] = country_pop[pop_col].sum()
for bracket, pop_col in pop_cols.items():
    for metric in ["heat", "cold", "net"]:
        for stat in ["mean", "lower", "upper"]:
            country_merged[f"{metric}_rate_{bracket}_{stat}"] = (
                country_merged[f"{metric}_deaths_{bracket}_{stat}"]
                / country_merged[pop_col]
                * 100000
            )
for metric in ["heat", "cold", "net"]:
    for stat in ["mean", "lower", "upper"]:
        country_merged[f"{metric}_rate_total_{stat}"] = (
            country_merged[f"{metric}_deaths_total_{stat}"]
            / country_merged["population_2020"]
            * 100000
        )
country_rate_cols = ["country", "N_Grids", "population_2020"]
country_rate_cols += [
    c
    for c in country_merged.columns
    if c.startswith(("heat_rate_", "cold_rate_", "net_rate_"))
]
country_rates_output = country_merged[country_rate_cols]
country_rates_output.to_csv(output_country_rates, index=False)
for _, r in country_rates_output.iterrows():
    pass
total_time = time.time() - start_time
for f in [
    output_excess,
    output_grid_deaths,
    output_country_deaths,
    output_grid_rates,
    output_country_rates,
]:
    pass
