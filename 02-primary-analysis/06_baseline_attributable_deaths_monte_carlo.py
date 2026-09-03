import pandas as pd
import numpy as np
from pathlib import Path
import time

N_SIMULATIONS = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
desktop_path = Path("data")
heat_excess_file = desktop_path / "Europe_Grid_Heat_Excess_2016-2023.csv"
cold_excess_file = desktop_path / "Europe_Grid_Cold_Excess_2016-2023.csv"
climate_zones_file = desktop_path / "Europe_Grid_Climate_Zones.csv"
rr_coeff_file = desktop_path / "RR_Coeff.csv"
cvd_deaths_file = desktop_path / "Grid_CVD_Age.csv"
output_grid = desktop_path / "Baseline_Grid_Attributable_Deaths_by_Age.csv"
output_country = desktop_path / "Baseline_Country_Attributable_Deaths.csv"
output_regional = desktop_path / "Baseline_Regional_Attributable_Deaths.csv"
output_net_grid = desktop_path / "Baseline_Grid_Net_Temperature_Deaths.csv"
output_net_country = desktop_path / "Baseline_Country_Net_Temperature_Deaths.csv"
output_net_regional = desktop_path / "Baseline_Regional_Net_Temperature_Deaths.csv"
start_time = time.time()
heat = pd.read_csv(heat_excess_file)
cold = pd.read_csv(cold_excess_file)
zones = pd.read_csv(climate_zones_file)
rr_coeff = pd.read_csv(rr_coeff_file)
cvd = pd.read_csv(cvd_deaths_file)
regions_file = desktop_path / "UN_Geoscheme_Classification.csv"
un_regions = pd.read_csv(regions_file)
age_groups = {
    "under_20": [
        "cvd_deaths_mean_under_20",
        "cvd_deaths_max_under_20",
        "cvd_deaths_min_under_20",
    ],
    "20_54": ["cvd_deaths_mean_20_54", "cvd_deaths_max_20_54", "cvd_deaths_min_20_54"],
    "55_64": ["cvd_deaths_mean_55_64", "cvd_deaths_max_55_64", "cvd_deaths_min_55_64"],
    "65_74": ["cvd_deaths_mean_65_74", "cvd_deaths_max_65_74", "cvd_deaths_min_65_74"],
    "75plus": [
        "cvd_deaths_mean_75plus",
        "cvd_deaths_max_75plus",
        "cvd_deaths_min_75plus",
    ],
}
for age, cols in age_groups.items():
    if all((col in cvd.columns for col in cols)):
        pass
    else:
        pass
df = heat[["grid_id", "Avg_Daily_Heat_Excess"]].merge(
    cold[["grid_id", "Avg_Daily_Cold_Excess"]], on="grid_id", how="inner"
)
df = df.merge(zones[["grid_id", "climate_zone"]], on="grid_id", how="inner")
cvd_cols = ["grid_id", "Country"] + [
    col for cols in age_groups.values() for col in cols
]
cvd_cols = [col for col in cvd_cols if col in cvd.columns]
df = df.merge(cvd[cvd_cols], on="grid_id", how="inner")
df = df.merge(un_regions, on="Country", how="left")
if df["UN_Region"].isna().sum() > 0:
    pass
sim_start = time.time()
results = []
progress_interval = max(1, len(df) // 20)
for idx, row in df.iterrows():
    if (idx + 1) % progress_interval == 0:
        elapsed = time.time() - sim_start
        rate = (idx + 1) / elapsed
        remaining = (len(df) - idx - 1) / rate
        pct = (idx + 1) / len(df) * 100
    grid_id = row["grid_id"]
    country = row["Country"]
    climate_zone = row["climate_zone"]
    heat_excess = row["Avg_Daily_Heat_Excess"]
    cold_excess = row["Avg_Daily_Cold_Excess"]
    zone_rr = rr_coeff[rr_coeff["climate_zone"] == climate_zone]
    if len(zone_rr) == 0:
        continue
    beta_heat = zone_rr["beta_heat"].iloc[0]
    beta_heat_lower = zone_rr["beta_heat_lower"].iloc[0]
    beta_heat_upper = zone_rr["beta_heat_upper"].iloc[0]
    beta_cold = zone_rr["beta_cold"].iloc[0]
    beta_cold_lower = zone_rr["beta_cold_lower"].iloc[0]
    beta_cold_upper = zone_rr["beta_cold_upper"].iloc[0]
    beta_heat_se = (beta_heat_upper - beta_heat_lower) / (2 * 1.96)
    beta_cold_se = (beta_cold_upper - beta_cold_lower) / (2 * 1.96)
    grid_results = {
        age: {"heat": [], "cold": [], "net": []} for age in age_groups.keys()
    }
    for sim in range(N_SIMULATIONS):
        beta_heat_sample = np.random.normal(beta_heat, beta_heat_se)
        beta_cold_sample = np.random.normal(beta_cold, beta_cold_se)
        RR_heat = np.exp(beta_heat_sample * heat_excess)
        RR_cold = np.exp(beta_cold_sample * cold_excess)
        PAF_heat = (RR_heat - 1) / RR_heat if RR_heat > 0 else 0
        PAF_cold = (RR_cold - 1) / RR_cold if RR_cold > 0 else 0
        for age, cols in age_groups.items():
            mean_col, max_col, min_col = cols
            if mean_col not in row.index:
                continue
            cvd_mean = row[mean_col]
            cvd_max = row[max_col]
            cvd_min = row[min_col]
            if cvd_max == cvd_min:
                cvd_sample = cvd_mean
            else:
                cvd_sample = np.random.triangular(cvd_min, cvd_mean, cvd_max)
            heat_deaths = cvd_sample * PAF_heat
            cold_deaths = cvd_sample * PAF_cold
            net_deaths = heat_deaths + cold_deaths
            grid_results[age]["heat"].append(heat_deaths)
            grid_results[age]["cold"].append(cold_deaths)
            grid_results[age]["net"].append(net_deaths)
    result_row = {
        "grid_id": grid_id,
        "Country": country,
        "UN_Region": row.get("UN_Region", None),
        "climate_zone": climate_zone,
        "heat_excess": heat_excess,
        "cold_excess": cold_excess,
    }
    for age in age_groups.keys():
        heat_sims = grid_results[age]["heat"]
        result_row[f"heat_deaths_{age}_mean"] = np.mean(heat_sims)
        result_row[f"heat_deaths_{age}_lower"] = np.percentile(heat_sims, 2.5)
        result_row[f"heat_deaths_{age}_upper"] = np.percentile(heat_sims, 97.5)
        cold_sims = grid_results[age]["cold"]
        result_row[f"cold_deaths_{age}_mean"] = np.mean(cold_sims)
        result_row[f"cold_deaths_{age}_lower"] = np.percentile(cold_sims, 2.5)
        result_row[f"cold_deaths_{age}_upper"] = np.percentile(cold_sims, 97.5)
        net_sims = grid_results[age]["net"]
        result_row[f"net_deaths_{age}_mean"] = np.mean(net_sims)
        result_row[f"net_deaths_{age}_lower"] = np.percentile(net_sims, 2.5)
        result_row[f"net_deaths_{age}_upper"] = np.percentile(net_sims, 97.5)
    total_heat_sims = [
        sum((grid_results[age]["heat"][i] for age in age_groups.keys()))
        for i in range(N_SIMULATIONS)
    ]
    total_cold_sims = [
        sum((grid_results[age]["cold"][i] for age in age_groups.keys()))
        for i in range(N_SIMULATIONS)
    ]
    total_net_sims = [
        sum((grid_results[age]["net"][i] for age in age_groups.keys()))
        for i in range(N_SIMULATIONS)
    ]
    result_row["heat_deaths_total_mean"] = np.mean(total_heat_sims)
    result_row["heat_deaths_total_lower"] = np.percentile(total_heat_sims, 2.5)
    result_row["heat_deaths_total_upper"] = np.percentile(total_heat_sims, 97.5)
    result_row["cold_deaths_total_mean"] = np.mean(total_cold_sims)
    result_row["cold_deaths_total_lower"] = np.percentile(total_cold_sims, 2.5)
    result_row["cold_deaths_total_upper"] = np.percentile(total_cold_sims, 97.5)
    result_row["net_deaths_total_mean"] = np.mean(total_net_sims)
    result_row["net_deaths_total_lower"] = np.percentile(total_net_sims, 2.5)
    result_row["net_deaths_total_upper"] = np.percentile(total_net_sims, 97.5)
    results.append(result_row)
sim_time = time.time() - sim_start
results_df = pd.DataFrame(results)
total_heat_mean = results_df["heat_deaths_total_mean"].sum()
total_cold_mean = results_df["cold_deaths_total_mean"].sum()
total_net = total_heat_mean + total_cold_mean
zone_summary = (
    results_df.groupby("climate_zone")
    .agg(
        {
            "heat_deaths_total_mean": "sum",
            "cold_deaths_total_mean": "sum",
            "grid_id": "count",
        }
    )
    .round(0)
)
zone_summary.columns = ["Heat Deaths", "Cold Deaths", "N Grids"]
country_heat = (
    results_df.groupby("Country")["heat_deaths_total_mean"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
for country, deaths in country_heat.items():
    pass
country_results = []
for country in results_df["Country"].unique():
    country_data = results_df[results_df["Country"] == country]
    country_row = {"Country": country, "N_Grids": len(country_data)}
    for age in age_groups.keys():
        country_row[f"heat_deaths_{age}_mean"] = country_data[
            f"heat_deaths_{age}_mean"
        ].sum()
        country_row[f"heat_deaths_{age}_lower"] = country_data[
            f"heat_deaths_{age}_lower"
        ].sum()
        country_row[f"heat_deaths_{age}_upper"] = country_data[
            f"heat_deaths_{age}_upper"
        ].sum()
        country_row[f"cold_deaths_{age}_mean"] = country_data[
            f"cold_deaths_{age}_mean"
        ].sum()
        country_row[f"cold_deaths_{age}_lower"] = country_data[
            f"cold_deaths_{age}_lower"
        ].sum()
        country_row[f"cold_deaths_{age}_upper"] = country_data[
            f"cold_deaths_{age}_upper"
        ].sum()
        country_row[f"net_deaths_{age}_mean"] = country_data[
            f"net_deaths_{age}_mean"
        ].sum()
        country_row[f"net_deaths_{age}_lower"] = country_data[
            f"net_deaths_{age}_lower"
        ].sum()
        country_row[f"net_deaths_{age}_upper"] = country_data[
            f"net_deaths_{age}_upper"
        ].sum()
    country_row["heat_deaths_total_mean"] = country_data["heat_deaths_total_mean"].sum()
    country_row["heat_deaths_total_lower"] = country_data[
        "heat_deaths_total_lower"
    ].sum()
    country_row["heat_deaths_total_upper"] = country_data[
        "heat_deaths_total_upper"
    ].sum()
    country_row["cold_deaths_total_mean"] = country_data["cold_deaths_total_mean"].sum()
    country_row["cold_deaths_total_lower"] = country_data[
        "cold_deaths_total_lower"
    ].sum()
    country_row["cold_deaths_total_upper"] = country_data[
        "cold_deaths_total_upper"
    ].sum()
    country_row["net_deaths_total_mean"] = country_data["net_deaths_total_mean"].sum()
    country_row["net_deaths_total_lower"] = country_data["net_deaths_total_lower"].sum()
    country_row["net_deaths_total_upper"] = country_data["net_deaths_total_upper"].sum()
    country_results.append(country_row)
country_df = pd.DataFrame(country_results)
regional_results = []
for region in results_df["UN_Region"].dropna().unique():
    region_data = results_df[results_df["UN_Region"] == region]
    region_row = {"UN_Region": region, "N_Grids": len(region_data)}
    for age in age_groups.keys():
        region_row[f"heat_deaths_{age}_mean"] = region_data[
            f"heat_deaths_{age}_mean"
        ].sum()
        region_row[f"heat_deaths_{age}_lower"] = region_data[
            f"heat_deaths_{age}_lower"
        ].sum()
        region_row[f"heat_deaths_{age}_upper"] = region_data[
            f"heat_deaths_{age}_upper"
        ].sum()
        region_row[f"cold_deaths_{age}_mean"] = region_data[
            f"cold_deaths_{age}_mean"
        ].sum()
        region_row[f"cold_deaths_{age}_lower"] = region_data[
            f"cold_deaths_{age}_lower"
        ].sum()
        region_row[f"cold_deaths_{age}_upper"] = region_data[
            f"cold_deaths_{age}_upper"
        ].sum()
        region_row[f"net_deaths_{age}_mean"] = region_data[
            f"net_deaths_{age}_mean"
        ].sum()
        region_row[f"net_deaths_{age}_lower"] = region_data[
            f"net_deaths_{age}_lower"
        ].sum()
        region_row[f"net_deaths_{age}_upper"] = region_data[
            f"net_deaths_{age}_upper"
        ].sum()
    region_row["heat_deaths_total_mean"] = region_data["heat_deaths_total_mean"].sum()
    region_row["heat_deaths_total_lower"] = region_data["heat_deaths_total_lower"].sum()
    region_row["heat_deaths_total_upper"] = region_data["heat_deaths_total_upper"].sum()
    region_row["cold_deaths_total_mean"] = region_data["cold_deaths_total_mean"].sum()
    region_row["cold_deaths_total_lower"] = region_data["cold_deaths_total_lower"].sum()
    region_row["cold_deaths_total_upper"] = region_data["cold_deaths_total_upper"].sum()
    region_row["net_deaths_total_mean"] = region_data["net_deaths_total_mean"].sum()
    region_row["net_deaths_total_lower"] = region_data["net_deaths_total_lower"].sum()
    region_row["net_deaths_total_upper"] = region_data["net_deaths_total_upper"].sum()
    regional_results.append(region_row)
regional_df = pd.DataFrame(regional_results)
total_row = {"Country": "TOTAL", "N_Grids": len(results_df)}
for age in age_groups.keys():
    total_row[f"heat_deaths_{age}_mean"] = results_df[f"heat_deaths_{age}_mean"].sum()
    total_row[f"heat_deaths_{age}_lower"] = results_df[f"heat_deaths_{age}_lower"].sum()
    total_row[f"heat_deaths_{age}_upper"] = results_df[f"heat_deaths_{age}_upper"].sum()
    total_row[f"cold_deaths_{age}_mean"] = results_df[f"cold_deaths_{age}_mean"].sum()
    total_row[f"cold_deaths_{age}_lower"] = results_df[f"cold_deaths_{age}_lower"].sum()
    total_row[f"cold_deaths_{age}_upper"] = results_df[f"cold_deaths_{age}_upper"].sum()
    total_row[f"net_deaths_{age}_mean"] = results_df[f"net_deaths_{age}_mean"].sum()
    total_row[f"net_deaths_{age}_lower"] = results_df[f"net_deaths_{age}_lower"].sum()
    total_row[f"net_deaths_{age}_upper"] = results_df[f"net_deaths_{age}_upper"].sum()
total_row["heat_deaths_total_mean"] = results_df["heat_deaths_total_mean"].sum()
total_row["heat_deaths_total_lower"] = results_df["heat_deaths_total_lower"].sum()
total_row["heat_deaths_total_upper"] = results_df["heat_deaths_total_upper"].sum()
total_row["cold_deaths_total_mean"] = results_df["cold_deaths_total_mean"].sum()
total_row["cold_deaths_total_lower"] = results_df["cold_deaths_total_lower"].sum()
total_row["cold_deaths_total_upper"] = results_df["cold_deaths_total_upper"].sum()
total_row["net_deaths_total_mean"] = results_df["net_deaths_total_mean"].sum()
total_row["net_deaths_total_lower"] = results_df["net_deaths_total_lower"].sum()
total_row["net_deaths_total_upper"] = results_df["net_deaths_total_upper"].sum()
country_df = pd.concat([country_df, pd.DataFrame([total_row])], ignore_index=True)
results_df.to_csv(output_grid, index=False)
country_df.to_csv(output_country, index=False)
regional_df.to_csv(output_regional, index=False)
net_cols_grid = ["grid_id", "Country", "climate_zone"] + [
    col for col in results_df.columns if "net_deaths" in col
]
net_grid_df = results_df[net_cols_grid]
net_grid_df.to_csv(output_net_grid, index=False)
net_cols_country = ["Country", "N_Grids"] + [
    col for col in country_df.columns if "net_deaths" in col
]
net_country_df = country_df[net_cols_country]
net_country_df.to_csv(output_net_country, index=False)
net_cols_regional = ["UN_Region", "N_Grids"] + [
    col for col in regional_df.columns if "net_deaths" in col
]
net_regional_df = regional_df[net_cols_regional]
net_regional_df.to_csv(output_net_regional, index=False)
total_time = time.time() - start_time
