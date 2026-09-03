import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc

N_SIMULATIONS = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
desktop_path = Path("data")
excess_dir = desktop_path / "Model_Excess_Results"
output_dir = desktop_path / "Model_Mortality_Results"
output_dir.mkdir(exist_ok=True)
models = ["CNRM-ESM2-1", "GFDL-ESM4", "MIROC6", "NorESM2-MM", "UKESM1-0-LL"]
scenarios = ["SSP245", "SSP585"]
climate_zones_file = desktop_path / "Europe_Grid_Climate_Zones.csv"
rr_coeff_file = desktop_path / "RR_Coeff.csv"
un_regions_file = desktop_path / "UN_Geoscheme_Classification.csv"
cvd_deaths_ssp2_file = desktop_path / "SSP2_Median_Fert_CVD_Deaths_Grid_2025_2080.csv"
cvd_deaths_ssp5_file = desktop_path / "SSP5_Median_Fert_CVD_Deaths_Grid_2025_2080.csv"
population_ssp2_file = (
    desktop_path / "Europe_Grid_SSP2_Median_Fert_Age_Pop_2025_2080.csv"
)
population_ssp5_file = (
    desktop_path / "Europe_Grid_SSP5_Median_Fert_Age_Pop_2025_2080.csv"
)
age_groups = {
    "under_20": "pop_under_20",
    "20_54": "pop_20_54",
    "55_64": "pop_55_64",
    "65_74": "pop_65_74",
    "75plus": "pop_75plus",
}
missing_files = []
for name, filepath in [
    ("Climate zones", climate_zones_file),
    ("RR coefficients", rr_coeff_file),
    ("UN regions", un_regions_file),
    ("SSP2 CVD deaths", cvd_deaths_ssp2_file),
    ("SSP5 CVD deaths", cvd_deaths_ssp5_file),
    ("SSP2 population", population_ssp2_file),
    ("SSP5 population", population_ssp5_file),
]:
    if filepath.exists():
        pass
    else:
        missing_files.append(str(filepath))
for model in models:
    for scenario in scenarios:
        filepath = excess_dir / f"{model}_{scenario}_HeatCold_Excess.csv"
        if filepath.exists():
            pass
        else:
            missing_files.append(str(filepath))
if missing_files:
    for f in missing_files:
        pass
    exit(1)
zones = pd.read_csv(climate_zones_file)
un_regions = pd.read_csv(un_regions_file)
rr_coeff = pd.read_csv(rr_coeff_file)
cvd_ssp2 = pd.read_csv(cvd_deaths_ssp2_file)
cvd_ssp5 = pd.read_csv(cvd_deaths_ssp5_file)
pop_ssp2 = pd.read_csv(population_ssp2_file)
pop_ssp5 = pd.read_csv(population_ssp5_file)
overall_start = time.time()
files_processed = 0
for model in models:
    for scenario in scenarios:
        files_processed += 1
        model_start = time.time()
        excess_file = excess_dir / f"{model}_{scenario}_HeatCold_Excess.csv"
        excess_df = pd.read_csv(excess_file)
        cvd_data = cvd_ssp2 if scenario == "SSP245" else cvd_ssp5
        pop_data = pop_ssp2 if scenario == "SSP245" else pop_ssp5
        df = excess_df.merge(
            zones[["grid_id", "climate_zone"]], on="grid_id", how="left"
        )
        country_lookup = cvd_data[["grid_id", "Country"]].drop_duplicates()
        df = df.merge(country_lookup, on="grid_id", how="left")
        df = df.merge(un_regions, on="Country", how="left")
        all_grid_results = []
        unique_years = sorted(df["year"].unique())
        for year in unique_years:
            year_data = df[df["year"] == year].copy()
            cvd_cols_mean = [
                f"cvd_deaths_mean_{age}_{year}" for age in age_groups.keys()
            ]
            cvd_cols_min = [f"cvd_deaths_min_{age}_{year}" for age in age_groups.keys()]
            cvd_cols_max = [f"cvd_deaths_max_{age}_{year}" for age in age_groups.keys()]
            year_data = year_data.merge(
                cvd_data[["grid_id"] + cvd_cols_mean + cvd_cols_min + cvd_cols_max],
                on="grid_id",
                how="inner",
            )
            pop_cols = [f"{pop_col}_{year}" for pop_col in age_groups.values()]
            year_data = year_data.merge(
                pop_data[["grid_id"] + pop_cols], on="grid_id", how="inner"
            )
            grid_results = []
            for idx, row in year_data.iterrows():
                if idx % 1000 == 0 and idx > 0:
                    pass
                grid_id = row["grid_id"]
                climate_zone = row["climate_zone"]
                heat_excess = row["avg_daily_heat_excess"]
                cold_excess = row["avg_daily_cold_excess"]
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
                grid_sims = {
                    age: {"heat": [], "cold": [], "net": []}
                    for age in age_groups.keys()
                }
                for sim in range(N_SIMULATIONS):
                    beta_heat_sample = np.random.normal(beta_heat, beta_heat_se)
                    beta_cold_sample = np.random.normal(beta_cold, beta_cold_se)
                    RR_heat = np.exp(beta_heat_sample * heat_excess)
                    RR_cold = np.exp(beta_cold_sample * cold_excess)
                    PAF_heat = (RR_heat - 1) / RR_heat if RR_heat > 1 else 0
                    PAF_cold = (RR_cold - 1) / RR_cold if RR_cold > 1 else 0
                    for age in age_groups.keys():
                        mean_col = f"cvd_deaths_mean_{age}_{year}"
                        min_col = f"cvd_deaths_min_{age}_{year}"
                        max_col = f"cvd_deaths_max_{age}_{year}"
                        if mean_col not in row.index or pd.isna(row[mean_col]):
                            continue
                        cvd_mean = row[mean_col]
                        cvd_min = row[min_col]
                        cvd_max = row[max_col]
                        if cvd_max == cvd_min:
                            cvd_sample = cvd_mean
                        else:
                            cvd_sample = np.random.triangular(
                                cvd_min, cvd_mean, cvd_max
                            )
                        heat_deaths = PAF_heat * cvd_sample
                        cold_deaths = PAF_cold * cvd_sample
                        net_deaths = heat_deaths + cold_deaths
                        grid_sims[age]["heat"].append(heat_deaths)
                        grid_sims[age]["cold"].append(cold_deaths)
                        grid_sims[age]["net"].append(net_deaths)
                result_row = {
                    "grid_id": grid_id,
                    "year": year,
                    "model": model,
                    "scenario": scenario,
                    "Country": row["Country"],
                    "UN_Region": row.get("UN_Region", None),
                    "climate_zone": climate_zone,
                }
                for age in age_groups.keys():
                    heat_array = np.array(grid_sims[age]["heat"])
                    cold_array = np.array(grid_sims[age]["cold"])
                    net_array = np.array(grid_sims[age]["net"])
                    result_row[f"heat_deaths_{age}_mean"] = np.mean(heat_array)
                    result_row[f"heat_deaths_{age}_lower"] = np.percentile(
                        heat_array, 2.5
                    )
                    result_row[f"heat_deaths_{age}_upper"] = np.percentile(
                        heat_array, 97.5
                    )
                    result_row[f"cold_deaths_{age}_mean"] = np.mean(cold_array)
                    result_row[f"cold_deaths_{age}_lower"] = np.percentile(
                        cold_array, 2.5
                    )
                    result_row[f"cold_deaths_{age}_upper"] = np.percentile(
                        cold_array, 97.5
                    )
                    result_row[f"net_deaths_{age}_mean"] = np.mean(net_array)
                    result_row[f"net_deaths_{age}_lower"] = np.percentile(
                        net_array, 2.5
                    )
                    result_row[f"net_deaths_{age}_upper"] = np.percentile(
                        net_array, 97.5
                    )
                    pop_col = f"{age_groups[age]}_{year}"
                    result_row[f"population_{age}"] = row.get(pop_col, np.nan)
                result_row["heat_deaths_total_mean"] = sum(
                    (result_row[f"heat_deaths_{age}_mean"] for age in age_groups.keys())
                )
                result_row["heat_deaths_total_lower"] = sum(
                    (
                        result_row[f"heat_deaths_{age}_lower"]
                        for age in age_groups.keys()
                    )
                )
                result_row["heat_deaths_total_upper"] = sum(
                    (
                        result_row[f"heat_deaths_{age}_upper"]
                        for age in age_groups.keys()
                    )
                )
                result_row["cold_deaths_total_mean"] = sum(
                    (result_row[f"cold_deaths_{age}_mean"] for age in age_groups.keys())
                )
                result_row["cold_deaths_total_lower"] = sum(
                    (
                        result_row[f"cold_deaths_{age}_lower"]
                        for age in age_groups.keys()
                    )
                )
                result_row["cold_deaths_total_upper"] = sum(
                    (
                        result_row[f"cold_deaths_{age}_upper"]
                        for age in age_groups.keys()
                    )
                )
                result_row["net_deaths_total_mean"] = sum(
                    (result_row[f"net_deaths_{age}_mean"] for age in age_groups.keys())
                )
                result_row["net_deaths_total_lower"] = sum(
                    (result_row[f"net_deaths_{age}_lower"] for age in age_groups.keys())
                )
                result_row["net_deaths_total_upper"] = sum(
                    (result_row[f"net_deaths_{age}_upper"] for age in age_groups.keys())
                )
                result_row["population_total"] = sum(
                    (
                        result_row.get(f"population_{age}", 0)
                        for age in age_groups.keys()
                    )
                )
                grid_results.append(result_row)
            all_grid_results.extend(grid_results)
            del year_data, grid_results
            gc.collect()
        grid_df = pd.DataFrame(all_grid_results)
        for age in age_groups.keys():
            pop_col = f"population_{age}"
            grid_df[f"heat_rate_{age}_mean"] = (
                grid_df[f"heat_deaths_{age}_mean"] / grid_df[pop_col] * 100000
            ).replace([np.inf, -np.inf], np.nan)
            grid_df[f"cold_rate_{age}_mean"] = (
                grid_df[f"cold_deaths_{age}_mean"] / grid_df[pop_col] * 100000
            ).replace([np.inf, -np.inf], np.nan)
            grid_df[f"net_rate_{age}_mean"] = (
                grid_df[f"net_deaths_{age}_mean"] / grid_df[pop_col] * 100000
            ).replace([np.inf, -np.inf], np.nan)
        grid_df["heat_rate_total_mean"] = (
            grid_df["heat_deaths_total_mean"] / grid_df["population_total"] * 100000
        ).replace([np.inf, -np.inf], np.nan)
        grid_df["cold_rate_total_mean"] = (
            grid_df["cold_deaths_total_mean"] / grid_df["population_total"] * 100000
        ).replace([np.inf, -np.inf], np.nan)
        grid_df["net_rate_total_mean"] = (
            grid_df["net_deaths_total_mean"] / grid_df["population_total"] * 100000
        ).replace([np.inf, -np.inf], np.nan)
        output_file = output_dir / f"{model}_{scenario}_Mortality_Projections.csv"
        grid_df.to_csv(output_file, index=False)
        model_time = time.time() - model_start
        del grid_df, all_grid_results
        gc.collect()
total_time = time.time() - overall_start
for model in models:
    for scenario in scenarios:
        output_file = output_dir / f"{model}_{scenario}_Mortality_Projections.csv"
        if output_file.exists():
            size_mb = output_file.stat().st_size / 1024**2
