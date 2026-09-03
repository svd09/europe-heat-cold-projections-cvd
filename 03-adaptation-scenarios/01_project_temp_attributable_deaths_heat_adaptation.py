import pandas as pd
import numpy as np
from pathlib import Path
import time

N_SIMULATIONS = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
ADAPTATION_SCENARIOS = {
    "NoAdaptation": 0.0,
    "Adapt10pct": 0.1,
    "Adapt50pct": 0.5,
    "Adapt90pct": 0.9,
}
desktop_path = Path("data")
heat_excess_ssp245_file = desktop_path / "Excess_Heat_SSP245_2025-80.csv"
cold_excess_ssp245_file = desktop_path / "Excess_Cold_SSP245_2025-80.csv"
heat_excess_ssp585_file = desktop_path / "Excess_Heat_SSP585_2025-80.csv"
cold_excess_ssp585_file = desktop_path / "Excess_Cold_SSP585_2025-80.csv"
climate_zones_file = desktop_path / "Europe_Grid_Climate_Zones.csv"
rr_coeff_file = desktop_path / "RR_Coeff.csv"
un_regions_file = desktop_path / "UN_Geoscheme_Classification.csv"
cvd_deaths_ssp2_file = desktop_path / "SSP2_Median_Fert_CVD_Deaths_Grid_2025_2080.csv"
cvd_deaths_ssp5_file = desktop_path / "SSP5_Median_Fert_CVD_Deaths_Grid_2025_2080.csv"


def build_output_paths(ssp_tag, adaptation_name):
    suffix = f"_{adaptation_name}_2025-2080.csv"
    return {
        "grid": desktop_path / f"Projection_{ssp_tag}_Grid_Deaths{suffix}",
        "country": desktop_path / f"Projection_{ssp_tag}_Country_Deaths{suffix}",
        "region": desktop_path / f"Projection_{ssp_tag}_Region_Deaths{suffix}",
        "net_grid": desktop_path / f"Projection_{ssp_tag}_Grid_Net_Deaths{suffix}",
        "net_country": desktop_path
        / f"Projection_{ssp_tag}_Country_Net_Deaths{suffix}",
        "net_region": desktop_path / f"Projection_{ssp_tag}_Region_Net_Deaths{suffix}",
    }


zones = pd.read_csv(climate_zones_file)
un_regions = pd.read_csv(un_regions_file)
rr_coeff = pd.read_csv(rr_coeff_file)
age_groups = ["under_20", "20_54", "55_64", "65_74", "75plus"]
years = range(2025, 2081)


def process_scenario(
    scenario_name,
    heat_file,
    cold_file,
    cvd_file,
    heat_reduction=0.0,
    adaptation_name="NoAdaptation",
):
    scenario_start = time.time()
    heat = pd.read_csv(heat_file)
    cold = pd.read_csv(cold_file)
    cvd = pd.read_csv(cvd_file)
    heat = heat.merge(zones[["grid_id", "climate_zone"]], on="grid_id", how="inner")
    cold = cold.merge(zones[["grid_id", "climate_zone"]], on="grid_id", how="inner")
    all_grid_results = []
    all_country_results = []
    all_region_results = []
    for year in years:
        year_start = time.time()
        heat_year = heat[heat["year"] == year].copy()
        cold_year = cold[cold["year"] == year].copy()
        if len(heat_year) == 0 or len(cold_year) == 0:
            continue
        df_year = heat_year[["grid_id", "climate_zone", "avg_daily_heat_excess"]].copy()
        df_year = df_year.merge(
            cold_year[["grid_id", "avg_daily_cold_excess"]], on="grid_id", how="inner"
        )
        df_year = df_year.rename(
            columns={
                "avg_daily_heat_excess": "heat_excess",
                "avg_daily_cold_excess": "cold_excess",
            }
        )
        cvd_cols_year = ["grid_id", "Country"]
        for age in age_groups:
            cvd_cols_year.extend(
                [
                    f"cvd_deaths_mean_{age}_{year}",
                    f"cvd_deaths_max_{age}_{year}",
                    f"cvd_deaths_min_{age}_{year}",
                ]
            )
        df_year = df_year.merge(cvd[cvd_cols_year], on="grid_id", how="inner")
        for age in age_groups:
            df_year = df_year.rename(
                columns={
                    f"cvd_deaths_mean_{age}_{year}": f"cvd_deaths_mean_{age}",
                    f"cvd_deaths_max_{age}_{year}": f"cvd_deaths_max_{age}",
                    f"cvd_deaths_min_{age}_{year}": f"cvd_deaths_min_{age}",
                }
            )
        year_results = []
        for idx, row in df_year.iterrows():
            grid_id = row["grid_id"]
            country = row["Country"]
            climate_zone = row["climate_zone"]
            heat_excess = row["heat_excess"]
            cold_excess = row["cold_excess"]
            zone_rr = rr_coeff[rr_coeff["climate_zone"] == climate_zone]
            if len(zone_rr) == 0:
                continue
            beta_heat = zone_rr["beta_heat"].iloc[0]
            beta_heat_lower = zone_rr["beta_heat_lower"].iloc[0]
            beta_heat_upper = zone_rr["beta_heat_upper"].iloc[0]
            heat_scale = 1.0 - heat_reduction
            beta_heat = beta_heat * heat_scale
            beta_heat_lower = beta_heat_lower * heat_scale
            beta_heat_upper = beta_heat_upper * heat_scale
            beta_cold = zone_rr["beta_cold"].iloc[0]
            beta_cold_lower = zone_rr["beta_cold_lower"].iloc[0]
            beta_cold_upper = zone_rr["beta_cold_upper"].iloc[0]
            beta_heat_se = (beta_heat_upper - beta_heat_lower) / (2 * 1.96)
            beta_cold_se = (beta_cold_upper - beta_cold_lower) / (2 * 1.96)
            grid_sims = {age: {"heat": [], "cold": [], "net": []} for age in age_groups}
            for sim in range(N_SIMULATIONS):
                beta_heat_sample = np.random.normal(beta_heat, beta_heat_se)
                beta_cold_sample = np.random.normal(beta_cold, beta_cold_se)
                RR_heat = np.exp(beta_heat_sample * heat_excess)
                RR_cold = np.exp(beta_cold_sample * cold_excess)
                PAF_heat = (RR_heat - 1) / RR_heat if RR_heat > 1 else 0
                PAF_cold = (RR_cold - 1) / RR_cold if RR_cold > 1 else 0
                for age in age_groups:
                    cvd_mean = row[f"cvd_deaths_mean_{age}"]
                    cvd_min = row[f"cvd_deaths_min_{age}"]
                    cvd_max = row[f"cvd_deaths_max_{age}"]
                    if cvd_min == cvd_max:
                        cvd_sample = cvd_mean
                    else:
                        cvd_sample = np.random.triangular(cvd_min, cvd_mean, cvd_max)
                    heat_deaths = PAF_heat * cvd_sample
                    cold_deaths = PAF_cold * cvd_sample
                    net_deaths = heat_deaths + cold_deaths
                    grid_sims[age]["heat"].append(heat_deaths)
                    grid_sims[age]["cold"].append(cold_deaths)
                    grid_sims[age]["net"].append(net_deaths)
            grid_result = {
                "year": year,
                "grid_id": grid_id,
                "Country": country,
                "climate_zone": climate_zone,
            }
            for age in age_groups:
                heat_array = np.array(grid_sims[age]["heat"])
                cold_array = np.array(grid_sims[age]["cold"])
                net_array = np.array(grid_sims[age]["net"])
                grid_result[f"heat_deaths_{age}_mean"] = np.mean(heat_array)
                grid_result[f"heat_deaths_{age}_lower"] = np.percentile(heat_array, 2.5)
                grid_result[f"heat_deaths_{age}_upper"] = np.percentile(
                    heat_array, 97.5
                )
                grid_result[f"cold_deaths_{age}_mean"] = np.mean(cold_array)
                grid_result[f"cold_deaths_{age}_lower"] = np.percentile(cold_array, 2.5)
                grid_result[f"cold_deaths_{age}_upper"] = np.percentile(
                    cold_array, 97.5
                )
                grid_result[f"net_deaths_{age}_mean"] = np.mean(net_array)
                grid_result[f"net_deaths_{age}_lower"] = np.percentile(net_array, 2.5)
                grid_result[f"net_deaths_{age}_upper"] = np.percentile(net_array, 97.5)
            grid_result["heat_deaths_total_mean"] = sum(
                (grid_result[f"heat_deaths_{age}_mean"] for age in age_groups)
            )
            grid_result["heat_deaths_total_lower"] = sum(
                (grid_result[f"heat_deaths_{age}_lower"] for age in age_groups)
            )
            grid_result["heat_deaths_total_upper"] = sum(
                (grid_result[f"heat_deaths_{age}_upper"] for age in age_groups)
            )
            grid_result["cold_deaths_total_mean"] = sum(
                (grid_result[f"cold_deaths_{age}_mean"] for age in age_groups)
            )
            grid_result["cold_deaths_total_lower"] = sum(
                (grid_result[f"cold_deaths_{age}_lower"] for age in age_groups)
            )
            grid_result["cold_deaths_total_upper"] = sum(
                (grid_result[f"cold_deaths_{age}_upper"] for age in age_groups)
            )
            grid_result["net_deaths_total_mean"] = sum(
                (grid_result[f"net_deaths_{age}_mean"] for age in age_groups)
            )
            grid_result["net_deaths_total_lower"] = sum(
                (grid_result[f"net_deaths_{age}_lower"] for age in age_groups)
            )
            grid_result["net_deaths_total_upper"] = sum(
                (grid_result[f"net_deaths_{age}_upper"] for age in age_groups)
            )
            year_results.append(grid_result)
        year_df = pd.DataFrame(year_results)
        all_grid_results.append(year_df)
        country_year_results = []
        for country in year_df["Country"].unique():
            country_data = year_df[year_df["Country"] == country]
            country_row = {
                "year": year,
                "Country": country,
                "N_Grids": len(country_data),
            }
            for age in age_groups:
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
            country_row["heat_deaths_total_mean"] = country_data[
                "heat_deaths_total_mean"
            ].sum()
            country_row["heat_deaths_total_lower"] = country_data[
                "heat_deaths_total_lower"
            ].sum()
            country_row["heat_deaths_total_upper"] = country_data[
                "heat_deaths_total_upper"
            ].sum()
            country_row["cold_deaths_total_mean"] = country_data[
                "cold_deaths_total_mean"
            ].sum()
            country_row["cold_deaths_total_lower"] = country_data[
                "cold_deaths_total_lower"
            ].sum()
            country_row["cold_deaths_total_upper"] = country_data[
                "cold_deaths_total_upper"
            ].sum()
            country_row["net_deaths_total_mean"] = country_data[
                "net_deaths_total_mean"
            ].sum()
            country_row["net_deaths_total_lower"] = country_data[
                "net_deaths_total_lower"
            ].sum()
            country_row["net_deaths_total_upper"] = country_data[
                "net_deaths_total_upper"
            ].sum()
            country_year_results.append(country_row)
        total_row = {"year": year, "Country": "TOTAL", "N_Grids": len(year_df)}
        for age in age_groups:
            total_row[f"heat_deaths_{age}_mean"] = year_df[
                f"heat_deaths_{age}_mean"
            ].sum()
            total_row[f"heat_deaths_{age}_lower"] = year_df[
                f"heat_deaths_{age}_lower"
            ].sum()
            total_row[f"heat_deaths_{age}_upper"] = year_df[
                f"heat_deaths_{age}_upper"
            ].sum()
            total_row[f"cold_deaths_{age}_mean"] = year_df[
                f"cold_deaths_{age}_mean"
            ].sum()
            total_row[f"cold_deaths_{age}_lower"] = year_df[
                f"cold_deaths_{age}_lower"
            ].sum()
            total_row[f"cold_deaths_{age}_upper"] = year_df[
                f"cold_deaths_{age}_upper"
            ].sum()
            total_row[f"net_deaths_{age}_mean"] = year_df[
                f"net_deaths_{age}_mean"
            ].sum()
            total_row[f"net_deaths_{age}_lower"] = year_df[
                f"net_deaths_{age}_lower"
            ].sum()
            total_row[f"net_deaths_{age}_upper"] = year_df[
                f"net_deaths_{age}_upper"
            ].sum()
        total_row["heat_deaths_total_mean"] = year_df["heat_deaths_total_mean"].sum()
        total_row["heat_deaths_total_lower"] = year_df["heat_deaths_total_lower"].sum()
        total_row["heat_deaths_total_upper"] = year_df["heat_deaths_total_upper"].sum()
        total_row["cold_deaths_total_mean"] = year_df["cold_deaths_total_mean"].sum()
        total_row["cold_deaths_total_lower"] = year_df["cold_deaths_total_lower"].sum()
        total_row["cold_deaths_total_upper"] = year_df["cold_deaths_total_upper"].sum()
        total_row["net_deaths_total_mean"] = year_df["net_deaths_total_mean"].sum()
        total_row["net_deaths_total_lower"] = year_df["net_deaths_total_lower"].sum()
        total_row["net_deaths_total_upper"] = year_df["net_deaths_total_upper"].sum()
        country_year_results.append(total_row)
        all_country_results.append(pd.DataFrame(country_year_results))
        year_df_with_region = year_df.merge(un_regions, on="Country", how="left")
        region_year_results = []
        for region in year_df_with_region["UN_Region"].dropna().unique():
            region_data = year_df_with_region[
                year_df_with_region["UN_Region"] == region
            ]
            region_row = {
                "year": year,
                "UN_Region": region,
                "N_Grids": len(region_data),
            }
            for age in age_groups:
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
            region_row["heat_deaths_total_mean"] = region_data[
                "heat_deaths_total_mean"
            ].sum()
            region_row["heat_deaths_total_lower"] = region_data[
                "heat_deaths_total_lower"
            ].sum()
            region_row["heat_deaths_total_upper"] = region_data[
                "heat_deaths_total_upper"
            ].sum()
            region_row["cold_deaths_total_mean"] = region_data[
                "cold_deaths_total_mean"
            ].sum()
            region_row["cold_deaths_total_lower"] = region_data[
                "cold_deaths_total_lower"
            ].sum()
            region_row["cold_deaths_total_upper"] = region_data[
                "cold_deaths_total_upper"
            ].sum()
            region_row["net_deaths_total_mean"] = region_data[
                "net_deaths_total_mean"
            ].sum()
            region_row["net_deaths_total_lower"] = region_data[
                "net_deaths_total_lower"
            ].sum()
            region_row["net_deaths_total_upper"] = region_data[
                "net_deaths_total_upper"
            ].sum()
            region_year_results.append(region_row)
        region_total_row = {"year": year, "UN_Region": "TOTAL", "N_Grids": len(year_df)}
        for age in age_groups:
            region_total_row[f"heat_deaths_{age}_mean"] = year_df[
                f"heat_deaths_{age}_mean"
            ].sum()
            region_total_row[f"heat_deaths_{age}_lower"] = year_df[
                f"heat_deaths_{age}_lower"
            ].sum()
            region_total_row[f"heat_deaths_{age}_upper"] = year_df[
                f"heat_deaths_{age}_upper"
            ].sum()
            region_total_row[f"cold_deaths_{age}_mean"] = year_df[
                f"cold_deaths_{age}_mean"
            ].sum()
            region_total_row[f"cold_deaths_{age}_lower"] = year_df[
                f"cold_deaths_{age}_lower"
            ].sum()
            region_total_row[f"cold_deaths_{age}_upper"] = year_df[
                f"cold_deaths_{age}_upper"
            ].sum()
            region_total_row[f"net_deaths_{age}_mean"] = year_df[
                f"net_deaths_{age}_mean"
            ].sum()
            region_total_row[f"net_deaths_{age}_lower"] = year_df[
                f"net_deaths_{age}_lower"
            ].sum()
            region_total_row[f"net_deaths_{age}_upper"] = year_df[
                f"net_deaths_{age}_upper"
            ].sum()
        region_total_row["heat_deaths_total_mean"] = year_df[
            "heat_deaths_total_mean"
        ].sum()
        region_total_row["heat_deaths_total_lower"] = year_df[
            "heat_deaths_total_lower"
        ].sum()
        region_total_row["heat_deaths_total_upper"] = year_df[
            "heat_deaths_total_upper"
        ].sum()
        region_total_row["cold_deaths_total_mean"] = year_df[
            "cold_deaths_total_mean"
        ].sum()
        region_total_row["cold_deaths_total_lower"] = year_df[
            "cold_deaths_total_lower"
        ].sum()
        region_total_row["cold_deaths_total_upper"] = year_df[
            "cold_deaths_total_upper"
        ].sum()
        region_total_row["net_deaths_total_mean"] = year_df[
            "net_deaths_total_mean"
        ].sum()
        region_total_row["net_deaths_total_lower"] = year_df[
            "net_deaths_total_lower"
        ].sum()
        region_total_row["net_deaths_total_upper"] = year_df[
            "net_deaths_total_upper"
        ].sum()
        region_year_results.append(region_total_row)
        all_region_results.append(pd.DataFrame(region_year_results))
        time.time() - year_start
    grid_results_all = pd.concat(all_grid_results, ignore_index=True)
    country_results_all = pd.concat(all_country_results, ignore_index=True)
    region_results_all = pd.concat(all_region_results, ignore_index=True)
    heat_cold_cols = ["year", "grid_id", "Country", "climate_zone"]
    for age in age_groups:
        heat_cold_cols.extend(
            [
                f"heat_deaths_{age}_mean",
                f"heat_deaths_{age}_lower",
                f"heat_deaths_{age}_upper",
                f"cold_deaths_{age}_mean",
                f"cold_deaths_{age}_lower",
                f"cold_deaths_{age}_upper",
            ]
        )
    heat_cold_cols.extend(
        [
            "heat_deaths_total_mean",
            "heat_deaths_total_lower",
            "heat_deaths_total_upper",
            "cold_deaths_total_mean",
            "cold_deaths_total_lower",
            "cold_deaths_total_upper",
        ]
    )
    grid_heat_cold = grid_results_all[heat_cold_cols]
    net_cols = ["year", "grid_id", "Country", "climate_zone"]
    for age in age_groups:
        net_cols.extend(
            [
                f"net_deaths_{age}_mean",
                f"net_deaths_{age}_lower",
                f"net_deaths_{age}_upper",
            ]
        )
    net_cols.extend(
        ["net_deaths_total_mean", "net_deaths_total_lower", "net_deaths_total_upper"]
    )
    grid_net = grid_results_all[net_cols]
    country_heat_cold_cols = ["year", "Country", "N_Grids"]
    for age in age_groups:
        country_heat_cold_cols.extend(
            [
                f"heat_deaths_{age}_mean",
                f"heat_deaths_{age}_lower",
                f"heat_deaths_{age}_upper",
                f"cold_deaths_{age}_mean",
                f"cold_deaths_{age}_lower",
                f"cold_deaths_{age}_upper",
            ]
        )
    country_heat_cold_cols.extend(
        [
            "heat_deaths_total_mean",
            "heat_deaths_total_lower",
            "heat_deaths_total_upper",
            "cold_deaths_total_mean",
            "cold_deaths_total_lower",
            "cold_deaths_total_upper",
        ]
    )
    country_heat_cold = country_results_all[country_heat_cold_cols]
    country_net_cols = ["year", "Country", "N_Grids"]
    for age in age_groups:
        country_net_cols.extend(
            [
                f"net_deaths_{age}_mean",
                f"net_deaths_{age}_lower",
                f"net_deaths_{age}_upper",
            ]
        )
    country_net_cols.extend(
        ["net_deaths_total_mean", "net_deaths_total_lower", "net_deaths_total_upper"]
    )
    country_net = country_results_all[country_net_cols]
    region_heat_cold_cols = ["year", "UN_Region", "N_Grids"]
    for age in age_groups:
        region_heat_cold_cols.extend(
            [
                f"heat_deaths_{age}_mean",
                f"heat_deaths_{age}_lower",
                f"heat_deaths_{age}_upper",
                f"cold_deaths_{age}_mean",
                f"cold_deaths_{age}_lower",
                f"cold_deaths_{age}_upper",
            ]
        )
    region_heat_cold_cols.extend(
        [
            "heat_deaths_total_mean",
            "heat_deaths_total_lower",
            "heat_deaths_total_upper",
            "cold_deaths_total_mean",
            "cold_deaths_total_lower",
            "cold_deaths_total_upper",
        ]
    )
    region_heat_cold = region_results_all[region_heat_cold_cols]
    region_net_cols = ["year", "UN_Region", "N_Grids"]
    for age in age_groups:
        region_net_cols.extend(
            [
                f"net_deaths_{age}_mean",
                f"net_deaths_{age}_lower",
                f"net_deaths_{age}_upper",
            ]
        )
    region_net_cols.extend(
        ["net_deaths_total_mean", "net_deaths_total_lower", "net_deaths_total_upper"]
    )
    region_net = region_results_all[region_net_cols]
    time.time() - scenario_start
    return (
        grid_heat_cold,
        country_heat_cold,
        region_heat_cold,
        grid_net,
        country_net,
        region_net,
    )


total_start = time.time()
all_saved_files = []
for adaptation_name, heat_reduction in ADAPTATION_SCENARIOS.items():
    np.random.seed(RANDOM_SEED)
    (
        grid_ssp245,
        country_ssp245,
        region_ssp245,
        net_grid_ssp245,
        net_country_ssp245,
        net_region_ssp245,
    ) = process_scenario(
        "SSP2-4.5",
        heat_excess_ssp245_file,
        cold_excess_ssp245_file,
        cvd_deaths_ssp2_file,
        heat_reduction=heat_reduction,
        adaptation_name=adaptation_name,
    )
    (
        grid_ssp585,
        country_ssp585,
        region_ssp585,
        net_grid_ssp585,
        net_country_ssp585,
        net_region_ssp585,
    ) = process_scenario(
        "SSP5-8.5",
        heat_excess_ssp585_file,
        cold_excess_ssp585_file,
        cvd_deaths_ssp5_file,
        heat_reduction=heat_reduction,
        adaptation_name=adaptation_name,
    )
    paths_245 = build_output_paths("SSP245", adaptation_name)
    paths_585 = build_output_paths("SSP585", adaptation_name)
    grid_ssp245.to_csv(paths_245["grid"], index=False)
    country_ssp245.to_csv(paths_245["country"], index=False)
    region_ssp245.to_csv(paths_245["region"], index=False)
    net_grid_ssp245.to_csv(paths_245["net_grid"], index=False)
    net_country_ssp245.to_csv(paths_245["net_country"], index=False)
    net_region_ssp245.to_csv(paths_245["net_region"], index=False)
    grid_ssp585.to_csv(paths_585["grid"], index=False)
    country_ssp585.to_csv(paths_585["country"], index=False)
    region_ssp585.to_csv(paths_585["region"], index=False)
    net_grid_ssp585.to_csv(paths_585["net_grid"], index=False)
    net_country_ssp585.to_csv(paths_585["net_country"], index=False)
    net_region_ssp585.to_csv(paths_585["net_region"], index=False)
    all_saved_files.extend([p.name for p in paths_245.values()])
    all_saved_files.extend([p.name for p in paths_585.values()])
total_time = time.time() - total_start
for fname in all_saved_files:
    pass
