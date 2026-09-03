import pandas as pd
import numpy as np
from pathlib import Path
import time

desktop_path = Path("data")
ADAPTATION_SCENARIOS = ["NoAdaptation", "Adapt10pct", "Adapt50pct", "Adapt90pct"]
age_groups = {
    "under_20": "pop_under_20",
    "20_54": "pop_20_54",
    "55_64": "pop_55_64",
    "65_74": "pop_65_74",
    "75plus": "pop_75plus",
}
years = range(2025, 2081)


def process_scenario_rates(
    scenario_name,
    deaths_grid_file,
    deaths_country_file,
    deaths_region_file,
    deaths_net_grid_file,
    deaths_net_country_file,
    deaths_net_region_file,
    population_file,
    un_regions_file,
    output_grid_rates,
    output_country_rates,
    output_region_rates,
    output_net_grid_rates,
    output_net_country_rates,
    output_net_region_rates,
    adaptation_name="NoAdaptation",
):
    scenario_start = time.time()
    deaths_grid = pd.read_csv(deaths_grid_file)
    deaths_country = pd.read_csv(deaths_country_file)
    deaths_region = pd.read_csv(deaths_region_file)
    deaths_net_grid = pd.read_csv(deaths_net_grid_file)
    deaths_net_country = pd.read_csv(deaths_net_country_file)
    deaths_net_region = pd.read_csv(deaths_net_region_file)
    pop = pd.read_csv(population_file)
    un_regions = pd.read_csv(un_regions_file)
    initial_inf = sum(
        (np.isinf(pop[col]).sum() for col in pop.columns if col.startswith("pop_"))
    )
    if initial_inf > 0:
        age_groups_list = ["under_20", "20_54", "55_64", "65_74", "75plus"]
        for age in age_groups_list:
            age_cols = [
                col
                for col in pop.columns
                if col.startswith(f"pop_{age}_") and col.split("_")[-1].isdigit()
            ]
            age_cols_sorted = sorted(age_cols, key=lambda x: int(x.split("_")[-1]))
            for idx in range(len(pop)):
                last_valid = None
                for col in age_cols_sorted:
                    val = pop.at[idx, col]
                    if (
                        isinstance(val, (int, float))
                        and (not np.isinf(val))
                        and (not np.isnan(val))
                    ):
                        last_valid = val
                    elif last_valid is not None:
                        pop.at[idx, col] = last_valid
        final_inf = sum(
            (np.isinf(pop[col]).sum() for col in pop.columns if col.startswith("pop_"))
        )
        if final_inf == 0:
            pass
        else:
            for col in pop.columns:
                if col.startswith("pop_"):
                    inf_mask = np.isinf(pop[col])
                    if inf_mask.any():
                        valid_values = pop.loc[~inf_mask, col]
                        if len(valid_values) > 0:
                            median_val = valid_values.median()
                            pop.loc[inf_mask, col] = median_val
    else:
        pass
    all_grid_rates = []
    all_net_grid_rates = []
    for year in years:
        pop_cols_year = {"grid_id": "grid_id", "Country": "Country"}
        for age, pop_prefix in age_groups.items():
            pop_cols_year[f"{pop_prefix}_{year}"] = pop_prefix
        pop_year = pop[list(pop_cols_year.keys())].copy()
        pop_year = pop_year.rename(columns=pop_cols_year)
        pop_year["population_total"] = sum(
            (pop_year[pop_prefix] for pop_prefix in age_groups.values())
        )
        deaths_year = deaths_grid[deaths_grid["year"] == year].copy()
        net_deaths_year = deaths_net_grid[deaths_net_grid["year"] == year].copy()
        if len(deaths_year) == 0:
            continue
        grid_merged = deaths_year.merge(
            pop_year, on=["grid_id", "Country"], how="inner"
        )
        net_grid_merged = net_deaths_year.merge(
            pop_year, on=["grid_id", "Country"], how="inner"
        )
        grid_rates = grid_merged.copy()
        for age, pop_col in age_groups.items():
            grid_rates[f"heat_rate_{age}_mean"] = (
                grid_rates[f"heat_deaths_{age}_mean"] / grid_rates[pop_col] * 100000
            )
            grid_rates[f"heat_rate_{age}_lower"] = (
                grid_rates[f"heat_deaths_{age}_lower"] / grid_rates[pop_col] * 100000
            )
            grid_rates[f"heat_rate_{age}_upper"] = (
                grid_rates[f"heat_deaths_{age}_upper"] / grid_rates[pop_col] * 100000
            )
            grid_rates[f"cold_rate_{age}_mean"] = (
                grid_rates[f"cold_deaths_{age}_mean"] / grid_rates[pop_col] * 100000
            )
            grid_rates[f"cold_rate_{age}_lower"] = (
                grid_rates[f"cold_deaths_{age}_lower"] / grid_rates[pop_col] * 100000
            )
            grid_rates[f"cold_rate_{age}_upper"] = (
                grid_rates[f"cold_deaths_{age}_upper"] / grid_rates[pop_col] * 100000
            )
        grid_rates["heat_rate_total_mean"] = (
            grid_rates["heat_deaths_total_mean"]
            / grid_rates["population_total"]
            * 100000
        )
        grid_rates["heat_rate_total_lower"] = (
            grid_rates["heat_deaths_total_lower"]
            / grid_rates["population_total"]
            * 100000
        )
        grid_rates["heat_rate_total_upper"] = (
            grid_rates["heat_deaths_total_upper"]
            / grid_rates["population_total"]
            * 100000
        )
        grid_rates["cold_rate_total_mean"] = (
            grid_rates["cold_deaths_total_mean"]
            / grid_rates["population_total"]
            * 100000
        )
        grid_rates["cold_rate_total_lower"] = (
            grid_rates["cold_deaths_total_lower"]
            / grid_rates["population_total"]
            * 100000
        )
        grid_rates["cold_rate_total_upper"] = (
            grid_rates["cold_deaths_total_upper"]
            / grid_rates["population_total"]
            * 100000
        )
        rate_cols = ["year", "grid_id", "Country", "climate_zone", "population_total"]
        rate_cols += [
            col
            for col in grid_rates.columns
            if col.startswith("heat_rate_") or col.startswith("cold_rate_")
        ]
        grid_rates_output = grid_rates[rate_cols]
        all_grid_rates.append(grid_rates_output)
        net_grid_rates = net_grid_merged.copy()
        for age, pop_col in age_groups.items():
            net_grid_rates[f"net_rate_{age}_mean"] = (
                net_grid_rates[f"net_deaths_{age}_mean"]
                / net_grid_rates[pop_col]
                * 100000
            )
            net_grid_rates[f"net_rate_{age}_lower"] = (
                net_grid_rates[f"net_deaths_{age}_lower"]
                / net_grid_rates[pop_col]
                * 100000
            )
            net_grid_rates[f"net_rate_{age}_upper"] = (
                net_grid_rates[f"net_deaths_{age}_upper"]
                / net_grid_rates[pop_col]
                * 100000
            )
        net_grid_rates["net_rate_total_mean"] = (
            net_grid_rates["net_deaths_total_mean"]
            / net_grid_rates["population_total"]
            * 100000
        )
        net_grid_rates["net_rate_total_lower"] = (
            net_grid_rates["net_deaths_total_lower"]
            / net_grid_rates["population_total"]
            * 100000
        )
        net_grid_rates["net_rate_total_upper"] = (
            net_grid_rates["net_deaths_total_upper"]
            / net_grid_rates["population_total"]
            * 100000
        )
        net_rate_cols = [
            "year",
            "grid_id",
            "Country",
            "climate_zone",
            "population_total",
        ]
        net_rate_cols += [
            col for col in net_grid_rates.columns if col.startswith("net_rate_")
        ]
        net_grid_rates_output = net_grid_rates[net_rate_cols]
        all_net_grid_rates.append(net_grid_rates_output)
    grid_rates_all = pd.concat(all_grid_rates, ignore_index=True)
    net_grid_rates_all = pd.concat(all_net_grid_rates, ignore_index=True)
    all_country_rates = []
    all_net_country_rates = []
    for year in years:
        country_year = deaths_country[deaths_country["year"] == year].copy()
        net_country_year = deaths_net_country[deaths_net_country["year"] == year].copy()
        if len(country_year) == 0:
            continue
        pop_year_cols = ["Country"]
        for age, pop_prefix in age_groups.items():
            pop_year_cols.append(f"{pop_prefix}_{year}")
        country_pop = pop[
            ["grid_id", "Country"]
            + [f"{pop_prefix}_{year}" for pop_prefix in age_groups.values()]
        ].copy()
        country_pop_agg = (
            country_pop.groupby("Country")
            .agg(
                {
                    **{
                        f"{pop_prefix}_{year}": "sum"
                        for pop_prefix in age_groups.values()
                    }
                }
            )
            .reset_index()
        )
        rename_dict = {
            f"{pop_prefix}_{year}": pop_prefix for pop_prefix in age_groups.values()
        }
        country_pop_agg = country_pop_agg.rename(columns=rename_dict)
        country_pop_agg["population_total"] = sum(
            (country_pop_agg[pop_prefix] for pop_prefix in age_groups.values())
        )
        country_merged = country_year.merge(country_pop_agg, on="Country", how="left")
        net_country_merged = net_country_year.merge(
            country_pop_agg, on="Country", how="left"
        )
        if (country_merged["Country"] == "TOTAL").any():
            total_pop = country_pop_agg["population_total"].sum()
            total_pop_ages = {
                age: country_pop_agg[pop_prefix].sum()
                for age, pop_prefix in age_groups.items()
            }
            country_merged.loc[
                country_merged["Country"] == "TOTAL", "population_total"
            ] = total_pop
            for age, pop_col in age_groups.items():
                country_merged.loc[country_merged["Country"] == "TOTAL", pop_col] = (
                    total_pop_ages[age]
                )
            net_country_merged.loc[
                net_country_merged["Country"] == "TOTAL", "population_total"
            ] = total_pop
            for age, pop_col in age_groups.items():
                net_country_merged.loc[
                    net_country_merged["Country"] == "TOTAL", pop_col
                ] = total_pop_ages[age]
        country_rates = country_merged.copy()
        for age, pop_col in age_groups.items():
            country_rates[f"heat_rate_{age}_mean"] = (
                country_rates[f"heat_deaths_{age}_mean"]
                / country_rates[pop_col]
                * 100000
            )
            country_rates[f"heat_rate_{age}_lower"] = (
                country_rates[f"heat_deaths_{age}_lower"]
                / country_rates[pop_col]
                * 100000
            )
            country_rates[f"heat_rate_{age}_upper"] = (
                country_rates[f"heat_deaths_{age}_upper"]
                / country_rates[pop_col]
                * 100000
            )
            country_rates[f"cold_rate_{age}_mean"] = (
                country_rates[f"cold_deaths_{age}_mean"]
                / country_rates[pop_col]
                * 100000
            )
            country_rates[f"cold_rate_{age}_lower"] = (
                country_rates[f"cold_deaths_{age}_lower"]
                / country_rates[pop_col]
                * 100000
            )
            country_rates[f"cold_rate_{age}_upper"] = (
                country_rates[f"cold_deaths_{age}_upper"]
                / country_rates[pop_col]
                * 100000
            )
        country_rates["heat_rate_total_mean"] = (
            country_rates["heat_deaths_total_mean"]
            / country_rates["population_total"]
            * 100000
        )
        country_rates["heat_rate_total_lower"] = (
            country_rates["heat_deaths_total_lower"]
            / country_rates["population_total"]
            * 100000
        )
        country_rates["heat_rate_total_upper"] = (
            country_rates["heat_deaths_total_upper"]
            / country_rates["population_total"]
            * 100000
        )
        country_rates["cold_rate_total_mean"] = (
            country_rates["cold_deaths_total_mean"]
            / country_rates["population_total"]
            * 100000
        )
        country_rates["cold_rate_total_lower"] = (
            country_rates["cold_deaths_total_lower"]
            / country_rates["population_total"]
            * 100000
        )
        country_rates["cold_rate_total_upper"] = (
            country_rates["cold_deaths_total_upper"]
            / country_rates["population_total"]
            * 100000
        )
        country_rate_cols = ["year", "Country", "N_Grids", "population_total"]
        country_rate_cols += [
            col
            for col in country_rates.columns
            if col.startswith("heat_rate_") or col.startswith("cold_rate_")
        ]
        country_rates_output = country_rates[country_rate_cols]
        all_country_rates.append(country_rates_output)
        net_country_rates = net_country_merged.copy()
        for age, pop_col in age_groups.items():
            net_country_rates[f"net_rate_{age}_mean"] = (
                net_country_rates[f"net_deaths_{age}_mean"]
                / net_country_rates[pop_col]
                * 100000
            )
            net_country_rates[f"net_rate_{age}_lower"] = (
                net_country_rates[f"net_deaths_{age}_lower"]
                / net_country_rates[pop_col]
                * 100000
            )
            net_country_rates[f"net_rate_{age}_upper"] = (
                net_country_rates[f"net_deaths_{age}_upper"]
                / net_country_rates[pop_col]
                * 100000
            )
        net_country_rates["net_rate_total_mean"] = (
            net_country_rates["net_deaths_total_mean"]
            / net_country_rates["population_total"]
            * 100000
        )
        net_country_rates["net_rate_total_lower"] = (
            net_country_rates["net_deaths_total_lower"]
            / net_country_rates["population_total"]
            * 100000
        )
        net_country_rates["net_rate_total_upper"] = (
            net_country_rates["net_deaths_total_upper"]
            / net_country_rates["population_total"]
            * 100000
        )
        net_country_rate_cols = ["year", "Country", "N_Grids", "population_total"]
        net_country_rate_cols += [
            col for col in net_country_rates.columns if col.startswith("net_rate_")
        ]
        net_country_rates_output = net_country_rates[net_country_rate_cols]
        all_net_country_rates.append(net_country_rates_output)
    country_rates_all = pd.concat(all_country_rates, ignore_index=True)
    net_country_rates_all = pd.concat(all_net_country_rates, ignore_index=True)
    all_region_rates = []
    all_net_region_rates = []
    for year in years:
        region_year = deaths_region[deaths_region["year"] == year].copy()
        net_region_year = deaths_net_region[deaths_net_region["year"] == year].copy()
        if len(region_year) == 0:
            continue
        pop_year_cols = ["Country"]
        for age, pop_prefix in age_groups.items():
            pop_year_cols.append(f"{pop_prefix}_{year}")
        country_pop = pop[
            ["grid_id", "Country"]
            + [f"{pop_prefix}_{year}" for pop_prefix in age_groups.values()]
        ].copy()
        country_pop_with_region = country_pop.merge(
            un_regions, on="Country", how="left"
        )
        region_pop_agg = (
            country_pop_with_region.groupby("UN_Region")
            .agg(
                {
                    **{
                        f"{pop_prefix}_{year}": "sum"
                        for pop_prefix in age_groups.values()
                    }
                }
            )
            .reset_index()
        )
        rename_dict = {
            f"{pop_prefix}_{year}": pop_prefix for pop_prefix in age_groups.values()
        }
        region_pop_agg = region_pop_agg.rename(columns=rename_dict)
        region_pop_agg["population_total"] = sum(
            (region_pop_agg[pop_prefix] for pop_prefix in age_groups.values())
        )
        region_merged = region_year.merge(region_pop_agg, on="UN_Region", how="left")
        net_region_merged = net_region_year.merge(
            region_pop_agg, on="UN_Region", how="left"
        )
        if (region_merged["UN_Region"] == "TOTAL").any():
            total_pop = region_pop_agg["population_total"].sum()
            total_pop_ages = {
                age: region_pop_agg[pop_prefix].sum()
                for age, pop_prefix in age_groups.items()
            }
            region_merged.loc[
                region_merged["UN_Region"] == "TOTAL", "population_total"
            ] = total_pop
            for age, pop_col in age_groups.items():
                region_merged.loc[region_merged["UN_Region"] == "TOTAL", pop_col] = (
                    total_pop_ages[age]
                )
            net_region_merged.loc[
                net_region_merged["UN_Region"] == "TOTAL", "population_total"
            ] = total_pop
            for age, pop_col in age_groups.items():
                net_region_merged.loc[
                    net_region_merged["UN_Region"] == "TOTAL", pop_col
                ] = total_pop_ages[age]
        region_rates = region_merged.copy()
        for age, pop_col in age_groups.items():
            region_rates[f"heat_rate_{age}_mean"] = (
                region_rates[f"heat_deaths_{age}_mean"] / region_rates[pop_col] * 100000
            )
            region_rates[f"heat_rate_{age}_lower"] = (
                region_rates[f"heat_deaths_{age}_lower"]
                / region_rates[pop_col]
                * 100000
            )
            region_rates[f"heat_rate_{age}_upper"] = (
                region_rates[f"heat_deaths_{age}_upper"]
                / region_rates[pop_col]
                * 100000
            )
            region_rates[f"cold_rate_{age}_mean"] = (
                region_rates[f"cold_deaths_{age}_mean"] / region_rates[pop_col] * 100000
            )
            region_rates[f"cold_rate_{age}_lower"] = (
                region_rates[f"cold_deaths_{age}_lower"]
                / region_rates[pop_col]
                * 100000
            )
            region_rates[f"cold_rate_{age}_upper"] = (
                region_rates[f"cold_deaths_{age}_upper"]
                / region_rates[pop_col]
                * 100000
            )
        region_rates["heat_rate_total_mean"] = (
            region_rates["heat_deaths_total_mean"]
            / region_rates["population_total"]
            * 100000
        )
        region_rates["heat_rate_total_lower"] = (
            region_rates["heat_deaths_total_lower"]
            / region_rates["population_total"]
            * 100000
        )
        region_rates["heat_rate_total_upper"] = (
            region_rates["heat_deaths_total_upper"]
            / region_rates["population_total"]
            * 100000
        )
        region_rates["cold_rate_total_mean"] = (
            region_rates["cold_deaths_total_mean"]
            / region_rates["population_total"]
            * 100000
        )
        region_rates["cold_rate_total_lower"] = (
            region_rates["cold_deaths_total_lower"]
            / region_rates["population_total"]
            * 100000
        )
        region_rates["cold_rate_total_upper"] = (
            region_rates["cold_deaths_total_upper"]
            / region_rates["population_total"]
            * 100000
        )
        region_rate_cols = ["year", "UN_Region", "N_Grids", "population_total"]
        region_rate_cols += [
            col
            for col in region_rates.columns
            if col.startswith("heat_rate_") or col.startswith("cold_rate_")
        ]
        region_rates_output = region_rates[region_rate_cols]
        all_region_rates.append(region_rates_output)
        net_region_rates = net_region_merged.copy()
        for age, pop_col in age_groups.items():
            net_region_rates[f"net_rate_{age}_mean"] = (
                net_region_rates[f"net_deaths_{age}_mean"]
                / net_region_rates[pop_col]
                * 100000
            )
            net_region_rates[f"net_rate_{age}_lower"] = (
                net_region_rates[f"net_deaths_{age}_lower"]
                / net_region_rates[pop_col]
                * 100000
            )
            net_region_rates[f"net_rate_{age}_upper"] = (
                net_region_rates[f"net_deaths_{age}_upper"]
                / net_region_rates[pop_col]
                * 100000
            )
        net_region_rates["net_rate_total_mean"] = (
            net_region_rates["net_deaths_total_mean"]
            / net_region_rates["population_total"]
            * 100000
        )
        net_region_rates["net_rate_total_lower"] = (
            net_region_rates["net_deaths_total_lower"]
            / net_region_rates["population_total"]
            * 100000
        )
        net_region_rates["net_rate_total_upper"] = (
            net_region_rates["net_deaths_total_upper"]
            / net_region_rates["population_total"]
            * 100000
        )
        net_region_rate_cols = ["year", "UN_Region", "N_Grids", "population_total"]
        net_region_rate_cols += [
            col for col in net_region_rates.columns if col.startswith("net_rate_")
        ]
        net_region_rates_output = net_region_rates[net_region_rate_cols]
        all_net_region_rates.append(net_region_rates_output)
    region_rates_all = pd.concat(all_region_rates, ignore_index=True)
    net_region_rates_all = pd.concat(all_net_region_rates, ignore_index=True)
    grid_rates_all.to_csv(output_grid_rates, index=False)
    country_rates_all.to_csv(output_country_rates, index=False)
    region_rates_all.to_csv(output_region_rates, index=False)
    net_grid_rates_all.to_csv(output_net_grid_rates, index=False)
    net_country_rates_all.to_csv(output_net_country_rates, index=False)
    net_region_rates_all.to_csv(output_net_region_rates, index=False)
    time.time() - scenario_start
    return (
        grid_rates_all,
        country_rates_all,
        region_rates_all,
        net_grid_rates_all,
        net_country_rates_all,
        net_region_rates_all,
    )


total_start = time.time()
all_saved_files = []
for adaptation_name in ADAPTATION_SCENARIOS:
    death_suffix = f"_{adaptation_name}_2025-2080.csv"
    rate_suffix = f"_{adaptation_name}_2025-2080.csv"
    (
        grid_rates_ssp245,
        country_rates_ssp245,
        region_rates_ssp245,
        net_grid_rates_ssp245,
        net_country_rates_ssp245,
        net_region_rates_ssp245,
    ) = process_scenario_rates(
        "SSP2-4.5",
        desktop_path / f"Projection_SSP245_Grid_Deaths{death_suffix}",
        desktop_path / f"Projection_SSP245_Country_Deaths{death_suffix}",
        desktop_path / f"Projection_SSP245_Region_Deaths{death_suffix}",
        desktop_path / f"Projection_SSP245_Grid_Net_Deaths{death_suffix}",
        desktop_path / f"Projection_SSP245_Country_Net_Deaths{death_suffix}",
        desktop_path / f"Projection_SSP245_Region_Net_Deaths{death_suffix}",
        desktop_path / "Europe_Grid_SSP2_Median_Fert_Age_Pop_2025_2080.csv",
        desktop_path / "UN_Geoscheme_Classification.csv",
        desktop_path / f"Projection_SSP245_Grid_Rates{rate_suffix}",
        desktop_path / f"Projection_SSP245_Country_Rates{rate_suffix}",
        desktop_path / f"Projection_SSP245_Region_Rates{rate_suffix}",
        desktop_path / f"Projection_SSP245_Grid_Net_Rates{rate_suffix}",
        desktop_path / f"Projection_SSP245_Country_Net_Rates{rate_suffix}",
        desktop_path / f"Projection_SSP245_Region_Net_Rates{rate_suffix}",
        adaptation_name=adaptation_name,
    )
    (
        grid_rates_ssp585,
        country_rates_ssp585,
        region_rates_ssp585,
        net_grid_rates_ssp585,
        net_country_rates_ssp585,
        net_region_rates_ssp585,
    ) = process_scenario_rates(
        "SSP5-8.5",
        desktop_path / f"Projection_SSP585_Grid_Deaths{death_suffix}",
        desktop_path / f"Projection_SSP585_Country_Deaths{death_suffix}",
        desktop_path / f"Projection_SSP585_Region_Deaths{death_suffix}",
        desktop_path / f"Projection_SSP585_Grid_Net_Deaths{death_suffix}",
        desktop_path / f"Projection_SSP585_Country_Net_Deaths{death_suffix}",
        desktop_path / f"Projection_SSP585_Region_Net_Deaths{death_suffix}",
        desktop_path / "Europe_Grid_SSP5_Median_Fert_Age_Pop_2025_2080.csv",
        desktop_path / "UN_Geoscheme_Classification.csv",
        desktop_path / f"Projection_SSP585_Grid_Rates{rate_suffix}",
        desktop_path / f"Projection_SSP585_Country_Rates{rate_suffix}",
        desktop_path / f"Projection_SSP585_Region_Rates{rate_suffix}",
        desktop_path / f"Projection_SSP585_Grid_Net_Rates{rate_suffix}",
        desktop_path / f"Projection_SSP585_Country_Net_Rates{rate_suffix}",
        desktop_path / f"Projection_SSP585_Region_Net_Rates{rate_suffix}",
        adaptation_name=adaptation_name,
    )
    all_saved_files.extend(
        [
            f"Projection_SSP245_Grid_Rates{rate_suffix}",
            f"Projection_SSP245_Country_Rates{rate_suffix}",
            f"Projection_SSP245_Region_Rates{rate_suffix}",
            f"Projection_SSP245_Grid_Net_Rates{rate_suffix}",
            f"Projection_SSP245_Country_Net_Rates{rate_suffix}",
            f"Projection_SSP245_Region_Net_Rates{rate_suffix}",
            f"Projection_SSP585_Grid_Rates{rate_suffix}",
            f"Projection_SSP585_Country_Rates{rate_suffix}",
            f"Projection_SSP585_Region_Rates{rate_suffix}",
            f"Projection_SSP585_Grid_Net_Rates{rate_suffix}",
            f"Projection_SSP585_Country_Net_Rates{rate_suffix}",
            f"Projection_SSP585_Region_Net_Rates{rate_suffix}",
        ]
    )
total_time = time.time() - total_start
for fname in all_saved_files:
    pass
