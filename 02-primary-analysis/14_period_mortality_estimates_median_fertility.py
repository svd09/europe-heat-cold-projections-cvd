import pandas as pd
import numpy as np
from pathlib import Path

DESKTOP = Path("data")
UN_GEOSCHEME = DESKTOP / "UN_Geoscheme_Classification.csv"
SSP2_POP = DESKTOP / "Europe_Grid_SSP2_Median_Fert_Age_Pop_2025_2080.csv"
SSP5_POP = DESKTOP / "Europe_Grid_SSP5_Median_Fert_Age_Pop_2025_2080.csv"
SSP245_DEATHS = DESKTOP / "Projection_SSP245_Grid_Deaths_2025-2080.csv"
SSP585_DEATHS = DESKTOP / "Projection_SSP585_Grid_Deaths_2025-2080.csv"
MID_CENTURY_START = 2046
MID_CENTURY_END = 2055
LATE_CENTURY_START = 2071
LATE_CENTURY_END = 2080
AGE_GROUPS = ["under_20", "20_54", "55_64", "65_74", "75plus"]
un_regions = pd.read_csv(UN_GEOSCHEME)
temp_df = pd.read_csv(SSP245_DEATHS)
grid_country = temp_df[["grid_id", "Country"]].drop_duplicates()
grid_country.columns = ["Grid_ID", "Country"]
grid_geography = grid_country.merge(un_regions, on="Country", how="left")


def reshape_population_long(pop_wide):
    pop_wide = pop_wide.copy()
    age_groups_list = ["under_20", "20_54", "55_64", "65_74", "75plus"]
    inf_fixed = 0
    for age in age_groups_list:
        age_cols = [
            col
            for col in pop_wide.columns
            if col.startswith(f"pop_{age}_") and col.split("_")[-1].isdigit()
        ]
        age_cols_sorted = sorted(age_cols, key=lambda x: int(x.split("_")[-1]))
        for idx in range(len(pop_wide)):
            last_valid = None
            for col in age_cols_sorted:
                val = pop_wide.at[idx, col]
                if (
                    isinstance(val, (int, float))
                    and (not np.isinf(val))
                    and (not np.isnan(val))
                ):
                    last_valid = val
                elif last_valid is not None and (np.isinf(val) or np.isnan(val)):
                    pop_wide.at[idx, col] = last_valid
                    inf_fixed += 1
    if inf_fixed > 0:
        pass
    remaining_inf = 0
    for age in age_groups_list:
        age_cols = [
            col
            for col in pop_wide.columns
            if col.startswith(f"pop_{age}_") and col.split("_")[-1].isdigit()
        ]
        for col in age_cols:
            remaining_inf += np.isinf(pop_wide[col]).sum()
    if remaining_inf > 0:
        for age in age_groups_list:
            age_cols = [
                col
                for col in pop_wide.columns
                if col.startswith(f"pop_{age}_") and col.split("_")[-1].isdigit()
            ]
            for col in age_cols:
                if np.isinf(pop_wide[col]).any():
                    inf_mask = np.isinf(pop_wide[col])
                    valid_values = pop_wide.loc[~inf_mask, col]
                    if len(valid_values) > 0:
                        median_val = valid_values.median()
                        pop_wide.loc[inf_mask, col] = median_val
                        inf_fixed += inf_mask.sum()
        final_inf = 0
        for age in age_groups_list:
            age_cols = [
                col
                for col in pop_wide.columns
                if col.startswith(f"pop_{age}_") and col.split("_")[-1].isdigit()
            ]
            for col in age_cols:
                final_inf += np.isinf(pop_wide[col]).sum()
        if final_inf == 0:
            pass
        else:
            pass
    else:
        pass
    pop_long_rows = []
    for _, row in pop_wide.iterrows():
        grid_id = row["grid_id"]
        country = row["Country"]
        for col in pop_wide.columns:
            if col.startswith("pop_"):
                parts = col.split("_")
                year = int(parts[-1])
                age_group = "_".join(parts[1:-1])
                pop_long_rows.append(
                    {
                        "Grid_ID": grid_id,
                        "Country": country,
                        "Year": year,
                        "Age_Group": age_group,
                        "Population": row[col],
                    }
                )
    return pd.DataFrame(pop_long_rows)


ssp2_wide = pd.read_csv(SSP2_POP)
ssp2_pop = reshape_population_long(ssp2_wide)
ssp5_wide = pd.read_csv(SSP5_POP)
ssp5_pop = reshape_population_long(ssp5_wide)


def reshape_deaths_long(df):
    df = df.copy()
    df.rename(columns={"grid_id": "Grid_ID", "year": "Year"}, inplace=True)
    rows = []
    for _, row in df.iterrows():
        grid_id = row["Grid_ID"]
        year = row["Year"]
        country = row["Country"]
        for age in AGE_GROUPS:
            heat_mean_col = f"heat_deaths_{age}_mean"
            heat_lower_col = f"heat_deaths_{age}_lower"
            heat_upper_col = f"heat_deaths_{age}_upper"
            if heat_mean_col in row.index:
                rows.append(
                    {
                        "Grid_ID": grid_id,
                        "Year": year,
                        "Country": country,
                        "Age_Group": age,
                        "Temperature_Type": "Heat",
                        "Deaths_Mean": row[heat_mean_col],
                        "Deaths_Lower_CI": row[heat_lower_col],
                        "Deaths_Upper_CI": row[heat_upper_col],
                    }
                )
            cold_mean_col = f"cold_deaths_{age}_mean"
            cold_lower_col = f"cold_deaths_{age}_lower"
            cold_upper_col = f"cold_deaths_{age}_upper"
            if cold_mean_col in row.index:
                rows.append(
                    {
                        "Grid_ID": grid_id,
                        "Year": year,
                        "Country": country,
                        "Age_Group": age,
                        "Temperature_Type": "Cold",
                        "Deaths_Mean": row[cold_mean_col],
                        "Deaths_Lower_CI": row[cold_lower_col],
                        "Deaths_Upper_CI": row[cold_upper_col],
                    }
                )
    return pd.DataFrame(rows)


def calculate_period_stats(df_long, pop_df, period_start, period_end, scenario):
    period_years = list(range(period_start, period_end + 1))
    n_years = len(period_years)
    df_period = df_long[df_long["Year"].isin(period_years)].copy()
    pop_period = pop_df[pop_df["Year"].isin(period_years)].copy()
    grid_age_temp_deaths = (
        df_period.groupby(["Grid_ID", "Country", "Age_Group", "Temperature_Type"])
        .agg({"Deaths_Mean": "sum", "Deaths_Lower_CI": "sum", "Deaths_Upper_CI": "sum"})
        .reset_index()
    )
    grid_age_temp_deaths.rename(
        columns={
            "Deaths_Mean": "CVD_Deaths_Mean",
            "Deaths_Lower_CI": "CVD_Deaths_Lower_CI",
            "Deaths_Upper_CI": "CVD_Deaths_Upper_CI",
        },
        inplace=True,
    )
    grid_age_temp_deaths["Annual_Deaths_Mean"] = (
        grid_age_temp_deaths["CVD_Deaths_Mean"] / n_years
    )
    grid_age_temp_deaths["Annual_Deaths_Lower_CI"] = (
        grid_age_temp_deaths["CVD_Deaths_Lower_CI"] / n_years
    )
    grid_age_temp_deaths["Annual_Deaths_Upper_CI"] = (
        grid_age_temp_deaths["CVD_Deaths_Upper_CI"] / n_years
    )
    grid_age_pop = (
        pop_period.groupby(["Grid_ID", "Age_Group"])["Population"].mean().reset_index()
    )
    grid_age_pop.rename(columns={"Population": "Avg_Population"}, inplace=True)
    inf_in_avg = np.isinf(grid_age_pop["Avg_Population"]).sum()
    if inf_in_avg > 0:
        pop_max = (
            pop_period.groupby(["Grid_ID", "Age_Group"])["Population"]
            .max()
            .reset_index()
        )
        pop_max.rename(columns={"Population": "Max_Population"}, inplace=True)
        grid_age_pop = grid_age_pop.merge(
            pop_max, on=["Grid_ID", "Age_Group"], how="left"
        )
        grid_age_pop["Avg_Population"] = np.where(
            np.isinf(grid_age_pop["Avg_Population"]),
            grid_age_pop["Max_Population"],
            grid_age_pop["Avg_Population"],
        )
        grid_age_pop = grid_age_pop.drop("Max_Population", axis=1)
        inf_after_fix = np.isinf(grid_age_pop["Avg_Population"]).sum()
        if inf_after_fix > 0:
            pass
        else:
            pass
    pop_max = (
        pop_period.groupby(["Grid_ID", "Age_Group"])["Population"].max().reset_index()
    )
    pop_max.rename(columns={"Population": "Max_Population"}, inplace=True)
    grid_age_pop = grid_age_pop.merge(pop_max, on=["Grid_ID", "Age_Group"], how="left")
    grid_age_pop["Avg_Population"] = np.where(
        grid_age_pop["Avg_Population"] < 1,
        grid_age_pop["Max_Population"],
        grid_age_pop["Avg_Population"],
    )
    grid_age_pop = grid_age_pop.drop("Max_Population", axis=1)
    grid_age_temp_stats = grid_age_temp_deaths.merge(
        grid_age_pop, on=["Grid_ID", "Age_Group"], how="left"
    )
    grid_age_temp_stats["Rate_per_100k_Mean"] = (
        grid_age_temp_stats["Annual_Deaths_Mean"]
        / grid_age_temp_stats["Avg_Population"]
        * 100000
    )
    grid_age_temp_stats["Rate_per_100k_Lower_CI"] = (
        grid_age_temp_stats["Annual_Deaths_Lower_CI"]
        / grid_age_temp_stats["Avg_Population"]
        * 100000
    )
    grid_age_temp_stats["Rate_per_100k_Upper_CI"] = (
        grid_age_temp_stats["Annual_Deaths_Upper_CI"]
        / grid_age_temp_stats["Avg_Population"]
        * 100000
    )
    grid_age_temp_stats = grid_age_temp_stats.copy()
    grid_age_temp_stats["Rate_per_100k_Mean"] = (
        grid_age_temp_stats["Rate_per_100k_Mean"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    grid_age_temp_stats["Rate_per_100k_Lower_CI"] = (
        grid_age_temp_stats["Rate_per_100k_Lower_CI"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    grid_age_temp_stats["Rate_per_100k_Upper_CI"] = (
        grid_age_temp_stats["Rate_per_100k_Upper_CI"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    grid_age_temp_stats["Period"] = f"{period_start}-{period_end}"
    grid_age_temp_stats["Scenario"] = scenario
    heat_data = grid_age_temp_stats[
        grid_age_temp_stats["Temperature_Type"] == "Heat"
    ].copy()
    cold_data = grid_age_temp_stats[
        grid_age_temp_stats["Temperature_Type"] == "Cold"
    ].copy()
    total_data = heat_data.merge(
        cold_data,
        on=["Grid_ID", "Country", "Age_Group", "Avg_Population", "Period", "Scenario"],
        suffixes=("_heat", "_cold"),
    )
    total_data["CVD_Deaths_Mean"] = (
        total_data["CVD_Deaths_Mean_heat"] + total_data["CVD_Deaths_Mean_cold"]
    )
    total_data["CVD_Deaths_Lower_CI"] = (
        total_data["CVD_Deaths_Lower_CI_heat"] + total_data["CVD_Deaths_Lower_CI_cold"]
    )
    total_data["CVD_Deaths_Upper_CI"] = (
        total_data["CVD_Deaths_Upper_CI_heat"] + total_data["CVD_Deaths_Upper_CI_cold"]
    )
    total_data["Annual_Deaths_Mean"] = (
        total_data["Annual_Deaths_Mean_heat"] + total_data["Annual_Deaths_Mean_cold"]
    )
    total_data["Annual_Deaths_Lower_CI"] = (
        total_data["Annual_Deaths_Lower_CI_heat"]
        + total_data["Annual_Deaths_Lower_CI_cold"]
    )
    total_data["Annual_Deaths_Upper_CI"] = (
        total_data["Annual_Deaths_Upper_CI_heat"]
        + total_data["Annual_Deaths_Upper_CI_cold"]
    )
    total_data["Rate_per_100k_Mean"] = (
        total_data["Annual_Deaths_Mean"] / total_data["Avg_Population"] * 100000
    )
    total_data["Rate_per_100k_Lower_CI"] = (
        total_data["Annual_Deaths_Lower_CI"] / total_data["Avg_Population"] * 100000
    )
    total_data["Rate_per_100k_Upper_CI"] = (
        total_data["Annual_Deaths_Upper_CI"] / total_data["Avg_Population"] * 100000
    )
    total_data = total_data.copy()
    total_data["Rate_per_100k_Mean"] = total_data["Rate_per_100k_Mean"].replace(
        [np.inf, -np.inf], 0
    )
    total_data["Rate_per_100k_Lower_CI"] = total_data["Rate_per_100k_Lower_CI"].replace(
        [np.inf, -np.inf], 0
    )
    total_data["Rate_per_100k_Upper_CI"] = total_data["Rate_per_100k_Upper_CI"].replace(
        [np.inf, -np.inf], 0
    )
    total_data["Temperature_Type"] = "Total"
    total_data = total_data[
        [
            "Grid_ID",
            "Country",
            "Age_Group",
            "Temperature_Type",
            "Period",
            "Scenario",
            "CVD_Deaths_Mean",
            "CVD_Deaths_Lower_CI",
            "CVD_Deaths_Upper_CI",
            "Annual_Deaths_Mean",
            "Annual_Deaths_Lower_CI",
            "Annual_Deaths_Upper_CI",
            "Avg_Population",
            "Rate_per_100k_Mean",
            "Rate_per_100k_Lower_CI",
            "Rate_per_100k_Upper_CI",
        ]
    ]
    all_data = pd.concat([grid_age_temp_stats, total_data], ignore_index=True)
    return all_data


def aggregate_to_all_ages(df):
    geo_cols = []
    if "Grid_ID" in df.columns:
        geo_cols.append("Grid_ID")
    if "Country" in df.columns:
        geo_cols.append("Country")
    if "UN_Region" in df.columns:
        geo_cols.append("UN_Region")
    if "Geography" in df.columns:
        geo_cols.append("Geography")
    group_cols = geo_cols + ["Temperature_Type", "Period", "Scenario"]
    all_ages = (
        df.groupby(group_cols)
        .agg(
            {
                "CVD_Deaths_Mean": "sum",
                "CVD_Deaths_Lower_CI": "sum",
                "CVD_Deaths_Upper_CI": "sum",
                "Annual_Deaths_Mean": "sum",
                "Annual_Deaths_Lower_CI": "sum",
                "Annual_Deaths_Upper_CI": "sum",
                "Avg_Population": "sum",
            }
        )
        .reset_index()
    )
    all_ages["Rate_per_100k_Mean"] = (
        all_ages["Annual_Deaths_Mean"] / all_ages["Avg_Population"] * 100000
    )
    all_ages["Rate_per_100k_Lower_CI"] = (
        all_ages["Annual_Deaths_Lower_CI"] / all_ages["Avg_Population"] * 100000
    )
    all_ages["Rate_per_100k_Upper_CI"] = (
        all_ages["Annual_Deaths_Upper_CI"] / all_ages["Avg_Population"] * 100000
    )
    all_ages["Rate_per_100k_Mean"] = all_ages["Rate_per_100k_Mean"].replace(
        [np.inf, -np.inf], 0
    )
    all_ages["Rate_per_100k_Lower_CI"] = all_ages["Rate_per_100k_Lower_CI"].replace(
        [np.inf, -np.inf], 0
    )
    all_ages["Rate_per_100k_Upper_CI"] = all_ages["Rate_per_100k_Upper_CI"].replace(
        [np.inf, -np.inf], 0
    )
    all_ages["Age_Group"] = "All_Ages"
    return all_ages


def aggregate_to_country(grid_df, grid_geography):
    country_df = grid_df.merge(
        grid_geography[["Grid_ID", "Country", "UN_Region"]],
        on=["Grid_ID", "Country"],
        how="left",
    )
    group_cols = [
        "Country",
        "UN_Region",
        "Age_Group",
        "Temperature_Type",
        "Period",
        "Scenario",
    ]
    country_agg = (
        country_df.groupby(group_cols)
        .agg(
            {
                "CVD_Deaths_Mean": "sum",
                "CVD_Deaths_Lower_CI": "sum",
                "CVD_Deaths_Upper_CI": "sum",
                "Annual_Deaths_Mean": "sum",
                "Annual_Deaths_Lower_CI": "sum",
                "Annual_Deaths_Upper_CI": "sum",
                "Avg_Population": "sum",
            }
        )
        .reset_index()
    )
    country_agg["Rate_per_100k_Mean"] = (
        country_agg["Annual_Deaths_Mean"] / country_agg["Avg_Population"] * 100000
    )
    country_agg["Rate_per_100k_Lower_CI"] = (
        country_agg["Annual_Deaths_Lower_CI"] / country_agg["Avg_Population"] * 100000
    )
    country_agg["Rate_per_100k_Upper_CI"] = (
        country_agg["Annual_Deaths_Upper_CI"] / country_agg["Avg_Population"] * 100000
    )
    country_agg["Rate_per_100k_Mean"] = country_agg["Rate_per_100k_Mean"].replace(
        [np.inf, -np.inf], 0
    )
    country_agg["Rate_per_100k_Lower_CI"] = country_agg[
        "Rate_per_100k_Lower_CI"
    ].replace([np.inf, -np.inf], 0)
    country_agg["Rate_per_100k_Upper_CI"] = country_agg[
        "Rate_per_100k_Upper_CI"
    ].replace([np.inf, -np.inf], 0)
    return country_agg


def aggregate_to_region(country_df):
    group_cols = ["UN_Region", "Age_Group", "Temperature_Type", "Period", "Scenario"]
    region_agg = (
        country_df.groupby(group_cols)
        .agg(
            {
                "CVD_Deaths_Mean": "sum",
                "CVD_Deaths_Lower_CI": "sum",
                "CVD_Deaths_Upper_CI": "sum",
                "Annual_Deaths_Mean": "sum",
                "Annual_Deaths_Lower_CI": "sum",
                "Annual_Deaths_Upper_CI": "sum",
                "Avg_Population": "sum",
            }
        )
        .reset_index()
    )
    region_agg["Rate_per_100k_Mean"] = (
        region_agg["Annual_Deaths_Mean"] / region_agg["Avg_Population"] * 100000
    )
    region_agg["Rate_per_100k_Lower_CI"] = (
        region_agg["Annual_Deaths_Lower_CI"] / region_agg["Avg_Population"] * 100000
    )
    region_agg["Rate_per_100k_Upper_CI"] = (
        region_agg["Annual_Deaths_Upper_CI"] / region_agg["Avg_Population"] * 100000
    )
    region_agg["Rate_per_100k_Mean"] = region_agg["Rate_per_100k_Mean"].replace(
        [np.inf, -np.inf], 0
    )
    region_agg["Rate_per_100k_Lower_CI"] = region_agg["Rate_per_100k_Lower_CI"].replace(
        [np.inf, -np.inf], 0
    )
    region_agg["Rate_per_100k_Upper_CI"] = region_agg["Rate_per_100k_Upper_CI"].replace(
        [np.inf, -np.inf], 0
    )
    return region_agg


def aggregate_to_europe(region_df):
    group_cols = ["Age_Group", "Temperature_Type", "Period", "Scenario"]
    europe_agg = (
        region_df.groupby(group_cols)
        .agg(
            {
                "CVD_Deaths_Mean": "sum",
                "CVD_Deaths_Lower_CI": "sum",
                "CVD_Deaths_Upper_CI": "sum",
                "Annual_Deaths_Mean": "sum",
                "Annual_Deaths_Lower_CI": "sum",
                "Annual_Deaths_Upper_CI": "sum",
                "Avg_Population": "sum",
            }
        )
        .reset_index()
    )
    europe_agg["Rate_per_100k_Mean"] = (
        europe_agg["Annual_Deaths_Mean"] / europe_agg["Avg_Population"] * 100000
    )
    europe_agg["Rate_per_100k_Lower_CI"] = (
        europe_agg["Annual_Deaths_Lower_CI"] / europe_agg["Avg_Population"] * 100000
    )
    europe_agg["Rate_per_100k_Upper_CI"] = (
        europe_agg["Annual_Deaths_Upper_CI"] / europe_agg["Avg_Population"] * 100000
    )
    europe_agg["Rate_per_100k_Mean"] = europe_agg["Rate_per_100k_Mean"].replace(
        [np.inf, -np.inf], 0
    )
    europe_agg["Rate_per_100k_Lower_CI"] = europe_agg["Rate_per_100k_Lower_CI"].replace(
        [np.inf, -np.inf], 0
    )
    europe_agg["Rate_per_100k_Upper_CI"] = europe_agg["Rate_per_100k_Upper_CI"].replace(
        [np.inf, -np.inf], 0
    )
    europe_agg["Geography"] = "Europe"
    return europe_agg


all_results = []
scenarios = [
    ("SSP2-4.5", SSP245_DEATHS, ssp2_pop),
    ("SSP5-8.5", SSP585_DEATHS, ssp5_pop),
]
periods = [
    ("Mid-Century", MID_CENTURY_START, MID_CENTURY_END),
    ("Late-Century", LATE_CENTURY_START, LATE_CENTURY_END),
]
for scenario_name, deaths_file, pop_df in scenarios:
    deaths_wide = pd.read_csv(deaths_file)
    deaths_long = reshape_deaths_long(deaths_wide)
    for period_name, period_start, period_end in periods:
        period_stats = calculate_period_stats(
            deaths_long, pop_df, period_start, period_end, scenario_name
        )
        all_results.append(period_stats)
grid_all = pd.concat(all_results, ignore_index=True)
grid_by_age = grid_all.copy()
grid_all_ages = aggregate_to_all_ages(grid_all)
grid_final = pd.concat([grid_by_age, grid_all_ages], ignore_index=True)
country_by_age = aggregate_to_country(grid_by_age, grid_geography)
country_all_ages = aggregate_to_all_ages(country_by_age)
country_final = pd.concat([country_by_age, country_all_ages], ignore_index=True)
region_by_age = aggregate_to_region(country_by_age)
region_all_ages = aggregate_to_all_ages(region_by_age)
region_final = pd.concat([region_by_age, region_all_ages], ignore_index=True)
europe_by_age = aggregate_to_europe(region_by_age)
europe_all_ages = aggregate_to_all_ages(europe_by_age)
europe_final = pd.concat([europe_by_age, europe_all_ages], ignore_index=True)
grid_output = DESKTOP / "Period_Estimates_Grid_Level.csv"
grid_final.to_csv(grid_output, index=False)
country_output = DESKTOP / "Period_Estimates_Country_Level.csv"
country_final.to_csv(country_output, index=False)
region_output = DESKTOP / "Period_Estimates_Regional_Level.csv"
region_final.to_csv(region_output, index=False)
europe_output = DESKTOP / "Period_Estimates_Europe_Wide.csv"
europe_final.to_csv(europe_output, index=False)
summary_data = []
for scenario in ["SSP2-4.5", "SSP5-8.5"]:
    for period in ["2046-2055", "2071-2080"]:
        for temp_type in ["Heat", "Cold", "Total"]:
            subset = europe_final[
                (europe_final["Scenario"] == scenario)
                & (europe_final["Period"] == period)
                & (europe_final["Temperature_Type"] == temp_type)
                & (europe_final["Age_Group"] == "All_Ages")
            ]
            if len(subset) > 0:
                row = subset.iloc[0]
                summary_data.append(
                    {
                        "Scenario": scenario,
                        "Period": period,
                        "Temp_Type": temp_type,
                        "Total_Deaths": f"{row['CVD_Deaths_Mean']:,.0f}",
                        "95%_CI": f"({row['CVD_Deaths_Lower_CI']:,.0f}-{row['CVD_Deaths_Upper_CI']:,.0f})",
                        "Annual_Avg": f"{row['Annual_Deaths_Mean']:,.0f}",
                        "Rate_per_100k": f"{row['Rate_per_100k_Mean']:.2f}",
                    }
                )
summary_df = pd.DataFrame(summary_data)
