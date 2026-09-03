import pandas as pd
import numpy as np
from pathlib import Path

DESKTOP = Path("data")
INPUT_DIR = DESKTOP / "Model_Mortality_Results_Aggregated"
OUTPUT_DIR = DESKTOP / "Model_Period_Estimates"
OUTPUT_DIR.mkdir(exist_ok=True)
UN_GEOSCHEME = DESKTOP / "UN_Geoscheme_Classification.csv"
SSP2_POP = DESKTOP / "Europe_Grid_SSP2_Median_Fert_Age_Pop_2025_2080.csv"
SSP5_POP = DESKTOP / "Europe_Grid_SSP5_Median_Fert_Age_Pop_2025_2080.csv"
MODELS = ["CNRM-ESM2-1", "GFDL-ESM4", "MIROC6", "NorESM2-MM", "UKESM1-0-LL"]
SCENARIOS = ["SSP245", "SSP585"]
SCENARIO_LABELS = {"SSP245": "SSP2-4.5", "SSP585": "SSP5-8.5"}
MID_CENTURY_START = 2046
MID_CENTURY_END = 2055
LATE_CENTURY_START = 2071
LATE_CENTURY_END = 2080
AGE_GROUPS = ["under_20", "20_54", "55_64", "65_74", "75plus"]
un_regions = pd.read_csv(UN_GEOSCHEME)


def fix_inf_population(pop_wide):
    pop_wide = pop_wide.copy()
    age_groups_list = ["under_20", "20_54", "55_64", "65_74", "75plus"]
    inf_fixed = 0
    for age in age_groups_list:
        age_cols = sorted(
            [
                c
                for c in pop_wide.columns
                if c.startswith(f"pop_{age}_") and c.split("_")[-1].isdigit()
            ],
            key=lambda x: int(x.split("_")[-1]),
        )
        for idx in range(len(pop_wide)):
            last_valid = None
            for col in age_cols:
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
    remaining = sum(
        (np.isinf(pop_wide[c]).sum() for c in pop_wide.columns if c.startswith("pop_"))
    )
    if remaining > 0:
        for col in pop_wide.columns:
            if col.startswith("pop_"):
                mask = np.isinf(pop_wide[col])
                if mask.any():
                    pop_wide.loc[mask, col] = pop_wide.loc[~mask, col].median()
    return pop_wide


def reshape_population_long(pop_wide):
    pop_wide = fix_inf_population(pop_wide)
    rows = []
    for _, row in pop_wide.iterrows():
        grid_id = row["grid_id"]
        country = row["Country"]
        for col in pop_wide.columns:
            if col.startswith("pop_"):
                parts = col.split("_")
                year = int(parts[-1])
                age_group = "_".join(parts[1:-1])
                rows.append(
                    {
                        "Grid_ID": grid_id,
                        "Country": country,
                        "Year": year,
                        "Age_Group": age_group,
                        "Population": row[col],
                    }
                )
    return pd.DataFrame(rows)


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
            hm = f"heat_deaths_{age}_mean"
            hl = f"heat_deaths_{age}_lower"
            hu = f"heat_deaths_{age}_upper"
            if hm in row.index:
                rows.append(
                    {
                        "Grid_ID": grid_id,
                        "Year": year,
                        "Country": country,
                        "Age_Group": age,
                        "Temperature_Type": "Heat",
                        "Deaths_Mean": row[hm],
                        "Deaths_Lower_CI": row[hl],
                        "Deaths_Upper_CI": row[hu],
                    }
                )
            cm = f"cold_deaths_{age}_mean"
            cl = f"cold_deaths_{age}_lower"
            cu = f"cold_deaths_{age}_upper"
            if cm in row.index:
                rows.append(
                    {
                        "Grid_ID": grid_id,
                        "Year": year,
                        "Country": country,
                        "Age_Group": age,
                        "Temperature_Type": "Cold",
                        "Deaths_Mean": row[cm],
                        "Deaths_Lower_CI": row[cl],
                        "Deaths_Upper_CI": row[cu],
                    }
                )
    return pd.DataFrame(rows)


def calculate_period_stats(
    df_long, pop_df, period_start, period_end, scenario_label, model
):
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
    for stat, deaths_col in [
        ("Mean", "Annual_Deaths_Mean"),
        ("Lower_CI", "Annual_Deaths_Lower_CI"),
        ("Upper_CI", "Annual_Deaths_Upper_CI"),
    ]:
        grid_age_temp_stats[f"Rate_per_100k_{stat}"] = (
            (
                grid_age_temp_stats[deaths_col]
                / grid_age_temp_stats["Avg_Population"]
                * 100000
            )
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )
    grid_age_temp_stats["Period"] = f"{period_start}-{period_end}"
    grid_age_temp_stats["Scenario"] = scenario_label
    grid_age_temp_stats["Model"] = model
    heat_data = grid_age_temp_stats[
        grid_age_temp_stats["Temperature_Type"] == "Heat"
    ].copy()
    cold_data = grid_age_temp_stats[
        grid_age_temp_stats["Temperature_Type"] == "Cold"
    ].copy()
    total_data = heat_data.merge(
        cold_data,
        on=[
            "Grid_ID",
            "Country",
            "Age_Group",
            "Avg_Population",
            "Period",
            "Scenario",
            "Model",
        ],
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
    for stat, deaths_col in [
        ("Mean", "Annual_Deaths_Mean"),
        ("Lower_CI", "Annual_Deaths_Lower_CI"),
        ("Upper_CI", "Annual_Deaths_Upper_CI"),
    ]:
        total_data[f"Rate_per_100k_{stat}"] = (
            total_data[deaths_col] / total_data["Avg_Population"] * 100000
        ).replace([np.inf, -np.inf], 0)
    total_data["Temperature_Type"] = "Total"
    total_data = total_data[
        [
            "Grid_ID",
            "Country",
            "Age_Group",
            "Temperature_Type",
            "Period",
            "Scenario",
            "Model",
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
    return pd.concat([grid_age_temp_stats, total_data], ignore_index=True)


def aggregate_to_all_ages(df):
    geo_cols = [
        c for c in ["Grid_ID", "Country", "UN_Region", "Geography"] if c in df.columns
    ]
    group_cols = geo_cols + ["Temperature_Type", "Period", "Scenario", "Model"]
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
    for stat, deaths_col in [
        ("Mean", "Annual_Deaths_Mean"),
        ("Lower_CI", "Annual_Deaths_Lower_CI"),
        ("Upper_CI", "Annual_Deaths_Upper_CI"),
    ]:
        all_ages[f"Rate_per_100k_{stat}"] = (
            all_ages[deaths_col] / all_ages["Avg_Population"] * 100000
        ).replace([np.inf, -np.inf], 0)
    all_ages["Age_Group"] = "All_Ages"
    return all_ages


def aggregate_to_country(grid_df, grid_geography):
    if "UN_Region" in grid_df.columns:
        country_df = grid_df.copy()
    else:
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
        "Model",
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
    for stat, deaths_col in [
        ("Mean", "Annual_Deaths_Mean"),
        ("Lower_CI", "Annual_Deaths_Lower_CI"),
        ("Upper_CI", "Annual_Deaths_Upper_CI"),
    ]:
        country_agg[f"Rate_per_100k_{stat}"] = (
            country_agg[deaths_col] / country_agg["Avg_Population"] * 100000
        ).replace([np.inf, -np.inf], 0)
    return country_agg


def aggregate_to_region(country_df):
    group_cols = [
        "UN_Region",
        "Age_Group",
        "Temperature_Type",
        "Period",
        "Scenario",
        "Model",
    ]
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
    for stat, deaths_col in [
        ("Mean", "Annual_Deaths_Mean"),
        ("Lower_CI", "Annual_Deaths_Lower_CI"),
        ("Upper_CI", "Annual_Deaths_Upper_CI"),
    ]:
        region_agg[f"Rate_per_100k_{stat}"] = (
            region_agg[deaths_col] / region_agg["Avg_Population"] * 100000
        ).replace([np.inf, -np.inf], 0)
    return region_agg


def aggregate_to_europe(region_df):
    group_cols = ["Age_Group", "Temperature_Type", "Period", "Scenario", "Model"]
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
    for stat, deaths_col in [
        ("Mean", "Annual_Deaths_Mean"),
        ("Lower_CI", "Annual_Deaths_Lower_CI"),
        ("Upper_CI", "Annual_Deaths_Upper_CI"),
    ]:
        europe_agg[f"Rate_per_100k_{stat}"] = (
            europe_agg[deaths_col] / europe_agg["Avg_Population"] * 100000
        ).replace([np.inf, -np.inf], 0)
    europe_agg["Geography"] = "Europe"
    return europe_agg


all_results = []
n_combos = len(MODELS) * len(SCENARIOS)
combo_num = 0
periods = [
    ("Mid-Century", MID_CENTURY_START, MID_CENTURY_END),
    ("Late-Century", LATE_CENTURY_START, LATE_CENTURY_END),
]
grid_geography = None
for model in MODELS:
    for scenario in SCENARIOS:
        combo_num += 1
        scenario_label = SCENARIO_LABELS[scenario]
        pop_df = ssp2_pop if scenario == "SSP245" else ssp5_pop
        deaths_file = INPUT_DIR / f"{model}_{scenario}_Grid_Deaths.csv"
        if not deaths_file.exists():
            continue
        deaths_wide = pd.read_csv(deaths_file)
        if grid_geography is None:
            if "UN_Region" in deaths_wide.columns:
                grid_geography = (
                    deaths_wide[["grid_id", "Country", "UN_Region"]]
                    .drop_duplicates()
                    .rename(columns={"grid_id": "Grid_ID"})
                )
            else:
                grid_country = deaths_wide[["grid_id", "Country"]].drop_duplicates()
                grid_country.columns = ["Grid_ID", "Country"]
                grid_geography = grid_country.merge(
                    un_regions, on="Country", how="left"
                )
        target_years = list(range(MID_CENTURY_START, MID_CENTURY_END + 1)) + list(
            range(LATE_CENTURY_START, LATE_CENTURY_END + 1)
        )
        deaths_wide = deaths_wide[deaths_wide["year"].isin(target_years)].copy()
        deaths_long = reshape_deaths_long(deaths_wide)
        for period_name, period_start, period_end in periods:
            period_stats = calculate_period_stats(
                deaths_long, pop_df, period_start, period_end, scenario_label, model
            )
            all_results.append(period_stats)
grid_all = pd.concat(all_results, ignore_index=True)
grid_all_ages = aggregate_to_all_ages(grid_all)
grid_final = pd.concat([grid_all, grid_all_ages], ignore_index=True)
country_by_age = aggregate_to_country(grid_all, grid_geography)
country_all_ages = aggregate_to_all_ages(country_by_age)
country_final = pd.concat([country_by_age, country_all_ages], ignore_index=True)
region_by_age = aggregate_to_region(country_by_age)
region_all_ages = aggregate_to_all_ages(region_by_age)
region_final = pd.concat([region_by_age, region_all_ages], ignore_index=True)
europe_by_age = aggregate_to_europe(region_by_age)
europe_all_ages = aggregate_to_all_ages(europe_by_age)
europe_final = pd.concat([europe_by_age, europe_all_ages], ignore_index=True)
outputs = {
    "Period_Estimates_Grid_Level.csv": grid_final,
    "Period_Estimates_Country_Level.csv": country_final,
    "Period_Estimates_Regional_Level.csv": region_final,
    "Period_Estimates_Europe_Wide.csv": europe_final,
}
for filename, df in outputs.items():
    fp = OUTPUT_DIR / filename
    df.to_csv(fp, index=False)
summary_rows = []
for model in MODELS:
    for scenario_label in SCENARIO_LABELS.values():
        for period_label in ["2046-2055", "2071-2080"]:
            for temp_type in ["Heat", "Cold", "Total"]:
                subset = europe_final[
                    (europe_final["Model"] == model)
                    & (europe_final["Scenario"] == scenario_label)
                    & (europe_final["Period"] == period_label)
                    & (europe_final["Temperature_Type"] == temp_type)
                    & (europe_final["Age_Group"] == "All_Ages")
                ]
                if len(subset) > 0:
                    r = subset.iloc[0]
                    summary_rows.append(
                        {
                            "Model": model,
                            "Scenario": scenario_label,
                            "Period": period_label,
                            "Temp_Type": temp_type,
                            "Annual_Avg": f"{r['Annual_Deaths_Mean']:,.0f}",
                            "95%_CI": f"({r['Annual_Deaths_Lower_CI']:,.0f}–{r['Annual_Deaths_Upper_CI']:,.0f})",
                            "Rate_per_100k": f"{r['Rate_per_100k_Mean']:.2f}",
                        }
                    )
summary_df = pd.DataFrame(summary_rows)
for filename in outputs:
    pass
