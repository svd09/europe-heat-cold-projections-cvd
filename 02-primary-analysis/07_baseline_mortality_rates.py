import pandas as pd
from pathlib import Path

desktop_path = Path("data")
deaths_grid_file = desktop_path / "Baseline_Grid_Attributable_Deaths_by_Age.csv"
deaths_country_file = desktop_path / "Baseline_Country_Attributable_Deaths.csv"
deaths_region_file = desktop_path / "Baseline_Regional_Attributable_Deaths.csv"
deaths_net_grid_file = desktop_path / "Baseline_Grid_Net_Temperature_Deaths.csv"
deaths_net_country_file = desktop_path / "Baseline_Country_Net_Temperature_Deaths.csv"
deaths_net_region_file = desktop_path / "Baseline_Regional_Net_Temperature_Deaths.csv"
population_file = desktop_path / "Europe_Grid_with_Countries_and_Age_Populations.csv"
un_regions_file = desktop_path / "UN_Geoscheme_Classification.csv"
output_grid_rates = desktop_path / "Baseline_Grid_Mortality_Rates_per_100k.csv"
output_country_rates = desktop_path / "Baseline_Country_Mortality_Rates_per_100k.csv"
output_region_rates = desktop_path / "Baseline_Region_Mortality_Rates_per_100k.csv"
output_net_grid_rates = desktop_path / "Baseline_Grid_Net_Rates_per_100k.csv"
output_net_country_rates = desktop_path / "Baseline_Country_Net_Rates_per_100k.csv"
output_net_region_rates = desktop_path / "Baseline_Region_Net_Rates_per_100k.csv"
deaths_grid = pd.read_csv(deaths_grid_file)
deaths_country = pd.read_csv(deaths_country_file)
deaths_region = pd.read_csv(deaths_region_file)
deaths_net_grid = pd.read_csv(deaths_net_grid_file)
deaths_net_country = pd.read_csv(deaths_net_country_file)
deaths_net_region = pd.read_csv(deaths_net_region_file)
pop = pd.read_csv(population_file)
un_regions = pd.read_csv(un_regions_file)
pop_cols_expected = {
    "population_2020": "total",
    "pop_under_20": "under_20",
    "pop_20_54": "20_54",
    "pop_55_64": "55_64",
    "pop_65_74": "65_74",
    "pop_75plus": "75plus",
}
for col, age in pop_cols_expected.items():
    if col in pop.columns:
        pass
    else:
        pass
grid_merged = deaths_grid.merge(
    pop[["grid_id", "Country"] + list(pop_cols_expected.keys())],
    on=["grid_id", "Country"],
    how="inner",
)
net_grid_merged = deaths_net_grid.merge(
    pop[["grid_id", "Country"] + list(pop_cols_expected.keys())],
    on=["grid_id", "Country"],
    how="inner",
)
grid_rates = grid_merged.copy()
age_groups = {
    "under_20": "pop_under_20",
    "20_54": "pop_20_54",
    "55_64": "pop_55_64",
    "65_74": "pop_65_74",
    "75plus": "pop_75plus",
}
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
    grid_rates["heat_deaths_total_mean"] / grid_rates["population_2020"] * 100000
)
grid_rates["heat_rate_total_lower"] = (
    grid_rates["heat_deaths_total_lower"] / grid_rates["population_2020"] * 100000
)
grid_rates["heat_rate_total_upper"] = (
    grid_rates["heat_deaths_total_upper"] / grid_rates["population_2020"] * 100000
)
grid_rates["cold_rate_total_mean"] = (
    grid_rates["cold_deaths_total_mean"] / grid_rates["population_2020"] * 100000
)
grid_rates["cold_rate_total_lower"] = (
    grid_rates["cold_deaths_total_lower"] / grid_rates["population_2020"] * 100000
)
grid_rates["cold_rate_total_upper"] = (
    grid_rates["cold_deaths_total_upper"] / grid_rates["population_2020"] * 100000
)
rate_cols = ["grid_id", "Country", "climate_zone", "population_2020"]
rate_cols += [
    col
    for col in grid_rates.columns
    if col.startswith("heat_rate_") or col.startswith("cold_rate_")
]
grid_rates_output = grid_rates[rate_cols]
net_grid_rates = net_grid_merged.copy()
for age, pop_col in age_groups.items():
    net_grid_rates[f"net_rate_{age}_mean"] = (
        net_grid_rates[f"net_deaths_{age}_mean"] / net_grid_rates[pop_col] * 100000
    )
    net_grid_rates[f"net_rate_{age}_lower"] = (
        net_grid_rates[f"net_deaths_{age}_lower"] / net_grid_rates[pop_col] * 100000
    )
    net_grid_rates[f"net_rate_{age}_upper"] = (
        net_grid_rates[f"net_deaths_{age}_upper"] / net_grid_rates[pop_col] * 100000
    )
net_grid_rates["net_rate_total_mean"] = (
    net_grid_rates["net_deaths_total_mean"] / net_grid_rates["population_2020"] * 100000
)
net_grid_rates["net_rate_total_lower"] = (
    net_grid_rates["net_deaths_total_lower"]
    / net_grid_rates["population_2020"]
    * 100000
)
net_grid_rates["net_rate_total_upper"] = (
    net_grid_rates["net_deaths_total_upper"]
    / net_grid_rates["population_2020"]
    * 100000
)
net_rate_cols = ["grid_id", "Country", "climate_zone", "population_2020"]
net_rate_cols += [col for col in net_grid_rates.columns if col.startswith("net_rate_")]
net_grid_rates_output = net_grid_rates[net_rate_cols]
country_pop = (
    pop.groupby("Country")
    .agg(
        {
            "population_2020": "sum",
            "pop_under_20": "sum",
            "pop_20_54": "sum",
            "pop_55_64": "sum",
            "pop_65_74": "sum",
            "pop_75plus": "sum",
        }
    )
    .reset_index()
)
country_merged = deaths_country.merge(country_pop, on="Country", how="left")
total_row_mask = country_merged["Country"] == "TOTAL"
if total_row_mask.any():
    total_pop = country_pop["population_2020"].sum()
    total_pop_ages = {
        age: country_pop[pop_col].sum() for age, pop_col in age_groups.items()
    }
    country_merged.loc[total_row_mask, "population_2020"] = total_pop
    for age, pop_col in age_groups.items():
        country_merged.loc[total_row_mask, pop_col] = total_pop_ages[age]
country_rates = country_merged.copy()
for age, pop_col in age_groups.items():
    country_rates[f"heat_rate_{age}_mean"] = (
        country_rates[f"heat_deaths_{age}_mean"] / country_rates[pop_col] * 100000
    )
    country_rates[f"heat_rate_{age}_lower"] = (
        country_rates[f"heat_deaths_{age}_lower"] / country_rates[pop_col] * 100000
    )
    country_rates[f"heat_rate_{age}_upper"] = (
        country_rates[f"heat_deaths_{age}_upper"] / country_rates[pop_col] * 100000
    )
    country_rates[f"cold_rate_{age}_mean"] = (
        country_rates[f"cold_deaths_{age}_mean"] / country_rates[pop_col] * 100000
    )
    country_rates[f"cold_rate_{age}_lower"] = (
        country_rates[f"cold_deaths_{age}_lower"] / country_rates[pop_col] * 100000
    )
    country_rates[f"cold_rate_{age}_upper"] = (
        country_rates[f"cold_deaths_{age}_upper"] / country_rates[pop_col] * 100000
    )
country_rates["heat_rate_total_mean"] = (
    country_rates["heat_deaths_total_mean"] / country_rates["population_2020"] * 100000
)
country_rates["heat_rate_total_lower"] = (
    country_rates["heat_deaths_total_lower"] / country_rates["population_2020"] * 100000
)
country_rates["heat_rate_total_upper"] = (
    country_rates["heat_deaths_total_upper"] / country_rates["population_2020"] * 100000
)
country_rates["cold_rate_total_mean"] = (
    country_rates["cold_deaths_total_mean"] / country_rates["population_2020"] * 100000
)
country_rates["cold_rate_total_lower"] = (
    country_rates["cold_deaths_total_lower"] / country_rates["population_2020"] * 100000
)
country_rates["cold_rate_total_upper"] = (
    country_rates["cold_deaths_total_upper"] / country_rates["population_2020"] * 100000
)
country_rate_cols = ["Country", "N_Grids", "population_2020"]
country_rate_cols += [
    col
    for col in country_rates.columns
    if col.startswith("heat_rate_") or col.startswith("cold_rate_")
]
country_rates_output = country_rates[country_rate_cols]
if total_row_mask.any():
    total_heat = country_rates.loc[total_row_mask, "heat_rate_total_mean"].iloc[0]
    total_cold = country_rates.loc[total_row_mask, "cold_rate_total_mean"].iloc[0]
net_country_merged = deaths_net_country.merge(country_pop, on="Country", how="left")
if (net_country_merged["Country"] == "TOTAL").any():
    net_country_merged.loc[
        net_country_merged["Country"] == "TOTAL", "population_2020"
    ] = total_pop
    for age, pop_col in age_groups.items():
        net_country_merged.loc[net_country_merged["Country"] == "TOTAL", pop_col] = (
            total_pop_ages[age]
        )
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
    / net_country_rates["population_2020"]
    * 100000
)
net_country_rates["net_rate_total_lower"] = (
    net_country_rates["net_deaths_total_lower"]
    / net_country_rates["population_2020"]
    * 100000
)
net_country_rates["net_rate_total_upper"] = (
    net_country_rates["net_deaths_total_upper"]
    / net_country_rates["population_2020"]
    * 100000
)
net_country_rate_cols = ["Country", "N_Grids", "population_2020"]
net_country_rate_cols += [
    col for col in net_country_rates.columns if col.startswith("net_rate_")
]
net_country_rates_output = net_country_rates[net_country_rate_cols]
if (net_country_rates["Country"] == "TOTAL").any():
    total_net = net_country_rates.loc[
        net_country_rates["Country"] == "TOTAL", "net_rate_total_mean"
    ].iloc[0]
pop_with_regions = pop.merge(un_regions, on="Country", how="left")
region_pop = (
    pop_with_regions.groupby("UN_Region")
    .agg(
        {
            "population_2020": "sum",
            "pop_under_20": "sum",
            "pop_20_54": "sum",
            "pop_55_64": "sum",
            "pop_65_74": "sum",
            "pop_75plus": "sum",
        }
    )
    .reset_index()
)
for idx, row in region_pop.iterrows():
    pass
total_region_pop = region_pop["population_2020"].sum()
total_region_pop_ages = {
    "under_20": region_pop["pop_under_20"].sum(),
    "20_54": region_pop["pop_20_54"].sum(),
    "55_64": region_pop["pop_55_64"].sum(),
    "65_74": region_pop["pop_65_74"].sum(),
    "75plus": region_pop["pop_75plus"].sum(),
}
region_merged = deaths_region.merge(region_pop, on="UN_Region", how="left")
total_region_mask = region_merged["UN_Region"] == "TOTAL"
if total_region_mask.any():
    region_merged.loc[total_region_mask, "population_2020"] = total_region_pop
    for age, pop_col in age_groups.items():
        region_merged.loc[total_region_mask, pop_col] = total_region_pop_ages[age]
region_rates = region_merged.copy()
for age, pop_col in age_groups.items():
    region_rates[f"heat_rate_{age}_mean"] = (
        region_rates[f"heat_deaths_{age}_mean"] / region_rates[pop_col] * 100000
    )
    region_rates[f"heat_rate_{age}_lower"] = (
        region_rates[f"heat_deaths_{age}_lower"] / region_rates[pop_col] * 100000
    )
    region_rates[f"heat_rate_{age}_upper"] = (
        region_rates[f"heat_deaths_{age}_upper"] / region_rates[pop_col] * 100000
    )
    region_rates[f"cold_rate_{age}_mean"] = (
        region_rates[f"cold_deaths_{age}_mean"] / region_rates[pop_col] * 100000
    )
    region_rates[f"cold_rate_{age}_lower"] = (
        region_rates[f"cold_deaths_{age}_lower"] / region_rates[pop_col] * 100000
    )
    region_rates[f"cold_rate_{age}_upper"] = (
        region_rates[f"cold_deaths_{age}_upper"] / region_rates[pop_col] * 100000
    )
region_rates["heat_rate_total_mean"] = (
    region_rates["heat_deaths_total_mean"] / region_rates["population_2020"] * 100000
)
region_rates["heat_rate_total_lower"] = (
    region_rates["heat_deaths_total_lower"] / region_rates["population_2020"] * 100000
)
region_rates["heat_rate_total_upper"] = (
    region_rates["heat_deaths_total_upper"] / region_rates["population_2020"] * 100000
)
region_rates["cold_rate_total_mean"] = (
    region_rates["cold_deaths_total_mean"] / region_rates["population_2020"] * 100000
)
region_rates["cold_rate_total_lower"] = (
    region_rates["cold_deaths_total_lower"] / region_rates["population_2020"] * 100000
)
region_rates["cold_rate_total_upper"] = (
    region_rates["cold_deaths_total_upper"] / region_rates["population_2020"] * 100000
)
region_rate_cols = ["UN_Region", "N_Grids", "population_2020"]
region_rate_cols += [
    col
    for col in region_rates.columns
    if col.startswith("heat_rate_") or col.startswith("cold_rate_")
]
region_rates_output = region_rates[region_rate_cols]
for idx, row in region_rates_output[
    region_rates_output["UN_Region"] != "TOTAL"
].iterrows():
    pass
net_region_merged = deaths_net_region.merge(region_pop, on="UN_Region", how="left")
if (net_region_merged["UN_Region"] == "TOTAL").any():
    net_region_merged.loc[
        net_region_merged["UN_Region"] == "TOTAL", "population_2020"
    ] = total_region_pop
    for age, pop_col in age_groups.items():
        net_region_merged.loc[net_region_merged["UN_Region"] == "TOTAL", pop_col] = (
            total_region_pop_ages[age]
        )
net_region_rates = net_region_merged.copy()
for age, pop_col in age_groups.items():
    net_region_rates[f"net_rate_{age}_mean"] = (
        net_region_rates[f"net_deaths_{age}_mean"] / net_region_rates[pop_col] * 100000
    )
    net_region_rates[f"net_rate_{age}_lower"] = (
        net_region_rates[f"net_deaths_{age}_lower"] / net_region_rates[pop_col] * 100000
    )
    net_region_rates[f"net_rate_{age}_upper"] = (
        net_region_rates[f"net_deaths_{age}_upper"] / net_region_rates[pop_col] * 100000
    )
net_region_rates["net_rate_total_mean"] = (
    net_region_rates["net_deaths_total_mean"]
    / net_region_rates["population_2020"]
    * 100000
)
net_region_rates["net_rate_total_lower"] = (
    net_region_rates["net_deaths_total_lower"]
    / net_region_rates["population_2020"]
    * 100000
)
net_region_rates["net_rate_total_upper"] = (
    net_region_rates["net_deaths_total_upper"]
    / net_region_rates["population_2020"]
    * 100000
)
net_region_rate_cols = ["UN_Region", "N_Grids", "population_2020"]
net_region_rate_cols += [
    col for col in net_region_rates.columns if col.startswith("net_rate_")
]
net_region_rates_output = net_region_rates[net_region_rate_cols]
for idx, row in net_region_rates_output[
    net_region_rates_output["UN_Region"] != "TOTAL"
].iterrows():
    pass
grid_rates_output.to_csv(output_grid_rates, index=False)
country_rates_output.to_csv(output_country_rates, index=False)
region_rates_output.to_csv(output_region_rates, index=False)
net_grid_rates_output.to_csv(output_net_grid_rates, index=False)
net_country_rates_output.to_csv(output_net_country_rates, index=False)
net_region_rates_output.to_csv(output_net_region_rates, index=False)
if total_row_mask.any():
    total_row = country_rates_output[country_rates_output["Country"] == "TOTAL"].iloc[0]
    net_total_row = net_country_rates_output[
        net_country_rates_output["Country"] == "TOTAL"
    ].iloc[0]
top_heat = country_rates_output[country_rates_output["Country"] != "TOTAL"].nlargest(
    10, "heat_rate_total_mean"
)
for idx, row in top_heat.iterrows():
    pass
top_cold = country_rates_output[country_rates_output["Country"] != "TOTAL"].nlargest(
    10, "cold_rate_total_mean"
)
for idx, row in top_cold.iterrows():
    pass
for idx, row in region_rates_output[
    region_rates_output["UN_Region"] != "TOTAL"
].iterrows():
    region_name = row["UN_Region"]
    heat_rate = row["heat_rate_total_mean"]
    cold_rate = row["cold_rate_total_mean"]
    net_row = net_region_rates_output[
        net_region_rates_output["UN_Region"] == region_name
    ].iloc[0]
    net_rate = net_row["net_rate_total_mean"]
