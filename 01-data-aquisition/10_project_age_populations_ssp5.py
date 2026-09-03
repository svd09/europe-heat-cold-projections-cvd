import pandas as pd

ssp5_file = "data/Europe_Grid_SSP5_Population_Annual_2020_2080.csv"
baseline_file = "data/Grid_CVD_Age.csv"
prop_median_file = "data/Proportions_Median.csv"
prop_high_file = "data/Proportions_High_Fert.csv"
prop_low_file = "data/Proportions_Low_Fert.csv"
output_static = "data/Europe_Grid_SSP5_Static_Age_Pop_2025_2080.csv"
output_median = "data/Europe_Grid_SSP5_Median_Fert_Age_Pop_2025_2080.csv"
output_high = "data/Europe_Grid_SSP5_High_Fert_Age_Pop_2025_2080.csv"
output_low = "data/Europe_Grid_SSP5_Low_Fert_Age_Pop_2025_2080.csv"
ssp5 = pd.read_csv(ssp5_file)
baseline = pd.read_csv(baseline_file)
prop_median = pd.read_csv(prop_median_file)
prop_high = pd.read_csv(prop_high_file)
prop_low = pd.read_csv(prop_low_file)
data = baseline[["grid_id", "Country", "population_2020"]].merge(
    ssp5, on="grid_id", how="inner"
)
years = range(2025, 2081)
for year in years:
    col_name = f"ssp5_{year}"
    if col_name in data.columns:
        data[f"growth_factor_{year}"] = data[col_name] / data["ssp5_2020"]
    if year % 10 == 0:
        pass
for year in years:
    growth_col = f"growth_factor_{year}"
    if growth_col in data.columns:
        data[f"harmonized_pop_{year}"] = data["population_2020"] * data[growth_col]
    if year % 10 == 0:
        pass


def calculate_age_populations(data, prop_df, scenario_name, start_year=2025):
    age_groups = {
        "<20": "under_20",
        "20-54": "20_54",
        "55-64": "55_64",
        "65-74": "65_74",
        "75+": "75plus",
    }
    results = data[["grid_id", "Country"]].copy()
    for year in range(start_year, 2081):
        harmonized_col = f"harmonized_pop_{year}"
        if harmonized_col not in data.columns:
            continue
        for age_group, age_suffix in age_groups.items():
            col_name = f"pop_{age_suffix}_{year}"
            results[col_name] = 0.0
            for country in data["Country"].unique():
                prop_data = prop_df[
                    (prop_df["Country"] == country)
                    & (prop_df["Age Group"] == age_group)
                ]
                if len(prop_data) > 0 and str(year) in prop_df.columns:
                    proportion = prop_data[str(year)].iloc[0]
                    mask = data["Country"] == country
                    results.loc[mask, col_name] = (
                        data.loc[mask, harmonized_col] * proportion
                    )
        if year % 10 == 0:
            sum(
                (
                    results[f"pop_{age_suffix}_{year}"].sum()
                    for age_suffix in age_groups.values()
                )
            )
    return results


prop_static = prop_median.copy()
year_cols = [str(y) for y in range(2021, 2081)]
for year_col in year_cols:
    if year_col in prop_static.columns:
        prop_static[year_col] = prop_static["2020"]
results_static = calculate_age_populations(data, prop_static, "Static (2020 structure)")
results_median = calculate_age_populations(data, prop_median, "Median Fertility")
results_high = calculate_age_populations(data, prop_high, "High Fertility")
results_low = calculate_age_populations(data, prop_low, "Low Fertility")
results_static.to_csv(output_static, index=False)
results_median.to_csv(output_median, index=False)
results_high.to_csv(output_high, index=False)
results_low.to_csv(output_low, index=False)
