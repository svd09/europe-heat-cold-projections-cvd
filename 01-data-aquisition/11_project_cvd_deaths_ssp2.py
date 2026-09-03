import pandas as pd
import numpy as np

baseline_file = "data/Grid_CVD_Age.csv"
ssp2_static_file = "data/Europe_Grid_SSP2_Static_Age_Pop_2025_2080.csv"
ssp2_median_file = "data/Europe_Grid_SSP2_Median_Fert_Age_Pop_2025_2080.csv"
ssp2_high_file = "data/Europe_Grid_SSP2_High_Fert_Age_Pop_2025_2080.csv"
ssp2_low_file = "data/Europe_Grid_SSP2_Low_Fert_Age_Pop_2025_2080.csv"
output_static = "data/SSP2_Static_CVD_Deaths_Grid_2025_2080.csv"
output_median = "data/SSP2_Median_Fert_CVD_Deaths_Grid_2025_2080.csv"
output_high = "data/SSP2_High_Fert_CVD_Deaths_Grid_2025_2080.csv"
output_low = "data/SSP2_Low_Fert_CVD_Deaths_Grid_2025_2080.csv"
output_static_country = "data/SSP2_Static_CVD_Deaths_Country_2025_2080.csv"
output_median_country = "data/SSP2_Median_Fert_CVD_Deaths_Country_2025_2080.csv"
output_high_country = "data/SSP2_High_Fert_CVD_Deaths_Country_2025_2080.csv"
output_low_country = "data/SSP2_Low_Fert_CVD_Deaths_Country_2025_2080.csv"
baseline = pd.read_csv(baseline_file)
ssp2_static = pd.read_csv(ssp2_static_file)
ssp2_median = pd.read_csv(ssp2_median_file)
ssp2_high = pd.read_csv(ssp2_high_file)
ssp2_low = pd.read_csv(ssp2_low_file)
age_groups = {
    "under_20": {
        "pop_col": "pop_under_20",
        "cvd_mean": "cvd_deaths_mean_under_20",
        "cvd_max": "cvd_deaths_max_under_20",
        "cvd_min": "cvd_deaths_min_under_20",
    },
    "20_54": {
        "pop_col": "pop_20_54",
        "cvd_mean": "cvd_deaths_mean_20_54",
        "cvd_max": "cvd_deaths_max_20_54",
        "cvd_min": "cvd_deaths_min_20_54",
    },
    "55_64": {
        "pop_col": "pop_55_64",
        "cvd_mean": "cvd_deaths_mean_55_64",
        "cvd_max": "cvd_deaths_max_55_64",
        "cvd_min": "cvd_deaths_min_55_64",
    },
    "65_74": {
        "pop_col": "pop_65_74",
        "cvd_mean": "cvd_deaths_mean_65_74",
        "cvd_max": "cvd_deaths_max_65_74",
        "cvd_min": "cvd_deaths_min_65_74",
    },
    "75plus": {
        "pop_col": "pop_75plus",
        "cvd_mean": "cvd_deaths_mean_75plus",
        "cvd_max": "cvd_deaths_max_75plus",
        "cvd_min": "cvd_deaths_min_75plus",
    },
}


def project_cvd_deaths(baseline_df, pop_projection_df, scenario_name, is_static=False):
    data = baseline_df.merge(
        pop_projection_df, on="grid_id", how="inner", suffixes=("", "_proj")
    )
    base_cols = data[["grid_id", "Country"]].copy()
    all_new_columns = []
    years = range(2025, 2081)
    for year in years:
        year_columns = {}
        if is_static:
            growth_factors_this_year = {}
        for age_suffix, age_info in age_groups.items():
            proj_pop_col = f"pop_{age_suffix}_{year}"
            baseline_pop_col = age_info["pop_col"]
            growth_factor = data[proj_pop_col] / data[baseline_pop_col]
            growth_factor = growth_factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            if is_static:
                growth_factors_this_year[age_suffix] = growth_factor.copy()
            for cvd_type in ["mean", "max", "min"]:
                cvd_col = age_info[f"cvd_{cvd_type}"]
                output_col = f"cvd_deaths_{cvd_type}_{age_suffix}_{year}"
                year_columns[output_col] = data[cvd_col] * growth_factor
        all_new_columns.append(pd.DataFrame(year_columns))
        if is_static and year % 10 == 0:
            gf_df = pd.DataFrame(growth_factors_this_year)
            gf_std = gf_df.std(axis=1)
            max_std = gf_std.max()
            if max_std < 1e-05:
                pass
            else:
                pass
        elif year % 10 == 0:
            pass
    results = pd.concat([base_cols] + all_new_columns, axis=1)
    return results


def create_country_summary(grid_results, scenario_name):
    country_summary = (
        grid_results.groupby("Country").sum(numeric_only=True).reset_index()
    )
    total_row = {"Country": "Total"}
    for col in country_summary.columns:
        if col != "Country":
            total_row[col] = country_summary[col].sum()
    country_summary = pd.concat(
        [country_summary, pd.DataFrame([total_row])], ignore_index=True
    )
    return country_summary


results_static = project_cvd_deaths(baseline, ssp2_static, "Static", is_static=True)
country_static = create_country_summary(results_static, "Static")
results_median = project_cvd_deaths(baseline, ssp2_median, "Median", is_static=False)
country_median = create_country_summary(results_median, "Median")
results_high = project_cvd_deaths(baseline, ssp2_high, "High", is_static=False)
country_high = create_country_summary(results_high, "High")
results_low = project_cvd_deaths(baseline, ssp2_low, "Low", is_static=False)
country_low = create_country_summary(results_low, "Low")
results_static.to_csv(output_static, index=False)
results_median.to_csv(output_median, index=False)
results_high.to_csv(output_high, index=False)
results_low.to_csv(output_low, index=False)
country_static.to_csv(output_static_country, index=False)
country_median.to_csv(output_median_country, index=False)
country_high.to_csv(output_high_country, index=False)
country_low.to_csv(output_low_country, index=False)
