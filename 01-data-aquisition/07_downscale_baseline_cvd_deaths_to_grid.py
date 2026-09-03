import pandas as pd

grid_pop_file = "Grid_Age_Pop.csv"
national_cvd_file = "CVD_Deaths_Summary.csv"
output_file = "Europe_Grid_CVD_Deaths_by_Age.csv"
grid_pop = pd.read_csv(grid_pop_file)
cvd_national = pd.read_csv(national_cvd_file)
age_mapping = {
    "<20": "pop_under_20",
    "20-54": "pop_20_54",
    "55-64": "pop_55_64",
    "65-74": "pop_65_74",
    "75+": "pop_75plus",
}
for cvd_age, pop_col in age_mapping.items():
    pass
national_pop = {}
for country in grid_pop["Country"].unique():
    country_grids = grid_pop[grid_pop["Country"] == country]
    national_pop[country] = {}
    for age_group, pop_col in age_mapping.items():
        if pop_col in country_grids.columns:
            national_pop[country][age_group] = country_grids[pop_col].sum()
        else:
            national_pop[country][age_group] = 0
national_pop_df = []
for country, ages in national_pop.items():
    for age_group, pop in ages.items():
        national_pop_df.append(
            {"Country": country, "Age Group": age_group, "National_Population": pop}
        )
national_pop_df = pd.DataFrame(national_pop_df)
cvd_national["CVD_Deaths_Mean"] = cvd_national["Mean (2016-2023)"]
cvd_national["CVD_Deaths_Max"] = cvd_national["Max"]
cvd_national["CVD_Deaths_Min"] = cvd_national["Min"]
cvd_with_pop = cvd_national.merge(
    national_pop_df, on=["Country", "Age Group"], how="left"
)
missing = cvd_with_pop["National_Population"].isna().sum()
if missing > 0:
    cvd_with_pop.loc[
        cvd_with_pop["National_Population"].isna(), "National_Population"
    ] = 1
results = []
for idx, grid in grid_pop.iterrows():
    if (idx + 1) % 1000 == 0:
        pass
    grid_id = grid["grid_id"]
    country = grid["Country"]
    grid_result = {
        "grid_id": grid_id,
        "lon_idx": grid.get("lon_idx"),
        "lat_idx": grid.get("lat_idx"),
        "center_lon": grid.get("center_lon"),
        "center_lat": grid.get("center_lat"),
        "Country": country,
        "population_2020": grid.get("population_2020", 0),
    }
    for age_group, pop_col in age_mapping.items():
        grid_pop_age = grid.get(pop_col, 0)
        national_data = cvd_with_pop[
            (cvd_with_pop["Country"] == country)
            & (cvd_with_pop["Age Group"] == age_group)
        ]
        if len(national_data) == 0:
            cvd_deaths_mean = 0
            cvd_deaths_max = 0
            cvd_deaths_min = 0
        else:
            national_cvd_mean = national_data["CVD_Deaths_Mean"].iloc[0]
            national_cvd_max = national_data["CVD_Deaths_Max"].iloc[0]
            national_cvd_min = national_data["CVD_Deaths_Min"].iloc[0]
            national_pop_age = national_data["National_Population"].iloc[0]
            if national_pop_age > 0:
                proportion = grid_pop_age / national_pop_age
                cvd_deaths_mean = national_cvd_mean * proportion
                cvd_deaths_max = national_cvd_max * proportion
                cvd_deaths_min = national_cvd_min * proportion
            else:
                cvd_deaths_mean = 0
                cvd_deaths_max = 0
                cvd_deaths_min = 0
        age_suffix = pop_col.replace("pop_", "")
        grid_result[f"cvd_deaths_mean_{age_suffix}"] = cvd_deaths_mean
        grid_result[f"cvd_deaths_max_{age_suffix}"] = cvd_deaths_max
        grid_result[f"cvd_deaths_min_{age_suffix}"] = cvd_deaths_min
        grid_result[pop_col] = grid_pop_age
    results.append(grid_result)
grid_cvd = pd.DataFrame(results)
validation_results = []
for country in grid_cvd["Country"].unique():
    country_grids = grid_cvd[grid_cvd["Country"] == country]
    for age_group in age_mapping.keys():
        national_data = cvd_with_pop[
            (cvd_with_pop["Country"] == country)
            & (cvd_with_pop["Age Group"] == age_group)
        ]
        if len(national_data) > 0:
            national_cvd_mean = national_data["CVD_Deaths_Mean"].iloc[0]
            national_cvd_max = national_data["CVD_Deaths_Max"].iloc[0]
            national_cvd_min = national_data["CVD_Deaths_Min"].iloc[0]
            age_suffix = age_mapping[age_group].replace("pop_", "")
            grid_sum_mean = country_grids[f"cvd_deaths_mean_{age_suffix}"].sum()
            grid_sum_max = country_grids[f"cvd_deaths_max_{age_suffix}"].sum()
            grid_sum_min = country_grids[f"cvd_deaths_min_{age_suffix}"].sum()
            diff_mean = abs(national_cvd_mean - grid_sum_mean)
            pct_diff_mean = (
                diff_mean / national_cvd_mean * 100 if national_cvd_mean > 0 else 0
            )
            diff_max = abs(national_cvd_max - grid_sum_max)
            pct_diff_max = (
                diff_max / national_cvd_max * 100 if national_cvd_max > 0 else 0
            )
            diff_min = abs(national_cvd_min - grid_sum_min)
            pct_diff_min = (
                diff_min / national_cvd_min * 100 if national_cvd_min > 0 else 0
            )
            validation_results.append(
                {
                    "Country": country,
                    "Age Group": age_group,
                    "National_Mean": national_cvd_mean,
                    "Grid_Sum_Mean": grid_sum_mean,
                    "Pct_Diff_Mean": pct_diff_mean,
                    "National_Max": national_cvd_max,
                    "Grid_Sum_Max": grid_sum_max,
                    "Pct_Diff_Max": pct_diff_max,
                    "National_Min": national_cvd_min,
                    "Grid_Sum_Min": grid_sum_min,
                    "Pct_Diff_Min": pct_diff_min,
                }
            )
validation_df = pd.DataFrame(validation_results)
large_diff = validation_df[validation_df["Pct_Diff_Mean"] > 1]
if len(large_diff) == 0:
    pass
else:
    pass
cvd_mean_cols = [col for col in grid_cvd.columns if col.startswith("cvd_deaths_mean_")]
for col in cvd_mean_cols:
    total = grid_cvd[col].sum()
    age_label = (
        col.replace("cvd_deaths_mean_", "").replace("_", "-").replace("plus", "+")
    )
total_mean = sum((grid_cvd[col].sum() for col in cvd_mean_cols))
cvd_max_cols = [col for col in grid_cvd.columns if col.startswith("cvd_deaths_max_")]
cvd_min_cols = [col for col in grid_cvd.columns if col.startswith("cvd_deaths_min_")]
total_max = sum((grid_cvd[col].sum() for col in cvd_max_cols))
total_min = sum((grid_cvd[col].sum() for col in cvd_min_cols))
grid_cvd.to_csv(output_file, index=False)
validation_file = "CVD_Downscaling_Validation.csv"
validation_df.to_csv(validation_file, index=False)
cvd_mean_cols = [col for col in grid_cvd.columns if col.startswith("cvd_deaths_mean_")]
display_cols = ["grid_id", "Country", "population_2020"] + cvd_mean_cols[:3]
display_cols = [c for c in display_cols if c in grid_cvd.columns]
