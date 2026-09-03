import pandas as pd
from pathlib import Path

desktop_path = Path("data")
grid_cvd_file = desktop_path / "Grid_CVD_Age.csv"
output_file = desktop_path / "Baseline_CVD_Rates_by_Country_Age.csv"
grid_cvd = pd.read_csv(grid_cvd_file)
for col in grid_cvd.columns:
    pass
age_groups = {
    "under_20": {
        "pop": "pop_under_20",
        "cvd_mean": "cvd_deaths_mean_under_20",
        "cvd_max": "cvd_deaths_max_under_20",
        "cvd_min": "cvd_deaths_min_under_20",
    },
    "20_54": {
        "pop": "pop_20_54",
        "cvd_mean": "cvd_deaths_mean_20_54",
        "cvd_max": "cvd_deaths_max_20_54",
        "cvd_min": "cvd_deaths_min_20_54",
    },
    "55_64": {
        "pop": "pop_55_64",
        "cvd_mean": "cvd_deaths_mean_55_64",
        "cvd_max": "cvd_deaths_max_55_64",
        "cvd_min": "cvd_deaths_min_55_64",
    },
    "65_74": {
        "pop": "pop_65_74",
        "cvd_mean": "cvd_deaths_mean_65_74",
        "cvd_max": "cvd_deaths_max_65_74",
        "cvd_min": "cvd_deaths_min_65_74",
    },
    "75plus": {
        "pop": "pop_75plus",
        "cvd_mean": "cvd_deaths_mean_75plus",
        "cvd_max": "cvd_deaths_max_75plus",
        "cvd_min": "cvd_deaths_min_75plus",
    },
}
agg_dict = {"population_2020": "sum"}
for age, cols in age_groups.items():
    agg_dict[cols["pop"]] = "sum"
    agg_dict[cols["cvd_mean"]] = "sum"
    agg_dict[cols["cvd_max"]] = "sum"
    agg_dict[cols["cvd_min"]] = "sum"
country_data = grid_cvd.groupby("Country").agg(agg_dict).reset_index()
results = []
for idx, row in country_data.iterrows():
    country = row["Country"]
    total_pop = row["population_2020"]
    result = {
        "Country": country,
        "Total_Population": total_pop,
        "N_Grids": len(grid_cvd[grid_cvd["Country"] == country]),
    }
    total_cvd_mean = 0
    total_cvd_max = 0
    total_cvd_min = 0
    for age, cols in age_groups.items():
        pop = row[cols["pop"]]
        cvd_mean = row[cols["cvd_mean"]]
        cvd_max = row[cols["cvd_max"]]
        cvd_min = row[cols["cvd_min"]]
        total_cvd_mean += cvd_mean
        total_cvd_max += cvd_max
        total_cvd_min += cvd_min
        result[f"CVD_deaths_{age}_mean"] = cvd_mean
        result[f"CVD_deaths_{age}_max"] = cvd_max
        result[f"CVD_deaths_{age}_min"] = cvd_min
        result[f"Population_{age}"] = pop
        if pop > 0:
            result[f"CVD_rate_{age}_mean"] = cvd_mean / pop * 100000
            result[f"CVD_rate_{age}_max"] = cvd_max / pop * 100000
            result[f"CVD_rate_{age}_min"] = cvd_min / pop * 100000
        else:
            result[f"CVD_rate_{age}_mean"] = 0
            result[f"CVD_rate_{age}_max"] = 0
            result[f"CVD_rate_{age}_min"] = 0
    result["CVD_deaths_total_mean"] = total_cvd_mean
    result["CVD_deaths_total_max"] = total_cvd_max
    result["CVD_deaths_total_min"] = total_cvd_min
    result["CVD_rate_total_mean"] = total_cvd_mean / total_pop * 100000
    result["CVD_rate_total_max"] = total_cvd_max / total_pop * 100000
    result["CVD_rate_total_min"] = total_cvd_min / total_pop * 100000
    results.append(result)
results_df = pd.DataFrame(results)
total_row = {"Country": "TOTAL", "N_Grids": grid_cvd["grid_id"].nunique()}
total_row["Total_Population"] = country_data["population_2020"].sum()
total_cvd_all = 0
for age, cols in age_groups.items():
    total_pop_age = country_data[cols["pop"]].sum()
    total_cvd_mean = country_data[cols["cvd_mean"]].sum()
    total_cvd_max = country_data[cols["cvd_max"]].sum()
    total_cvd_min = country_data[cols["cvd_min"]].sum()
    total_cvd_all += total_cvd_mean
    total_row[f"Population_{age}"] = total_pop_age
    total_row[f"CVD_deaths_{age}_mean"] = total_cvd_mean
    total_row[f"CVD_deaths_{age}_max"] = total_cvd_max
    total_row[f"CVD_deaths_{age}_min"] = total_cvd_min
    if total_pop_age > 0:
        total_row[f"CVD_rate_{age}_mean"] = total_cvd_mean / total_pop_age * 100000
        total_row[f"CVD_rate_{age}_max"] = total_cvd_max / total_pop_age * 100000
        total_row[f"CVD_rate_{age}_min"] = total_cvd_min / total_pop_age * 100000
    else:
        total_row[f"CVD_rate_{age}_mean"] = 0
        total_row[f"CVD_rate_{age}_max"] = 0
        total_row[f"CVD_rate_{age}_min"] = 0
total_cvd_mean_all = results_df["CVD_deaths_total_mean"].sum()
total_cvd_max_all = results_df["CVD_deaths_total_max"].sum()
total_cvd_min_all = results_df["CVD_deaths_total_min"].sum()
total_row["CVD_deaths_total_mean"] = total_cvd_mean_all
total_row["CVD_deaths_total_max"] = total_cvd_max_all
total_row["CVD_deaths_total_min"] = total_cvd_min_all
total_row["CVD_rate_total_mean"] = (
    total_cvd_mean_all / total_row["Total_Population"] * 100000
)
total_row["CVD_rate_total_max"] = (
    total_cvd_max_all / total_row["Total_Population"] * 100000
)
total_row["CVD_rate_total_min"] = (
    total_cvd_min_all / total_row["Total_Population"] * 100000
)
results_df = pd.concat([results_df, pd.DataFrame([total_row])], ignore_index=True)
results_df.to_csv(output_file, index=False)
total_stats = results_df[results_df["Country"] == "TOTAL"].iloc[0]
for age in ["under_20", "20_54", "55_64", "65_74", "75plus"]:
    rate = total_stats[f"CVD_rate_{age}_mean"]
    deaths = total_stats[f"CVD_deaths_{age}_mean"]
countries_only = results_df[results_df["Country"] != "TOTAL"].copy()
top_10 = countries_only.nlargest(10, "CVD_rate_total_mean")
for idx, row in top_10.iterrows():
    pass
for age in ["under_20", "20_54", "55_64", "65_74", "75plus"]:
    top_5 = countries_only.nlargest(5, f"CVD_rate_{age}_mean")
    for idx, row in top_5.iterrows():
        rate = row[f"CVD_rate_{age}_mean"]
temp_pct = 168585 / total_stats["CVD_deaths_total_mean"] * 100
if temp_pct < 1:
    pass
elif temp_pct > 15:
    pass
else:
    pass
