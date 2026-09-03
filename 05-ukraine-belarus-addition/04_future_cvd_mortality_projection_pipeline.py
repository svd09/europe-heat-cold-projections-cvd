import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

N_SIMULATIONS = 1000
RANDOM_SEED = 42
DATA_DIR = Path("data")
AGE_BRACKETS = ["<20", "20-54", "55-64", "65-74", "75+"]
PERIODS = {"2046-2055": list(range(2046, 2056)), "2071-2080": list(range(2071, 2081))}
PERIOD_LABELS = {"2046-2055": "Mid-Century", "2071-2080": "Late-Century"}
SSP_MAP = {"SSP245": "SSP2", "SSP585": "SSP5"}
ERA5_FILE = DATA_DIR / "Ukraine_Belarus_ERA5Land_Baseline_2016_2023.csv"
ZONES_FILE = DATA_DIR / "Ukraine_Belarus_Zones.csv"
RR_FILE = DATA_DIR / "RR_Coeff.csv"
TEMP_FILE_TEMPLATE = (
    "Ukraine_Belarus_CMIP6_Ensemble_{ssp_long}_{period}_BiasCorrected.csv"
)
MODELS = {
    "DemographicAdjusted": {
        "cvd_file": lambda ssp_short: f"{ssp_short}_CVD_Deaths_Grid_2025_2080.csv",
        "pop_file": lambda ssp_short: f"Ukraine_Belarus_Grid_{ssp_short}.csv",
    },
    "DemographicFixed": {
        "cvd_file": lambda ssp_short: f"{ssp_short}Fixed_CVD_Deaths_Grid_2025_2080.csv",
        "pop_file": lambda ssp_short: f"Ukraine_Belarus_Grid_{ssp_short}Fixed.csv",
    },
}
tmrel_start = time.time()
era5 = pd.read_csv(ERA5_FILE)
era5["date"] = pd.to_datetime(era5["date"])
era5["temp_rounded"] = era5["temp_mean"].round(1)
grid_coords = era5[["grid_id", "center_lon", "center_lat"]].drop_duplicates()
coords_array = grid_coords[["center_lon", "center_lat"]].values
tree = cKDTree(coords_array)
grid_id_list = grid_coords["grid_id"].tolist()
missing_before = era5["temp_rounded"].isna().sum()
if missing_before > 0:
    era5_pivot = era5.pivot_table(
        index="date", columns="grid_id", values="temp_rounded", aggfunc="first"
    )
    grid_neighbors = {}
    for i, grid_id in enumerate(grid_id_list):
        distances, indices = tree.query(coords_array[i], k=min(11, len(grid_id_list)))
        grid_neighbors[grid_id] = [
            grid_id_list[idx] for idx in np.atleast_1d(indices)[1:]
        ]
    grids_with_missing = era5.loc[era5["temp_rounded"].isna(), "grid_id"].unique()
    imputed_count = 0
    for grid_id in grids_with_missing:
        grid_missing_mask = (era5["grid_id"] == grid_id) & era5["temp_rounded"].isna()
        missing_dates = era5.loc[grid_missing_mask, "date"]
        for neighbor_id in grid_neighbors.get(grid_id, []):
            still_missing = era5.loc[grid_missing_mask, "temp_rounded"].isna()
            if not still_missing.any():
                break
            if neighbor_id not in era5_pivot.columns:
                continue
            neighbor_vals = era5_pivot[neighbor_id].reindex(missing_dates).values
            fillable = era5.loc[grid_missing_mask].index[
                still_missing.values & pd.notna(neighbor_vals)
            ]
            if len(fillable) == 0:
                continue
            era5.loc[fillable, "temp_rounded"] = (
                era5_pivot[neighbor_id].reindex(era5.loc[fillable, "date"]).values
            )
            imputed_count += len(fillable)
else:
    pass
tmrel_rows = {}
for grid_id in era5["grid_id"].unique():
    g = era5.loc[era5["grid_id"] == grid_id, "temp_rounded"].dropna()
    if len(g) == 0:
        continue
    p54, p92 = (np.percentile(g, 54), np.percentile(g, 92))
    pf = g[(g >= p54) & (g <= p92)]
    mode_series = pf.mode()
    if len(mode_series) > 0:
        tmrel = mode_series.iloc[0]
        mode_count = (pf == tmrel).sum()
        if mode_count == 1:
            mode_whole = pf.round(0).mode()
            tmrel = mode_whole.iloc[0] if len(mode_whole) > 0 else pf.median()
    else:
        tmrel = pf.median()
    tmrel_rows[grid_id] = tmrel
tmrel_lookup = pd.Series(tmrel_rows, name="TMREL")
tmrel_lookup.index.name = "grid_id"
del era5
zones = pd.read_csv(ZONES_FILE)
if zones.duplicated("grid_id").any():
    zones = zones.drop_duplicates(subset="grid_id")
zones = zones[["grid_id", "climate_zone"]]
rr_coeff = pd.read_csv(RR_FILE).set_index("climate_zone")


def compute_excess_for_year(temp_year_df):
    t = temp_year_df.merge(tmrel_lookup.rename("TMREL"), on="grid_id", how="inner")
    t["heat_excess_daily"] = (t["mean"] - t["TMREL"]).clip(lower=0)
    t["cold_excess_daily"] = (t["TMREL"] - t["mean"]).clip(lower=0)
    agg = t.groupby("grid_id").agg(
        n_days=("mean", "size"),
        heat_sum=("heat_excess_daily", "sum"),
        cold_sum=("cold_excess_daily", "sum"),
    )
    agg["avg_daily_heat_excess"] = agg["heat_sum"] / agg["n_days"]
    agg["avg_daily_cold_excess"] = agg["cold_sum"] / agg["n_days"]
    return agg[["avg_daily_heat_excess", "avg_daily_cold_excess"]].reset_index()


def run_monte_carlo(merged_year_df):
    rng = np.random.default_rng(RANDOM_SEED)
    results = []
    for _, row in merged_year_df.iterrows():
        zone = row["climate_zone"]
        if zone not in rr_coeff.index:
            continue
        rr = rr_coeff.loc[zone]
        beta_heat_se = (rr["beta_heat_upper"] - rr["beta_heat_lower"]) / (2 * 1.96)
        beta_cold_se = (rr["beta_cold_upper"] - rr["beta_cold_lower"]) / (2 * 1.96)
        beta_heat_samples = rng.normal(rr["beta_heat"], beta_heat_se, N_SIMULATIONS)
        beta_cold_samples = rng.normal(rr["beta_cold"], beta_cold_se, N_SIMULATIONS)
        RR_heat = np.exp(beta_heat_samples * row["heat_excess"])
        RR_cold = np.exp(beta_cold_samples * row["cold_excess"])
        PAF_heat = np.where(RR_heat > 1, (RR_heat - 1) / RR_heat, 0)
        PAF_cold = np.where(RR_cold > 1, (RR_cold - 1) / RR_cold, 0)
        result_row = {
            "grid_id": row["grid_id"],
            "country": row["country"],
            "climate_zone": zone,
        }
        for bracket in AGE_BRACKETS:
            cvd_mean = row[f"cvd_deaths_mean_{bracket}"]
            cvd_max = row[f"cvd_deaths_max_{bracket}"]
            cvd_min = row[f"cvd_deaths_min_{bracket}"]
            if cvd_max == cvd_min:
                cvd_samples = np.full(N_SIMULATIONS, cvd_mean)
            else:
                cvd_samples = rng.triangular(cvd_min, cvd_mean, cvd_max, N_SIMULATIONS)
            heat_deaths = PAF_heat * cvd_samples
            cold_deaths = PAF_cold * cvd_samples
            net_deaths = heat_deaths + cold_deaths
            result_row[f"heat_deaths_{bracket}_mean"] = heat_deaths.mean()
            result_row[f"heat_deaths_{bracket}_lower"] = np.percentile(heat_deaths, 2.5)
            result_row[f"heat_deaths_{bracket}_upper"] = np.percentile(
                heat_deaths, 97.5
            )
            result_row[f"cold_deaths_{bracket}_mean"] = cold_deaths.mean()
            result_row[f"cold_deaths_{bracket}_lower"] = np.percentile(cold_deaths, 2.5)
            result_row[f"cold_deaths_{bracket}_upper"] = np.percentile(
                cold_deaths, 97.5
            )
            result_row[f"net_deaths_{bracket}_mean"] = net_deaths.mean()
            result_row[f"net_deaths_{bracket}_lower"] = np.percentile(net_deaths, 2.5)
            result_row[f"net_deaths_{bracket}_upper"] = np.percentile(net_deaths, 97.5)
        for metric in ["heat", "cold", "net"]:
            for stat in ["mean", "lower", "upper"]:
                result_row[f"{metric}_deaths_total_{stat}"] = sum(
                    (result_row[f"{metric}_deaths_{b}_{stat}"] for b in AGE_BRACKETS)
                )
        results.append(result_row)
    return pd.DataFrame(results)


def aggregate_country_year(grid_year_df):
    sum_cols = [
        c
        for c in grid_year_df.columns
        if c.startswith(("heat_deaths_", "cold_deaths_", "net_deaths_"))
    ]
    country_df = grid_year_df.groupby("country")[sum_cols].sum().reset_index()
    country_df.insert(1, "N_Grids", grid_year_df.groupby("country").size().values)
    total_row = {"country": "TOTAL", "N_Grids": len(grid_year_df)}
    for c in sum_cols:
        total_row[c] = grid_year_df[c].sum()
    return pd.concat([country_df, pd.DataFrame([total_row])], ignore_index=True)


def add_rates(deaths_df, pop_lookup_df, key_cols):
    merged = deaths_df.merge(pop_lookup_df, on=key_cols, how="left")
    for bracket in AGE_BRACKETS:
        pop_col = f"population_{bracket}"
        for metric in ["heat", "cold", "net"]:
            for stat in ["mean", "lower", "upper"]:
                merged[f"{metric}_rate_{bracket}_{stat}"] = (
                    merged[f"{metric}_deaths_{bracket}_{stat}"]
                    / merged[pop_col]
                    * 100000
                )
    for metric in ["heat", "cold", "net"]:
        for stat in ["mean", "lower", "upper"]:
            merged[f"{metric}_rate_total_{stat}"] = (
                merged[f"{metric}_deaths_total_{stat}"]
                / merged["population_total"]
                * 100000
            )
    return merged


def compute_period_stats(
    all_years_grid_deaths, all_years_grid_pop, period_years, period_label, scenario
):
    n_years = len(period_years)
    df_period = all_years_grid_deaths[all_years_grid_deaths["year"].isin(period_years)]
    pop_period = all_years_grid_pop[all_years_grid_pop["year"].isin(period_years)]
    rows = []
    for bracket in AGE_BRACKETS + ["All_Ages"]:
        for temp_type, prefix in [("Heat", "heat"), ("Cold", "cold")]:
            if bracket == "All_Ages":
                mean_sum = (
                    df_period.groupby("grid_id")[
                        [f"{prefix}_deaths_{b}_mean" for b in AGE_BRACKETS]
                    ]
                    .sum()
                    .sum(axis=1)
                )
                lower_sum = (
                    df_period.groupby("grid_id")[
                        [f"{prefix}_deaths_{b}_lower" for b in AGE_BRACKETS]
                    ]
                    .sum()
                    .sum(axis=1)
                )
                upper_sum = (
                    df_period.groupby("grid_id")[
                        [f"{prefix}_deaths_{b}_upper" for b in AGE_BRACKETS]
                    ]
                    .sum()
                    .sum(axis=1)
                )
            else:
                mean_sum = df_period.groupby("grid_id")[
                    f"{prefix}_deaths_{bracket}_mean"
                ].sum()
                lower_sum = df_period.groupby("grid_id")[
                    f"{prefix}_deaths_{bracket}_lower"
                ].sum()
                upper_sum = df_period.groupby("grid_id")[
                    f"{prefix}_deaths_{bracket}_upper"
                ].sum()
            annual_mean = mean_sum / n_years
            annual_lower = lower_sum / n_years
            annual_upper = upper_sum / n_years
            pop_col = (
                "population_total" if bracket == "All_Ages" else f"population_{bracket}"
            )
            avg_pop = pop_period.groupby("grid_id")[pop_col].mean()
            max_pop = pop_period.groupby("grid_id")[pop_col].max()
            avg_pop = avg_pop.where(avg_pop >= 1, max_pop)
            rate_mean = (
                (annual_mean / avg_pop * 100000).replace([np.inf, -np.inf], 0).fillna(0)
            )
            rate_lower = (
                (annual_lower / avg_pop * 100000)
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
            )
            rate_upper = (
                (annual_upper / avg_pop * 100000)
                .replace([np.inf, -np.inf], 0)
                .fillna(0)
            )
            block = pd.DataFrame(
                {
                    "grid_id": annual_mean.index,
                    "Age_Group": bracket,
                    "Temperature_Type": temp_type,
                    "Period": period_label,
                    "Scenario": scenario,
                    "Annual_Deaths_Mean": annual_mean.values,
                    "Annual_Deaths_Lower": annual_lower.values,
                    "Annual_Deaths_Upper": annual_upper.values,
                    "Avg_Population": avg_pop.reindex(annual_mean.index).values,
                    "Rate_per_100k_Mean": rate_mean.reindex(annual_mean.index).values,
                    "Rate_per_100k_Lower": rate_lower.reindex(annual_mean.index).values,
                    "Rate_per_100k_Upper": rate_upper.reindex(annual_mean.index).values,
                }
            )
            rows.append(block)
    period_df = pd.concat(rows, ignore_index=True)
    heat = period_df[period_df["Temperature_Type"] == "Heat"].set_index(
        ["grid_id", "Age_Group"]
    )
    cold = period_df[period_df["Temperature_Type"] == "Cold"].set_index(
        ["grid_id", "Age_Group"]
    )
    total = heat[
        [
            "Annual_Deaths_Mean",
            "Annual_Deaths_Lower",
            "Annual_Deaths_Upper",
            "Avg_Population",
        ]
    ].add(
        cold[
            [
                "Annual_Deaths_Mean",
                "Annual_Deaths_Lower",
                "Annual_Deaths_Upper",
                "Avg_Population",
            ]
        ]
    )
    total["Avg_Population"] = heat["Avg_Population"]
    total["Rate_per_100k_Mean"] = (
        (total["Annual_Deaths_Mean"] / total["Avg_Population"] * 100000)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    total["Rate_per_100k_Lower"] = (
        (total["Annual_Deaths_Lower"] / total["Avg_Population"] * 100000)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    total["Rate_per_100k_Upper"] = (
        (total["Annual_Deaths_Upper"] / total["Avg_Population"] * 100000)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    total = total.reset_index()
    total["Temperature_Type"] = "Total"
    total["Period"] = period_label
    total["Scenario"] = scenario
    period_df = pd.concat([period_df, total], ignore_index=True)
    grid_country = df_period[["grid_id", "country"]].drop_duplicates()
    period_df = period_df.merge(grid_country, on="grid_id", how="left")
    return period_df


def aggregate_period_to_country(period_grid_df):
    group_cols = ["country", "Age_Group", "Temperature_Type", "Period", "Scenario"]
    agg = (
        period_grid_df.groupby(group_cols)
        .agg(
            Annual_Deaths_Mean=("Annual_Deaths_Mean", "sum"),
            Annual_Deaths_Lower=("Annual_Deaths_Lower", "sum"),
            Annual_Deaths_Upper=("Annual_Deaths_Upper", "sum"),
            Avg_Population=("Avg_Population", "sum"),
        )
        .reset_index()
    )
    agg["Rate_per_100k_Mean"] = (
        (agg["Annual_Deaths_Mean"] / agg["Avg_Population"] * 100000)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    agg["Rate_per_100k_Lower"] = (
        (agg["Annual_Deaths_Lower"] / agg["Avg_Population"] * 100000)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    agg["Rate_per_100k_Upper"] = (
        (agg["Annual_Deaths_Upper"] / agg["Avg_Population"] * 100000)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    total_rows = []
    for (age, temp, period, scenario), sub in agg.groupby(
        ["Age_Group", "Temperature_Type", "Period", "Scenario"]
    ):
        total_rows.append(
            {
                "country": "TOTAL",
                "Age_Group": age,
                "Temperature_Type": temp,
                "Period": period,
                "Scenario": scenario,
                "Annual_Deaths_Mean": sub["Annual_Deaths_Mean"].sum(),
                "Annual_Deaths_Lower": sub["Annual_Deaths_Lower"].sum(),
                "Annual_Deaths_Upper": sub["Annual_Deaths_Upper"].sum(),
                "Avg_Population": sub["Avg_Population"].sum(),
            }
        )
    total_df = pd.DataFrame(total_rows)
    total_df["Rate_per_100k_Mean"] = (
        (total_df["Annual_Deaths_Mean"] / total_df["Avg_Population"] * 100000)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    total_df["Rate_per_100k_Lower"] = (
        (total_df["Annual_Deaths_Lower"] / total_df["Avg_Population"] * 100000)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    total_df["Rate_per_100k_Upper"] = (
        (total_df["Annual_Deaths_Upper"] / total_df["Avg_Population"] * 100000)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    return pd.concat([agg, total_df], ignore_index=True)


overall_start = time.time()
for model_name, model_cfg in MODELS.items():
    out_dir = DATA_DIR / f"Ukraine_Belarus_Projections_{model_name}"
    out_dir.mkdir(exist_ok=True)
    all_excess = []
    all_grid_deaths = []
    all_country_deaths = []
    all_grid_rates = []
    all_country_rates = []
    all_period_grid = []
    all_period_country = []
    for ssp_long, ssp_short in SSP_MAP.items():
        cvd = pd.read_csv(DATA_DIR / model_cfg["cvd_file"](ssp_short))
        pop_wide = pd.read_csv(DATA_DIR / model_cfg["pop_file"](ssp_short))
        scenario_grid_pop_rows = []
        for period_name, period_years in PERIODS.items():
            temp_file = DATA_DIR / TEMP_FILE_TEMPLATE.format(
                ssp_long=ssp_long, period=period_name
            )
            temp_df = pd.read_csv(temp_file)
            temp_df["date"] = pd.to_datetime(temp_df["date"])
            temp_df["year"] = temp_df["date"].dt.year
            for year in period_years:
                year_start = time.time()
                temp_year = temp_df[temp_df["year"] == year]
                if len(temp_year) == 0:
                    continue
                excess_year = compute_excess_for_year(temp_year)
                excess_year["year"] = year
                excess_year["scenario"] = ssp_long
                all_excess.append(excess_year.copy())
                merged = excess_year.merge(zones, on="grid_id", how="inner")
                merged = merged.rename(
                    columns={
                        "avg_daily_heat_excess": "heat_excess",
                        "avg_daily_cold_excess": "cold_excess",
                    }
                )
                cvd_cols = ["grid_id", "country"] + [
                    f"cvd_deaths_{stat}_{b}_{year}"
                    for b in AGE_BRACKETS
                    for stat in ["mean", "max", "min"]
                ]
                cvd_year = cvd[cvd_cols].copy()
                rename_map = {
                    f"cvd_deaths_{stat}_{b}_{year}": f"cvd_deaths_{stat}_{b}"
                    for b in AGE_BRACKETS
                    for stat in ["mean", "max", "min"]
                }
                cvd_year = cvd_year.rename(columns=rename_map)
                merged = merged.merge(cvd_year, on="grid_id", how="inner")
                grid_deaths_year = run_monte_carlo(merged)
                grid_deaths_year["year"] = year
                grid_deaths_year["scenario"] = ssp_long
                all_grid_deaths.append(grid_deaths_year.copy())
                country_deaths_year = aggregate_country_year(grid_deaths_year)
                country_deaths_year["year"] = year
                country_deaths_year["scenario"] = ssp_long
                all_country_deaths.append(country_deaths_year.copy())
                pop_cols_year = ["grid_id", "country"] + [
                    f"{b}_{year}" for b in AGE_BRACKETS
                ]
                pop_year = pop_wide[pop_cols_year].copy()
                pop_year = pop_year.rename(
                    columns={f"{b}_{year}": f"population_{b}" for b in AGE_BRACKETS}
                )
                pop_year["population_total"] = pop_year[
                    [f"population_{b}" for b in AGE_BRACKETS]
                ].sum(axis=1)
                pop_year["year"] = year
                scenario_grid_pop_rows.append(pop_year.copy())
                grid_rates_year = add_rates(
                    grid_deaths_year,
                    pop_year.drop(columns="year"),
                    ["grid_id", "country"],
                )
                grid_rates_year["year"] = year
                all_grid_rates.append(grid_rates_year.copy())
                country_pop_year = (
                    pop_year.drop(columns="year")
                    .groupby("country")[
                        [f"population_{b}" for b in AGE_BRACKETS] + ["population_total"]
                    ]
                    .sum()
                    .reset_index()
                )
                total_pop_row = {"country": "TOTAL"}
                for c in [f"population_{b}" for b in AGE_BRACKETS] + [
                    "population_total"
                ]:
                    total_pop_row[c] = country_pop_year[c].sum()
                country_pop_year = pd.concat(
                    [country_pop_year, pd.DataFrame([total_pop_row])], ignore_index=True
                )
                country_rates_year = add_rates(
                    country_deaths_year, country_pop_year, ["country"]
                )
                country_rates_year["year"] = year
                all_country_rates.append(country_rates_year.copy())
        scenario_grid_pop_all = pd.concat(scenario_grid_pop_rows, ignore_index=True)
        scenario_grid_deaths_all = pd.concat(
            [d for d in all_grid_deaths if (d["scenario"] == ssp_long).all()],
            ignore_index=True,
        )
        for period_name, period_years in PERIODS.items():
            period_stats = compute_period_stats(
                scenario_grid_deaths_all,
                scenario_grid_pop_all,
                period_years,
                PERIOD_LABELS[period_name],
                ssp_long,
            )
            all_period_grid.append(period_stats)
    excess_df = pd.concat(all_excess, ignore_index=True)
    grid_deaths_df = pd.concat(all_grid_deaths, ignore_index=True)
    country_deaths_df = pd.concat(all_country_deaths, ignore_index=True)
    grid_rates_df = pd.concat(all_grid_rates, ignore_index=True)
    country_rates_df = pd.concat(all_country_rates, ignore_index=True)
    period_grid_df = pd.concat(all_period_grid, ignore_index=True)
    period_country_df = aggregate_period_to_country(period_grid_df)
    excess_df.to_csv(out_dir / "Grid_Heat_Cold_Excess_by_Year.csv", index=False)
    grid_deaths_df.to_csv(out_dir / "Grid_Attributable_Deaths_by_Year.csv", index=False)
    country_deaths_df.to_csv(
        out_dir / "Country_Attributable_Deaths_by_Year.csv", index=False
    )
    grid_rates_df.to_csv(out_dir / "Grid_Mortality_Rates_by_Year.csv", index=False)
    country_rates_df.to_csv(
        out_dir / "Country_Mortality_Rates_by_Year.csv", index=False
    )
    period_grid_df.to_csv(out_dir / "Period_Mean_Grid_Mortality.csv", index=False)
    period_country_df.to_csv(out_dir / "Period_Mean_Country_Mortality.csv", index=False)
    summary = period_country_df[
        (period_country_df["country"] == "TOTAL")
        & (period_country_df["Age_Group"] == "All_Ages")
        & (period_country_df["Temperature_Type"] == "Total")
    ]
    for _, r in summary.iterrows():
        pass
total_time = time.time() - overall_start
