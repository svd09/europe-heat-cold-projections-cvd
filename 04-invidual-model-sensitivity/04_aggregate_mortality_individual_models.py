import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc

desktop_path = Path("data")
input_dir = desktop_path / "Model_Mortality_Results"
output_dir = desktop_path / "Model_Mortality_Results_Aggregated"
output_dir.mkdir(exist_ok=True)
models = ["CNRM-ESM2-1", "GFDL-ESM4", "MIROC6", "NorESM2-MM", "UKESM1-0-LL"]
scenarios = ["SSP245", "SSP585"]
un_regions_file = desktop_path / "UN_Geoscheme_Classification.csv"
population_ssp2_file = (
    desktop_path / "Europe_Grid_SSP2_Median_Fert_Age_Pop_2025_2080.csv"
)
population_ssp5_file = (
    desktop_path / "Europe_Grid_SSP5_Median_Fert_Age_Pop_2025_2080.csv"
)
age_groups = {
    "under_20": "pop_under_20",
    "20_54": "pop_20_54",
    "55_64": "pop_55_64",
    "65_74": "pop_65_74",
    "75plus": "pop_75plus",
}
missing = []
for f, label in [
    (un_regions_file, "UN regions"),
    (population_ssp2_file, "Population SSP2"),
    (population_ssp5_file, "Population SSP5"),
]:
    if f.exists():
        pass
    else:
        missing.append(str(f))
for model in models:
    for scenario in scenarios:
        fp = input_dir / f"{model}_{scenario}_Mortality_Projections.csv"
        if fp.exists():
            pass
        else:
            missing.append(str(fp))
if missing:
    for f in missing:
        pass
    exit(1)
un_regions = pd.read_csv(un_regions_file)
pop_ssp2 = pd.read_csv(population_ssp2_file)
pop_ssp5 = pd.read_csv(population_ssp5_file)


def fix_inf_population(pop_df, label=""):
    age_groups_list = ["under_20", "20_54", "55_64", "65_74", "75plus"]
    initial_inf = sum(
        (
            np.isinf(pop_df[col]).sum()
            for col in pop_df.columns
            if col.startswith("pop_")
        )
    )
    if initial_inf == 0:
        return pop_df
    pop_df = pop_df.copy()
    for age in age_groups_list:
        age_cols = sorted(
            [
                c
                for c in pop_df.columns
                if c.startswith(f"pop_{age}_") and c.split("_")[-1].isdigit()
            ],
            key=lambda x: int(x.split("_")[-1]),
        )
        for idx in range(len(pop_df)):
            last_valid = None
            for col in age_cols:
                val = pop_df.at[idx, col]
                if (
                    isinstance(val, (int, float))
                    and (not np.isinf(val))
                    and (not np.isnan(val))
                ):
                    last_valid = val
                elif last_valid is not None:
                    pop_df.at[idx, col] = last_valid
    final_inf = sum(
        (
            np.isinf(pop_df[col]).sum()
            for col in pop_df.columns
            if col.startswith("pop_")
        )
    )
    if final_inf > 0:
        for col in pop_df.columns:
            if col.startswith("pop_"):
                mask = np.isinf(pop_df[col])
                if mask.any():
                    median_val = pop_df.loc[~mask, col].median()
                    pop_df.loc[mask, col] = median_val
    else:
        pass
    return pop_df


pop_ssp2 = fix_inf_population(pop_ssp2, "SSP2")
pop_ssp5 = fix_inf_population(pop_ssp5, "SSP5")


def get_grid_pop_year(pop_df, year):
    cols = {"grid_id": "grid_id", "Country": "Country"}
    for age, prefix in age_groups.items():
        cols[f"{prefix}_{year}"] = prefix
    available = {k: v for k, v in cols.items() if k in pop_df.columns}
    pop_year = pop_df[list(available.keys())].copy().rename(columns=available)
    pop_year["population_total"] = sum(
        (pop_year[p] for p in age_groups.values() if p in pop_year.columns)
    )
    return pop_year


def get_country_pop_year(pop_df, year):
    pop_cols = [f"{p}_{year}" for p in age_groups.values()]
    available = [c for c in pop_cols if c in pop_df.columns]
    agg_dict = {c: "sum" for c in available}
    country_pop = (
        pop_df[["Country"] + available].groupby("Country").agg(agg_dict).reset_index()
    )
    rename = {f"{p}_{year}": p for p in age_groups.values()}
    country_pop = country_pop.rename(columns=rename)
    country_pop["population_total"] = sum(
        (country_pop[p] for p in age_groups.values() if p in country_pop.columns)
    )
    return country_pop


def get_region_pop_year(pop_df, un_regions_df, year):
    pop_cols = [f"{p}_{year}" for p in age_groups.values()]
    available = [c for c in pop_cols if c in pop_df.columns]
    merged = pop_df[["Country"] + available].merge(
        un_regions_df, on="Country", how="left"
    )
    agg_dict = {c: "sum" for c in available}
    region_pop = merged.groupby("UN_Region").agg(agg_dict).reset_index()
    rename = {f"{p}_{year}": p for p in age_groups.values()}
    region_pop = region_pop.rename(columns=rename)
    region_pop["population_total"] = sum(
        (region_pop[p] for p in age_groups.values() if p in region_pop.columns)
    )
    return region_pop


def aggregate_to_country(year_df, year, un_regions_df):
    rows = []
    for country, grp in year_df.groupby("Country"):
        row = {"year": year, "Country": country, "N_Grids": len(grp)}
        for age in age_groups:
            for metric in ["heat", "cold", "net"]:
                for stat in ["mean", "lower", "upper"]:
                    col = f"{metric}_deaths_{age}_{stat}"
                    if col in grp.columns:
                        row[col] = grp[col].sum()
        for metric in ["heat", "cold", "net"]:
            for stat in ["mean", "lower", "upper"]:
                col = f"{metric}_deaths_total_{stat}"
                if col in grp.columns:
                    row[col] = grp[col].sum()
        rows.append(row)
    total_row = {"year": year, "Country": "TOTAL", "N_Grids": len(year_df)}
    for age in age_groups:
        for metric in ["heat", "cold", "net"]:
            for stat in ["mean", "lower", "upper"]:
                col = f"{metric}_deaths_{age}_{stat}"
                if col in year_df.columns:
                    total_row[col] = year_df[col].sum()
    for metric in ["heat", "cold", "net"]:
        for stat in ["mean", "lower", "upper"]:
            col = f"{metric}_deaths_total_{stat}"
            if col in year_df.columns:
                total_row[col] = year_df[col].sum()
    rows.append(total_row)
    return pd.DataFrame(rows)


def aggregate_to_region(year_df, year, un_regions_df, _warned=set()):
    if "UN_Region" in year_df.columns:
        year_df_reg = year_df.copy()
    else:
        year_df_reg = year_df.merge(un_regions_df, on="Country", how="left")
    unmatched = year_df_reg[year_df_reg["UN_Region"].isna()]["Country"].unique()
    new_unmatched = [c for c in unmatched if c not in _warned]
    if new_unmatched:
        for c in sorted(new_unmatched):
            pass
        _warned.update(new_unmatched)
    year_df_reg = year_df_reg.dropna(subset=["UN_Region"])
    rows = []
    for region, grp in year_df_reg.groupby("UN_Region"):
        row = {"year": year, "UN_Region": region, "N_Grids": len(grp)}
        for age in age_groups:
            for metric in ["heat", "cold", "net"]:
                for stat in ["mean", "lower", "upper"]:
                    col = f"{metric}_deaths_{age}_{stat}"
                    if col in grp.columns:
                        row[col] = grp[col].sum()
        for metric in ["heat", "cold", "net"]:
            for stat in ["mean", "lower", "upper"]:
                col = f"{metric}_deaths_total_{stat}"
                if col in grp.columns:
                    row[col] = grp[col].sum()
        rows.append(row)
    total_row = {"year": year, "UN_Region": "TOTAL", "N_Grids": len(year_df)}
    for age in age_groups:
        for metric in ["heat", "cold", "net"]:
            for stat in ["mean", "lower", "upper"]:
                col = f"{metric}_deaths_{age}_{stat}"
                if col in year_df.columns:
                    total_row[col] = year_df[col].sum()
    for metric in ["heat", "cold", "net"]:
        for stat in ["mean", "lower", "upper"]:
            col = f"{metric}_deaths_total_{stat}"
            if col in year_df.columns:
                total_row[col] = year_df[col].sum()
    rows.append(total_row)
    return pd.DataFrame(rows)


def split_grid_deaths(grid_df):
    base_cols = ["year", "grid_id", "Country", "climate_zone"]
    hc_cols = base_cols.copy()
    for age in age_groups:
        for metric in ["heat", "cold"]:
            for stat in ["mean", "lower", "upper"]:
                c = f"{metric}_deaths_{age}_{stat}"
                if c in grid_df.columns:
                    hc_cols.append(c)
    for metric in ["heat", "cold"]:
        for stat in ["mean", "lower", "upper"]:
            c = f"{metric}_deaths_total_{stat}"
            if c in grid_df.columns:
                hc_cols.append(c)
    net_cols = base_cols.copy()
    for age in age_groups:
        for stat in ["mean", "lower", "upper"]:
            c = f"net_deaths_{age}_{stat}"
            if c in grid_df.columns:
                net_cols.append(c)
    for stat in ["mean", "lower", "upper"]:
        c = f"net_deaths_total_{stat}"
        if c in grid_df.columns:
            net_cols.append(c)
    return (grid_df[hc_cols], grid_df[net_cols])


def split_country_deaths(country_df):
    base_cols = ["year", "Country", "N_Grids"]
    hc_cols = base_cols.copy()
    for age in age_groups:
        for metric in ["heat", "cold"]:
            for stat in ["mean", "lower", "upper"]:
                c = f"{metric}_deaths_{age}_{stat}"
                if c in country_df.columns:
                    hc_cols.append(c)
    for metric in ["heat", "cold"]:
        for stat in ["mean", "lower", "upper"]:
            c = f"{metric}_deaths_total_{stat}"
            if c in country_df.columns:
                hc_cols.append(c)
    net_cols = base_cols.copy()
    for age in age_groups:
        for stat in ["mean", "lower", "upper"]:
            c = f"net_deaths_{age}_{stat}"
            if c in country_df.columns:
                net_cols.append(c)
    for stat in ["mean", "lower", "upper"]:
        c = f"net_deaths_total_{stat}"
        if c in country_df.columns:
            net_cols.append(c)
    return (country_df[hc_cols], country_df[net_cols])


def split_region_deaths(region_df):
    base_cols = ["year", "UN_Region", "N_Grids"]
    hc_cols = base_cols.copy()
    for age in age_groups:
        for metric in ["heat", "cold"]:
            for stat in ["mean", "lower", "upper"]:
                c = f"{metric}_deaths_{age}_{stat}"
                if c in region_df.columns:
                    hc_cols.append(c)
    for metric in ["heat", "cold"]:
        for stat in ["mean", "lower", "upper"]:
            c = f"{metric}_deaths_total_{stat}"
            if c in region_df.columns:
                hc_cols.append(c)
    net_cols = base_cols.copy()
    for age in age_groups:
        for stat in ["mean", "lower", "upper"]:
            c = f"net_deaths_{age}_{stat}"
            if c in region_df.columns:
                net_cols.append(c)
    for stat in ["mean", "lower", "upper"]:
        c = f"net_deaths_total_{stat}"
        if c in region_df.columns:
            net_cols.append(c)
    return (region_df[hc_cols], region_df[net_cols])


def calc_grid_rates(grid_hc_df, grid_net_df, pop_df, years):
    all_hc_rates = []
    all_net_rates = []
    for year in years:
        pop_year = get_grid_pop_year(pop_df, year)
        hc_year = grid_hc_df[grid_hc_df["year"] == year].copy()
        if len(hc_year) == 0:
            continue
        merged = hc_year.merge(pop_year, on=["grid_id", "Country"], how="inner")
        for age, pop_col in age_groups.items():
            for metric in ["heat", "cold"]:
                for stat in ["mean", "lower", "upper"]:
                    d_col = f"{metric}_deaths_{age}_{stat}"
                    r_col = f"{metric}_rate_{age}_{stat}"
                    if d_col in merged.columns and pop_col in merged.columns:
                        merged[r_col] = (
                            merged[d_col] / merged[pop_col] * 100000
                        ).replace([np.inf, -np.inf], np.nan)
        for metric in ["heat", "cold"]:
            for stat in ["mean", "lower", "upper"]:
                d_col = f"{metric}_deaths_total_{stat}"
                r_col = f"{metric}_rate_total_{stat}"
                if d_col in merged.columns:
                    merged[r_col] = (
                        merged[d_col] / merged["population_total"] * 100000
                    ).replace([np.inf, -np.inf], np.nan)
        rate_cols = ["year", "grid_id", "Country", "climate_zone", "population_total"]
        rate_cols += [
            c
            for c in merged.columns
            if c.startswith("heat_rate_") or c.startswith("cold_rate_")
        ]
        all_hc_rates.append(merged[[c for c in rate_cols if c in merged.columns]])
        net_year = grid_net_df[grid_net_df["year"] == year].copy()
        net_merged = net_year.merge(pop_year, on=["grid_id", "Country"], how="inner")
        for age, pop_col in age_groups.items():
            for stat in ["mean", "lower", "upper"]:
                d_col = f"net_deaths_{age}_{stat}"
                r_col = f"net_rate_{age}_{stat}"
                if d_col in net_merged.columns and pop_col in net_merged.columns:
                    net_merged[r_col] = (
                        net_merged[d_col] / net_merged[pop_col] * 100000
                    ).replace([np.inf, -np.inf], np.nan)
        for stat in ["mean", "lower", "upper"]:
            d_col = f"net_deaths_total_{stat}"
            r_col = f"net_rate_total_{stat}"
            if d_col in net_merged.columns:
                net_merged[r_col] = (
                    net_merged[d_col] / net_merged["population_total"] * 100000
                ).replace([np.inf, -np.inf], np.nan)
        net_rate_cols = [
            "year",
            "grid_id",
            "Country",
            "climate_zone",
            "population_total",
        ]
        net_rate_cols += [c for c in net_merged.columns if c.startswith("net_rate_")]
        all_net_rates.append(
            net_merged[[c for c in net_rate_cols if c in net_merged.columns]]
        )
    return (
        pd.concat(all_hc_rates, ignore_index=True),
        pd.concat(all_net_rates, ignore_index=True),
    )


def calc_country_rates(country_hc_df, country_net_df, pop_df, years):
    all_hc_rates = []
    all_net_rates = []
    for year in years:
        country_pop = get_country_pop_year(pop_df, year)
        hc_year = country_hc_df[country_hc_df["year"] == year].copy()
        if len(hc_year) == 0:
            continue
        merged = hc_year.merge(country_pop, on="Country", how="left")
        if (merged["Country"] == "TOTAL").any():
            total_pop_total = country_pop["population_total"].sum()
            merged.loc[merged["Country"] == "TOTAL", "population_total"] = (
                total_pop_total
            )
            for age, pop_col in age_groups.items():
                if pop_col in country_pop.columns:
                    merged.loc[merged["Country"] == "TOTAL", pop_col] = country_pop[
                        pop_col
                    ].sum()
        for age, pop_col in age_groups.items():
            for metric in ["heat", "cold"]:
                for stat in ["mean", "lower", "upper"]:
                    d_col = f"{metric}_deaths_{age}_{stat}"
                    r_col = f"{metric}_rate_{age}_{stat}"
                    if d_col in merged.columns and pop_col in merged.columns:
                        merged[r_col] = (
                            merged[d_col] / merged[pop_col] * 100000
                        ).replace([np.inf, -np.inf], np.nan)
        for metric in ["heat", "cold"]:
            for stat in ["mean", "lower", "upper"]:
                d_col = f"{metric}_deaths_total_{stat}"
                r_col = f"{metric}_rate_total_{stat}"
                if d_col in merged.columns:
                    merged[r_col] = (
                        merged[d_col] / merged["population_total"] * 100000
                    ).replace([np.inf, -np.inf], np.nan)
        rate_cols = ["year", "Country", "N_Grids", "population_total"]
        rate_cols += [
            c
            for c in merged.columns
            if c.startswith("heat_rate_") or c.startswith("cold_rate_")
        ]
        all_hc_rates.append(merged[[c for c in rate_cols if c in merged.columns]])
        net_year = country_net_df[country_net_df["year"] == year].copy()
        net_merged = net_year.merge(country_pop, on="Country", how="left")
        if (net_merged["Country"] == "TOTAL").any():
            net_merged.loc[net_merged["Country"] == "TOTAL", "population_total"] = (
                country_pop["population_total"].sum()
            )
            for age, pop_col in age_groups.items():
                if pop_col in country_pop.columns:
                    net_merged.loc[net_merged["Country"] == "TOTAL", pop_col] = (
                        country_pop[pop_col].sum()
                    )
        for age, pop_col in age_groups.items():
            for stat in ["mean", "lower", "upper"]:
                d_col = f"net_deaths_{age}_{stat}"
                r_col = f"net_rate_{age}_{stat}"
                if d_col in net_merged.columns and pop_col in net_merged.columns:
                    net_merged[r_col] = (
                        net_merged[d_col] / net_merged[pop_col] * 100000
                    ).replace([np.inf, -np.inf], np.nan)
        for stat in ["mean", "lower", "upper"]:
            d_col = f"net_deaths_total_{stat}"
            r_col = f"net_rate_total_{stat}"
            if d_col in net_merged.columns:
                net_merged[r_col] = (
                    net_merged[d_col] / net_merged["population_total"] * 100000
                ).replace([np.inf, -np.inf], np.nan)
        net_rate_cols = ["year", "Country", "N_Grids", "population_total"]
        net_rate_cols += [c for c in net_merged.columns if c.startswith("net_rate_")]
        all_net_rates.append(
            net_merged[[c for c in net_rate_cols if c in net_merged.columns]]
        )
    return (
        pd.concat(all_hc_rates, ignore_index=True),
        pd.concat(all_net_rates, ignore_index=True),
    )


def calc_region_rates(region_hc_df, region_net_df, pop_df, un_regions_df, years):
    all_hc_rates = []
    all_net_rates = []
    for year in years:
        region_pop = get_region_pop_year(pop_df, un_regions_df, year)
        hc_year = region_hc_df[region_hc_df["year"] == year].copy()
        if len(hc_year) == 0:
            continue
        merged = hc_year.merge(region_pop, on="UN_Region", how="left")
        if (merged["UN_Region"] == "TOTAL").any():
            merged.loc[merged["UN_Region"] == "TOTAL", "population_total"] = region_pop[
                "population_total"
            ].sum()
            for age, pop_col in age_groups.items():
                if pop_col in region_pop.columns:
                    merged.loc[merged["UN_Region"] == "TOTAL", pop_col] = region_pop[
                        pop_col
                    ].sum()
        for age, pop_col in age_groups.items():
            for metric in ["heat", "cold"]:
                for stat in ["mean", "lower", "upper"]:
                    d_col = f"{metric}_deaths_{age}_{stat}"
                    r_col = f"{metric}_rate_{age}_{stat}"
                    if d_col in merged.columns and pop_col in merged.columns:
                        merged[r_col] = (
                            merged[d_col] / merged[pop_col] * 100000
                        ).replace([np.inf, -np.inf], np.nan)
        for metric in ["heat", "cold"]:
            for stat in ["mean", "lower", "upper"]:
                d_col = f"{metric}_deaths_total_{stat}"
                r_col = f"{metric}_rate_total_{stat}"
                if d_col in merged.columns:
                    merged[r_col] = (
                        merged[d_col] / merged["population_total"] * 100000
                    ).replace([np.inf, -np.inf], np.nan)
        rate_cols = ["year", "UN_Region", "N_Grids", "population_total"]
        rate_cols += [
            c
            for c in merged.columns
            if c.startswith("heat_rate_") or c.startswith("cold_rate_")
        ]
        all_hc_rates.append(merged[[c for c in rate_cols if c in merged.columns]])
        net_year = region_net_df[region_net_df["year"] == year].copy()
        net_merged = net_year.merge(region_pop, on="UN_Region", how="left")
        if (net_merged["UN_Region"] == "TOTAL").any():
            net_merged.loc[net_merged["UN_Region"] == "TOTAL", "population_total"] = (
                region_pop["population_total"].sum()
            )
            for age, pop_col in age_groups.items():
                if pop_col in region_pop.columns:
                    net_merged.loc[net_merged["UN_Region"] == "TOTAL", pop_col] = (
                        region_pop[pop_col].sum()
                    )
        for age, pop_col in age_groups.items():
            for stat in ["mean", "lower", "upper"]:
                d_col = f"net_deaths_{age}_{stat}"
                r_col = f"net_rate_{age}_{stat}"
                if d_col in net_merged.columns and pop_col in net_merged.columns:
                    net_merged[r_col] = (
                        net_merged[d_col] / net_merged[pop_col] * 100000
                    ).replace([np.inf, -np.inf], np.nan)
        for stat in ["mean", "lower", "upper"]:
            d_col = f"net_deaths_total_{stat}"
            r_col = f"net_rate_total_{stat}"
            if d_col in net_merged.columns:
                net_merged[r_col] = (
                    net_merged[d_col] / net_merged["population_total"] * 100000
                ).replace([np.inf, -np.inf], np.nan)
        net_rate_cols = ["year", "UN_Region", "N_Grids", "population_total"]
        net_rate_cols += [c for c in net_merged.columns if c.startswith("net_rate_")]
        all_net_rates.append(
            net_merged[[c for c in net_rate_cols if c in net_merged.columns]]
        )
    return (
        pd.concat(all_hc_rates, ignore_index=True),
        pd.concat(all_net_rates, ignore_index=True),
    )


overall_start = time.time()
files_processed = 0
for model in models:
    for scenario in scenarios:
        files_processed += 1
        combo = f"{model} - {scenario}"
        combo_start = time.time()
        pop_df = pop_ssp2 if scenario == "SSP245" else pop_ssp5
        grid_file = input_dir / f"{model}_{scenario}_Mortality_Projections.csv"
        grid_all = pd.read_csv(grid_file)
        years = sorted(grid_all["year"].unique())
        grid_countries = set(grid_all["Country"].unique())
        un_countries = set(un_regions["Country"].unique())
        unmatched = grid_countries - un_countries
        if unmatched:
            pass
        COUNTRY_NAME_MAP = {
            "Czech Republic": "Czechia",
            "Moldova": "Republic of Moldova",
            "Bosnia & Herzegovina": "Bosnia and Herzegovina",
            "Bosnia-Herzegovina": "Bosnia and Herzegovina",
            "North Macedonia": "North Macedonia",
            "Republic of North Macedonia": "North Macedonia",
            "FYROM": "North Macedonia",
            "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
            "UK": "United Kingdom",
        }
        grid_all["Country"] = grid_all["Country"].replace(COUNTRY_NAME_MAP)
        still_unmatched = set(grid_all["Country"].unique()) - un_countries
        if still_unmatched:
            for c in sorted(still_unmatched):
                pass
        else:
            pass
        all_country = []
        all_region = []
        for year in years:
            year_df = grid_all[grid_all["year"] == year].copy()
            all_country.append(aggregate_to_country(year_df, year, un_regions))
            all_region.append(aggregate_to_region(year_df, year, un_regions))
        country_all = pd.concat(all_country, ignore_index=True)
        region_all = pd.concat(all_region, ignore_index=True)
        grid_hc, grid_net = split_grid_deaths(grid_all)
        country_hc, country_net = split_country_deaths(country_all)
        region_hc, region_net = split_region_deaths(region_all)
        grid_hc_rates, grid_net_rates = calc_grid_rates(
            grid_hc, grid_net, pop_df, years
        )
        country_hc_rates, country_net_rates = calc_country_rates(
            country_hc, country_net, pop_df, years
        )
        region_hc_rates, region_net_rates = calc_region_rates(
            region_hc, region_net, pop_df, un_regions, years
        )
        prefix = f"{model}_{scenario}"
        outputs = {
            f"{prefix}_Grid_Deaths.csv": grid_hc,
            f"{prefix}_Country_Deaths.csv": country_hc,
            f"{prefix}_Region_Deaths.csv": region_hc,
            f"{prefix}_Grid_Net_Deaths.csv": grid_net,
            f"{prefix}_Country_Net_Deaths.csv": country_net,
            f"{prefix}_Region_Net_Deaths.csv": region_net,
            f"{prefix}_Grid_Rates.csv": grid_hc_rates,
            f"{prefix}_Country_Rates.csv": country_hc_rates,
            f"{prefix}_Region_Rates.csv": region_hc_rates,
            f"{prefix}_Grid_Net_Rates.csv": grid_net_rates,
            f"{prefix}_Country_Net_Rates.csv": country_net_rates,
            f"{prefix}_Region_Net_Rates.csv": region_net_rates,
        }
        for filename, df in outputs.items():
            fp = output_dir / filename
            df.to_csv(fp, index=False)
            size_mb = fp.stat().st_size / 1024**2
        combo_time = time.time() - combo_start
        del grid_all, country_all, region_all
        del grid_hc, grid_net, country_hc, country_net, region_hc, region_net
        del grid_hc_rates, grid_net_rates, country_hc_rates, country_net_rates
        del region_hc_rates, region_net_rates
        gc.collect()
total_time = time.time() - overall_start
