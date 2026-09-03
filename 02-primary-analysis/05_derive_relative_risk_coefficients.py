import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

temp_file = "data/Europe_ERA5_2016-23.csv"
climate_file = "data/Europe_Grid_Climate_Zones.csv"
output_file = "data/Europe_RR_Coefficients_by_ClimateZone_WithCI.csv"
RR_VALUES = {
    "1st_percentile": {"rr": 1.32, "lower": 1.27, "upper": 1.38},
    "90th_percentile": {"rr": 1.0, "lower": 1.0, "upper": 1.0},
    "99th_percentile": {"rr": 1.11, "lower": 1.07, "upper": 1.14},
}
temp_df = pd.read_csv(temp_file)
climate_df = pd.read_csv(climate_file)
data = temp_df.merge(climate_df[["grid_id", "climate_zone"]], on="grid_id", how="left")
missing_climate = data["climate_zone"].isna().sum()
if missing_climate > 0:
    data = data.dropna(subset=["climate_zone"])
climate_zones = sorted(data["climate_zone"].unique())
percentile_results = []
for zone in climate_zones:
    zone_data = data[data["climate_zone"] == zone]["temp_mean"].dropna()
    if len(zone_data) == 0:
        continue
    p1 = np.percentile(zone_data, 1)
    p90 = np.percentile(zone_data, 90)
    p99 = np.percentile(zone_data, 99)
    percentile_results.append(
        {
            "climate_zone": zone,
            "n_observations": len(zone_data),
            "temp_1st": p1,
            "temp_90th_MMT": p90,
            "temp_99th": p99,
            "temp_mean": zone_data.mean(),
            "temp_std": zone_data.std(),
        }
    )
percentile_df = pd.DataFrame(percentile_results)
rr_coefficients = []
for idx, row in percentile_df.iterrows():
    zone = row["climate_zone"]
    t_1st = row["temp_1st"]
    t_mmt = row["temp_90th_MMT"]
    t_99th = row["temp_99th"]
    temps_cold = np.array([t_1st, t_mmt])
    delta_t_cold = t_mmt - temps_cold
    rr_cold_mean = np.array(
        [RR_VALUES["1st_percentile"]["rr"], RR_VALUES["90th_percentile"]["rr"]]
    )
    ln_rr_cold_mean = np.log(rr_cold_mean)
    slope_cold_mean, _, r_cold_mean, _, _ = linregress(delta_t_cold, ln_rr_cold_mean)
    beta_cold = slope_cold_mean
    rr_cold_lower = np.array(
        [RR_VALUES["1st_percentile"]["upper"], RR_VALUES["90th_percentile"]["rr"]]
    )
    ln_rr_cold_lower = np.log(rr_cold_lower)
    slope_cold_lower, _, _, _, _ = linregress(delta_t_cold, ln_rr_cold_lower)
    beta_cold_lower = slope_cold_lower
    rr_cold_upper = np.array(
        [RR_VALUES["1st_percentile"]["lower"], RR_VALUES["90th_percentile"]["rr"]]
    )
    ln_rr_cold_upper = np.log(rr_cold_upper)
    slope_cold_upper, _, _, _, _ = linregress(delta_t_cold, ln_rr_cold_upper)
    beta_cold_upper = slope_cold_upper
    rr_1st_predicted = np.exp(beta_cold * (t_mmt - t_1st))
    temps_heat = np.array([t_mmt, t_99th])
    delta_t_heat = temps_heat - t_mmt
    rr_heat_mean = np.array(
        [RR_VALUES["90th_percentile"]["rr"], RR_VALUES["99th_percentile"]["rr"]]
    )
    ln_rr_heat_mean = np.log(rr_heat_mean)
    slope_heat_mean, _, r_heat_mean, _, _ = linregress(delta_t_heat, ln_rr_heat_mean)
    beta_heat = slope_heat_mean
    rr_heat_lower = np.array(
        [RR_VALUES["90th_percentile"]["rr"], RR_VALUES["99th_percentile"]["lower"]]
    )
    ln_rr_heat_lower = np.log(rr_heat_lower)
    slope_heat_lower, _, _, _, _ = linregress(delta_t_heat, ln_rr_heat_lower)
    beta_heat_lower = slope_heat_lower
    rr_heat_upper = np.array(
        [RR_VALUES["90th_percentile"]["rr"], RR_VALUES["99th_percentile"]["upper"]]
    )
    ln_rr_heat_upper = np.log(rr_heat_upper)
    slope_heat_upper, _, _, _, _ = linregress(delta_t_heat, ln_rr_heat_upper)
    beta_heat_upper = slope_heat_upper
    rr_99th_predicted = np.exp(beta_heat * (t_99th - t_mmt))
    rr_coefficients.append(
        {
            "climate_zone": zone,
            "n_observations": row["n_observations"],
            "temp_1st": t_1st,
            "temp_MMT": t_mmt,
            "temp_99th": t_99th,
            "beta_cold": beta_cold,
            "beta_cold_lower": beta_cold_upper,
            "beta_cold_upper": beta_cold_lower,
            "beta_heat": beta_heat,
            "beta_heat_lower": beta_heat_lower,
            "beta_heat_upper": beta_heat_upper,
            "r2_cold": r_cold_mean**2,
            "r2_heat": r_heat_mean**2,
        }
    )
rr_coef_df = pd.DataFrame(rr_coefficients)
rr_coef_df.to_csv(output_file, index=False)
for idx, row in rr_coef_df.iterrows():
    zone = row["climate_zone"]
    t_1st = row["temp_1st"]
    t_mmt = row["temp_MMT"]
    t_99th = row["temp_99th"]
    beta_cold = row["beta_cold"]
    beta_cold_lower = row["beta_cold_lower"]
    beta_cold_upper = row["beta_cold_upper"]
    beta_heat = row["beta_heat"]
    beta_heat_lower = row["beta_heat_lower"]
    beta_heat_upper = row["beta_heat_upper"]
    temp_range_cold = np.linspace(t_1st - 5, t_mmt, 100)
    temp_range_heat = np.linspace(t_mmt, t_99th + 5, 100)
    rr_cold_mean = np.exp(beta_cold * (t_mmt - temp_range_cold))
    rr_heat_mean = np.exp(beta_heat * (temp_range_heat - t_mmt))
    rr_cold_lower = np.exp(beta_cold_lower * (t_mmt - temp_range_cold))
    rr_heat_lower = np.exp(beta_heat_lower * (temp_range_heat - t_mmt))
    rr_cold_upper = np.exp(beta_cold_upper * (t_mmt - temp_range_cold))
    rr_heat_upper = np.exp(beta_heat_upper * (temp_range_heat - t_mmt))
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        temp_range_cold, rr_cold_mean, "b-", linewidth=3, label="Cold (Mean)", zorder=5
    )
    ax.fill_between(
        temp_range_cold,
        rr_cold_lower,
        rr_cold_upper,
        color="blue",
        alpha=0.2,
        label="Cold (95% CI)",
        zorder=3,
    )
    ax.plot(
        temp_range_heat, rr_heat_mean, "r-", linewidth=3, label="Heat (Mean)", zorder=5
    )
    ax.fill_between(
        temp_range_heat,
        rr_heat_lower,
        rr_heat_upper,
        color="red",
        alpha=0.2,
        label="Heat (95% CI)",
        zorder=3,
    )
    ax.plot(
        t_1st,
        RR_VALUES["1st_percentile"]["rr"],
        "bo",
        markersize=12,
        label=f"1st %ile: {t_1st:.1f}°C, RR={RR_VALUES['1st_percentile']['rr']}",
        zorder=10,
    )
    ax.errorbar(
        t_1st,
        RR_VALUES["1st_percentile"]["rr"],
        yerr=[
            [RR_VALUES["1st_percentile"]["rr"] - RR_VALUES["1st_percentile"]["lower"]],
            [RR_VALUES["1st_percentile"]["upper"] - RR_VALUES["1st_percentile"]["rr"]],
        ],
        fmt="none",
        color="blue",
        capsize=5,
        linewidth=2,
        zorder=10,
    )
    ax.plot(
        t_mmt,
        RR_VALUES["90th_percentile"]["rr"],
        "go",
        markersize=12,
        label=f"MMT (90th): {t_mmt:.1f}°C, RR=1.00",
        zorder=10,
    )
    ax.plot(
        t_99th,
        RR_VALUES["99th_percentile"]["rr"],
        "ro",
        markersize=12,
        label=f"99th %ile: {t_99th:.1f}°C, RR={RR_VALUES['99th_percentile']['rr']}",
        zorder=10,
    )
    ax.errorbar(
        t_99th,
        RR_VALUES["99th_percentile"]["rr"],
        yerr=[
            [
                RR_VALUES["99th_percentile"]["rr"]
                - RR_VALUES["99th_percentile"]["lower"]
            ],
            [
                RR_VALUES["99th_percentile"]["upper"]
                - RR_VALUES["99th_percentile"]["rr"]
            ],
        ],
        fmt="none",
        color="red",
        capsize=5,
        linewidth=2,
        zorder=10,
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.axvline(x=t_mmt, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.set_xlabel("Temperature (°C)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Relative Risk (RR)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Climate Zone {zone}: Temperature-Mortality Relationship with 95% CI\nβ_cold={beta_cold:.4f} [{beta_cold_lower:.4f}, {beta_cold_upper:.4f}], β_heat={beta_heat:.4f} [{beta_heat_lower:.4f}, {beta_heat_upper:.4f}]",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"data/Europe_RR_Curve_Zone_{zone}_WithCI.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
for idx, row in rr_coef_df.iterrows():
    zone = row["climate_zone"]
