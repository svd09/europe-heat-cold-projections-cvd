# Temperature-Attributable CVD Mortality in Europe — Analysis Code

This repository contains the analysis pipeline for the manuscript on temperature-attributable
cardiovascular mortality across Europe. Code is organized into five folders, numbered in run
order. Scripts within each folder are also numbered in the order they should be executed.

Raw and intermediate data files are not included in this repository. Data is available on
request. All scripts reference input/output paths as `data/...` placeholders — set these to
your own local paths, and set `YOUR_GEE_PROJECT_ID` to your own Google Earth Engine project ID
in the extraction scripts, before running.

## Repository structure

```
01_data_acquisition/            Grid construction, raw temperature/population extraction
02_primary_analysis/            Main analysis: baseline + future mortality, ensemble climate
03_adaptation_scenarios/        Heat-adaptation sensitivity analysis (0/10/50/90% attenuation)
04_individual_model_sensitivity/  Per-CMIP6-model sensitivity analysis (no ensemble averaging)
05_ukraine_belarus_companion/   Sensitivity analysis extending the pipeline to Ukraine + Belarus
```

---

## 01_data_acquisition

Builds the 0.25° European grid and pulls all raw inputs: ERA5-Land baseline temperatures
(2016–2023), CMIP6 projected temperatures (2020–2080, 5 models × 2 SSPs), GHS-POP population,
NASA SSP population projections, and downscales national CVD mortality data to the grid.
Also projects age-stratified population and CVD deaths forward under 4 fertility scenarios
(static / median / high / low) for both SSP2 and SSP5.

Files 01–12, run in numeric order. See in-file configuration blocks for expected input files.

---

## 02_primary_analysis

The main analysis pipeline, run in numeric order (01–15):

1. **Ensemble creation + bias correction** (01–02): the 5 CMIP6 models are averaged into a
   single ensemble mean per SSP, then bias-corrected against ERA5-Land observations using
   **quantile mapping**: for each grid cell, empirical quantiles (100 quantiles) of the model's
   2020–2023 overlap-period temperatures are mapped onto ERA5's quantiles via linear
   interpolation (`scipy.interpolate.interp1d`, `kind='linear'`), with
   **`fill_value='extrapolate'`**.
2. **Baseline heat/cold excess** (03–04): TMREL (minimum-mortality reference temperature) per
   grid = modal temperature (1 d.p.) within the 54th–92nd percentile of each grid's 2016–2023
   ERA5 temperature distribution, with whole-number and median fallbacks if no unique mode
   exists. Heat/cold excess = daily temperature above/below TMREL (floored at 0), averaged over
   the full study period.
3. **RR derivation** (05): fits log-linear relative-risk curves (cold side and heat side,
   separately) by climate zone, anchored to literature RR values at the 1st/90th/99th
   temperature percentiles, with 95% CI bounds.
4. **Baseline attributable deaths + rates** (06–07): **the Monte Carlo procedure** — 1,000
   simulations (seed 42) per grid. Each simulation draws β_heat and β_cold from
   Normal(mean, SE), computes RR = exp(β × excess), PAF = (RR−1)/RR (clamped at 0 when RR ≤ 1),
   samples CVD deaths from a Triangular(min, mean, max) distribution, and computes attributable
   deaths = PAF × sampled CVD deaths. Net (heat + cold) deaths are summed **within** each
   simulation before taking percentiles — not by adding separately-computed percentile bounds.
5. **Future heat/cold excess** (08–09): same TMREL/excess logic as step 2, applied to the
   bias-corrected future ensemble temperatures, 2025–2080.
6. **Future attributable deaths + rates** (10–13): same Monte Carlo procedure as step 4, run
   under two demographic variants — **median fertility** (main analysis) and **static age
   structure** (2020 age structure held constant, population totals still grow). These two
   variants are also what the **aging decomposition** compares: the static-age run isolates
   the effect of population growth alone, so the difference between the median-fertility and
   static-age results at a given year/scenario attributes the remaining change to population
   aging.
7. **Period mortality estimates** (14–15): aggregates grid-level results to
   country/region/Europe-wide, for mid-century (2046–2055) and late-century (2071–2080), for
   both demographic variants. This is the source for the manuscript's period-summary figures
   and tables (e.g. Figure 4).

---

## 03_adaptation_scenarios

Re-runs the future attributable-deaths pipeline (equivalent to `02_primary_analysis/10–14`)
under 4 heat-adaptation levels: **NoAdaptation (0%), Adapt10pct, Adapt50pct, Adapt90pct**. At
each level, β_heat (and its CI bounds) is attenuated by `(1 − reduction)` *before* the Monte
Carlo draw, representing reduced population sensitivity to a given amount of heat excess.
Cold is never touched — adaptation is modeled for heat only. Because
`β_heat_scenario = β_heat × (1 − reduction)` stays positive for any reduction < 1.0,
RR_heat is guaranteed ≥ 1 at every level (heat can never look protective). Output files are
suffixed with the adaptation level; `Heat_Adaptation` appears as an explicit column alongside
`Scenario` in the period-estimate outputs.

---

## 04_individual_model_sensitivity

Repeats bias correction, excess calculation, Monte Carlo mortality, aggregation, and period
estimation **separately for each of the 5 CMIP6 models** (rather than the ensemble mean used
in Primary Analysis), for mid-century and late-century only. This tests whether the ensemble
result is sensitive to any single model.

Bias correction (01) uses the same quantile mapping and linear extrapolation as the primary
ensemble (`02_primary_analysis/02`), so future values outside the calibration range are
extrapolated rather than fixed, consistent with the primary analysis.

Files 01–05, run in numeric order.

---

## 05_ukraine_belarus_companion

A sensitivity analysis extending the same methodology to Ukraine and Belarus specifically
(excluded from the main European grid). Files 01–02 extract ERA5-Land baseline and CMIP6
ensemble-mean temperatures for the Ukraine/Belarus grid via Earth Engine. Files 03–04
consolidate the full baseline (2016–2023) and future (2025–2080, mid-/late-century) mortality
pipelines — TMREL, heat/cold excess, Monte Carlo attributable deaths, rates, and period
estimates — into two standalone scripts.

Files 01–04, run in numeric order.

---

## Data availability

Raw climate and population data are not included in this repository due to size and licensing.
Sources:

- **ERA5-Land**: ECMWF, via Google Earth Engine (`ECMWF/ERA5_LAND/DAILY_AGGR`)
- **CMIP6 (NEX-GDDP-CMIP6)**: NASA, via Google Earth Engine (`NASA/GDDP-CMIP6`)
- **GHS-POP**: European Commission Joint Research Centre
- **SSP population projections**: NASA SEDAC (downscaled 1km SSP population)
- **National CVD mortality**: Global Burden of Disease (GBD)

## Requirements

Python 3.11, with `pandas`, `numpy`, `scipy`, `geopandas`, `rasterio`, `xarray`, and
`earthengine-api` (for extraction scripts only).
