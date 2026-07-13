"""
Calculate Baseline Temperature-Attributable CVD Deaths (2016-2023)
With Monte Carlo Uncertainty Propagation

CORRECTED VERSION: Net deaths calculated by adding heat+cold WITHIN each simulation
before taking percentiles (not by adding percentile bounds)

Inputs:
- Heat/Cold Excess by grid
- Climate zones by grid
- RR coefficients by climate zone (with uncertainty bounds)
- CVD deaths by age (with uncertainty bounds)

Outputs:
- Grid-level deaths by age (mean + 95% CI)
- Country-level aggregates
- Total European summary
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import time

print("="*80)
print("BASELINE TEMPERATURE-ATTRIBUTABLE CVD MORTALITY (2016-2023)")
print("Monte Carlo Uncertainty Propagation")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SIMULATIONS = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

desktop_path = Path("")

# Input files
heat_excess_file = desktop_path / "Europe_Grid_Heat_Excess_2016-2023.csv"
cold_excess_file = desktop_path / "Europe_Grid_Cold_Excess_2016-2023.csv"
climate_zones_file = desktop_path / "Europe_Grid_Climate_Zones.csv"
rr_coeff_file = desktop_path / "RR_Coeff.csv"
cvd_deaths_file = desktop_path / "Grid_CVD_Age.csv"

# Output files
output_grid = desktop_path / "Baseline_Grid_Attributable_Deaths_by_Age.csv"
output_country = desktop_path / "Baseline_Country_Attributable_Deaths.csv"
output_regional = desktop_path / "Baseline_Regional_Attributable_Deaths.csv"
output_net_grid = desktop_path / "Baseline_Grid_Net_Temperature_Deaths.csv"
output_net_country = desktop_path / "Baseline_Country_Net_Temperature_Deaths.csv"
output_net_regional = desktop_path / "Baseline_Regional_Net_Temperature_Deaths.csv"

print(f"\nConfiguration:")
print(f"  • Monte Carlo simulations: {N_SIMULATIONS:,}")
print(f"  • Random seed: {RANDOM_SEED}")

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n{'='*80}")
print("LOADING DATA")
print("="*80)

start_time = time.time()

# Heat excess
print(f"\n1. Loading heat excess...")
heat = pd.read_csv(heat_excess_file)
print(f"   ✓ {len(heat):,} grids")

# Cold excess
print(f"\n2. Loading cold excess...")
cold = pd.read_csv(cold_excess_file)
print(f"   ✓ {len(cold):,} grids")

# Climate zones
print(f"\n3. Loading climate zones...")
zones = pd.read_csv(climate_zones_file)
print(f"   ✓ {len(zones):,} grids")
print(f"   Climate zones: {sorted(zones['climate_zone'].unique())}")

# RR coefficients
print(f"\n4. Loading RR coefficients...")
rr_coeff = pd.read_csv(rr_coeff_file)
print(f"   ✓ {len(rr_coeff)} climate zones")
print(f"\n   RR Coefficients by climate zone:")
print(rr_coeff.to_string(index=False))

# CVD deaths by age
print(f"\n5. Loading CVD deaths by age...")
cvd = pd.read_csv(cvd_deaths_file)
print(f"   ✓ {len(cvd):,} grids")

# UN Regional classification
print(f"\n6. Loading UN regional classification...")
regions_file = desktop_path / "UN_Geoscheme_Classification.csv"
un_regions = pd.read_csv(regions_file)
print(f"   ✓ {len(un_regions)} countries")
print(f"   Regions: {sorted(un_regions['UN_Region'].unique())}")

# Check for CVD death columns
age_groups = {
    'under_20': ['cvd_deaths_mean_under_20', 'cvd_deaths_max_under_20', 'cvd_deaths_min_under_20'],
    '20_54': ['cvd_deaths_mean_20_54', 'cvd_deaths_max_20_54', 'cvd_deaths_min_20_54'],
    '55_64': ['cvd_deaths_mean_55_64', 'cvd_deaths_max_55_64', 'cvd_deaths_min_55_64'],
    '65_74': ['cvd_deaths_mean_65_74', 'cvd_deaths_max_65_74', 'cvd_deaths_min_65_74'],
    '75plus': ['cvd_deaths_mean_75plus', 'cvd_deaths_max_75plus', 'cvd_deaths_min_75plus']
}

print(f"\n   Age groups detected:")
for age, cols in age_groups.items():
    if all(col in cvd.columns for col in cols):
        print(f"   ✓ {age}")
    else:
        print(f"   ✗ {age} - MISSING COLUMNS!")

print(f"\nData loaded in {time.time() - start_time:.1f} seconds")

# ============================================================================
# MERGE DATA
# ============================================================================

print(f"\n{'='*80}")
print("MERGING DATA")
print("="*80)

# Merge heat and cold excess
df = heat[['grid_id', 'Avg_Daily_Heat_Excess']].merge(
    cold[['grid_id', 'Avg_Daily_Cold_Excess']], 
    on='grid_id', 
    how='inner'
)
print(f"✓ Heat + Cold: {len(df):,} grids")

# Merge climate zones
df = df.merge(zones[['grid_id', 'climate_zone']], on='grid_id', how='inner')
print(f"✓ + Climate zones: {len(df):,} grids")

# Merge CVD deaths
cvd_cols = ['grid_id', 'Country'] + [col for cols in age_groups.values() for col in cols]
cvd_cols = [col for col in cvd_cols if col in cvd.columns]
df = df.merge(cvd[cvd_cols], on='grid_id', how='inner')
print(f"✓ + CVD deaths: {len(df):,} grids")

# Merge UN regional classification
df = df.merge(un_regions, on='Country', how='left')
print(f"✓ + UN Regions: {len(df):,} grids")
if df['UN_Region'].isna().sum() > 0:
    print(f"  ⚠️  {df['UN_Region'].isna().sum()} grids missing region classification")

print(f"\nFinal dataset: {len(df):,} grids")
print(f"Countries: {df['Country'].nunique()}")
print(f"UN Regions: {sorted(df['UN_Region'].dropna().unique())}")
print(f"Climate zones: {sorted(df['climate_zone'].unique())}")

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

print(f"\n{'='*80}")
print("MONTE CARLO SIMULATION")
print("="*80)

print(f"\nRunning {N_SIMULATIONS:,} simulations for {len(df):,} grids...")
print("This may take several minutes...")

sim_start = time.time()

# Prepare results storage
results = []

# Progress tracking
progress_interval = max(1, len(df) // 20)  # Update every 5%

for idx, row in df.iterrows():
    if (idx + 1) % progress_interval == 0:
        elapsed = time.time() - sim_start
        rate = (idx + 1) / elapsed
        remaining = (len(df) - idx - 1) / rate
        pct = ((idx + 1) / len(df)) * 100
        print(f"  Progress: {idx+1:,}/{len(df):,} ({pct:.1f}%) - ETA: {remaining/60:.1f} min")
    
    grid_id = row['grid_id']
    country = row['Country']
    climate_zone = row['climate_zone']
    heat_excess = row['Avg_Daily_Heat_Excess']
    cold_excess = row['Avg_Daily_Cold_Excess']
    
    # Get RR coefficients for this climate zone
    zone_rr = rr_coeff[rr_coeff['climate_zone'] == climate_zone]
    
    if len(zone_rr) == 0:
        print(f"  Warning: No RR coefficients for zone {climate_zone} in grid {grid_id}")
        continue
    
    beta_heat = zone_rr['beta_heat'].iloc[0]
    beta_heat_lower = zone_rr['beta_heat_lower'].iloc[0]
    beta_heat_upper = zone_rr['beta_heat_upper'].iloc[0]
    
    beta_cold = zone_rr['beta_cold'].iloc[0]
    beta_cold_lower = zone_rr['beta_cold_lower'].iloc[0]
    beta_cold_upper = zone_rr['beta_cold_upper'].iloc[0]
    
    # Calculate SE from CI bounds (assuming normal distribution)
    # 95% CI: mean ± 1.96 × SE  →  SE = (upper - lower) / (2 × 1.96)
    beta_heat_se = (beta_heat_upper - beta_heat_lower) / (2 * 1.96)
    beta_cold_se = (beta_cold_upper - beta_cold_lower) / (2 * 1.96)
    
    # Initialize storage for this grid's simulations (CORRECTED: Added 'net')
    grid_results = {age: {'heat': [], 'cold': [], 'net': []} for age in age_groups.keys()}
    
    # Run Monte Carlo simulations
    for sim in range(N_SIMULATIONS):
        # Sample RR coefficients from normal distribution
        beta_heat_sample = np.random.normal(beta_heat, beta_heat_se)
        beta_cold_sample = np.random.normal(beta_cold, beta_cold_se)
        
        # Calculate RR
        RR_heat = np.exp(beta_heat_sample * heat_excess)
        RR_cold = np.exp(beta_cold_sample * cold_excess)
        
        # Calculate PAF
        PAF_heat = (RR_heat - 1) / RR_heat if RR_heat > 0 else 0
        PAF_cold = (RR_cold - 1) / RR_cold if RR_cold > 0 else 0
        
        # For each age group
        for age, cols in age_groups.items():
            mean_col, max_col, min_col = cols
            
            if mean_col not in row.index:
                continue
            
            cvd_mean = row[mean_col]
            cvd_max = row[max_col]
            cvd_min = row[min_col]
            
            # Sample CVD deaths from triangular distribution
            # (if min/max are same as mean, just use mean)
            if cvd_max == cvd_min:
                cvd_sample = cvd_mean
            else:
                cvd_sample = np.random.triangular(cvd_min, cvd_mean, cvd_max)
            
            # Calculate attributable deaths
            heat_deaths = cvd_sample * PAF_heat
            cold_deaths = cvd_sample * PAF_cold
            net_deaths = heat_deaths + cold_deaths  # CORRECTED: Calculate net within simulation
            
            grid_results[age]['heat'].append(heat_deaths)
            grid_results[age]['cold'].append(cold_deaths)
            grid_results[age]['net'].append(net_deaths)  # CORRECTED: Store net deaths
    
    # Calculate summary statistics (mean and 95% CI)
    result_row = {
        'grid_id': grid_id,
        'Country': country,
        'UN_Region': row.get('UN_Region', None),  # Add UN Region
        'climate_zone': climate_zone,
        'heat_excess': heat_excess,
        'cold_excess': cold_excess
    }
    
    for age in age_groups.keys():
        # Heat deaths
        heat_sims = grid_results[age]['heat']
        result_row[f'heat_deaths_{age}_mean'] = np.mean(heat_sims)
        result_row[f'heat_deaths_{age}_lower'] = np.percentile(heat_sims, 2.5)
        result_row[f'heat_deaths_{age}_upper'] = np.percentile(heat_sims, 97.5)
        
        # Cold deaths
        cold_sims = grid_results[age]['cold']
        result_row[f'cold_deaths_{age}_mean'] = np.mean(cold_sims)
        result_row[f'cold_deaths_{age}_lower'] = np.percentile(cold_sims, 2.5)
        result_row[f'cold_deaths_{age}_upper'] = np.percentile(cold_sims, 97.5)
        
        # CORRECTED: Net deaths from simulations (not by adding bounds)
        net_sims = grid_results[age]['net']
        result_row[f'net_deaths_{age}_mean'] = np.mean(net_sims)
        result_row[f'net_deaths_{age}_lower'] = np.percentile(net_sims, 2.5)
        result_row[f'net_deaths_{age}_upper'] = np.percentile(net_sims, 97.5)
    
    # Calculate totals (sum across age groups)
    total_heat_sims = [sum(grid_results[age]['heat'][i] for age in age_groups.keys()) 
                       for i in range(N_SIMULATIONS)]
    total_cold_sims = [sum(grid_results[age]['cold'][i] for age in age_groups.keys()) 
                       for i in range(N_SIMULATIONS)]
    # CORRECTED: Net total from simulations
    total_net_sims = [sum(grid_results[age]['net'][i] for age in age_groups.keys()) 
                      for i in range(N_SIMULATIONS)]
    
    result_row['heat_deaths_total_mean'] = np.mean(total_heat_sims)
    result_row['heat_deaths_total_lower'] = np.percentile(total_heat_sims, 2.5)
    result_row['heat_deaths_total_upper'] = np.percentile(total_heat_sims, 97.5)
    
    result_row['cold_deaths_total_mean'] = np.mean(total_cold_sims)
    result_row['cold_deaths_total_lower'] = np.percentile(total_cold_sims, 2.5)
    result_row['cold_deaths_total_upper'] = np.percentile(total_cold_sims, 97.5)
    
    # CORRECTED: Net total from simulations (not by adding bounds)
    result_row['net_deaths_total_mean'] = np.mean(total_net_sims)
    result_row['net_deaths_total_lower'] = np.percentile(total_net_sims, 2.5)
    result_row['net_deaths_total_upper'] = np.percentile(total_net_sims, 97.5)
    
    results.append(result_row)

sim_time = time.time() - sim_start
print(f"\n✓ Simulations complete in {sim_time/60:.1f} minutes")

# ============================================================================
# CREATE RESULTS DATAFRAME
# ============================================================================

results_df = pd.DataFrame(results)

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print("="*80)

print(f"\nTotal grids analyzed: {len(results_df):,}")

# Calculate Europe-wide totals
total_heat_mean = results_df['heat_deaths_total_mean'].sum()
total_cold_mean = results_df['cold_deaths_total_mean'].sum()
total_net = total_heat_mean + total_cold_mean

print(f"\nEurope-wide annual deaths (2016-2023 baseline):")
print(f"  Heat-attributable:  {total_heat_mean:,.0f}")
print(f"  Cold-attributable:  {total_cold_mean:,.0f}")
print(f"  Net temperature:    {total_net:,.0f}")

# By climate zone
print(f"\nBy climate zone:")
zone_summary = results_df.groupby('climate_zone').agg({
    'heat_deaths_total_mean': 'sum',
    'cold_deaths_total_mean': 'sum',
    'grid_id': 'count'
}).round(0)
zone_summary.columns = ['Heat Deaths', 'Cold Deaths', 'N Grids']
print(zone_summary.to_string())

# Top 10 countries
print(f"\nTop 10 countries by heat-attributable deaths:")
country_heat = results_df.groupby('Country')['heat_deaths_total_mean'].sum().sort_values(ascending=False).head(10)
for country, deaths in country_heat.items():
    print(f"  {country:<20} {deaths:>10,.0f}")

# ============================================================================
# AGGREGATE TO COUNTRY LEVEL
# ============================================================================

print(f"\n{'='*80}")
print("AGGREGATING TO COUNTRY LEVEL")
print("="*80)

country_results = []

for country in results_df['Country'].unique():
    country_data = results_df[results_df['Country'] == country]
    
    country_row = {'Country': country, 'N_Grids': len(country_data)}
    
    # Sum across grids for each age group
    for age in age_groups.keys():
        country_row[f'heat_deaths_{age}_mean'] = country_data[f'heat_deaths_{age}_mean'].sum()
        country_row[f'heat_deaths_{age}_lower'] = country_data[f'heat_deaths_{age}_lower'].sum()
        country_row[f'heat_deaths_{age}_upper'] = country_data[f'heat_deaths_{age}_upper'].sum()
        
        country_row[f'cold_deaths_{age}_mean'] = country_data[f'cold_deaths_{age}_mean'].sum()
        country_row[f'cold_deaths_{age}_lower'] = country_data[f'cold_deaths_{age}_lower'].sum()
        country_row[f'cold_deaths_{age}_upper'] = country_data[f'cold_deaths_{age}_upper'].sum()
        
        # CORRECTED: Net deaths (already calculated per grid from MC)
        country_row[f'net_deaths_{age}_mean'] = country_data[f'net_deaths_{age}_mean'].sum()
        country_row[f'net_deaths_{age}_lower'] = country_data[f'net_deaths_{age}_lower'].sum()
        country_row[f'net_deaths_{age}_upper'] = country_data[f'net_deaths_{age}_upper'].sum()
    
    # Totals
    country_row['heat_deaths_total_mean'] = country_data['heat_deaths_total_mean'].sum()
    country_row['heat_deaths_total_lower'] = country_data['heat_deaths_total_lower'].sum()
    country_row['heat_deaths_total_upper'] = country_data['heat_deaths_total_upper'].sum()
    
    country_row['cold_deaths_total_mean'] = country_data['cold_deaths_total_mean'].sum()
    country_row['cold_deaths_total_lower'] = country_data['cold_deaths_total_lower'].sum()
    country_row['cold_deaths_total_upper'] = country_data['cold_deaths_total_upper'].sum()
    
    # CORRECTED: Net total deaths (already calculated per grid from MC)
    country_row['net_deaths_total_mean'] = country_data['net_deaths_total_mean'].sum()
    country_row['net_deaths_total_lower'] = country_data['net_deaths_total_lower'].sum()
    country_row['net_deaths_total_upper'] = country_data['net_deaths_total_upper'].sum()
    
    country_results.append(country_row)

country_df = pd.DataFrame(country_results)
print(f"✓ Aggregated to {len(country_df)} countries")

# ============================================================================
# AGGREGATE TO UN REGIONAL LEVEL
# ============================================================================

print(f"\n{'='*80}")
print("AGGREGATING TO UN REGIONAL LEVEL")
print("="*80)

regional_results = []

for region in results_df['UN_Region'].dropna().unique():
    region_data = results_df[results_df['UN_Region'] == region]
    
    region_row = {'UN_Region': region, 'N_Grids': len(region_data)}
    
    # Sum across grids for each age group
    for age in age_groups.keys():
        region_row[f'heat_deaths_{age}_mean'] = region_data[f'heat_deaths_{age}_mean'].sum()
        region_row[f'heat_deaths_{age}_lower'] = region_data[f'heat_deaths_{age}_lower'].sum()
        region_row[f'heat_deaths_{age}_upper'] = region_data[f'heat_deaths_{age}_upper'].sum()
        
        region_row[f'cold_deaths_{age}_mean'] = region_data[f'cold_deaths_{age}_mean'].sum()
        region_row[f'cold_deaths_{age}_lower'] = region_data[f'cold_deaths_{age}_lower'].sum()
        region_row[f'cold_deaths_{age}_upper'] = region_data[f'cold_deaths_{age}_upper'].sum()
        
        # CORRECTED: Net deaths (already calculated per grid from MC)
        region_row[f'net_deaths_{age}_mean'] = region_data[f'net_deaths_{age}_mean'].sum()
        region_row[f'net_deaths_{age}_lower'] = region_data[f'net_deaths_{age}_lower'].sum()
        region_row[f'net_deaths_{age}_upper'] = region_data[f'net_deaths_{age}_upper'].sum()
    
    # Totals
    region_row['heat_deaths_total_mean'] = region_data['heat_deaths_total_mean'].sum()
    region_row['heat_deaths_total_lower'] = region_data['heat_deaths_total_lower'].sum()
    region_row['heat_deaths_total_upper'] = region_data['heat_deaths_total_upper'].sum()
    
    region_row['cold_deaths_total_mean'] = region_data['cold_deaths_total_mean'].sum()
    region_row['cold_deaths_total_lower'] = region_data['cold_deaths_total_lower'].sum()
    region_row['cold_deaths_total_upper'] = region_data['cold_deaths_total_upper'].sum()
    
    # CORRECTED: Net total deaths (already calculated per grid from MC)
    region_row['net_deaths_total_mean'] = region_data['net_deaths_total_mean'].sum()
    region_row['net_deaths_total_lower'] = region_data['net_deaths_total_lower'].sum()
    region_row['net_deaths_total_upper'] = region_data['net_deaths_total_upper'].sum()
    
    regional_results.append(region_row)

regional_df = pd.DataFrame(regional_results)
print(f"✓ Aggregated to {len(regional_df)} UN regions")

# ============================================================================
# CREATE TOTAL SUMMARY AND ADD TO COUNTRY FILE
# ============================================================================

print(f"\n{'='*80}")
print("CREATING TOTAL SUMMARY")
print("="*80)

# Create Total row matching country_df structure
total_row = {'Country': 'TOTAL', 'N_Grids': len(results_df)}

# Sum across all grids for each age group
for age in age_groups.keys():
    total_row[f'heat_deaths_{age}_mean'] = results_df[f'heat_deaths_{age}_mean'].sum()
    total_row[f'heat_deaths_{age}_lower'] = results_df[f'heat_deaths_{age}_lower'].sum()
    total_row[f'heat_deaths_{age}_upper'] = results_df[f'heat_deaths_{age}_upper'].sum()
    
    total_row[f'cold_deaths_{age}_mean'] = results_df[f'cold_deaths_{age}_mean'].sum()
    total_row[f'cold_deaths_{age}_lower'] = results_df[f'cold_deaths_{age}_lower'].sum()
    total_row[f'cold_deaths_{age}_upper'] = results_df[f'cold_deaths_{age}_upper'].sum()
    
    # CORRECTED: Net deaths (already calculated per grid from MC)
    total_row[f'net_deaths_{age}_mean'] = results_df[f'net_deaths_{age}_mean'].sum()
    total_row[f'net_deaths_{age}_lower'] = results_df[f'net_deaths_{age}_lower'].sum()
    total_row[f'net_deaths_{age}_upper'] = results_df[f'net_deaths_{age}_upper'].sum()

# Totals
total_row['heat_deaths_total_mean'] = results_df['heat_deaths_total_mean'].sum()
total_row['heat_deaths_total_lower'] = results_df['heat_deaths_total_lower'].sum()
total_row['heat_deaths_total_upper'] = results_df['heat_deaths_total_upper'].sum()

total_row['cold_deaths_total_mean'] = results_df['cold_deaths_total_mean'].sum()
total_row['cold_deaths_total_lower'] = results_df['cold_deaths_total_lower'].sum()
total_row['cold_deaths_total_upper'] = results_df['cold_deaths_total_upper'].sum()

# CORRECTED: Net total deaths (already calculated per grid from MC)
total_row['net_deaths_total_mean'] = results_df['net_deaths_total_mean'].sum()
total_row['net_deaths_total_lower'] = results_df['net_deaths_total_lower'].sum()
total_row['net_deaths_total_upper'] = results_df['net_deaths_total_upper'].sum()

# Append Total row to country dataframe
country_df = pd.concat([country_df, pd.DataFrame([total_row])], ignore_index=True)
print(f"✓ Added TOTAL row to country summary")

# ============================================================================
# NOTE: Net temperature deaths already calculated correctly from MC simulations
# ============================================================================
# Net deaths are now calculated WITHIN each Monte Carlo simulation
# (lines 232-235) and then percentiles are taken from the net simulations.
# This is the correct approach - NOT adding heat+cold bounds separately.
# Both grid-level and country-level results already contain net_deaths columns.

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("SAVING RESULTS")
print("="*80)

# Grid-level heat + cold + net (CORRECTED: Now includes net columns from MC)
results_df.to_csv(output_grid, index=False)
print(f"✓ Grid-level (heat + cold + net): {output_grid.name}")
print(f"  Rows: {len(results_df):,}")
print(f"  Columns: {len(results_df.columns)}")

# Country-level heat + cold + net (CORRECTED: Now includes net columns from MC)
country_df.to_csv(output_country, index=False)
print(f"✓ Country-level (heat + cold + net, includes TOTAL): {output_country.name}")
print(f"  Rows: {len(country_df):,} (includes TOTAL)")
print(f"  Columns: {len(country_df.columns)}")

# Regional-level heat + cold + net
regional_df.to_csv(output_regional, index=False)
print(f"✓ Regional-level (heat + cold + net): {output_regional.name}")
print(f"  Rows: {len(regional_df):,}")
print(f"  Columns: {len(regional_df.columns)}")

# Create net-only output files by selecting net columns
net_cols_grid = ['grid_id', 'Country', 'climate_zone'] + [col for col in results_df.columns if 'net_deaths' in col]
net_grid_df = results_df[net_cols_grid]
net_grid_df.to_csv(output_net_grid, index=False)
print(f"✓ Grid-level (net only): {output_net_grid.name}")
print(f"  Rows: {len(net_grid_df):,}")

net_cols_country = ['Country', 'N_Grids'] + [col for col in country_df.columns if 'net_deaths' in col]
net_country_df = country_df[net_cols_country]
net_country_df.to_csv(output_net_country, index=False)
print(f"✓ Country-level (net only, includes TOTAL): {output_net_country.name}")
print(f"  Rows: {len(net_country_df):,}")

net_cols_regional = ['UN_Region', 'N_Grids'] + [col for col in regional_df.columns if 'net_deaths' in col]
net_regional_df = regional_df[net_cols_regional]
net_regional_df.to_csv(output_net_regional, index=False)
print(f"✓ Regional-level (net only): {output_net_regional.name}")
print(f"  Rows: {len(net_regional_df):,}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

total_time = time.time() - start_time

print(f"\n{'='*80}")
print("✅ BASELINE ANALYSIS COMPLETE")
print("="*80)

print(f"\nTotal runtime: {total_time/60:.1f} minutes")

print(f"\nKEY FINDINGS (2016-2023 baseline):")
print(f"  • Heat-attributable CVD deaths:  {total_heat_mean:>10,.0f} deaths/year")
print(f"  • Cold-attributable CVD deaths:  {total_cold_mean:>10,.0f} deaths/year")
print(f"  • Net temperature-attributable:  {total_net:>10,.0f} deaths/year")

print(f"\nOUTPUT FILES:")
print(f"  1. {output_grid.name}")
print(f"     → Grid-level heat + cold + net deaths by age with 95% CI")
print(f"  2. {output_country.name}")
print(f"     → Country-level heat + cold + net deaths (includes TOTAL row)")
print(f"  3. {output_regional.name}")
print(f"     → Regional-level heat + cold + net deaths (UN regions)")
print(f"  4. {output_net_grid.name}")
print(f"     → Grid-level net temperature deaths only")
print(f"  5. {output_net_country.name}")
print(f"     → Country-level net temperature deaths only (includes TOTAL row)")
print(f"  6. {output_net_regional.name}")
print(f"     → Regional-level net temperature deaths only (UN regions)")

print(f"\nNEXT STEPS:")
print(f"  • Review baseline results")
print(f"  • Calculate future projections (2025-2080)")
print(f"  • Fit GAM to temporal trends")
print(f"  • Create visualizations")

print("="*80)
