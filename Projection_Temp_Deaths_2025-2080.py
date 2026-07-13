"""
Calculate Future Temperature-Attributable CVD Deaths (2025-2080)
SSP2-4.5 and SSP5-8.5 Scenarios
With Monte Carlo Uncertainty Propagation

CORRECTED VERSION: Net deaths calculated by adding heat+cold WITHIN each simulation
before taking percentiles (not by adding percentile bounds)

Inputs:
- Heat/Cold Excess by grid and year (2025-2080)
- Climate zones by grid
- RR coefficients by climate zone (with uncertainty bounds)
- CVD deaths by age and year (with uncertainty bounds)
- Population projections by age and year

Outputs:
- Grid-level deaths by age and year (mean + 95% CI) - consolidated file
- Country-level aggregates by year - consolidated file
- Separate files for SSP245 and SSP585
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import time

print("="*80)
print("FUTURE TEMPERATURE-ATTRIBUTABLE CVD MORTALITY (2025-2080)")
print("SSP2-4.5 and SSP5-8.5 Scenarios")
print("Monte Carlo Uncertainty Propagation - CORRECTED VERSION")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SIMULATIONS = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

desktop_path = Path("")

# Input files
heat_excess_ssp245_file = desktop_path / "Excess_Heat_SSP245_2025-80.csv"
cold_excess_ssp245_file = desktop_path / "Excess_Cold_SSP245_2025-80.csv"
heat_excess_ssp585_file = desktop_path / "Excess_Heat_SSP585_2025-80.csv"
cold_excess_ssp585_file = desktop_path / "Excess_Cold_SSP585_2025-80.csv"

climate_zones_file = desktop_path / "Europe_Grid_Climate_Zones.csv"
rr_coeff_file = desktop_path / "RR_Coeff.csv"
un_regions_file = desktop_path / "UN_Geoscheme_Classification.csv"

cvd_deaths_ssp2_file = desktop_path / "SSP2_Median_Fert_CVD_Deaths_Grid_2025_2080.csv"
cvd_deaths_ssp5_file = desktop_path / "SSP5_Median_Fert_CVD_Deaths_Grid_2025_2080.csv"

# Output files
output_grid_ssp245 = desktop_path / "Projection_SSP245_Grid_Deaths_2025-2080.csv"
output_country_ssp245 = desktop_path / "Projection_SSP245_Country_Deaths_2025-2080.csv"
output_region_ssp245 = desktop_path / "Projection_SSP245_Region_Deaths_2025-2080.csv"
output_net_grid_ssp245 = desktop_path / "Projection_SSP245_Grid_Net_Deaths_2025-2080.csv"
output_net_country_ssp245 = desktop_path / "Projection_SSP245_Country_Net_Deaths_2025-2080.csv"
output_net_region_ssp245 = desktop_path / "Projection_SSP245_Region_Net_Deaths_2025-2080.csv"

output_grid_ssp585 = desktop_path / "Projection_SSP585_Grid_Deaths_2025-2080.csv"
output_country_ssp585 = desktop_path / "Projection_SSP585_Country_Deaths_2025-2080.csv"
output_region_ssp585 = desktop_path / "Projection_SSP585_Region_Deaths_2025-2080.csv"
output_net_grid_ssp585 = desktop_path / "Projection_SSP585_Grid_Net_Deaths_2025-2080.csv"
output_net_country_ssp585 = desktop_path / "Projection_SSP585_Country_Net_Deaths_2025-2080.csv"
output_net_region_ssp585 = desktop_path / "Projection_SSP585_Region_Net_Deaths_2025-2080.csv"

print(f"\nConfiguration:")
print(f"  • Monte Carlo simulations: {N_SIMULATIONS:,}")
print(f"  • Random seed: {RANDOM_SEED}")
print(f"  • Years: 2025-2080 (56 years)")
print(f"  • Scenarios: SSP2-4.5, SSP5-8.5")

# ============================================================================
# LOAD COMMON DATA (CLIMATE ZONES, RR COEFFICIENTS)
# ============================================================================

print(f"\n{'='*80}")
print("LOADING COMMON DATA")
print("="*80)

# Climate zones
print(f"\nLoading climate zones...")
zones = pd.read_csv(climate_zones_file)
print(f"   ✓ {len(zones):,} grids")
print(f"   Climate zones: {sorted(zones['climate_zone'].unique())}")

# UN Region classifications
print(f"\nLoading UN region classifications...")
un_regions = pd.read_csv(un_regions_file)
print(f"   ✓ {len(un_regions)} countries")
print(f"   UN Regions: {sorted(un_regions['UN_Region'].unique())}")

# RR coefficients
print(f"\nLoading RR coefficients...")
rr_coeff = pd.read_csv(rr_coeff_file)
print(f"   ✓ {len(rr_coeff)} climate zones")
print(f"\n   RR Coefficients by climate zone:")
print(rr_coeff.to_string(index=False))

# Age groups
age_groups = ['under_20', '20_54', '55_64', '65_74', '75plus']
years = range(2025, 2081)  # 2025-2080 inclusive

print(f"\n   Age groups: {age_groups}")
print(f"   Years: {min(years)}-{max(years)} ({len(list(years))} years)")

# ============================================================================
# FUNCTION TO PROCESS ONE SCENARIO
# ============================================================================

def process_scenario(scenario_name, heat_file, cold_file, cvd_file):
    """
    Process one scenario (SSP245 or SSP585)
    
    Returns:
        grid_results_df: Grid-level heat + cold deaths for all years
        country_results_df: Country-level heat + cold deaths for all years (includes TOTAL)
        region_results_df: Region-level heat + cold deaths for all years (includes TOTAL)
        net_grid_results_df: Grid-level net temperature deaths for all years
        net_country_results_df: Country-level net temperature deaths for all years (includes TOTAL)
        net_region_results_df: Region-level net temperature deaths for all years (includes TOTAL)
    """
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {scenario_name}")
    print("="*80)
    
    scenario_start = time.time()
    
    # Load scenario-specific data
    print(f"\nLoading {scenario_name} data...")
    
    print(f"  1. Heat excess...")
    heat = pd.read_csv(heat_file)
    print(f"     ✓ {len(heat):,} grids")
    
    print(f"  2. Cold excess...")
    cold = pd.read_csv(cold_file)
    print(f"     ✓ {len(cold):,} grids")
    
    print(f"  3. CVD deaths...")
    cvd = pd.read_csv(cvd_file)
    print(f"     ✓ {len(cvd):,} grids")
    
    # Merge common data (climate zones) to heat data
    # Heat and cold are already in long format with year column
    heat = heat.merge(zones[['grid_id', 'climate_zone']], on='grid_id', how='inner')
    cold = cold.merge(zones[['grid_id', 'climate_zone']], on='grid_id', how='inner')
    print(f"\n  ✓ Merged climate zones")
    print(f"    Heat data: {len(heat):,} rows (grids × years)")
    print(f"    Cold data: {len(cold):,} rows (grids × years)")
    
    # Storage for all years
    all_grid_results = []
    all_country_results = []
    all_region_results = []
    
    # Process each year
    print(f"\n{'='*80}")
    print(f"PROCESSING YEARS (2025-2080)")
    print("="*80)
    
    for year in years:
        year_start = time.time()
        print(f"\n  Processing {year}...")
        
        # Filter excess data for this year
        heat_year = heat[heat['year'] == year].copy()
        cold_year = cold[cold['year'] == year].copy()
        
        if len(heat_year) == 0 or len(cold_year) == 0:
            print(f"    ✗ No excess data for {year}, skipping...")
            continue
        
        # Create year dataframe with heat and cold excess
        df_year = heat_year[['grid_id', 'climate_zone', 'avg_daily_heat_excess']].copy()
        df_year = df_year.merge(
            cold_year[['grid_id', 'avg_daily_cold_excess']], 
            on='grid_id', 
            how='inner'
        )
        df_year = df_year.rename(columns={
            'avg_daily_heat_excess': 'heat_excess',
            'avg_daily_cold_excess': 'cold_excess'
        })
        
        # Merge CVD deaths for this year
        cvd_cols_year = ['grid_id', 'Country']
        for age in age_groups:
            cvd_cols_year.extend([
                f'cvd_deaths_mean_{age}_{year}',
                f'cvd_deaths_max_{age}_{year}',
                f'cvd_deaths_min_{age}_{year}'
            ])
        
        df_year = df_year.merge(cvd[cvd_cols_year], on='grid_id', how='inner')
        
        # Rename CVD columns to match baseline format (without year suffix)
        for age in age_groups:
            df_year = df_year.rename(columns={
                f'cvd_deaths_mean_{age}_{year}': f'cvd_deaths_mean_{age}',
                f'cvd_deaths_max_{age}_{year}': f'cvd_deaths_max_{age}',
                f'cvd_deaths_min_{age}_{year}': f'cvd_deaths_min_{age}'
            })
        
        print(f"    Data prepared: {len(df_year):,} grids")
        
        # Run Monte Carlo simulation for this year
        year_results = []
        
        for idx, row in df_year.iterrows():
            grid_id = row['grid_id']
            country = row['Country']
            climate_zone = row['climate_zone']
            heat_excess = row['heat_excess']
            cold_excess = row['cold_excess']
            
            # Get RR coefficients for this climate zone
            zone_rr = rr_coeff[rr_coeff['climate_zone'] == climate_zone]
            
            if len(zone_rr) == 0:
                continue
            
            beta_heat = zone_rr['beta_heat'].iloc[0]
            beta_heat_lower = zone_rr['beta_heat_lower'].iloc[0]
            beta_heat_upper = zone_rr['beta_heat_upper'].iloc[0]
            
            beta_cold = zone_rr['beta_cold'].iloc[0]
            beta_cold_lower = zone_rr['beta_cold_lower'].iloc[0]
            beta_cold_upper = zone_rr['beta_cold_upper'].iloc[0]
            
            # Calculate SE from CI bounds
            beta_heat_se = (beta_heat_upper - beta_heat_lower) / (2 * 1.96)
            beta_cold_se = (beta_cold_upper - beta_cold_lower) / (2 * 1.96)
            
            # Initialize storage for this grid's simulations
            # CRITICAL CHANGE: Now storing 'net' in addition to heat and cold
            grid_sims = {age: {'heat': [], 'cold': [], 'net': []} for age in age_groups}
            
            # Run Monte Carlo simulations
            for sim in range(N_SIMULATIONS):
                # Sample RR coefficients from normal distribution
                beta_heat_sample = np.random.normal(beta_heat, beta_heat_se)
                beta_cold_sample = np.random.normal(beta_cold, beta_cold_se)
                
                # Calculate RR
                RR_heat = np.exp(beta_heat_sample * heat_excess)
                RR_cold = np.exp(beta_cold_sample * cold_excess)
                
                # Calculate PAF
                PAF_heat = (RR_heat - 1) / RR_heat if RR_heat > 1 else 0
                PAF_cold = (RR_cold - 1) / RR_cold if RR_cold > 1 else 0
                
                # For each age group
                for age in age_groups:
                    # Sample CVD deaths from triangular distribution
                    cvd_mean = row[f'cvd_deaths_mean_{age}']
                    cvd_min = row[f'cvd_deaths_min_{age}']
                    cvd_max = row[f'cvd_deaths_max_{age}']
                    
                    # Handle case where min=mean=max (no variance)
                    if cvd_min == cvd_max:
                        cvd_sample = cvd_mean
                    else:
                        cvd_sample = np.random.triangular(cvd_min, cvd_mean, cvd_max)
                    
                    # Calculate attributable deaths
                    heat_deaths = PAF_heat * cvd_sample
                    cold_deaths = PAF_cold * cvd_sample
                    net_deaths = heat_deaths + cold_deaths  # CRITICAL: Add WITHIN simulation
                    
                    grid_sims[age]['heat'].append(heat_deaths)
                    grid_sims[age]['cold'].append(cold_deaths)
                    grid_sims[age]['net'].append(net_deaths)  # Store net simulation
            
            # Calculate statistics from simulations
            grid_result = {
                'year': year,
                'grid_id': grid_id,
                'Country': country,
                'climate_zone': climate_zone
            }
            
            # Age-specific results
            for age in age_groups:
                heat_array = np.array(grid_sims[age]['heat'])
                cold_array = np.array(grid_sims[age]['cold'])
                net_array = np.array(grid_sims[age]['net'])  # Get net simulations
                
                grid_result[f'heat_deaths_{age}_mean'] = np.mean(heat_array)
                grid_result[f'heat_deaths_{age}_lower'] = np.percentile(heat_array, 2.5)
                grid_result[f'heat_deaths_{age}_upper'] = np.percentile(heat_array, 97.5)
                
                grid_result[f'cold_deaths_{age}_mean'] = np.mean(cold_array)
                grid_result[f'cold_deaths_{age}_lower'] = np.percentile(cold_array, 2.5)
                grid_result[f'cold_deaths_{age}_upper'] = np.percentile(cold_array, 97.5)
                
                # CRITICAL CHANGE: Net percentiles from net simulations
                grid_result[f'net_deaths_{age}_mean'] = np.mean(net_array)
                grid_result[f'net_deaths_{age}_lower'] = np.percentile(net_array, 2.5)
                grid_result[f'net_deaths_{age}_upper'] = np.percentile(net_array, 97.5)
            
            # Total deaths (sum across age groups)
            # For heat and cold, we still sum the bounds (this is correct for independent age groups)
            grid_result['heat_deaths_total_mean'] = sum(grid_result[f'heat_deaths_{age}_mean'] for age in age_groups)
            grid_result['heat_deaths_total_lower'] = sum(grid_result[f'heat_deaths_{age}_lower'] for age in age_groups)
            grid_result['heat_deaths_total_upper'] = sum(grid_result[f'heat_deaths_{age}_upper'] for age in age_groups)
            
            grid_result['cold_deaths_total_mean'] = sum(grid_result[f'cold_deaths_{age}_mean'] for age in age_groups)
            grid_result['cold_deaths_total_lower'] = sum(grid_result[f'cold_deaths_{age}_lower'] for age in age_groups)
            grid_result['cold_deaths_total_upper'] = sum(grid_result[f'cold_deaths_{age}_upper'] for age in age_groups)
            
            # CRITICAL CHANGE: Net totals from net age-specific values
            grid_result['net_deaths_total_mean'] = sum(grid_result[f'net_deaths_{age}_mean'] for age in age_groups)
            grid_result['net_deaths_total_lower'] = sum(grid_result[f'net_deaths_{age}_lower'] for age in age_groups)
            grid_result['net_deaths_total_upper'] = sum(grid_result[f'net_deaths_{age}_upper'] for age in age_groups)
            
            year_results.append(grid_result)
        
        year_df = pd.DataFrame(year_results)
        all_grid_results.append(year_df)
        
        # Country aggregation for this year
        country_year_results = []
        
        for country in year_df['Country'].unique():
            country_data = year_df[year_df['Country'] == country]
            
            country_row = {
                'year': year,
                'Country': country,
                'N_Grids': len(country_data)
            }
            
            # Heat and cold deaths
            for age in age_groups:
                country_row[f'heat_deaths_{age}_mean'] = country_data[f'heat_deaths_{age}_mean'].sum()
                country_row[f'heat_deaths_{age}_lower'] = country_data[f'heat_deaths_{age}_lower'].sum()
                country_row[f'heat_deaths_{age}_upper'] = country_data[f'heat_deaths_{age}_upper'].sum()
                
                country_row[f'cold_deaths_{age}_mean'] = country_data[f'cold_deaths_{age}_mean'].sum()
                country_row[f'cold_deaths_{age}_lower'] = country_data[f'cold_deaths_{age}_lower'].sum()
                country_row[f'cold_deaths_{age}_upper'] = country_data[f'cold_deaths_{age}_upper'].sum()
                
                # Net deaths
                country_row[f'net_deaths_{age}_mean'] = country_data[f'net_deaths_{age}_mean'].sum()
                country_row[f'net_deaths_{age}_lower'] = country_data[f'net_deaths_{age}_lower'].sum()
                country_row[f'net_deaths_{age}_upper'] = country_data[f'net_deaths_{age}_upper'].sum()
            
            country_row['heat_deaths_total_mean'] = country_data['heat_deaths_total_mean'].sum()
            country_row['heat_deaths_total_lower'] = country_data['heat_deaths_total_lower'].sum()
            country_row['heat_deaths_total_upper'] = country_data['heat_deaths_total_upper'].sum()
            
            country_row['cold_deaths_total_mean'] = country_data['cold_deaths_total_mean'].sum()
            country_row['cold_deaths_total_lower'] = country_data['cold_deaths_total_lower'].sum()
            country_row['cold_deaths_total_upper'] = country_data['cold_deaths_total_upper'].sum()
            
            country_row['net_deaths_total_mean'] = country_data['net_deaths_total_mean'].sum()
            country_row['net_deaths_total_lower'] = country_data['net_deaths_total_lower'].sum()
            country_row['net_deaths_total_upper'] = country_data['net_deaths_total_upper'].sum()
            
            country_year_results.append(country_row)
        
        # Add TOTAL (Europe-wide) row for this year
        total_row = {
            'year': year,
            'Country': 'TOTAL',
            'N_Grids': len(year_df)
        }
        
        for age in age_groups:
            total_row[f'heat_deaths_{age}_mean'] = year_df[f'heat_deaths_{age}_mean'].sum()
            total_row[f'heat_deaths_{age}_lower'] = year_df[f'heat_deaths_{age}_lower'].sum()
            total_row[f'heat_deaths_{age}_upper'] = year_df[f'heat_deaths_{age}_upper'].sum()
            
            total_row[f'cold_deaths_{age}_mean'] = year_df[f'cold_deaths_{age}_mean'].sum()
            total_row[f'cold_deaths_{age}_lower'] = year_df[f'cold_deaths_{age}_lower'].sum()
            total_row[f'cold_deaths_{age}_upper'] = year_df[f'cold_deaths_{age}_upper'].sum()
            
            total_row[f'net_deaths_{age}_mean'] = year_df[f'net_deaths_{age}_mean'].sum()
            total_row[f'net_deaths_{age}_lower'] = year_df[f'net_deaths_{age}_lower'].sum()
            total_row[f'net_deaths_{age}_upper'] = year_df[f'net_deaths_{age}_upper'].sum()
        
        total_row['heat_deaths_total_mean'] = year_df['heat_deaths_total_mean'].sum()
        total_row['heat_deaths_total_lower'] = year_df['heat_deaths_total_lower'].sum()
        total_row['heat_deaths_total_upper'] = year_df['heat_deaths_total_upper'].sum()
        
        total_row['cold_deaths_total_mean'] = year_df['cold_deaths_total_mean'].sum()
        total_row['cold_deaths_total_lower'] = year_df['cold_deaths_total_lower'].sum()
        total_row['cold_deaths_total_upper'] = year_df['cold_deaths_total_upper'].sum()
        
        total_row['net_deaths_total_mean'] = year_df['net_deaths_total_mean'].sum()
        total_row['net_deaths_total_lower'] = year_df['net_deaths_total_lower'].sum()
        total_row['net_deaths_total_upper'] = year_df['net_deaths_total_upper'].sum()
        
        country_year_results.append(total_row)
        
        all_country_results.append(pd.DataFrame(country_year_results))
        
        # Regional aggregation for this year (UN Regions)
        # First merge region info to year_df
        year_df_with_region = year_df.merge(un_regions, on='Country', how='left')
        
        region_year_results = []
        
        for region in year_df_with_region['UN_Region'].dropna().unique():
            region_data = year_df_with_region[year_df_with_region['UN_Region'] == region]
            
            region_row = {
                'year': year,
                'UN_Region': region,
                'N_Grids': len(region_data)
            }
            
            # Heat and cold deaths by age
            for age in age_groups:
                region_row[f'heat_deaths_{age}_mean'] = region_data[f'heat_deaths_{age}_mean'].sum()
                region_row[f'heat_deaths_{age}_lower'] = region_data[f'heat_deaths_{age}_lower'].sum()
                region_row[f'heat_deaths_{age}_upper'] = region_data[f'heat_deaths_{age}_upper'].sum()
                
                region_row[f'cold_deaths_{age}_mean'] = region_data[f'cold_deaths_{age}_mean'].sum()
                region_row[f'cold_deaths_{age}_lower'] = region_data[f'cold_deaths_{age}_lower'].sum()
                region_row[f'cold_deaths_{age}_upper'] = region_data[f'cold_deaths_{age}_upper'].sum()
                
                # Net deaths
                region_row[f'net_deaths_{age}_mean'] = region_data[f'net_deaths_{age}_mean'].sum()
                region_row[f'net_deaths_{age}_lower'] = region_data[f'net_deaths_{age}_lower'].sum()
                region_row[f'net_deaths_{age}_upper'] = region_data[f'net_deaths_{age}_upper'].sum()
            
            # Total deaths
            region_row['heat_deaths_total_mean'] = region_data['heat_deaths_total_mean'].sum()
            region_row['heat_deaths_total_lower'] = region_data['heat_deaths_total_lower'].sum()
            region_row['heat_deaths_total_upper'] = region_data['heat_deaths_total_upper'].sum()
            
            region_row['cold_deaths_total_mean'] = region_data['cold_deaths_total_mean'].sum()
            region_row['cold_deaths_total_lower'] = region_data['cold_deaths_total_lower'].sum()
            region_row['cold_deaths_total_upper'] = region_data['cold_deaths_total_upper'].sum()
            
            region_row['net_deaths_total_mean'] = region_data['net_deaths_total_mean'].sum()
            region_row['net_deaths_total_lower'] = region_data['net_deaths_total_lower'].sum()
            region_row['net_deaths_total_upper'] = region_data['net_deaths_total_upper'].sum()
            
            region_year_results.append(region_row)
        
        # Add TOTAL (all regions combined) row for this year
        region_total_row = {
            'year': year,
            'UN_Region': 'TOTAL',
            'N_Grids': len(year_df)
        }
        
        for age in age_groups:
            region_total_row[f'heat_deaths_{age}_mean'] = year_df[f'heat_deaths_{age}_mean'].sum()
            region_total_row[f'heat_deaths_{age}_lower'] = year_df[f'heat_deaths_{age}_lower'].sum()
            region_total_row[f'heat_deaths_{age}_upper'] = year_df[f'heat_deaths_{age}_upper'].sum()
            
            region_total_row[f'cold_deaths_{age}_mean'] = year_df[f'cold_deaths_{age}_mean'].sum()
            region_total_row[f'cold_deaths_{age}_lower'] = year_df[f'cold_deaths_{age}_lower'].sum()
            region_total_row[f'cold_deaths_{age}_upper'] = year_df[f'cold_deaths_{age}_upper'].sum()
            
            region_total_row[f'net_deaths_{age}_mean'] = year_df[f'net_deaths_{age}_mean'].sum()
            region_total_row[f'net_deaths_{age}_lower'] = year_df[f'net_deaths_{age}_lower'].sum()
            region_total_row[f'net_deaths_{age}_upper'] = year_df[f'net_deaths_{age}_upper'].sum()
        
        region_total_row['heat_deaths_total_mean'] = year_df['heat_deaths_total_mean'].sum()
        region_total_row['heat_deaths_total_lower'] = year_df['heat_deaths_total_lower'].sum()
        region_total_row['heat_deaths_total_upper'] = year_df['heat_deaths_total_upper'].sum()
        
        region_total_row['cold_deaths_total_mean'] = year_df['cold_deaths_total_mean'].sum()
        region_total_row['cold_deaths_total_lower'] = year_df['cold_deaths_total_lower'].sum()
        region_total_row['cold_deaths_total_upper'] = year_df['cold_deaths_total_upper'].sum()
        
        region_total_row['net_deaths_total_mean'] = year_df['net_deaths_total_mean'].sum()
        region_total_row['net_deaths_total_lower'] = year_df['net_deaths_total_lower'].sum()
        region_total_row['net_deaths_total_upper'] = year_df['net_deaths_total_upper'].sum()
        
        region_year_results.append(region_total_row)
        
        all_region_results.append(pd.DataFrame(region_year_results))
        
        year_time = time.time() - year_start
        print(f"    ✓ {year} complete in {year_time:.1f}s ({len(year_df):,} grids)")
    
    # Combine all years
    print(f"\n{'='*80}")
    print(f"COMBINING RESULTS")
    print("="*80)
    
    grid_results_all = pd.concat(all_grid_results, ignore_index=True)
    country_results_all = pd.concat(all_country_results, ignore_index=True)
    region_results_all = pd.concat(all_region_results, ignore_index=True)
    
    # Separate heat+cold from net for separate files
    # Grid results - heat and cold only
    heat_cold_cols = ['year', 'grid_id', 'Country', 'climate_zone']
    for age in age_groups:
        heat_cold_cols.extend([
            f'heat_deaths_{age}_mean', f'heat_deaths_{age}_lower', f'heat_deaths_{age}_upper',
            f'cold_deaths_{age}_mean', f'cold_deaths_{age}_lower', f'cold_deaths_{age}_upper'
        ])
    heat_cold_cols.extend([
        'heat_deaths_total_mean', 'heat_deaths_total_lower', 'heat_deaths_total_upper',
        'cold_deaths_total_mean', 'cold_deaths_total_lower', 'cold_deaths_total_upper'
    ])
    
    grid_heat_cold = grid_results_all[heat_cold_cols]
    
    # Grid results - net only
    net_cols = ['year', 'grid_id', 'Country', 'climate_zone']
    for age in age_groups:
        net_cols.extend([
            f'net_deaths_{age}_mean', f'net_deaths_{age}_lower', f'net_deaths_{age}_upper'
        ])
    net_cols.extend([
        'net_deaths_total_mean', 'net_deaths_total_lower', 'net_deaths_total_upper'
    ])
    
    grid_net = grid_results_all[net_cols]
    
    # Country results - heat and cold only
    country_heat_cold_cols = ['year', 'Country', 'N_Grids']
    for age in age_groups:
        country_heat_cold_cols.extend([
            f'heat_deaths_{age}_mean', f'heat_deaths_{age}_lower', f'heat_deaths_{age}_upper',
            f'cold_deaths_{age}_mean', f'cold_deaths_{age}_lower', f'cold_deaths_{age}_upper'
        ])
    country_heat_cold_cols.extend([
        'heat_deaths_total_mean', 'heat_deaths_total_lower', 'heat_deaths_total_upper',
        'cold_deaths_total_mean', 'cold_deaths_total_lower', 'cold_deaths_total_upper'
    ])
    
    country_heat_cold = country_results_all[country_heat_cold_cols]
    
    # Country results - net only
    country_net_cols = ['year', 'Country', 'N_Grids']
    for age in age_groups:
        country_net_cols.extend([
            f'net_deaths_{age}_mean', f'net_deaths_{age}_lower', f'net_deaths_{age}_upper'
        ])
    country_net_cols.extend([
        'net_deaths_total_mean', 'net_deaths_total_lower', 'net_deaths_total_upper'
    ])
    
    country_net = country_results_all[country_net_cols]
    
    # Region results - heat and cold only
    region_heat_cold_cols = ['year', 'UN_Region', 'N_Grids']
    for age in age_groups:
        region_heat_cold_cols.extend([
            f'heat_deaths_{age}_mean', f'heat_deaths_{age}_lower', f'heat_deaths_{age}_upper',
            f'cold_deaths_{age}_mean', f'cold_deaths_{age}_lower', f'cold_deaths_{age}_upper'
        ])
    region_heat_cold_cols.extend([
        'heat_deaths_total_mean', 'heat_deaths_total_lower', 'heat_deaths_total_upper',
        'cold_deaths_total_mean', 'cold_deaths_total_lower', 'cold_deaths_total_upper'
    ])
    
    region_heat_cold = region_results_all[region_heat_cold_cols]
    
    # Region results - net only
    region_net_cols = ['year', 'UN_Region', 'N_Grids']
    for age in age_groups:
        region_net_cols.extend([
            f'net_deaths_{age}_mean', f'net_deaths_{age}_lower', f'net_deaths_{age}_upper'
        ])
    region_net_cols.extend([
        'net_deaths_total_mean', 'net_deaths_total_lower', 'net_deaths_total_upper'
    ])
    
    region_net = region_results_all[region_net_cols]
    
    scenario_time = time.time() - scenario_start
    
    print(f"\n  ✓ {scenario_name} complete in {scenario_time/60:.1f} minutes")
    print(f"    Grid heat+cold: {len(grid_heat_cold):,} rows")
    print(f"    Country heat+cold: {len(country_heat_cold):,} rows")
    print(f"    Region heat+cold: {len(region_heat_cold):,} rows")
    print(f"    Grid net: {len(grid_net):,} rows")
    print(f"    Country net: {len(country_net):,} rows")
    print(f"    Region net: {len(region_net):,} rows")
    
    return grid_heat_cold, country_heat_cold, region_heat_cold, grid_net, country_net, region_net

# ============================================================================
# PROCESS BOTH SCENARIOS
# ============================================================================

total_start = time.time()

# SSP2-4.5
print(f"\n{'#'*80}")
print(f"SCENARIO 1: SSP2-4.5")
print(f"{'#'*80}")

grid_ssp245, country_ssp245, region_ssp245, net_grid_ssp245, net_country_ssp245, net_region_ssp245 = process_scenario(
    "SSP2-4.5",
    heat_excess_ssp245_file,
    cold_excess_ssp245_file,
    cvd_deaths_ssp2_file
)

# SSP5-8.5
print(f"\n{'#'*80}")
print(f"SCENARIO 2: SSP5-8.5")
print(f"{'#'*80}")

grid_ssp585, country_ssp585, region_ssp585, net_grid_ssp585, net_country_ssp585, net_region_ssp585 = process_scenario(
    "SSP5-8.5",
    heat_excess_ssp585_file,
    cold_excess_ssp585_file,
    cvd_deaths_ssp5_file
)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("SAVING RESULTS")
print("="*80)

print(f"\nSaving SSP2-4.5 outputs...")
grid_ssp245.to_csv(output_grid_ssp245, index=False)
print(f"  ✓ {output_grid_ssp245.name}")

country_ssp245.to_csv(output_country_ssp245, index=False)
print(f"  ✓ {output_country_ssp245.name}")

region_ssp245.to_csv(output_region_ssp245, index=False)
print(f"  ✓ {output_region_ssp245.name}")

net_grid_ssp245.to_csv(output_net_grid_ssp245, index=False)
print(f"  ✓ {output_net_grid_ssp245.name}")

net_country_ssp245.to_csv(output_net_country_ssp245, index=False)
print(f"  ✓ {output_net_country_ssp245.name}")

net_region_ssp245.to_csv(output_net_region_ssp245, index=False)
print(f"  ✓ {output_net_region_ssp245.name}")

print(f"\nSaving SSP5-8.5 outputs...")
grid_ssp585.to_csv(output_grid_ssp585, index=False)
print(f"  ✓ {output_grid_ssp585.name}")

country_ssp585.to_csv(output_country_ssp585, index=False)
print(f"  ✓ {output_country_ssp585.name}")

region_ssp585.to_csv(output_region_ssp585, index=False)
print(f"  ✓ {output_region_ssp585.name}")

net_grid_ssp585.to_csv(output_net_grid_ssp585, index=False)
print(f"  ✓ {output_net_grid_ssp585.name}")

net_country_ssp585.to_csv(output_net_country_ssp585, index=False)
print(f"  ✓ {output_net_country_ssp585.name}")

net_region_ssp585.to_csv(output_net_region_ssp585, index=False)
print(f"  ✓ {output_net_region_ssp585.name}")

# ============================================================================
# SUMMARY
# ============================================================================

total_time = time.time() - total_start

print(f"\n{'='*80}")
print("PROJECTION COMPLETE!")
print("="*80)

print(f"\nTotal execution time: {total_time/60:.1f} minutes")
print(f"\nOutputs saved:")
print(f"  SSP2-4.5:")
print(f"    • {output_grid_ssp245.name}")
print(f"    • {output_country_ssp245.name}")
print(f"    • {output_region_ssp245.name}")
print(f"    • {output_net_grid_ssp245.name}")
print(f"    • {output_net_country_ssp245.name}")
print(f"    • {output_net_region_ssp245.name}")
print(f"  SSP5-8.5:")
print(f"    • {output_grid_ssp585.name}")
print(f"    • {output_country_ssp585.name}")
print(f"    • {output_region_ssp585.name}")
print(f"    • {output_net_grid_ssp585.name}")
print(f"    • {output_net_country_ssp585.name}")
print(f"    • {output_net_region_ssp585.name}")

print(f"\n✅ All projections calculated with corrected net uncertainty propagation!")
print(f"   Net deaths calculated as heat+cold WITHIN each simulation before taking percentiles")
