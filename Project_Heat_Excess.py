#!/usr/bin/env python3
"""
Calculate Future Heat Excess (2025-2080) for Europe Grid Cells

METHODOLOGY:
1. Load baseline TMREL per grid (from 2016-2023 ERA5 analysis)
2. For each year (2025-2080):
   - Load daily temperatures for that year
   - For each grid:
     - Calculate daily heat excess: temp - TMREL (only when temp > TMREL)
     - Sum to annual heat excess
3. Output: grid_id, year, annual_heat_excess

ASSUMPTIONS:
- TMREL stays constant (no adaptation scenario)
- Heat excess = sum of daily excesses above TMREL
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

print("="*80)
print("CALCULATING FUTURE HEAT EXCESS (2025-2080)")
print("Using Baseline TMREL from 2016-2023")
print("="*80)

# ============================================================================
# FILE PATHS
# ============================================================================

desktop_path = Path("")
data_dir = Path("")

# Baseline TMREL file (from baseline analysis)
baseline_path = Path("/Users/gokulparameswaran/Documents/Case Western/Projects/Epidemiology/Europe Project/2) Primary Analysis/Baseline Period/Analysis Bundle")
tmrel_file = baseline_path / "Europe_Grid_Heat_Excess_2016-2023.csv"

# Projection scenarios
scenarios = {
    'SSP245': data_dir / "ENSEMBLE_SSP245_2020-80_BiasCorrect.csv",
    'SSP585': data_dir / "ENSEMBLE_SSP585_2020-80_BiasCorrect.csv"
}

print(f"\n{'='*80}")
print("STEP 1: Load Baseline TMREL")
print("="*80)

print(f"\nLoading TMREL from: {tmrel_file.name}")

try:
    tmrel_df = pd.read_csv(tmrel_file)
    print(f"✓ Loaded TMREL for {len(tmrel_df):,} grid cells")
except FileNotFoundError:
    print(f"✗ Error: TMREL file not found!")
    print(f"   Expected: {tmrel_file}")
    print(f"\n   Please run Heat_Excess.py first to generate baseline TMREL")
    exit()

# Keep only grid_id and TMREL columns
tmrel_lookup = tmrel_df[['grid_id', 'TMREL']].copy()
tmrel_lookup = tmrel_lookup.set_index('grid_id')

print(f"\nTMREL Summary:")
print(f"  Min TMREL: {tmrel_lookup['TMREL'].min():.1f}°C")
print(f"  Max TMREL: {tmrel_lookup['TMREL'].max():.1f}°C")
print(f"  Mean TMREL: {tmrel_lookup['TMREL'].mean():.1f}°C")
print(f"  Median TMREL: {tmrel_lookup['TMREL'].median():.1f}°C")

print(f"\nSample TMREL values:")
print(tmrel_lookup.head(10))

# ============================================================================
# PROCESS EACH SCENARIO
# ============================================================================

for scenario_name, scenario_file in scenarios.items():
    
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print("="*80)
    
    print(f"\nProjection file: {scenario_file.name}")
    
    if not scenario_file.exists():
        print(f"✗ File not found: {scenario_file}")
        continue
    
    # Storage for results
    all_results = []
    
    # Process years 2025-2080 (skip 2020-2024 to focus on future)
    years_to_process = range(2025, 2081)
    
    print(f"\n{'='*80}")
    print(f"Processing {len(years_to_process)} years (2025-2080)")
    print("="*80)
    
    overall_start = time.time()
    
    for year in years_to_process:
        year_start = time.time()
        
        print(f"\nYear {year}...")
        
        # ====================================================================
        # STEP 2: Load this year's daily temperatures
        # ====================================================================
        
        # We need to read the entire file and filter by year
        # Strategy: Read in chunks to avoid memory issues
        
        print(f"  Loading daily temperatures for {year}...")
        
        # Initialize empty list for this year's data
        year_data = []
        
        # Read file in chunks
        chunksize = 5_000_000  # 5 million rows at a time
        chunk_num = 0
        
        for chunk in pd.read_csv(scenario_file, chunksize=chunksize):
            chunk_num += 1
            
            # Parse date column
            chunk['date'] = pd.to_datetime(chunk['date'])
            
            # Extract year
            chunk['year'] = chunk['date'].dt.year
            
            # Filter to this year only
            year_chunk = chunk[chunk['year'] == year].copy()
            
            if len(year_chunk) > 0:
                year_data.append(year_chunk)
                print(f"    Chunk {chunk_num}: Found {len(year_chunk):,} rows for {year}", end='\r')
        
        # Combine all chunks for this year
        if len(year_data) == 0:
            print(f"  ✗ No data found for {year}")
            continue
        
        year_df = pd.concat(year_data, ignore_index=True)
        print(f"\n  ✓ Loaded {len(year_df):,} rows ({year_df['grid_id'].nunique():,} grids × {year_df['date'].nunique()} days)")
        
        # ====================================================================
        # STEP 3: Calculate heat excess for each grid
        # ====================================================================
        
        print(f"  Calculating heat excess per grid...")
        
        grid_results = []
        
        unique_grids = year_df['grid_id'].unique()
        
        for idx, grid_id in enumerate(unique_grids):
            if (idx + 1) % 2000 == 0:
                print(f"    Processed {idx+1:,}/{len(unique_grids):,} grids...", end='\r')
            
            # Get TMREL for this grid
            if grid_id not in tmrel_lookup.index:
                # Skip grids without TMREL (shouldn't happen)
                continue
            
            tmrel = tmrel_lookup.loc[grid_id, 'TMREL']
            
            # Get temperatures for this grid in this year
            grid_temps = year_df[year_df['grid_id'] == grid_id]['mean'].values
            
            # Calculate heat excess for each day
            # Heat excess = temp - TMREL (only when temp > TMREL)
            heat_excess_daily = np.maximum(grid_temps - tmrel, 0)
            
            # Sum to annual heat excess
            total_heat_excess = heat_excess_daily.sum()
            
            # Calculate AVERAGE daily heat excess (divide by number of days)
            n_days = len(grid_temps)
            avg_daily_heat_excess = total_heat_excess / n_days if n_days > 0 else 0
            
            # Count days above TMREL
            days_above_tmrel = (grid_temps > tmrel).sum()
            
            # Store result
            grid_results.append({
                'grid_id': grid_id,
                'year': year,
                'scenario': scenario_name,
                'tmrel': tmrel,
                'total_heat_excess': total_heat_excess,
                'avg_daily_heat_excess': avg_daily_heat_excess,
                'days_above_tmrel': days_above_tmrel,
                'n_days': n_days,
                'mean_temp': grid_temps.mean(),
                'max_temp': grid_temps.max()
            })
        
        print(f"\n  ✓ Calculated heat excess for {len(grid_results):,} grids")
        
        # Add to overall results
        all_results.extend(grid_results)
        
        year_time = time.time() - year_start
        print(f"  Year {year} completed in {year_time:.1f} seconds")
    
    # ====================================================================
    # STEP 4: Save results for this scenario
    # ====================================================================
    
    results_df = pd.DataFrame(all_results)
    
    output_file = desktop_path / f"Future_Heat_Excess_{scenario_name}_2025-2080.csv"
    results_df.to_csv(output_file, index=False)
    
    overall_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print(f"SCENARIO {scenario_name} COMPLETE")
    print("="*80)
    
    print(f"\n✓ Saved: {output_file.name}")
    print(f"  Rows: {len(results_df):,}")
    print(f"  Years: {results_df['year'].min()} to {results_df['year'].max()}")
    print(f"  Grids: {results_df['grid_id'].nunique():,}")
    print(f"  Total time: {overall_time/60:.1f} minutes")
    
    # Show summary statistics
    print(f"\nHeat Excess Summary ({scenario_name}):")
    print(f"  Mean avg daily heat excess: {results_df['avg_daily_heat_excess'].mean():.4f}°C")
    print(f"  Median avg daily heat excess: {results_df['avg_daily_heat_excess'].median():.4f}°C")
    print(f"  Max avg daily heat excess: {results_df['avg_daily_heat_excess'].max():.4f}°C")
    
    # Show trend over time
    annual_avg = results_df.groupby('year')['avg_daily_heat_excess'].mean()
    print(f"\nTrend over time (Europe-wide average daily heat excess):")
    print(f"  2025: {annual_avg.iloc[0]:.4f}°C")
    print(f"  2050: {annual_avg.loc[2050]:.4f}°C")
    print(f"  2080: {annual_avg.iloc[-1]:.4f}°C")

print(f"\n{'='*80}")
print("ALL SCENARIOS COMPLETE")
print("="*80)

print(f"\nOutput files:")
for scenario_name in scenarios.keys():
    output_file = desktop_path / f"Future_Heat_Excess_{scenario_name}_2025-2080.csv"
    if output_file.exists():
        print(f"  ✓ {output_file.name}")

print(f"\nNext steps:")
print(f"  1. Run cold excess projection script")
print(f"  2. Use these files for Monte Carlo mortality calculations")
print("="*80)
