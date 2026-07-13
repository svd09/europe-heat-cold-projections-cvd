"""
Calculate Cold Excess for Europe Grid Cells - OPTIMIZED VERSION
Baseline Period: 2016-2023
- TMREL = Modal temperature (1 d.p.) from 54th-92nd percentile
- Cold excess = Daily temperatures below TMREL
- Missing data imputed from nearest neighbor grid cell (VECTORIZED)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import time

print("="*80)
print("CALCULATING COLD EXCESS FOR EUROPE GRID CELLS")
print("WITH NEAREST NEIGHBOR IMPUTATION (OPTIMIZED)")
print("="*80)

# ============================================================================
# FILE PATHS
# ============================================================================

desktop_path = Path("")

# Input file
input_file = desktop_path / "Europe_ERA5_2016-23.csv"

print(f"\nReading gridded temperature data...")
print(f"File: {input_file.name}")

start_time = time.time()

try:
    df = pd.read_csv(input_file)
    print(f"✓ Successfully loaded data: {df.shape[0]:,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print(f"✗ Error: File not found - {input_file}")
    exit()

print(f"Loaded in {time.time() - start_time:.1f} seconds")
print(f"\nColumns: {list(df.columns)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ============================================================================
# VERIFY REQUIRED COLUMNS
# ============================================================================

required_cols = ['grid_id', 'date', 'temp_mean']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"\n✗ Error: Missing required columns: {missing_cols}")
    exit()

# Verify coordinate columns exist
if 'center_lon' not in df.columns or 'center_lat' not in df.columns:
    print(f"\n✗ Error: Need 'center_lon' and 'center_lat' columns for spatial imputation")
    exit()

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Round temperature to 1 decimal place
df['temp_rounded'] = df['temp_mean'].round(1)

print(f"\n{'='*80}")
print("STEP 0: OPTIMIZED Imputing missing temperature data from nearest neighbors")
print("="*80)

# Get unique grid cells with their coordinates
grid_coords = df[['grid_id', 'center_lon', 'center_lat']].drop_duplicates()
print(f"Total grid cells: {len(grid_coords):,}")

# Build KD-tree for spatial queries
coords_array = grid_coords[['center_lon', 'center_lat']].values
tree = cKDTree(coords_array)
grid_id_list = grid_coords['grid_id'].tolist()

# Count missing values before imputation
missing_before = df['temp_rounded'].isna().sum()
print(f"Missing temperature values before imputation: {missing_before:,}")

# ============================================================================
# OPTIMIZED IMPUTATION - VECTORIZED APPROACH
# ============================================================================

if missing_before > 0:
    print("Imputing missing values using optimized vectorized approach...")
    impute_start = time.time()
    
    # Create a pivot table for fast lookups: grid_id x date
    print("  Building lookup table...")
    df_pivot = df.pivot_table(
        index='date', 
        columns='grid_id', 
        values='temp_rounded',
        aggfunc='first'
    )
    
    # Find neighbors for each grid (do this once)
    print("  Finding nearest neighbors for each grid...")
    grid_neighbors = {}
    for i, grid_id in enumerate(grid_id_list):
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i+1:,}/{len(grid_id_list):,} grids...")
        
        current_coords = coords_array[i]
        distances, indices = tree.query(current_coords, k=11)  # Self + 10 neighbors
        
        # Store neighbor grid IDs (excluding self)
        neighbor_ids = [grid_id_list[idx] for idx in indices[1:]]
        grid_neighbors[grid_id] = neighbor_ids
    
    # Impute missing values using merge-based approach
    print("  Imputing missing values...")
    
    # Get grids with missing data
    grids_with_missing = df[df['temp_rounded'].isna()]['grid_id'].unique()
    print(f"  Grids with missing data: {len(grids_with_missing):,}")
    
    imputed_count = 0
    
    # For each grid with missing data
    for i, grid_id in enumerate(grids_with_missing):
        if (i + 1) % 50 == 0 or (i + 1) == len(grids_with_missing):
            print(f"    Imputing grid {i+1:,}/{len(grids_with_missing):,}...")
        
        # Get neighbors for this grid
        neighbors = grid_neighbors.get(grid_id, [])
        if len(neighbors) == 0:
            continue
        
        # Get rows for this grid that need imputation
        grid_missing_mask = (df['grid_id'] == grid_id) & (df['temp_rounded'].isna())
        missing_dates = df.loc[grid_missing_mask, 'date'].values
        
        if len(missing_dates) == 0:
            continue
        
        # Try each neighbor in order until all values are filled
        for neighbor_id in neighbors:
            # Check if still have missing values
            still_missing = df.loc[grid_missing_mask, 'temp_rounded'].isna()
            if not still_missing.any():
                break  # All filled
            
            # Get neighbor data for the missing dates
            if neighbor_id not in df_pivot.columns:
                continue
            
            # Get neighbor temps for missing dates
            neighbor_data = df_pivot[neighbor_id].reindex(missing_dates)
            
            # Fill missing values where neighbor has valid data
            still_missing_idx = df[grid_missing_mask & df['temp_rounded'].isna()].index
            
            for idx, date in zip(still_missing_idx, df.loc[still_missing_idx, 'date']):
                if date in neighbor_data.index:
                    neighbor_val = neighbor_data.loc[date]
                    if pd.notna(neighbor_val):
                        df.at[idx, 'temp_rounded'] = neighbor_val
                        imputed_count += 1
    
    impute_time = time.time() - impute_start
    print(f"✓ Imputed {imputed_count:,} missing temperature values in {impute_time:.1f} seconds")
    
    # Check remaining missing
    missing_after = df['temp_rounded'].isna().sum()
    print(f"Missing temperature values after imputation: {missing_after:,}")
    
    if missing_after > 0:
        print(f"⚠️  Warning: {missing_after:,} values could not be imputed (no valid neighbors)")
        print("   These will be excluded from analysis")
else:
    print("✓ No missing temperature values - skipping imputation")

print(f"\n{'='*80}")
print("STEP 1: Calculating TMREL for each grid cell")
print("TMREL = Modal temperature (1 d.p.) from 54th-92nd percentile")
print("="*80)

# Calculate total study period days
start_date = df['date'].min()
end_date = df['date'].max()
STUDY_PERIOD_DAYS = (end_date - start_date).days + 1
print(f"Study period: {start_date.date()} to {end_date.date()} ({STUDY_PERIOD_DAYS} days)")

# ============================================================================
# CALCULATE COLD EXCESS FOR EACH GRID CELL
# ============================================================================

grid_stats = []

unique_grids = df['grid_id'].unique()
print(f"\nProcessing {len(unique_grids):,} grid cells...")

calc_start = time.time()

for idx, grid_id in enumerate(unique_grids):
    if (idx + 1) % 1000 == 0:
        elapsed = time.time() - calc_start
        rate = (idx + 1) / elapsed
        remaining = (len(unique_grids) - idx - 1) / rate
        print(f"  Processed {idx + 1:,}/{len(unique_grids):,} grid cells... "
              f"ETA: {remaining/60:.1f} min")
    
    grid_data = df[df['grid_id'] == grid_id].copy()
    
    # Get grid coordinates
    center_lon = grid_data['center_lon'].iloc[0]
    center_lat = grid_data['center_lat'].iloc[0]
    
    # Remove rows with missing temperature data (if any remain after imputation)
    grid_data_clean = grid_data.dropna(subset=['temp_rounded'])
    
    # Skip grid if no valid data
    if len(grid_data_clean) == 0:
        print(f"  Warning: Skipping grid {grid_id} - no valid temperature data")
        continue
    
    # ============= CALCULATE TMREL (54th-92nd PERCENTILE MODAL TEMP) =============
    # Calculate 54th and 92nd percentiles
    p54 = np.percentile(grid_data_clean['temp_rounded'], 54)
    p92 = np.percentile(grid_data_clean['temp_rounded'], 92)
    
    # Filter to only temperatures in 54th-92nd percentile range
    percentile_filtered = grid_data_clean[
        (grid_data_clean['temp_rounded'] >= p54) & 
        (grid_data_clean['temp_rounded'] <= p92)
    ].copy()
    
    # Calculate mode (TMREL) from filtered data at 1 decimal place
    mode_series = percentile_filtered['temp_rounded'].mode()
    
    if len(mode_series) > 0:
        tmrel = mode_series.iloc[0]
        mode_count = (percentile_filtered['temp_rounded'] == tmrel).sum()
        
        # If mode appears only once, try rounding to whole numbers
        if mode_count == 1:
            percentile_filtered['temp_whole'] = percentile_filtered['temp_rounded'].round(0)
            mode_series_whole = percentile_filtered['temp_whole'].mode()
            if len(mode_series_whole) > 0:
                tmrel = mode_series_whole.iloc[0]
                mode_count = (percentile_filtered['temp_whole'] == tmrel).sum()
            else:
                # Fallback to median
                tmrel = percentile_filtered['temp_rounded'].median()
                mode_count = 0
    else:
        # Fallback to median if no mode found
        tmrel = percentile_filtered['temp_rounded'].median()
        mode_count = 0
    # =============================================================================
    
    # Total number of days with data
    total_days = len(grid_data_clean)
    
    # STEP 2: Calculate COLD excess BELOW TMREL
    # Cold excess = TMREL - Temperature (inverted from heat excess)
    grid_data_clean['cold_excess'] = tmrel - grid_data_clean['temp_rounded']
    
    # Only count positive cold excess (temperatures BELOW TMREL)
    grid_data_clean.loc[grid_data_clean['cold_excess'] <= 0, 'cold_excess'] = 0
    
    # Total cold excess across entire study period
    total_cold_excess = grid_data_clean['cold_excess'].sum()
    
    # STEP 3: Calculate average daily cold excess
    # Divide total cold excess by total study period days
    avg_daily_cold_excess = total_cold_excess / STUDY_PERIOD_DAYS
    
    # Additional useful metrics
    days_below_tmrel = (grid_data_clean['temp_rounded'] < tmrel).sum()
    pct_days_below_tmrel = (days_below_tmrel / total_days) * 100 if total_days > 0 else 0
    max_cold_excess = grid_data_clean['cold_excess'].max()
    mean_temp = grid_data_clean['temp_rounded'].mean()
    min_temp = grid_data_clean['temp_rounded'].min()
    max_temp = grid_data_clean['temp_rounded'].max()
    
    # Data quality check
    missing_days = STUDY_PERIOD_DAYS - total_days
    
    grid_stats.append({
        'grid_id': grid_id,
        'center_lon': center_lon,
        'center_lat': center_lat,
        'TMREL': round(tmrel, 1),
        'TMREL_Frequency': mode_count,
        'Total_Days': total_days,
        'Missing_Days': missing_days,
        'Total_Cold_Excess_Celsius': round(total_cold_excess, 2),
        'Avg_Daily_Cold_Excess': round(avg_daily_cold_excess, 4),
        'Days_Below_TMREL': days_below_tmrel,
        'Percent_Days_Below_TMREL': round(pct_days_below_tmrel, 2),
        'Max_Cold_Excess': round(max_cold_excess, 2),
        'Mean_Temperature': round(mean_temp, 2),
        'Min_Temperature': round(min_temp, 2),
        'Max_Temperature': round(max_temp, 2),
        'P54_Temperature': round(p54, 2),
        'P92_Temperature': round(p92, 2)
    })

calc_time = time.time() - calc_start
print(f"✓ Processed all {len(unique_grids):,} grid cells in {calc_time/60:.1f} minutes")

# ============================================================================
# CREATE RESULTS DATAFRAME
# ============================================================================

results_df = pd.DataFrame(grid_stats)

print(f"\n{'='*80}")
print("RESULTS SUMMARY - COLD EXCESS")
print("="*80)
print(f"\nTotal grid cells analyzed: {len(results_df):,}")

print(f"\nDescriptive Statistics:")
print(f"{'Metric':<40} {'Mean':<12} {'Median':<12} {'Min':<12} {'Max':<12}")
print("-" * 88)
print(f"{'TMREL (°C)':<40} {results_df['TMREL'].mean():<12.2f} {results_df['TMREL'].median():<12.2f} {results_df['TMREL'].min():<12.2f} {results_df['TMREL'].max():<12.2f}")
print(f"{'Mean Temperature (°C)':<40} {results_df['Mean_Temperature'].mean():<12.2f} {results_df['Mean_Temperature'].median():<12.2f} {results_df['Mean_Temperature'].min():<12.2f} {results_df['Mean_Temperature'].max():<12.2f}")
print(f"{'Total Cold Excess (°C)':<40} {results_df['Total_Cold_Excess_Celsius'].mean():<12.2f} {results_df['Total_Cold_Excess_Celsius'].median():<12.2f} {results_df['Total_Cold_Excess_Celsius'].min():<12.2f} {results_df['Total_Cold_Excess_Celsius'].max():<12.2f}")
print(f"{'Avg Daily Cold Excess (°C)':<40} {results_df['Avg_Daily_Cold_Excess'].mean():<12.4f} {results_df['Avg_Daily_Cold_Excess'].median():<12.4f} {results_df['Avg_Daily_Cold_Excess'].min():<12.4f} {results_df['Avg_Daily_Cold_Excess'].max():<12.4f}")
print(f"{'% Days Below TMREL':<40} {results_df['Percent_Days_Below_TMREL'].mean():<12.2f} {results_df['Percent_Days_Below_TMREL'].median():<12.2f} {results_df['Percent_Days_Below_TMREL'].min():<12.2f} {results_df['Percent_Days_Below_TMREL'].max():<12.2f}")

print(f"\nSample of results (first 10 grid cells):")
print(results_df[['grid_id', 'TMREL', 'Total_Cold_Excess_Celsius', 
                  'Avg_Daily_Cold_Excess', 'Percent_Days_Below_TMREL']].head(10).to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_file = desktop_path / "Europe_Grid_Cold_Excess_2016-2023.csv"
results_df.to_csv(output_file, index=False)

print(f"\n{'='*80}")
print(f"✓ Results saved to: {output_file.name}")
print("="*80)

total_time = time.time() - start_time
print(f"\nTOTAL RUNTIME: {total_time/60:.1f} minutes")

print(f"\nDATA QUALITY:")
print(f"  • Missing values before imputation: {missing_before:,}")
print(f"  • Values imputed from neighbors: {imputed_count if missing_before > 0 else 0:,}")
print(f"  • Missing values after imputation: {df['temp_rounded'].isna().sum():,}")

print(f"\nKEY OUTPUT COLUMNS:")
print(f"  • grid_id - Grid cell identifier")
print(f"  • TMREL - Modal temperature from 54th-92nd percentile (°C, 1 d.p.)")
print(f"  • Avg_Daily_Cold_Excess - For RR calculation (°C)")
print(f"  • Total_Cold_Excess_Celsius - Sum of all daily cold excesses (°C)")

print(f"\nNEXT STEPS:")
print(f"  1. Merge with heat excess results")
print(f"  2. Calculate RR and PAF")
print("="*80)
