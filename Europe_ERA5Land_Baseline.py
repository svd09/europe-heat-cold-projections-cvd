"""
Daily Air Temperature Extraction for Europe Grid - BASELINE
Dataset: ERA5-Land (Historical Observations)
Years: 2016-2023 (8 years)
Uses DATE-MAPPING approach
Output: 8 CSV files (one per year)
Grid: Europe trimmed shapefile (12,411 cells)
"""

import ee
import time

# ============================================================================
# INITIALIZE EARTH ENGINE
# ============================================================================
PROJECT_ID = ''

try:
    ee.Initialize(project=PROJECT_ID)
    print(f"✓ Earth Engine initialized with project: {PROJECT_ID}\n")
except Exception as e:
    print(f"✗ Error: {e}")
    print("Run: earthengine authenticate")
    exit()

# ============================================================================
# LOAD GRID SHAPEFILE
# ============================================================================
print("Loading Europe grid...")

GRID_ASSET = "projects/climate-data-download-2/assets/Europe"

try:
    grid = ee.FeatureCollection(GRID_ASSET)
    num_cells = grid.size().getInfo()
    print(f"✓ Loaded grid: {num_cells:,} cells")
    print(f"  Asset: {GRID_ASSET}")
    print(f"  Resolution: 0.25° (~25-27 km)")
    print(f"  Coverage: EU27 + UK + EFTA + Western Balkans + Moldova")
    print(f"  Excludes: Ukraine, Belarus, Russia, Turkey\n")
except Exception as e:
    print(f"✗ Error loading grid: {e}")
    print("Make sure you've uploaded Europe_Shape.zip to GEE as asset 'Europe'")
    exit()

# ============================================================================
# LOAD ERA5-LAND DATASET
# ============================================================================
print("Loading ERA5-Land dataset...")
ERA5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
print("✓ ERA5-Land loaded\n")

# ============================================================================
# BASELINE YEARS
# ============================================================================
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

print(f"Baseline period: {YEARS[0]}-{YEARS[-1]}")
print(f"Total years: {len(YEARS)}")
print()

# ============================================================================
# TEMPERATURE EXTRACTION FUNCTION (DATE-BASED)
# ============================================================================
def extract_air_temp(date):
    """
    Extract mean air temperature for each grid cell for this date
    Uses date-mapping approach
    """
    date = ee.Date(date)
    date_str = date.format('YYYY-MM-dd')
    
    # Get ERA5-Land image for this day
    era5_day = ERA5.filterDate(date, date.advance(1, 'day')).first()
    
    # Check if data exists
    era5_exists = ee.Algorithms.If(era5_day, 1, 0)
    
    # Extract temperature and convert Kelvin to Celsius
    air_temp = ee.Image(ee.Algorithms.If(
        era5_exists,
        era5_day.select('temperature_2m').subtract(273.15),
        ee.Image.constant(-999)  # Missing data flag
    ))
    
    # Extract temperature for each grid cell
    grid_temps = air_temp.reduceRegions(
        collection=grid,
        reducer=ee.Reducer.mean(),
        scale=11132  # ERA5-Land resolution ~11km
    )
    
    # Rename 'mean' to 'temp_mean' and add date
    def rename_and_add_date(feature):
        return feature.set({
            'date': date_str,
            'temp_mean': feature.get('mean')
        }).select(['grid_id', 'lon_idx', 'lat_idx', 'center_lon', 'center_lat', 'date', 'temp_mean'])
    
    return grid_temps.map(rename_and_add_date)

# ============================================================================
# PROCESS EACH YEAR
# ============================================================================

print("="*70)
print("PROCESSING ERA5-LAND BASELINE (2016-2023)")
print("="*70)
print(f"Grid cells: {num_cells:,}")
print(f"Years: {len(YEARS)}")
print()

tasks = []

# Loop through each year
for year in YEARS:
    
    print(f"\n{'='*70}")
    print(f"YEAR: {year} (ERA5-Land Historical)")
    print(f"{'='*70}")
    
    # Define date range for this year
    start_date = f'{year}-01-01'
    end_date = f'{year+1}-01-01'  # Next year to include Dec 31
    
    # Create list of dates
    print(f"  Creating date sequence...")
    dates = ee.List.sequence(
        ee.Date(start_date).millis(),
        ee.Date(end_date).advance(-1, 'day').millis(),  # Don't include next year's Jan 1
        1000 * 60 * 60 * 24  # One day in milliseconds
    ).map(lambda d: ee.Date(d))
    
    num_days = dates.size().getInfo()
    print(f"  ✓ Processing {num_days} days")
    
    # Extract temperatures for all dates
    print(f"  Extracting daily temperatures for {num_cells:,} grid cells...")
    print(f"  Converting from Kelvin to Celsius...")
    
    all_grid_temps = ee.FeatureCollection(dates.map(extract_air_temp)).flatten()
    
    # Create export task name
    task_name = f'Europe_ERA5Land_Baseline_{year}'
    
    print(f"  Creating export task: {task_name}")
    
    # Export structure
    selectors = [
        'grid_id',         # Unique grid cell ID
        'lon_idx',         # Longitude index
        'lat_idx',         # Latitude index
        'center_lon',      # Grid cell center longitude
        'center_lat',      # Grid cell center latitude
        'date',            # Date (YYYY-MM-DD)
        'temp_mean'        # Daily mean temperature (°C)
    ]
    
    task = ee.batch.Export.table.toDrive(
        collection=all_grid_temps,
        description=task_name,
        selectors=selectors,
        fileFormat='CSV',
        folder='Europe_ERA5Land_Baseline'
    )
    
    task.start()
    tasks.append((year, task_name))
    
    print(f"  ✓ Task started: {task_name}")
    
    # Small delay between tasks
    time.sleep(2)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ALL EXPORT TASKS CREATED")
print("="*70)
print(f"\nTotal tasks: {len(tasks)}")

print("\n📅 Tasks by year:")
for year, task_name in tasks:
    print(f"  • {year}: {task_name}")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Go to: https://code.earthengine.google.com/tasks")
print("2. Click 'RUN' on each task to start exports")
print("3. Wait for completion (~30 min to 2 hours per task)")
print("4. Download from Google Drive folder: 'Europe_ERA5Land_Baseline'")

print("\n" + "="*70)
print("FILE DETAILS (each file):")
print("="*70)
print(f"  • Rows: ~{num_cells * 365:,} ({num_cells:,} cells × 365 days)")
print(f"  • Size: ~200-250 MB per file")
print("  • Columns:")
print("    - grid_id: Unique grid cell identifier")
print("    - lon_idx: Longitude index")
print("    - lat_idx: Latitude index")
print("    - center_lon: Grid cell center longitude")
print("    - center_lat: Grid cell center latitude")
print("    - date: Date (YYYY-MM-DD)")
print("    - temp_mean: Daily mean temperature (°C)")

print("\n" + "="*70)
print("✅ GRID DETAILS")
print("="*70)
print(f"  • Total cells: {num_cells:,}")
print("  • Resolution: 0.25°")
print("  • Trimmed to exact country boundaries")
print("  • Coverage: 34.5°N to 71.0°N")
print("  • Countries: EU27 + UK + Iceland + Norway + Switzerland")
print("  •            + Western Balkans + Moldova")
print("  • Excluded: Ukraine, Belarus, Russia, Turkey")

print("\n" + "="*70)
print("DATA SOURCE:")
print("="*70)
print("  • ECMWF ERA5-Land Daily Aggregated")
print("  • Resolution: ~11km (0.1° native)")
print("  • Aggregated to 0.25° grid cells")
print("  • Variables: temperature_2m (daily mean)")
print("  • Period: 2016-2023 (8 years)")
print("  • Temperature converted: Kelvin → Celsius")
print("  • Missing data flagged as -999")

print("\n" + "="*70)
print("EXPECTED OUTPUT:")
print("="*70)
print("  Each CSV will contain:")
print(f"  • ~{num_cells:,} grid cells")
print("  • 365 days per cell (366 for leap years)")
print(f"  • Total rows per file: ~{num_cells * 365:,}")
print("  • One row = one cell, one day")
print("  • Temperature in Celsius")
print("  • Ready for climate-health analysis!")

print("\n" + "="*70)
print("SCRIPT COMPLETE")
print("="*70)
print("\nAll tasks queued! Monitor progress at:")
print("https://code.earthengine.google.com/tasks")
print("="*70)
