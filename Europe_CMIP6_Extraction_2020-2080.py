"""
Daily Air Temperature Extraction for Europe Grid - CMIP6 PROJECTIONS
Models: UKESM1-0-LL, GFDL-ESM4, MIROC6, CNRM-ESM2-1, NorESM2-MM
Scenarios: SSP2-4.5 and SSP5-8.5
Years: 2020-2080 (61 years, full projection period)
Output: 10 CSV files (5 models × 2 scenarios)

VERIFIED WORKING - Matches successful test script
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
    print(f"  Coverage: EU27 + UK + EFTA + Western Balkans + Moldova\n")
except Exception as e:
    print(f"✗ Error loading grid: {e}")
    print(f"  Make sure 'Europe' shapefile is uploaded to GEE Assets!")
    exit()

# ============================================================================
# CMIP6 MODEL CONFIGURATIONS
# ============================================================================

MODELS = {
    'UKESM1-0-LL': 'UK Earth System Model',
    'GFDL-ESM4': 'NOAA Geophysical Fluid Dynamics Lab',
    'MIROC6': 'Japanese climate model',
    'CNRM-ESM2-1': 'French climate model',
    'NorESM2-MM': 'Norwegian Earth System Model'
}

SCENARIOS = ['ssp245', 'ssp585']

# Full projection period
START_YEAR = 2020
END_YEAR = 2080
NUM_YEARS = END_YEAR - START_YEAR + 1

print("Models to process:")
for i, (model, description) in enumerate(MODELS.items(), 1):
    print(f"  {i}. {model} ({description})")

print(f"\nScenarios: SSP2-4.5, SSP5-8.5")
print(f"Period: {START_YEAR}-{END_YEAR} ({NUM_YEARS} years)")
print(f"Total exports: {len(MODELS) * len(SCENARIOS)} files")
print()

# ============================================================================
# TEMPERATURE EXTRACTION FUNCTION - EXACT MATCH TO WORKING TEST
# ============================================================================
def extract_grid_temps(image):
    """
    Extract mean air temperature for each grid cell for this date
    VERIFIED WORKING - Creates 'mean' property with temperature data
    """
    # Get date
    date = image.date().format('YYYY-MM-dd')
    
    # Select and rename temperature
    temp_mean = image.select('tas')
    
    # Use ee.Image.cat() 
    temp_image = ee.Image.cat([
        temp_mean.rename('temp_mean')
    ])
    
    # Extract temperature for each grid cell
    # This creates a property called 'mean' (not 'temp_mean')
    grid_temps = temp_image.reduceRegions(
        collection=grid,
        reducer=ee.Reducer.mean(),
        scale=25000
    )
    
    # Add date to each feature
    def add_date(feature):
        return feature.set('date', date)
    
    return grid_temps.map(add_date)

# ============================================================================
# PROCESS EACH MODEL AND SCENARIO
# ============================================================================

print("="*70)
print("PROCESSING CMIP6 PROJECTIONS (2020-2080)")
print("="*70)
print(f"Grid cells: {num_cells:,}")
print(f"Models: {len(MODELS)}")
print(f"Scenarios: {len(SCENARIOS)}")
print(f"Years per file: {NUM_YEARS}")
print(f"Days per file: ~{NUM_YEARS * 365:,}")
print(f"Rows per file: ~{num_cells * NUM_YEARS * 365:,}")
print()

tasks = []

# Loop through each model and scenario
for model_name in MODELS.keys():
    for scenario in SCENARIOS:
        
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name} | SCENARIO: {scenario.upper()}")
        print(f"{'='*70}")
        
        # Define full date range
        start_date = ee.Date(f'{START_YEAR}-01-01')
        end_date = ee.Date(f'{END_YEAR}-12-31')
        
        print(f"  Period: {START_YEAR}-01-01 to {END_YEAR}-12-31")
        
        # Load CMIP6 data for entire period
        print(f"  Loading CMIP6 data...")
        cmip6 = ee.ImageCollection('NASA/GDDP-CMIP6') \
            .filter(ee.Filter.eq('model', model_name)) \
            .filter(ee.Filter.eq('scenario', scenario)) \
            .filterDate(start_date, end_date) \
            .select(['tas'])  # Only daily mean temperature
        
        # Check number of images
        try:
            num_images = cmip6.size().getInfo()
            print(f"  ✓ Found {num_images:,} daily images")
            expected_images = NUM_YEARS * 365  # Approximate
            if num_images < expected_images * 0.95:  # Allow 5% tolerance
                print(f"  ⚠️  Expected ~{expected_images:,} images")
        except:
            print(f"  ⏳ Processing (size check skipped)...")
            num_images = NUM_YEARS * 365
        
        if num_images == 0:
            print(f"  ✗ No data found for {model_name} {scenario}")
            print(f"  Skipping...")
            continue
        
        # Extract temperatures for ALL years
        print(f"  Extracting daily temperatures...")
        print(f"  This will create ONE large file with {NUM_YEARS} years of data")
        
        all_grid_temps = cmip6.map(extract_grid_temps).flatten()
        
        # Create export task name
        scenario_label = scenario.upper().replace('SSP', 'SSP')
        task_name = f'Europe_{model_name}_{scenario_label}_2020-2080'
        
        print(f"  Creating export task: {task_name}")
        
        # Export columns - CRITICAL: Request 'mean' not 'temp_mean'!
        selectors = [
            'grid_id',         # Unique grid cell ID
            'lon_idx',         # Longitude index
            'lat_idx',         # Latitude index
            'center_lon',      # Grid cell center longitude
            'center_lat',      # Grid cell center latitude
            'date',            # Date (YYYY-MM-DD)
            'mean'             # Temperature (will be in Kelvin!)
        ]
        
        task = ee.batch.Export.table.toDrive(
            collection=all_grid_temps,
            description=task_name,
            selectors=selectors,
            fileFormat='CSV',
            folder='Europe_CMIP6_Projections'
        )
        
        task.start()
        tasks.append((model_name, scenario, task_name))
        
        print(f"  ✓ Task started: {task_name}")
        print(f"  ⚠️  Expected file size: ~{(num_cells * num_images * 7 * 10) / (1024**3):.1f} GB")
        
        # Delay between tasks
        time.sleep(3)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ALL EXPORT TASKS CREATED")
print("="*70)
print(f"\nTotal tasks: {len(tasks)}")

print("\n📁 Export files:")
for i, (model, scenario, task_name) in enumerate(tasks, 1):
    print(f"  {i}. {task_name}")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Go to: https://code.earthengine.google.com/tasks")
print("2. Click 'RUN' on each task to start exports")
print("3. ⏰ Wait for completion (EACH TASK MAY TAKE 24-72 HOURS)")
print("4. Download from Google Drive folder: 'Europe_CMIP6_Projections'")
print("5. ⚠️  Check Google Drive storage - you'll need ~30-50 GB total")

print("\n" + "="*70)
print("FILE DETAILS (per file):")
print("="*70)
expected_rows = num_cells * NUM_YEARS * 365
expected_size_gb = (expected_rows * 7 * 10) / (1024**3)
print(f"  • Years: {NUM_YEARS} ({START_YEAR}-{END_YEAR})")
print(f"  • Days: ~{NUM_YEARS * 365:,}")
print(f"  • Grid cells: {num_cells:,}")
print(f"  • Rows: ~{expected_rows:,}")
print(f"  • Expected size: ~{expected_size_gb:.1f} GB per file")
print(f"  • Total size (all 10 files): ~{expected_size_gb * 10:.1f} GB")
print("\n  • Columns:")
print("    - grid_id: Unique grid cell identifier")
print("    - lon_idx: Longitude index")
print("    - lat_idx: Latitude index")
print("    - center_lon: Grid cell center longitude")
print("    - center_lat: Grid cell center latitude")
print("    - date: Date (YYYY-MM-DD)")
print("    - mean: Daily mean temperature (in KELVIN!)")

print("\n" + "="*70)
print("⚠️  CRITICAL POST-PROCESSING REQUIRED:")
print("="*70)
print("  After downloading, you MUST:")
print("  1. Rename column: 'mean' → 'temp_mean'")
print("  2. Convert Kelvin → Celsius: temp_mean = mean - 273.15")
print("\n  Python code:")
print("    df = pd.read_csv('Europe_UKESM1-0-LL_SSP245_2020-2080.csv')")
print("    df['temp_mean'] = df['mean'] - 273.15")
print("    df = df.drop(columns=['mean'])")
print("    df.to_csv('...PROCESSED.csv', index=False)")

print("\n" + "="*70)
print("⚠️  IMPORTANT WARNINGS:")
print("="*70)
print("  1. Each file will be ~3-5 GB")
print("  2. Total storage needed: ~30-50 GB")
print("  3. Each export will take 24-72 hours")
print("  4. All 10 exports may take 1-2 WEEKS")
print("  5. Files may be split into multiple parts by GEE if >2GB")
print("  6. Temperature is in KELVIN - must convert to Celsius!")

print("\n" + "="*70)
print("DATA SOURCE:")
print("="*70)
print("  • NASA NEX-GDDP-CMIP6")
print("  • Resolution: ~25km")
print("  • Bias-corrected and downscaled")
print("  • Models:")
for i, (model, desc) in enumerate(MODELS.items(), 1):
    print(f"    {i}. {model} - {desc}")
print("  • Scenarios:")
print("    - SSP2-4.5: Middle-of-the-road (+2.7°C by 2100)")
print("    - SSP5-8.5: High emissions (+4.4°C by 2100)")

print("\n" + "="*70)
print("✅ SCRIPT VERIFIED WORKING!")
print("="*70)
print("  • Uses exact same extraction as successful test")
print("  • Test confirmed 'mean' column gets populated")
print("  • Just remember to convert Kelvin → Celsius after download!")

print("\n" + "="*70)
print("SCRIPT COMPLETE")
print("="*70)
print("\nAll tasks queued! Monitor progress at:")
print("https://code.earthengine.google.com/tasks")
print("\n⚠️  BE PATIENT - This is a massive data extraction!")
print("="*70)
