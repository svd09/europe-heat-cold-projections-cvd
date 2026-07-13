"""
Europe Grid - BULLETPROOF VERSION
Generates EVERY SINGLE CELL in bounding box using simple iteration
No fancy sampling - just brute force cell generation
"""

import ee

# Initialize
PROJECT_ID = ''
ee.Initialize(project=PROJECT_ID)

print("="*70)
print("="*70)

# Define bounds
WEST = -25.0
EAST = 40.25
SOUTH = 34.0
NORTH = 71.25
RESOLUTION = 0.25

print(f"\nBounds: {SOUTH}°N to {NORTH}°N, {WEST}°E to {EAST}°E")
print(f"Resolution: {RESOLUTION}°")

# Calculate grid dimensions
n_lon = int((EAST - WEST) / RESOLUTION)
n_lat = int((NORTH - SOUTH) / RESOLUTION)

print(f"Dimensions: {n_lon} × {n_lat} = {n_lon * n_lat:,} cells")
print("\nGenerating cells using server-side iteration...")

# Create list of all cell features
cells = []

# Generate cell centers
lons = ee.List.sequence(WEST + RESOLUTION/2, EAST - RESOLUTION/2, RESOLUTION)
lats = ee.List.sequence(SOUTH + RESOLUTION/2, NORTH - RESOLUTION/2, RESOLUTION)

# Create grid using nested mapping
def create_cells_for_lat(lat):
    lat = ee.Number(lat)
    
    def create_cell_at_lon(lon):
        lon = ee.Number(lon)
        
        # Create cell polygon
        cell = ee.Geometry.Rectangle([
            lon.subtract(RESOLUTION/2),
            lat.subtract(RESOLUTION/2),
            lon.add(RESOLUTION/2),
            lat.add(RESOLUTION/2)
        ])
        
        # Calculate indices
        lon_idx = lon.subtract(WEST).subtract(RESOLUTION/2).divide(RESOLUTION).floor()
        lat_idx = lat.subtract(SOUTH).subtract(RESOLUTION/2).divide(RESOLUTION).floor()
        grid_id = lat_idx.multiply(n_lon).add(lon_idx)
        
        return ee.Feature(cell, {
            'grid_id': grid_id,
            'center_lon': lon,
            'center_lat': lat,
            'lon_idx': lon_idx,
            'lat_idx': lat_idx
        })
    
    return lons.map(create_cell_at_lon)

# Generate all cells
all_cells_nested = lats.map(create_cells_for_lat)
all_cells_flat = all_cells_nested.flatten()
grid = ee.FeatureCollection(all_cells_flat)

print("✓ Grid generated")

# Export
print("\n" + "="*70)
print("EXPORTING")
print("="*70)

task = ee.batch.Export.table.toDrive(
    collection=grid,
    description='EUROPE_COMPLETE_GRID_025deg',
    fileFormat='GeoJSON',
    folder='Europe_Climate_Grid'
)

task.start()
print("\n✓ Export started: EUROPE_COMPLETE_GRID_025deg")
print(f"\nThis will create {n_lon * n_lat:,} cells")
print("EVERY cell from 34.0°N to 71.25°N will be included")
print("\nGo to: https://code.earthengine.google.com/tasks")
print("Click RUN and wait ~15-20 minutes")
print("="*70)
