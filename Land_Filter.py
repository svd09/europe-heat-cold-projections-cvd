"""
FINAL OCEAN FILTER
Takes the complete grid and removes ONLY ocean cells
Keeps ALL land including Cyprus, Crete, Malta, Iceland
"""

import geopandas as gpd
from shapely.geometry import box

print("FILTERING OCEAN CELLS")
print("="*70)

# Load the complete grid (whichever file you have that shows 34.0°N coverage)
# Try different filenames
import os
files = [f for f in os.listdir('.') if f.endswith('.geojson')]
print(f"Available files: {files}\n")

# You should update this to match your actual filename
GRID_FILE = 'EUROPE_COMPLETE_GRID_025deg.geojson'  

grid = gpd.read_file(GRID_FILE)
print(f"Loaded: {GRID_FILE}")
print(f"Total cells: {len(grid):,}")
print(f"Bounds: {grid.total_bounds[1]:.1f}°N to {grid.total_bounds[3]:.1f}°N\n")

# ============================================================================
# LOAD NATURAL EARTH BOUNDARIES
# ============================================================================

print("Loading Natural Earth boundaries...")
try:
    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
except:
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

print(f"✓ Loaded {len(world)} countries\n")

# ============================================================================
# FILTER TO EUROPE AND CLIP
# ============================================================================

print("Filtering to Europe and clipping...")

# European countries we care about
european_names = {
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus',
    'Czech Republic', 'Czechia', 'Denmark', 'Estonia', 'Finland',
    'France', 'Germany', 'Greece', 'Hungary', 'Ireland',
    'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
    'Netherlands', 'Poland', 'Portugal', 'Romania', 
    'Slovakia', 'Slovenia', 'Spain', 'Sweden',
    'United Kingdom', 'Iceland', 'Norway', 'Switzerland',
    'Albania', 'Bosnia and Herzegovina', 'Bosnia and Herz.',
    'Montenegro', 'North Macedonia', 'Macedonia', 'Serbia',
    'Moldova', 'Ukraine', 'Belarus', 'Kosovo'
}

# Get name field
name_field = 'NAME' if 'NAME' in world.columns else 'name'

# Filter
europe = world[world[name_field].isin(european_names)].copy()

# Clip to bounding box
bbox = box(-25, 34, 40, 71)
europe['geometry'] = europe.geometry.intersection(bbox)
europe = europe[~europe.geometry.is_empty]

# Create land mask
land_mask = europe.geometry.unary_union

print(f"✓ Created land mask from {len(europe)} countries\n")

# ============================================================================
# FILTER GRID
# ============================================================================

print("Filtering grid to land cells...")
print("(This may take 1-2 minutes for 30k+ cells)\n")

# Keep cells that intersect land
grid_land = grid[grid.geometry.intersects(land_mask)].copy()

print(f"✓ Done!")
print(f"  Started with: {len(grid):,} cells")
print(f"  Kept (land):  {len(grid_land):,} cells")
print(f"  Removed:      {len(grid) - len(grid_land):,} cells\n")

# ============================================================================
# VERIFY SOUTHERN COVERAGE
# ============================================================================

print("Verifying southern islands:")

cyprus = grid_land[(grid_land['center_lon'] >= 32) & (grid_land['center_lon'] <= 35) & 
                    (grid_land['center_lat'] >= 34.5) & (grid_land['center_lat'] <= 36)]
print(f"  Cyprus (32-35°E, 34.5-36°N): {len(cyprus)} cells")

crete = grid_land[(grid_land['center_lon'] >= 23) & (grid_land['center_lon'] <= 27) & 
                   (grid_land['center_lat'] >= 34.5) & (grid_land['center_lat'] <= 36)]
print(f"  Crete (23-27°E, 34.5-36°N): {len(crete)} cells")

malta = grid_land[(grid_land['center_lon'] >= 14) & (grid_land['center_lon'] <= 15) & 
                   (grid_land['center_lat'] >= 35.5) & (grid_land['center_lat'] <= 36.5)]
print(f"  Malta (14-15°E, 35.5-36.5°N): {len(malta)} cells")

iceland = grid_land[(grid_land['center_lon'] >= -25) & (grid_land['center_lon'] <= -13) & 
                     (grid_land['center_lat'] >= 63) & (grid_land['center_lat'] <= 67)]
print(f"  Iceland (-25--13°E, 63-67°N): {len(iceland)} cells\n")

# ============================================================================
# SAVE
# ============================================================================

print("="*70)
print("SAVING FINAL GRID")
print("="*70)

grid_land.to_file('EUROPE_FINAL_LAND_025deg.geojson', driver='GeoJSON')
print("✓ Saved: EUROPE_FINAL_LAND_025deg.geojson")

grid_land.to_file('EUROPE_FINAL_LAND_025deg.shp', driver='ESRI Shapefile')
print("✓ Saved: EUROPE_FINAL_LAND_025deg.shp")

grid_land[['grid_id', 'center_lon', 'center_lat', 'lon_idx', 'lat_idx']].to_csv(
    'EUROPE_FINAL_LAND_025deg.csv', index=False
)
print("✓ Saved: EUROPE_FINAL_LAND_025deg.csv")

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"\nFinal grid: {len(grid_land):,} cells")
print(f"Coverage: {grid_land.total_bounds[1]:.1f}°N to {grid_land.total_bounds[3]:.1f}°N")
print("\nUpdate your plotting script to use:")
print("  GRID_FILE = 'EUROPE_FINAL_LAND_025deg.geojson'")
print("="*70)
