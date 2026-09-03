import ee

PROJECT_ID = "YOUR_GEE_PROJECT_ID"
ee.Initialize(project=PROJECT_ID)
WEST = -25.0
EAST = 40.25
SOUTH = 34.0
NORTH = 71.25
RESOLUTION = 0.25
n_lon = int((EAST - WEST) / RESOLUTION)
n_lat = int((NORTH - SOUTH) / RESOLUTION)
cells = []
lons = ee.List.sequence(WEST + RESOLUTION / 2, EAST - RESOLUTION / 2, RESOLUTION)
lats = ee.List.sequence(SOUTH + RESOLUTION / 2, NORTH - RESOLUTION / 2, RESOLUTION)


def create_cells_for_lat(lat):
    lat = ee.Number(lat)

    def create_cell_at_lon(lon):
        lon = ee.Number(lon)
        cell = ee.Geometry.Rectangle(
            [
                lon.subtract(RESOLUTION / 2),
                lat.subtract(RESOLUTION / 2),
                lon.add(RESOLUTION / 2),
                lat.add(RESOLUTION / 2),
            ]
        )
        lon_idx = lon.subtract(WEST).subtract(RESOLUTION / 2).divide(RESOLUTION).floor()
        lat_idx = (
            lat.subtract(SOUTH).subtract(RESOLUTION / 2).divide(RESOLUTION).floor()
        )
        grid_id = lat_idx.multiply(n_lon).add(lon_idx)
        return ee.Feature(
            cell,
            {
                "grid_id": grid_id,
                "center_lon": lon,
                "center_lat": lat,
                "lon_idx": lon_idx,
                "lat_idx": lat_idx,
            },
        )

    return lons.map(create_cell_at_lon)


all_cells_nested = lats.map(create_cells_for_lat)
all_cells_flat = all_cells_nested.flatten()
grid = ee.FeatureCollection(all_cells_flat)
task = ee.batch.Export.table.toDrive(
    collection=grid,
    description="EUROPE_COMPLETE_GRID_025deg",
    fileFormat="GeoJSON",
    folder="Europe_Climate_Grid",
)
task.start()
