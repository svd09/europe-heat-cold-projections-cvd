import geopandas as gpd
from shapely.geometry import box
import os

files = [f for f in os.listdir(".") if f.endswith(".geojson")]
GRID_FILE = "EUROPE_COMPLETE_GRID_025deg.geojson"
grid = gpd.read_file(GRID_FILE)
try:
    world = gpd.read_file(
        "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    )
except:
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
european_names = {
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "United Kingdom",
    "Iceland",
    "Norway",
    "Switzerland",
    "Albania",
    "Bosnia and Herzegovina",
    "Bosnia and Herz.",
    "Montenegro",
    "North Macedonia",
    "Macedonia",
    "Serbia",
    "Moldova",
    "Ukraine",
    "Belarus",
    "Kosovo",
}
name_field = "NAME" if "NAME" in world.columns else "name"
europe = world[world[name_field].isin(european_names)].copy()
bbox = box(-25, 34, 40, 71)
europe["geometry"] = europe.geometry.intersection(bbox)
europe = europe[~europe.geometry.is_empty]
land_mask = europe.geometry.unary_union
grid_land = grid[grid.geometry.intersects(land_mask)].copy()
cyprus = grid_land[
    (grid_land["center_lon"] >= 32)
    & (grid_land["center_lon"] <= 35)
    & (grid_land["center_lat"] >= 34.5)
    & (grid_land["center_lat"] <= 36)
]
crete = grid_land[
    (grid_land["center_lon"] >= 23)
    & (grid_land["center_lon"] <= 27)
    & (grid_land["center_lat"] >= 34.5)
    & (grid_land["center_lat"] <= 36)
]
malta = grid_land[
    (grid_land["center_lon"] >= 14)
    & (grid_land["center_lon"] <= 15)
    & (grid_land["center_lat"] >= 35.5)
    & (grid_land["center_lat"] <= 36.5)
]
iceland = grid_land[
    (grid_land["center_lon"] >= -25)
    & (grid_land["center_lon"] <= -13)
    & (grid_land["center_lat"] >= 63)
    & (grid_land["center_lat"] <= 67)
]
grid_land.to_file("EUROPE_FINAL_LAND_025deg.geojson", driver="GeoJSON")
grid_land.to_file("EUROPE_FINAL_LAND_025deg.shp", driver="ESRI Shapefile")
grid_land[["grid_id", "center_lon", "center_lat", "lon_idx", "lat_idx"]].to_csv(
    "EUROPE_FINAL_LAND_025deg.csv", index=False
)
