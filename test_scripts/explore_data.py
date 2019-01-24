# explore_data.py
import rasterio
from rasterio.plot import show_hist, show
import numpy
import matplotlib.pyplot as plt

from rasterstats import zonal_stats
import osmnx as ox
import geopandas as gpd
from pyproj import Proj

from mpl_toolkits.basemap import Basemap

data_dir = './'
data = rasterio.open(data_dir+'crop_yield_Data/1_1.tif')
temp = rasterio.open(data_dir+'crop_yield_data_temperature/1_1.tif')
mask = rasterio.open(data_dir+'crop_yield_Data_mask/1_1.tif')

# view the shape of the data
data.meta

# visualise the data
# rasterio.plot import show_hist
rasterio.plot.show((data,1),cmap='Reds')

# visualise the first timestep (7 bands) (RASTERIO = 1 based indexing)
hist = data.read(range(1,8))
show_hist(hist, bins=50, lw=0.0, stacked=False, alpha=0.3,
  histtype='stepfilled', title="Histogram")

# NDVI
nir = data.read(2)
red = data.read(1)
ndvi = (nir-red)/(nir+red)

# Combine with OSM data
# https://nominatim.openstreetmap.org/
# https://automating-gis-processes.github.io/CSC18/lessons/L6/zonal-statistics.html

osm_query = "alabama river"
alabama_river = ox.gdf_from_place(osm_query)
alabama_river = alabama_river.to_crs(crs=data.crs.data)

osm_query = "Prattville Airport"
airport = ox.gdf_from_place(osm_query)
airport = airport.to_crs(crs=data.crs.data)

# show the roads and buildings in Antauga County
ox.plot_graph(ox.graph_from_place('Autauga County'))

alabama_river.plot(ax=ax, facecolor='blue', edgecolor='blue', linewidth=2)
airport.plot(ax=ax, facecolor='gray', edgecolor='red', linewidth=0.5)
show((data, 1), ax=ax)
plt.show()









