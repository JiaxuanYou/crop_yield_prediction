# Show geolocation
from pyproj import Proj
import rasterio
from mpl_toolkits.basemap import Basemap

data_dir = './'
data = rasterio.open(data_dir+'crop_yield_Data/1_1.tif')

def compute_map_bounds(meta):
  """"""
  try:
    map_width = meta['width'] * meta['affine'][0]
    map_height = meta['height'] * meta['affine'][0]
    xmin = meta['affine'][2]
    ymax = meta['affine'][5]
  except:
    map_width = meta['width'] * meta['transform'][0]
    map_height = meta['height'] * meta['transform'][0]
    xmin = meta['transform'][2]
    ymax = meta['transform'][5]

  xmax = xmin + map_width
  ymin = ymax - map_height
  llproj = (xmin, ymin)
  urproj = (xmax, ymax)
  extent = [xmin, xmax, ymin, ymax] # [left, right, bottom, top]

  return llproj, urproj, extent

def compute_longlat(crs, llproj, urproj):
  """ """
  # Instantiate projection class and compute longlat coordinates of
  # the raster's ll and ur corners
  p = Proj(**crs)
  llll = p(*llproj, inverse=True)
  urll = p(*urproj, inverse=True)

  return llll, urll

def get_geocoords(data):
  """ from a rasterio object return the required parameters for Basemap
  # https://gis.stackexchange.com/questions/224619/how-to-map-albers-projection-raster-in-basemap-of-python
  Arguments:
  ---------
  : data (rasterio obj): data wanting to plot on basemap

  Returns:
  -------
  : :
  """
  meta = data.meta
  crs = meta['crs']

  llproj, urproj, extent = compute_map_bounds(meta)
  llll, urll = compute_longlat(crs, llproj, urproj)

  return crs, llll, urll

def plot_rasterio_on_basemap(data, **kwargs):
  """
  Help: https://gis.stackexchange.com/questions/224619/how-to-map-albers-projection-raster-in-basemap-of-python
  """
  crs, llll, urll = get_geocoords(data)
  try:
    projection = crs['proj']
  except:
    projection = crs['init']
  m = Basemap(llcrnrlon=llll[0], llcrnrlat=llll[1], urcrnrlon=urll[0], urcrnrlat=urll[1],
              projection=projection, resolution='l', **kwargs)
              # There might be other parameters to set depending on your CRS

  m.drawcoastlines()
  m.imshow(data, origin='upper', extent = extent)
  m.colorbar()

  plt.show()

plot_rasterio_on_basemap(data)
