import ee
import time
import sys
import numpy as np
import pandas as pd
import itertools
import os
import urllib.request, urllib.parse, urllib.error

ee.Initialize()

def export_oneimage(img,folder,name,region,scale,crs):
  task = ee.batch.Export.image(img, name, {
      'driveFolder':folder,
      'driveFileNamePrefix':name,
      'region': region,
      'scale':scale,
      'crs':crs
  })
  task.start()
  while task.status()['state'] == 'RUNNING':
    print('Running...')
    # Perhaps task.cancel() at some point.
    time.sleep(10)
  print('Done.', task.status())




# locations = pd.read_csv('locations_major.csv')
locations = pd.read_csv('../data/subset_locations.csv',header=None)


# Transforms an Image Collection with 1 band per Image into a single Image with items as bands
# Author: Jamie Vleeshouwer

def appendBand(current, previous):
    # Rename the band
    previous=ee.Image(previous)
    current = current.select([0])
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous,None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
    return accum

county_region = ee.FeatureCollection('ft:18Ayj5e7JxxtTPm1BdMnnzWbZMrxMB49eqGDTsaSp')

imgcoll = ee.ImageCollection('MODIS/051/MCD12Q1') \
    .filterBounds(ee.Geometry.Rectangle(-106.5, 50,-64, 23))\
    .filterDate('2001-12-31','2015-12-31')
img=imgcoll.iterate(appendBand)
img=ee.Image(img)

# img_0=ee.Image(ee.Number(0))
# img_5000=ee.Image(ee.Number(5000))
#
# img=img.min(img_5000)
# img=img.max(img_0)

# img=ee.Image(ee.Number(100))
# img=ee.ImageCollection('LC8_L1T').mosaic()

for loc1, loc2, lat, lon in locations.values:
    fname = '{}_{}'.format(int(loc1), int(loc2))

    offset = 0.11
    scale  = 500
    crs='EPSG:4326'

    # filter for a county
    region = county_region.filterMetadata('STATE num', 'equals', loc1)
    region = ee.FeatureCollection(region).filterMetadata('COUNTY num', 'equals', loc2)
    region = region.first()
    region = region.geometry().coordinates().getInfo()[0]

    # region = str([
    #     [lat - offset, lon + offset],
    #     [lat + offset, lon + offset],
    #     [lat + offset, lon - offset],
    #     [lat - offset, lon - offset]])
    while True:
        try:
            export_oneimage(img, 'crop_yield/Data_county_mask', fname, region, scale, crs)
        except:
            print('retry')
            time.sleep(10)
            continue
        break
    # while True:
    #     try:
    #         export_oneimage(img,'Data_test',fname,region,scale,crs)
    #     except:
    #         print 'retry'
    #         time.sleep(10)
    #         continue
    #     break
