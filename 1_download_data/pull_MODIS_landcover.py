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




locations = pd.read_csv('locations.csv')


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

imgcoll = ee.ImageCollection('MODIS/051/MCD12Q1') \
    .filterBounds(ee.Geometry.Rectangle(-106.5, 50,-64, 23))
img=imgcoll.iterate(appendBand)

for loc1, loc2, lat, lon in locations.values:
    fname = '{}_{}'.format(int(loc1), int(loc2))

    offset = 0.11
    scale  = 500
    crs='EPSG:4326'

    region = str([
        [lat - offset, lon + offset],
        [lat + offset, lon + offset],
        [lat + offset, lon - offset],
        [lat - offset, lon - offset]])

    while True:
        try:
            export_oneimage(img,'Data_mask',fname,region,scale,crs)
        except:
            print('retry')
            time.sleep(10)
            continue
        break