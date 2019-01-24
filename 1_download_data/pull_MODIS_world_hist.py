import ee
import time
import sys
import numpy as np
import pandas as pd
import itertools
import os
import urllib.request, urllib.parse, urllib.error

ee.Initialize()

# locations = pd.read_csv('locations_remedy.csv')
locations = pd.read_csv('world_locations.csv',header=None)

def appendBand(current, previous):
    # Rename the band
    previous=ee.Image(previous)
    current = current.select([0,1,2,3,4,5,6])
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous,None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
    return accum

# county_region = ee.FeatureCollection('ft:18Ayj5e7JxxtTPm1BdMnnzWbZMrxMB49eqGDTsaSp')
world_region = ee.FeatureCollection('ft:1tdSwUL7MVpOauSgRzqVTOwdfy17KDbw-1d9omPw')

imgcoll = ee.ImageCollection('MODIS/MOD09A1') \
    .filterDate('2001-12-31','2015-12-31')
img=imgcoll.iterate(appendBand)
img=ee.Image(img)

for country,index in locations.values:
    scale  = 500
    crs='EPSG:4326'

    # filter for a country
    region = world_region.filterMetadata('Country', 'equals', country)
    if region==None:
        print(country,index,'not found')
        continue
    region = region.first()
    # region = region.geometry().coordinates().getInfo()[0]

    img_temp = img.clip(region)
    hist = ee.Feature(None, {'mean': img_temp.reduceRegion(ee.Reducer.fixedHistogram(1,4999,32), region, scale, crs,None,False,1e12,16)})

    hist_info = hist.getInfo()['features']
    print(hist_info)
