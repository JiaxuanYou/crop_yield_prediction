#!bin/bash
# data_download.sh

# RUN:
# ---
#   bash bash_scripts/data_dowload.sh

nohup python pull_MODIS.py > pull_MODIS.out &
nohup python pull_MODIS_entire_county.py > pull_MODIS_entire_county.out &
nohup python pull_MODIS_entire_county_clip.py > pull_MODIS_entire_county_clip.out &
nohup python pull_MODIS_landcover.py > pull_MODIS_landcover.out &
nohup python pull_MODIS_landcover_entire_county.py > pull_MODIS_landcover_entire_county.out &
nohup python pull_MODIS_landcover_entire_county_clip.py > pull_MODIS_landcover_entire_county_clip.out &
nohup python pull_MODIS_temperature_entire_county.py > pull_MODIS_temperature_entire_county.out &
nohup python pull_MODIS_temperature_entire_county_clip.py > pull_MODIS_temperature_entire_county_clip.out &

wait

echo "**** PROCESS DONE ALL FILES DOWNLOADED ****"
