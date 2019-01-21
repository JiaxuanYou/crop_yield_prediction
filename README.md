# Crop yield Prediction with Deep Learning
The necessary code for our paper, [Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data](http://jiaxuanyou.me/files/Jiaxuan_AAAI17.pdf), AAAI 2017 (Best Student Paper Award in Computational Sustainability Track). We are glad to win the "Best Big Data Solution" in [World Bank Big Data Innovation Chanllenge](http://bigdatainnovationchallenge.org/) as well.

Here is a brief introduction on the utilities for each folder.

- **"/1 download data"** How we download data from Google Earth Engine to Google Drive. Users then need to export data from Google Drive to their local folder, e.g., their clusters. The trick there is that we first concatenated all images across all the years available (say 2003 to 2015), then download the huge image at once, which could be hundreds of times faster.
- **"/2 clean data"** How the raw data is preprocessed, including slicing the huge images to get individual images, 3-D histogram calculations, etc.
- **"/3 model"** The CNN/LSTM model structure, written in tensorflow (v0.9). The Gaussian Process model, written in python.
- **"/4 model_batch"** Since we are training different models for each year and each month, a batch code is used for training.
- **"/5 model_semi_supervised"** A recent contribution, extending the model with semi-supervised deep generative model, however it doesn't work well.  We are happy to discuss the model if you can make it work.
- **"/6 result_analysis"** Plot results, plot yield map, etc.

For more information, please contact Jiaxuan You.

youjiaxuan@gmail.com.

# Notes to myself (tommy)

If want to run using python3 you want to install the environment from requirements_3.txt.

Otherwise install the environment from requirements.txt.

### Do we have to specify our own EE login details?
### Do we need lots of space on our own Google drive?
### How do we pull from the EE - Google Drive - Local machines?

