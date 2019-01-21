import numpy as np
import csv
from BeautifulSoup import BeautifulSoup
from GP_crop_v3 import *
 

# Read CNN_err prediction
CNN = {}
GP = {}
save_path = '/atlas/u/jiaxuan/data/train_results/final/monthly/'
save_path = 'C:/360Downloads/final/monthly/'
path_current = save_path+str(0)+str(30)+str(2014)+'result_prediction.npz'
data = np.load(path_current)

year = data['year_out']
real = data['real_out']
pred = data['pred_out']
index=data['index_out']

validate = np.nonzero(year == 2014)[0]
year = year[validate]
real = real[validate]
pred = pred[validate]
index = index[validate]
err_CNN = pred-real

rmse,me,err_GP = GaussianProcess(2014,path_current)


print 'CNN',err_CNN.min(),err_CNN.max()
print 'GP',err_GP.min(),err_GP.max()

for i in range(year.shape[0]):
    loc1 = str(int(index[i,0]))
    loc2 = str(int(index[i,1]))
    if len(loc1)==1:
        loc1='0'+loc1
    if len(loc2)==1:
        loc2='00'+loc2
    if len(loc2)==2:
        loc2='0'+loc2
    fips = loc1+loc2
    CNN[fips] = err_CNN[i]
    GP[fips] = err_GP[i]

'''CNN'''
# Load the SVG map
svg = open('counties.svg', 'r').read()
# Load into Beautiful Soup
soup = BeautifulSoup(svg, selfClosingTags=['defs','sodipodi:namedview'])
# Find counties
paths = soup.findAll('path')
# Map colors
colors = ["#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac"]

# County style
path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'
# Color the counties based on unemployment rate
for p in paths:    
    if p['id'] not in ["State_Lines", "separator"]:
        try:
            rate = CNN[p['id']]
        except:
            continue
        if rate > 15:
            color_class = 7
        elif rate > 10:
            color_class = 6
        elif rate > 5:
            color_class = 5
        elif rate > 0:
            color_class = 4
        elif rate > -5:
            color_class = 3
        elif rate > -10:
            color_class = 2            
        elif rate > -15:
            color_class = 1
        else:
            color_class = 0

        color = colors[color_class]
        p['style'] = path_style + color
 
soup=soup.prettify()
with open('CNN_err.svg', 'wb') as f:
    f.write(soup)

'''GP'''
# Load the SVG map
svg = open('counties.svg', 'r').read()
# Load into Beautiful Soup
soup = BeautifulSoup(svg, selfClosingTags=['defs','sodipodi:namedview'])
# Find counties
paths = soup.findAll('path')
# Map colors
colors = ["#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac"]

# County style
path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'
# Color the counties based on unemployment rate
for p in paths:    
    if p['id'] not in ["State_Lines", "separator"]:
        try:
            rate = GP[p['id']]
        except:
            continue
        if rate > 15:
            color_class = 7
        elif rate > 10:
            color_class = 6
        elif rate > 5:
            color_class = 5
        elif rate > 0:
            color_class = 4
        elif rate > -5:
            color_class = 3
        elif rate > -10:
            color_class = 2            
        elif rate > -15:
            color_class = 1
        else:
            color_class = 0

        color = colors[color_class]
        p['style'] = path_style + color
 
soup=soup.prettify()
with open('GP_err.svg', 'wb') as f:
    f.write(soup)
