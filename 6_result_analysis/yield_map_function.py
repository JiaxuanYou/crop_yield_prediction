import numpy as np
import csv
from bs4 import BeautifulSoup
# from GP_crop_v3 import *
from sklearn import linear_model
from sklearn import ensemble

def yield_map(path_load,path_save,predict_year,flag):
    # Read CNN_err prediction
    CNN = {}
    # save_path = '/atlas/u/jiaxuan/data/train_results/final/monthly/'
    # path_current = save_path+str(0)+str(30)+str(2014)+'result_prediction.npz'
    data = np.load(path_load)

    year = data['year_out']
    real = data['real_out']
    pred = data['pred_out']
    index=data['index_out']

    validate = np.nonzero(year == predict_year)[0]
    year = year[validate]
    real = real[validate]
    pred = pred[validate]
    index = index[validate]
    # err_CNN = pred-real
    if flag=='real':
        err_CNN = real
    elif flag=='pred':
        err_CNN = pred

    print 'CNN',err_CNN.min(),err_CNN.max()
    print 'RMSE',np.sqrt(np.mean((pred-real)**2))

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

    '''CNN'''
    # Load the SVG map
    svg = open('counties.svg', 'r').read()
    # Load into Beautiful Soup
    soup = BeautifulSoup(svg, selfClosingTags=['defs','sodipodi:namedview'])
    # Find counties
    paths = soup.findAll('path')
    # Map colors
    # # plot error: 8 classes
    # colors = ["#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac"]
    # plot yield: 11 classes
    colors = ['#a50026','#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4','#313695']

    # County style
    path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'
    # Color the counties based on unemployment rate
    for p in paths:
        if p['id'] not in ["State_Lines", "separator"]:
            try:
                rate = CNN[p['id']]
            except:
                continue

            # # plot error
            # if rate > 15:
            #     color_class = 7
            # elif rate > 10:
            #     color_class = 6
            # elif rate > 5:
            #     color_class = 5
            # elif rate > 0:
            #     color_class = 4
            # elif rate > -5:
            #     color_class = 3
            # elif rate > -10:
            #     color_class = 2
            # elif rate > -15:
            #     color_class = 1
            # else:
            #     color_class = 0

            # # plot soybean yield
            # if rate > 60:
            #     color_class = 0
            # elif rate > 55:
            #     color_class = 1
            # elif rate > 50:
            #     color_class = 2
            # elif rate > 45:
            #     color_class = 3
            # elif rate > 40:
            #     color_class = 4
            # elif rate > 35:
            #     color_class = 5
            # elif rate > 30:
            #     color_class = 6
            # elif rate > 25:
            #     color_class = 7
            # elif rate > 20:
            #     color_class = 8
            # elif rate > 15:
            #     color_class = 9
            # else:
            #     color_class = 10

            # plot corn yield
            if rate > 200:
                color_class = 0
            elif rate > 180:
                color_class = 1
            elif rate > 160:
                color_class = 2
            elif rate > 140:
                color_class = 3
            elif rate > 120:
                color_class = 4
            elif rate > 100:
                color_class = 5
            elif rate > 80:
                color_class = 6
            elif rate > 60:
                color_class = 7
            elif rate > 40:
                color_class = 8
            elif rate > 20:
                color_class = 9
            else:
                color_class = 10

            color = colors[color_class]
            p['style'] = path_style + color

    soup=soup.prettify()
    with open(path_save, 'wb') as f:
        f.write(soup)


def yield_map_raw(real,index,path_save,predict_year):
    # Read CNN_err prediction
    CNN = {}
    err_CNN = real

    print 'CNN',err_CNN.min(),err_CNN.max()




    for i in range(real.shape[0]):
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

    '''CNN'''
    # Load the SVG map
    svg = open('counties.svg', 'r').read()
    # Load into Beautiful Soup
    soup = BeautifulSoup(svg, selfClosingTags=['defs','sodipodi:namedview'])
    # Find counties
    paths = soup.findAll('path')
    # Map colors
    # # plot error: 8 classes
    # colors = ["#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac"]
    # plot yield: 11 classes
    colors = ['#a50026','#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4','#313695']

    # County style
    path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'
    # Color the counties based on unemployment rate
    for p in paths:
        if p['id'] not in ["State_Lines", "separator"]:
            try:
                rate = CNN[p['id']]
            except:
                continue

            # # plot error
            # if rate > 15:
            #     color_class = 7
            # elif rate > 10:
            #     color_class = 6
            # elif rate > 5:
            #     color_class = 5
            # elif rate > 0:
            #     color_class = 4
            # elif rate > -5:
            #     color_class = 3
            # elif rate > -10:
            #     color_class = 2
            # elif rate > -15:
            #     color_class = 1
            # else:
            #     color_class = 0

            # plot yield
            if rate > 60:
                color_class = 0
            elif rate > 55:
                color_class = 1
            elif rate > 50:
                color_class = 2
            elif rate > 45:
                color_class = 3
            elif rate > 40:
                color_class = 4
            elif rate > 35:
                color_class = 5
            elif rate > 30:
                color_class = 6
            elif rate > 25:
                color_class = 7
            elif rate > 20:
                color_class = 8
            elif rate > 15:
                color_class = 9
            else:
                color_class = 10

            color = colors[color_class]
            p['style'] = path_style + color

    soup=soup.prettify()
    with open(path_save, 'wb') as f:
        f.write(soup)

if __name__ == "__main__":
    path = '/atlas/u/jiaxuan/data/train_results/final/new_L1_L2/'

    # # load baseline
    # '''LOAD 2009-2015, no weather'''
    # path_data = '/atlas/u/jiaxuan/data/google_drive/img_output/'
    # # load mean data
    # filename = 'histogram_all_mean.npz'
    # content = np.load(path_data + filename)
    # image_all = content['output_image']
    # yield_all = content['output_yield']
    # year_all = content['output_year']
    # locations_all = content['output_locations']
    # index_all = content['output_index']

    # # copy index
    # path_load = path+str(0)+str(10)+str(2014)+'result_prediction.npz'
    # content_ref=np.load(path_load)
    # year_ref=content_ref['year_out']
    # index_ref=content_ref['index_out']
    # ref=np.concatenate((year_ref[:,np.newaxis], index_ref),axis=1)

    # print 'before',index_all.shape[0]
    # # remove extra index
    # list_delete=[]
    # for i in range(index_all.shape[0]):
    #     key = np.array([year_all[i],index_all[i,0],index_all[i,1]])
    #     index = np.where(np.all(ref[:,0:3] == key, axis=1))
    #     if index[0].shape[0] == 0:
    #         list_delete.append(i)
    # image_all=np.delete(image_all,list_delete,0)
    # yield_all=np.delete(yield_all,list_delete,0)
    # year_all = np.delete(year_all,list_delete, 0)
    # locations_all = np.delete(locations_all, list_delete, 0)
    # index_all = np.delete(index_all, list_delete, 0)
    # print 'after',index_all.shape[0]

    # # calc NDVI
    # image_NDVI = np.zeros([image_all.shape[0],32])
    # for i in range(32):
    #     image_NDVI[:,i] = (image_all[:,1+9*i]-image_all[:,9*i])/(image_all[:,1+9*i]+image_all[:,9*i])




    for predict_year in range(2009,2014):
        # validate = np.nonzero(year_all == predict_year)[0]
        # train = np.nonzero(year_all < predict_year)[0]
        for day in range(10,31,4):
        #     # Ridge regression, NDVI
        #     feature = image_NDVI[:,0:day]

        #     lr = linear_model.Ridge(10)
        #     lr.fit(feature[train],yield_all[train])
        #     Y_pred_reg = lr.predict(feature[validate])

        #     rmse = np.sqrt(np.mean((Y_pred_reg-yield_all[validate])**2))
        #     me = np.mean(Y_pred_reg-yield_all[validate])/np.mean(yield_all[validate])*100
            # print 'Ridge',predict_year,day,rmse,me

            # # print baseline figure
            # path_save = path+'map_baseline/'+str(0)+str(predict_year)+str(day)+'baseline.svg'
            # yield_map_raw(Y_pred_reg, index_all[validate], path_save, predict_year)


            # print CNN figure
            path_load = path+str(2)+str(day)+str(predict_year)+'result_prediction.npz'
            path_save = path+'map_real/'+str(0)+str(predict_year)+str(day)+'real.svg'
            yield_map(path_load, path_save, predict_year,'real')
            print predict_year,day

            # print CNN figure
            path_load = path+str(2)+str(day)+str(predict_year)+'result_prediction.npz'
            path_save = path+'map_pred/'+str(0)+str(predict_year)+str(day)+'pred.svg'
            yield_map(path_load, path_save, predict_year,'pred')
            print predict_year,day

