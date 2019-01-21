import numpy as np
from sklearn import linear_model
from sklearn import ensemble
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import tensorflow as tf
from GP_crop_v3 import *

# def dense(input_data, H, N=None, name="dense"):
#     if not N:
#         N = input_data.get_shape()[-1]
#     with tf.variable_scope(name):
#         W = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.variance_scaling_initializer())
#         b = tf.get_variable("b", [1, H])
#         return tf.matmul(input_data, W, name="matmul") + b

# def batch_normalization(input_data, axes=[0], name="batch"):
#     with tf.variable_scope(name):
#         mean, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
#         return tf.nn.batch_normalization(input_data, mean, variance, None, None, 1e-6, name="batch")
# class NeuralModel():
#     def __init__(self,W, name):

#         self.x = tf.placeholder(tf.float32, [None, W], name="x")
#         self.y = tf.placeholder(tf.float32, [None])
#         self.lr = tf.placeholder(tf.float32, [])
#         self.keep_prob = tf.placeholder(tf.float32, [])

#         fc1 = dense(self.x, 256, name="fc1")
#         fc1_r = tf.nn.relu(fc1)
#         fc2 = dense(fc1_r, 256, name="fc2")
#         fc2_r = tf.nn.relu(fc2)
#         fc3 = dense(fc2_r, 256, name="fc3")
#         fc3_r = tf.nn.relu(fc3)
#         self.pred = tf.squeeze(dense(fc3_r, 1, name="pred"))

#         # l2
#         self.loss_err = tf.nn.l2_loss(self.pred - self.y)

#         self.loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
#         self.loss = self.loss_err+self.loss_reg
#         # self.loss = self.loss_err

#         self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

# '''LOAD 2009-2013'''
# path_data = '/atlas/u/jiaxuan/data/MODIS_data_county_processed_compressed/'
# # load mean data
# filename = 'histogram_all_soilweather_mean' + '.npz'
# content = np.load(path_data + filename)
# image_all = content['output_image']
# yield_all = content['output_yield']
# year_all = content['output_year']
# # locations_weather = content['output_locations']
# soil_all = content['output_soil']
# weather_all = content['output_weather']

# weather_mean=np.mean(weather_all,0)
# weather_std=np.std(weather_all,0)
# weather_all=(weather_all - weather_mean)/weather_std
# print weather_all.min(),weather_all.max()

# soil_mean=np.mean(soil_all,0)
# soil_std=np.std(soil_all,0)
# soil_all=(soil_all - soil_mean)/soil_std
# print soil_all.min(),soil_all.max()


'''LOAD 2009-2015, no weather'''
path_data = '/atlas/u/jiaxuan/data/google_drive/img_output/'
# load mean data
filename = 'histogram_all_mean.npz'
content = np.load(path_data + filename)
image_all = content['output_image']
yield_all = content['output_yield']
year_all = content['output_year']
locations_all = content['output_locations']
index_all = content['output_index']
# keep major counties
list_keep=[]
for i in range(image_all.shape[0]):
        if (index_all[i,0]==5)or(index_all[i,0]==17)or(index_all[i,0]==18)or(index_all[i,0]==19)or(index_all[i,0]==20)or(index_all[i,0]==27)or(index_all[i,0]==29)or(index_all[i,0]==31)or(index_all[i,0]==38)or(index_all[i,0]==39)or(index_all[i,0]==46):
                list_keep.append(i)
image_all=image_all[list_keep]
yield_all=yield_all[list_keep]
year_all = year_all[list_keep]
locations_all = locations_all[list_keep]
index_all = index_all[list_keep]

# calc NDVI
image_NDVI = np.zeros([image_all.shape[0],32])
for i in range(32):
        image_NDVI[:,i] = (image_all[:,1+9*i]-image_all[:,9*i])/(image_all[:,1+9*i]+image_all[:,9*i])


RMSE_ridge  = np.zeros([7,6])
ME_ridge  = np.zeros([7,6])
RMSE_tree  = np.zeros([7,6])
ME_tree  = np.zeros([7,6])
RMSE_DNN  = np.zeros([7,6])
ME_DNN  = np.zeros([7,6])
RMSE_ridge_raw  = np.zeros([7,6])
ME_ridge_raw  = np.zeros([7,6])
RMSE_tree_raw  = np.zeros([7,6])
ME_tree_raw  = np.zeros([7,6])
RMSE_DNN_raw  = np.zeros([7,6])
ME_DNN_raw  = np.zeros([7,6])

RMSE_ridge_NDVI = np.zeros([7,6])
ME_ridge_NDVI = np.zeros([7,6])
RMSE_ridge_weather = np.zeros([7,6])
ME_ridge_weather = np.zeros([7,6])
RMSE_ridge_NDVI_weather = np.zeros([7,6])
ME_ridge_NDVI_weather = np.zeros([7,6])
for i,predict_year in enumerate(range(2009,2014)):
	validate = np.nonzero(year_all == predict_year)[0]
	train = np.nonzero(year_all < predict_year)[0]
	for j,day in enumerate(range(10,31,4)):
		print day
		# Ridge regression, NDVI
		feature = image_NDVI[:,0:day]

		lr = linear_model.Ridge(10)
		lr.fit(feature[train],yield_all[train])
		Y_pred_reg = lr.predict(feature[validate])
		rmse = np.sqrt(np.mean((Y_pred_reg-yield_all[validate,])**2))
		me = np.mean(Y_pred_reg-yield_all[validate,])/np.mean(yield_all[validate,])*100
		RMSE_ridge_NDVI[i,j] = rmse
		ME_ridge_NDVI[i,j] = me
        print 'Ridge',rmse,me

                # # Ridge regression weather
                # feature = np.concatenate((soil_all,weather_all[:,0:((day*8+49)/30)*5]),axis=1)

                # lr = linear_model.Ridge(10)
                # lr.fit(feature[train],yield_all[train])
                # Y_pred_reg = lr.predict(feature[validate])
                # rmse = np.sqrt(np.mean((Y_pred_reg-yield_all[validate,])**2))
                # me = np.mean(Y_pred_reg-yield_all[validate,])/np.mean(yield_all[validate,])*100
                # RMSE_ridge_weather[i,j] = rmse
                # ME_ridge_weather[i,j] = me

                # # Ridge regression NDVI+weather
                # feature = np.concatenate((image_NDVI[:,0:day],soil_all,weather_all[:,0:((day*8+49)/30)*5]),axis=1)

                # lr = linear_model.Ridge(10)
                # lr.fit(feature[train],yield_all[train])
                # Y_pred_reg = lr.predict(feature[validate])
                # rmse = np.sqrt(np.mean((Y_pred_reg-yield_all[validate,])**2))
                # me = np.mean(Y_pred_reg-yield_all[validate,])/np.mean(yield_all[validate,])*100
                # RMSE_ridge_NDVI_weather[i,j] = rmse
                # ME_ridge_NDVI_weather[i,j] = me

		# Boosting Regression Tree, NDVI
		feature = image_NDVI[:,0:day]
		# feature = image_all[:,0:day*9]
		# feature = np.concatenate((image_NDVI[:,0:day],soil_all,weather_all[:,0:((day*8+49)/30)*5]),axis=1)
		params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
		lr = ensemble.GradientBoostingRegressor(**params)
		lr.fit(feature[train],yield_all[train])
		Y_pred_reg = lr.predict(feature[validate])
		rmse = np.sqrt(np.mean((Y_pred_reg-yield_all[validate,])**2))
		me = np.mean(Y_pred_reg-yield_all[validate,])/np.mean(yield_all[validate,])*100
		RMSE_tree[i,j] = rmse
		ME_tree[i,j] = me
        print 'Tree',rmse,me

		# # DNN
		# # feature = np.concatenate((image_NDVI[:,0:day],soil_all,weather_all[:,0:((day*8+49)/30)*5]),axis=1)
		# feature = image_NDVI[:,0:day]
		# g = tf.Graph()
		# with g.as_default():
		# 	model= NeuralModel(feature.shape[1],'net')
		# 	sess = tf.Session()
		# 	sess.run(tf.initialize_all_variables())
		# 	lr = 1e-3
		# 	for k in range(500):
		# 		# if k==500:
		# 		# 	lr/=10
		# 		_, train_loss, train_loss_err = sess.run([model.train_op, model.loss, model.loss_err], feed_dict={
  #                                   model.x:feature[train],
  #                                   model.y:yield_all[train],
  #                                   model.lr:lr,
  #                                   model.keep_prob: 1
  #                                   })
		# 		val_loss, val_loss_err,pred = sess.run([model.loss, model.loss_err,model.pred], feed_dict={
  #                                   model.x:feature[validate],
  #                                   model.y:yield_all[validate],
  #                                   model.keep_prob: 1
  #                                   })
		# 		if k%100 == 0:
		# 			print k
		# 			print 'train',train_loss,train_loss_err
		# 			print 'val', val_loss,val_loss_err
		# 			rmse = np.sqrt(np.mean((pred - yield_all[validate])**2))
		# 			me = np.mean(pred - yield_all[validate])/np.mean(yield_all[validate,])*100
		# 			print 'RMSE',rmse,'ME',me
		# RMSE_DNN[i,j] = rmse
		# ME_DNN[i,j] = me

          #       # Ridge regression Raw image
          #       feature = image_all[:,0:day*9]

          #       lr = linear_model.Ridge(10)
          #       lr.fit(feature[train],yield_all[train])
          #       Y_pred_reg = lr.predict(feature[validate])
          #       rmse = np.sqrt(np.mean((Y_pred_reg-yield_all[validate,])**2))
          #       me = np.mean(Y_pred_reg-yield_all[validate,])/np.mean(yield_all[validate,])*100
          #       RMSE_ridge_raw[i,j] = rmse
          #       ME_ridge_raw[i,j] = me

          #       # Boosting Regression Tree, NDVI
          #       feature = image_all[:,0:day*9]
          #       # feature = image_all[:,0:day*9]
          #       # feature = np.concatenate((image_NDVI[:,0:day],soil_all,weather_all[:,0:((day*8+49)/30)*5]),axis=1)
          #       params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          # 'learning_rate': 0.01, 'loss': 'ls'}
          #       lr = ensemble.GradientBoostingRegressor(**params)
          #       lr.fit(feature[train],yield_all[train])
          #       Y_pred_reg = lr.predict(feature[validate])
          #       rmse = np.sqrt(np.mean((Y_pred_reg-yield_all[validate,])**2))
          #       me = np.mean(Y_pred_reg-yield_all[validate,])/np.mean(yield_all[validate,])*100
          #       RMSE_tree_raw[i,j] = rmse
          #       ME_tree_raw[i,j] = me

          #       # DNN
          #       # feature = np.concatenate((image_NDVI[:,0:day],soil_all,weather_all[:,0:((day*8+49)/30)*5]),axis=1)
          #       feature = image_all[:,0:day*9]
          #       g = tf.Graph()
          #       with g.as_default():
          #               model= NeuralModel(feature.shape[1],'net')
          #               sess = tf.Session()
          #               sess.run(tf.initialize_all_variables())
          #               lr = 1e-3
          #               for k in range(4000):
          #                       if k==500:
          #                             lr/=10
          #                       _, train_loss, train_loss_err = sess.run([model.train_op, model.loss, model.loss_err], feed_dict={
          #                           model.x:feature[train],
          #                           model.y:yield_all[train],
          #                           model.lr:lr,
          #                           model.keep_prob: 1
          #                           })
          #                       val_loss, val_loss_err,pred = sess.run([model.loss, model.loss_err,model.pred], feed_dict={
          #                           model.x:feature[validate],
          #                           model.y:yield_all[validate],
          #                           model.keep_prob: 1
          #                           })
          #                       if k%100 == 0:
          #                               print k
          #                               print 'train',train_loss,train_loss_err
          #                               print 'val', val_loss,val_loss_err
          #                               rmse = np.sqrt(np.mean((pred - yield_all[validate])**2))
          #                               me = np.mean(pred - yield_all[validate])/np.mean(yield_all[validate,])*100
          #                               print 'RMSE',rmse,'ME',me
          #       RMSE_DNN_raw[i,j] = rmse
          #       ME_DNN_raw[i,j] = me


# ## CNN
# RMSE_CNN = np.zeros([7,6])
# ME_CNN = np.zeros([7,6])
# RMSE_CNN_GP = np.zeros([7,6])
# ME_CNN_GP = np.zeros([7,6])

# save_path = '/atlas/u/jiaxuan/data/train_results/final/monthly/'
# for loop in range(0,2):
# 	for i,predict_year in enumerate(range(2009,2016)):
#                 for j,time in enumerate(range(10,31,4)):
#                         if predict_year==2012:
#                                 path_current = save_path+str(loop+2)+str(time)+str(predict_year)+'result_prediction.npz'
#                         else:
# 			     path_current = save_path+str(loop)+str(time)+str(predict_year)+'result_prediction.npz'
# 			data = np.load(path_current)
                        
# 			year = data['year_out']
# 			real = data['real_out']
# 			pred = data['pred_out']
#                         locations = data['locations_out']
#                         index_out=data['index_out']
#                         feature_image = data['feature_out']

# 			validate = np.nonzero(year == predict_year)[0]
# 			train = np.nonzero(year < predict_year)[0]
# 			#CNN
# 			rmse=np.sqrt(np.mean((real[validate]-pred[validate])**2))
# 			me = np.mean(pred[validate]-real[validate])/np.mean(real[validate])*100
# 			RMSE_CNN[i,j]+=rmse
# 			ME_CNN[i,j]+=me

# 			#CNN+GP
# 			rmse,me = GaussianProcess(predict_year,path_current)
# 			RMSE_CNN_GP[i,j]+=rmse
# 			ME_CNN_GP[i,j]+=me/np.mean(real[validate])*100
# RMSE_CNN /= 2
# ME_CNN /= 2
# RMSE_CNN_GP /= 2
# ME_CNN_GP /= 2


# ## CNN+weather
# RMSE_CNN_noweather = np.zeros([5,6])
# ME_CNN_noweather = np.zeros([5,6])
# RMSE_CNN_weather = np.zeros([5,6])
# ME_CNN_weather = np.zeros([5,6])

# save_path = '/atlas/u/jiaxuan/data/train_results/final/monthly/'
# for loop in range(0,2):
#         #open any set
#         path_current = save_path+str(0)+str(10)+str(2009)+'result_prediction.npz'
#         data = np.load(path_current)
#         year = data['year_out']
#         real = data['real_out']
#         pred = data['pred_out']
#         locations = data['locations_out']
#         index_out=data['index_out']
#         feature_image = data['feature_out']
#         #find related soil and weather
        
#         weather_dir = '/atlas/u/jiaxuan/data/MODIS_data_county_processed_compressed/'
#         soil = np.genfromtxt(weather_dir+'soil_output.csv', delimiter=',')
#         weather = np.genfromtxt(weather_dir+'daymet_mean.csv', delimiter=',')
#         feature_soil = np.zeros([locations.shape[0],3])
#         feature_weather = np.zeros([locations.shape[0],60])

#         list_delete = []
#         for k in range(locations.shape[0]):
#                 if k%1000==0:
#                         print k

#                 key = np.array([int(year[k]),int(index_out[k,0]),int(index_out[k,1])])
#                 index = np.where(np.all(soil[:,0:3].astype('int') == key, axis=1))
#                 soil_temp = soil[index,3:6]
#                 if soil_temp.shape==(1,0,3):
#                         list_delete.append(k)
#                         continue

#                 weather_temp = np.zeros([5*12])
#                 for idx,month in enumerate(range(1,13)):
#                         key = np.array([int(year[k]),int(month),int(index_out[k,0]),int(index_out[k,1])])
#                         index = np.where(np.all(weather[:,0:4].astype(int) == key, axis=1))
#                         weather_temp[idx*5:(idx+1)*5] = weather[index,4:9].flatten()

#                 feature_soil[k]=soil_temp
#                 feature_weather[k]=weather_temp
#         feature_soil = np.delete(feature_soil, list_delete, 0)
#         feature_weather = np.delete(feature_weather, list_delete, 0)
#         for i,predict_year in enumerate(range(2009,2014)):
#                 for j,time in enumerate(range(10,31,4)):
#                         if predict_year==2012:
#                                 path_current = save_path+str(loop+2)+str(time)+str(predict_year)+'result_prediction.npz'
#                         else:
#                              path_current = save_path+str(loop)+str(time)+str(predict_year)+'result_prediction.npz'
#                         data = np.load(path_current)
                        
#                         year = data['year_out']
#                         real = data['real_out']
#                         pred = data['pred_out']
#                         locations = data['locations_out']
#                         index_out=data['index_out']
#                         feature_image = data['feature_out']

#                         year=np.delete(year,list_delete,0)
#                         real=np.delete(real,list_delete,0)
#                         pred = np.delete(pred,list_delete, 0)
#                         locations = np.delete(locations, list_delete, 0)
#                         index_out = np.delete(index_out, list_delete, 0)
#                         feature_image = np.delete(feature_image, list_delete, 0)
#                         print year.shape,real.shape,pred.shape,locations.shape
#                         print index_out.shape,feature_image.shape,feature_soil.shape,feature_weather.shape

#                         validate = np.nonzero(year == predict_year)[0]
#                         train = np.nonzero(year < predict_year)[0]


#                         # img
#                         feature = feature_image
#                         lr = linear_model.Ridge(alpha =2000)
#                         lr.fit(feature[train],real[train])
#                         pred_ridge = lr.predict(feature[validate])
#                         rmse = np.sqrt(np.mean((pred_ridge-real[validate])**2))
#                         me = np.mean(pred_ridge-real[validate])/np.mean(real[validate])*100
#                         print 'img'
#                         print rmse,me
#                         RMSE_CNN_noweather[i,j]+=rmse
#                         ME_CNN_noweather[i,j]+=me
                        
#                         # img+weather
#                         feature = np.concatenate((feature_image, feature_soil, feature_weather[:,0:((day*8+49)/30)*5]),axis=1)
#                         lr = linear_model.Ridge(alpha =2000)
#                         lr.fit(feature[train],real[train])
#                         pred_ridge = lr.predict(feature[validate])

#                         rmse = np.sqrt(np.mean((pred_ridge-real[validate])**2))
#                         me = np.mean(pred_ridge-real[validate])/np.mean(real[validate])*100
#                         print 'img+weather'
#                         print rmse,me
#                         RMSE_CNN_weather[i,j]+=rmse
#                         ME_CNN_weather[i,j]+=me
# RMSE_CNN_noweather /= 2
# ME_CNN_noweather /= 2
# RMSE_CNN_weather /= 2
# ME_CNN_weather /= 2



# ## LSTM
# RMSE_LSTM = np.zeros([7,6])
# ME_LSTM = np.zeros([7,6])
# RMSE_LSTM_GP = np.zeros([7,6])
# ME_LSTM_GP = np.zeros([7,6])

# save_path = '/atlas/u/jiaxuan/data/train_results/final/lstm/'
# for loop in range(0,3):
#       for i,predict_year in enumerate(range(2009,2016)):
#               for j,time in enumerate(range(10,31,4)):
#                       path_current = save_path+str(loop)+str(time)+str(predict_year)+'result_prediction.npz'
#                       data = np.load(path_current)
#                       year_all = data['year_out']
#                       real = data['real_out']
#                       pred = data['pred_out']

#                       validate = np.nonzero(year_all == predict_year)[0]
#                       train = np.nonzero(year_all < predict_year)[0]
                        
#                       rmse=np.sqrt(np.mean((real[validate]-pred[validate])**2))
#                       me = np.mean(pred[validate]-real[validate])
#                       RMSE_LSTM[i,j]=+rmse
#                       ME_LSTM[i,j]=+me/np.mean(real[validate])*100

#                       #LSTM+GP
#                       rmse,me = GaussianProcess(predict_year,path_current)
#                       RMSE_LSTM_GP[i,j]=+rmse
#                       ME_LSTM_GP[i,j]=+me/np.mean(real[validate])*100
# RMSE_LSTM /=3
# ME_LSTM /= 3
# RMSE_LSTM_GP /= 3
# ME_LSTM_GP /= 3

# ME_USDA = np.array([0,0,0,4.19,4.52,2.65])
# ME_USDA_std = np.array([0,0,0,3.16,4.06,1.93])


# # ##################### archive ################
# # # np.savez('Compare_result_final.npz',RMSE_all=RMSE_all,RMSE_CNN=RMSE_CNN,RMSE_DNN=RMSE_DNN,RMSE_raw=RMSE_raw,RMSE_ridge=RMSE_ridge,RMSE_tree=RMSE_tree,RMSE_GP=RMSE_GP,RMSE_LSTM=RMSE_LSTM,RMSE_LSTM_GP=RMSE_LSTM_GP,
# # # 	ME_all=ME_all,ME_CNN=ME_CNN,ME_DNN=ME_DNN,ME_raw=ME_raw,ME_ridge=ME_ridge,ME_tree=ME_tree,ME_USDA=ME_USDA,ME_GP=ME_GP,ME_LSTM=ME_LSTM,ME_LSTM_GP=ME_LSTM_GP)

# # # data = np.load('Compare_result_final.npz')
# # # RMSE_all=data['RMSE_all']
# # # RMSE_CNN=data['RMSE_CNN']
# # # RMSE_DNN=data['RMSE_DNN']
# # # RMSE_raw=data['RMSE_raw']
# # # RMSE_ridge=data['RMSE_ridge']
# # # RMSE_tree=data['RMSE_tree']
# # # RMSE_GP=data['RMSE_GP']
# # # ME_all=data['ME_all']
# # # ME_CNN=data['ME_CNN']
# # # ME_DNN=data['ME_DNN']
# # # ME_raw=data['ME_raw']
# # # ME_ridge=data['ME_ridge']
# # # ME_tree=data['ME_tree']
# # # ME_USDA=data['ME_USDA']
# # # ME_GP=data['ME_GP']
# # ##################### archive ################

# # np.savez('Compare_result_final.npz',
# #         RMSE_ridge=RMSE_ridge,RMSE_ridge_raw=RMSE_ridge_raw,RMSE_tree=RMSE_tree,RMSE_tree_raw=RMSE_tree_raw,
# #         RMSE_DNN=RMSE_DNN,RMSE_DNN_raw=RMSE_DNN_raw,RMSE_CNN=RMSE_CNN,RMSE_CNN_GP=RMSE_CNN_GP,
# #         RMSE_CNN_noweather=RMSE_CNN_noweather,RMSE_CNN_weather=RMSE_CNN_weather,
# #         RMSE_LSTM=RMSE_LSTM,RMSE_LSTM_GP=RMSE_LSTM_GP,
# #         ME_ridge=ME_ridge,ME_ridge_raw=ME_ridge_raw,ME_tree=ME_tree,ME_tree_raw=ME_tree_raw,
# #         ME_DNN=ME_DNN,ME_DNN_raw=ME_DNN_raw,ME_CNN=ME_CNN,ME_CNN_GP=ME_CNN_GP,
# #         ME_CNN_noweather=ME_CNN_noweather,ME_CNN_weather=ME_CNN_weather,
# #         ME_LSTM=ME_LSTM,ME_LSTM_GP=ME_LSTM_GP)

# # print 'SAVED!!'

# # np.savez('Compare_result_ridge.npz',
# #         RMSE_ridge_NDVI=RMSE_ridge_NDVI,RMSE_ridge_weather=RMSE_ridge_weather,RMSE_ridge_NDVI_weather=RMSE_ridge_NDVI_weather,
# #         ME_ridge_NDVI=ME_ridge_NDVI,ME_ridge_weather=ME_ridge_weather,ME_ridge_NDVI_weather=ME_ridge_NDVI_weather)
# # print 'SAVED!!'


# data = np.load('Compare_result_final.npz')
# data_ridge = np.load('Compare_result_ridge.npz')
# RMSE_ridge=data['RMSE_ridge']
# RMSE_ridge_raw=data['RMSE_ridge_raw']
# RMSE_tree=data['RMSE_tree']
# RMSE_tree_raw=data['RMSE_tree_raw']
# RMSE_DNN=data['RMSE_DNN']
# RMSE_CNN=data['RMSE_CNN']
# RMSE_CNN_GP=data['RMSE_CNN_GP']
# RMSE_CNN_noweather=data['RMSE_CNN_noweather']
# RMSE_CNN_weather=data['RMSE_CNN_weather']
# RMSE_LSTM=data['RMSE_LSTM']*3
# RMSE_LSTM_GP=data['RMSE_LSTM_GP']*3

# ME_ridge=data['ME_ridge']
# ME_ridge_raw=data['ME_ridge_raw']
# ME_tree=data['ME_tree']
# ME_tree_raw=data['ME_tree_raw']
# ME_DNN=data['ME_DNN']
# ME_CNN=data['ME_CNN']
# ME_CNN_GP=data['ME_CNN_GP']
# ME_CNN_noweather=data['ME_CNN_noweather']
# ME_CNN_weather=data['ME_CNN_weather']
# ME_LSTM=data['ME_LSTM']*3
# ME_LSTM_GP=data['ME_LSTM_GP']*3

# RMSE_ridge_NDVI = data_ridge['RMSE_ridge_NDVI'][0:5,:]
# RMSE_ridge_weather = data_ridge['RMSE_ridge_weather'][0:5,:]
# RMSE_ridge_NDVI_weather = data_ridge['RMSE_ridge_NDVI_weather'][0:5,:]

# print ME_CNN.shape
# print ME_CNN



# # result = np.concatenate((RMSE_ridge[:,5:6], RMSE_tree[:,5:6], RMSE_DNN[:,5:6],RMSE_LSTM[:,5:6],RMSE_LSTM_GP[:,5:6],RMSE_CNN[:,5:6],RMSE_CNN_GP[:,5:6]),axis=1)
# # result = result[2:,:]
# # result_mean = np.mean(result,axis=0,keepdims=True)
# # result = np.concatenate((result, result_mean),axis=0)
# # print np.round(result,2)
# # np.save('paper_result.npy', result)
# # print 'saved'



# # plt.plot(range(2011,2016),RMSE_ridge[2:,5])
# # plt.plot(range(2011,2016),RMSE_tree[2:,5])
# # plt.plot(range(2011,2016),RMSE_DNN[2:,5])
# # plt.plot(range(2011,2016),RMSE_LSTM[2:,5])
# # plt.plot(range(2011,2016),RMSE_LSTM_GP[2:,5])
# # plt.plot(range(2011,2016),RMSE_CNN[2:,5])
# # plt.plot(range(2011,2016),RMSE_CNN_GP[2:,5])

# # plt.legend(['RMSE_ridge','RMSE_tree','RMSE_DNN','RMSE_LSTM','RMSE_LSTM_GP','RMSE_CNN','RMSE_CNN_GP'])
# # plt.show()

# # # bar plot
# # n_groups = 6
# # fig, ax = plt.subplots()
# # index = np.arange(n_groups)
# # bar_width = 0.12
# # opacity = 0.6
# # error_config = {'ecolor': '0.3'}

# # rects1 = plt.bar(index, np.mean(RMSE_ridge[2:],axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='b',
# #                  yerr=0,
# #                  error_kw=error_config,
# #                  label='Ridge')
# # rects2 = plt.bar(index + bar_width, np.mean(RMSE_tree[2:],axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='g',
# #                  yerr=0,
# #                  error_kw=error_config,
# #                  label='Tree')
# # rects3 = plt.bar(index + bar_width*2, np.mean(RMSE_DNN[2:],axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='r',
# #                  yerr=0,
# #                  error_kw=error_config,
# #                  label='DNN')
# # rects4 = plt.bar(index + bar_width*3, np.mean(RMSE_LSTM[2:],axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='c',
# #                  yerr=0,
# #                  error_kw=error_config,
# #                  label='LSTM')
# # rects5 = plt.bar(index + bar_width*4, np.mean(RMSE_LSTM_GP[2:],axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='m',
# #                  yerr=0,
# #                  error_kw=error_config,
# #                  label='LSTM+GP')
# # rects6 = plt.bar(index + bar_width*5, np.mean(RMSE_CNN[2:],axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='y',
# #                  yerr=0,
# #                  error_kw=error_config,
# #                  label='CNN')
# # rects7 = plt.bar(index + bar_width*6, np.mean(RMSE_CNN_GP[2:],axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='k',
# #                  yerr=0,
# #                  error_kw=error_config,
# #                  label='CNN+GP')

# # plt.xlabel('Predicting month',fontsize=16)
# # plt.ylabel('Root Mean Square Error',fontsize=16)
# # # plt.title('Root Mean Square Error')
# # plt.xticks(index + bar_width*3.5, ('May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct'))
# # plt.legend(fontsize=14)

# # axes = plt.gca()
# # axes.set_ylim([0,15])

# # plt.tight_layout()
# # plt.show()


# # # bar plot
# # n_groups = 6
# # fig, ax = plt.subplots()
# # index = np.arange(n_groups)
# # bar_width = 0.16
# # opacity = 0.6
# # error_config = {'ecolor': '0.3'}

# # rects1 = plt.bar(index, np.mean(RMSE_ridge_weather,axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='b',
# #                  yerr=0,
# #                  error_kw=error_config,
# #                  label='Weather')
# # rects2 = plt.bar(index + bar_width, np.mean(RMSE_ridge_NDVI,axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='g',
# #                  yerr=0,
# #                  error_kw=error_config,
# #                  label='NDVI')
# # rects3 = plt.bar(index + bar_width*2, np.mean(RMSE_ridge_NDVI_weather,axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='r',
# #                  yerr=0,
# #                  error_kw=error_config,
# #                  label='NDVI+Weather')
# # rects4 = plt.bar(index + bar_width*3, np.mean(RMSE_CNN_noweather,axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='c',
# #                  yerr=0,
# #                  error_kw=error_config,
# #                  label='Image')
# # rects5 = plt.bar(index + bar_width*4, np.mean(RMSE_CNN_weather,axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='m',
# #                  yerr=0,
# #                  error_kw=error_config,
# #                  label='Image+Weather')

# # plt.xlabel('Predicting month',fontsize=16)
# # plt.ylabel('Root Mean Square Error',fontsize=16)
# # # plt.title('Root Mean Square Error')
# # plt.xticks(index + bar_width*2.5, ('May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct'))
# # plt.legend(fontsize=14)

# # axes = plt.gca()
# # axes.set_ylim([0,15])

# # plt.tight_layout()
# # plt.show()


# # # bar plot
# # n_groups = 6
# # fig, ax = plt.subplots()
# # index = np.arange(n_groups)
# # bar_width = 0.1
# # opacity = 0.5
# # error_config = {'ecolor': '0.3'}

# # rects1 = plt.bar(index, np.mean(RMSE_CNN,axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='b',
# #                  yerr=np.std(RMSE_CNN,axis=0),
# #                  error_kw=error_config,
# #                  label='RMSE_CNN')
# # rects2 = plt.bar(index + bar_width, np.mean(RMSE_CNN_GP,axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='g',
# #                  yerr=np.std(RMSE_CNN_GP,axis=0),
# #                  error_kw=error_config,
# #                  label='RMSE_CNN_GP')
# # rects2 = plt.bar(index + bar_width*2, np.mean(RMSE_CNN_noweather,axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='r',
# #                  yerr=np.std(RMSE_CNN_noweather,axis=0),
# #                  error_kw=error_config,
# #                  label='RMSE_CNN_noweather')
# # rects2 = plt.bar(index + bar_width*3, np.mean(RMSE_CNN_weather,axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='c',
# #                  yerr=np.std(RMSE_CNN_weather,axis=0),
# #                  error_kw=error_config,
# #                  label='RMSE_CNN_weather')
# # rects4 = plt.bar(index + bar_width*4, np.mean(RMSE_LSTM,axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='m',
# #                  yerr=np.std(RMSE_LSTM,axis=0),
# #                  error_kw=error_config,
# #                  label='RMSE_LSTM')
# # rects3 = plt.bar(index + bar_width*5, np.mean(RMSE_LSTM_GP,axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='y',
# #                  yerr=np.std(RMSE_LSTM_GP,axis=0),
# #                  error_kw=error_config,
# #                  label='RMSE_LSTM_GP')
# # # rects3 = plt.bar(index + bar_width*6, np.mean(RMSE_GP,axis=0), bar_width,
# # #                  alpha=opacity,
# # #                  color='k',
# # #                  yerr=np.std(RMSE_GP,axis=0),
# # #                  error_kw=error_config,
# # #                  label='GP')


# # plt.xlabel('Month')
# # plt.ylabel('RMSE')
# # plt.title('Root Mean Square Error')
# # plt.xticks(index + bar_width*1.5, ('May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct'))
# # plt.legend()

# # plt.tight_layout()
# # plt.show()

# # bar plot
# n_groups = 3
# fig, ax = plt.subplots()
# index = np.arange(n_groups)
# bar_width = 0.10
# opacity = 0.5
# error_config = {'ecolor': '0.3'}

# rects1 = plt.bar(index, np.mean(np.absolute(ME_ridge[0:6,3:6]),axis=0), bar_width,
#                  alpha=opacity,
#                  color='b',
#                  yerr=np.std(np.absolute(ME_ridge[0:6,3:6]),axis=0),
#                  error_kw=error_config,
#                  label='ridge')
# rects2 = plt.bar(index + bar_width, np.mean(np.absolute(ME_tree[0:6,3:6]),axis=0), bar_width,
#                  alpha=opacity,
#                  color='g',
#                  yerr=np.std(np.absolute(ME_tree[0:6,3:6]),axis=0),
#                  error_kw=error_config,
#                  label='tree')
# rects3 = plt.bar(index + bar_width*2, np.mean(np.absolute(ME_DNN[0:6,3:6]),axis=0), bar_width,
#                  alpha=opacity,
#                  color='r',
#                  yerr=np.std(np.absolute(ME_DNN[0:6,3:6]),axis=0),
#                  error_kw=error_config,
#                  label='DNN')
# rects4 = plt.bar(index + bar_width*3, np.mean(np.absolute(ME_CNN_GP[0:6,3:6]),axis=0), bar_width,
#                  alpha=opacity,
#                  color='c',
#                  yerr=np.std(np.absolute(ME_CNN_GP[0:6,3:6]),axis=0),
#                  error_kw=error_config,
#                  label='CNN+GP')
# rects5 = plt.bar(index + bar_width*4, np.mean(np.absolute(ME_CNN[0:6,3:6]),axis=0), bar_width,
#                  alpha=opacity,
#                  color='m',
#                  yerr=np.std(np.absolute(ME_CNN[0:6,3:6]),axis=0),
#                  error_kw=error_config,
#                  label='CNN')
# rects6 = plt.bar(index + bar_width*5, ME_USDA[3:6], bar_width,
#                  alpha=opacity,
#                  color='k',
#                  yerr=ME_USDA_std[3:6],
#                  error_kw=error_config,
#                  label='USDA')
# # rects5 = plt.bar(index + bar_width*6, np.mean(np.absolute(ME_GP),axis=0), bar_width,
# #                  alpha=opacity,
# #                  color='b',
# #                  yerr=np.std(np.absolute(ME_GP),axis=0),
# #                  error_kw=error_config,
# #                  label='GP')

# plt.xlabel('Month')
# plt.ylabel('MAPE')
# plt.title('Mean Absolute Percent Error (2009-2014)')
# plt.xticks(index + bar_width*3, ('Aug', 'Sept', 'Oct'))
# plt.legend()

# plt.tight_layout()
# plt.show()






# # # print 'NDVI+weather'
# # # RMSE  = np.zeros([24])
# # # for day in range(32,8,-1):
# # # 	# sklearn linear regression

# # # 	feature = np.concatenate((image_NDVI[:,0:day],weather_all[:,0:((day*8+49)/30)*5]),axis=1)

# # # 	lr = linear_model.Ridge(10)
# # # 	lr.fit(feature[train],yield_all[train])
# # # 	Y_pred_reg = lr.predict(feature[validate])
# # # 	rmse = np.sqrt(np.mean((Y_pred_reg-yield_all[validate])**2))
# # # 	# print "The RMSE of ridge regression is", rmse
# # # 	RMSE[day-9] = rmse
# # # plt.plot(np.arange(9,33)*8+49,RMSE)

# # # print 'NDVI+soil'
# # # RMSE  = np.zeros([24])
# # # for day in range(32,8,-1):
# # # 	# sklearn linear regression

# # # 	feature = np.concatenate((image_NDVI[:,0:day],soil_all),axis=1)

# # # 	lr = linear_model.Ridge(10)
# # # 	lr.fit(feature[train],yield_all[train])
# # # 	Y_pred_reg = lr.predict(feature[validate])
# # # 	rmse = np.sqrt(np.mean((Y_pred_reg-yield_all[validate])**2))
# # # 	# print "The RMSE of ridge regression is", rmse
# # # 	RMSE[day-9] = rmse
# # # plt.plot(np.arange(9,33)*8+49,RMSE)

# # # print 'NDVI+soil+weather'
# # # RMSE  = np.zeros([24])
# # # for day in range(32,8,-1):
# # # 	# sklearn linear regression

# # # 	feature = np.concatenate((image_NDVI[:,0:day],soil_all,weather_all[:,0:((day*8+49)/30)*5]),axis=1)

# # # 	lr = linear_model.Ridge(10)
# # # 	lr.fit(feature[train],yield_all[train])
# # # 	Y_pred_reg = lr.predict(feature[validate])
# # # 	rmse = np.sqrt(np.mean((Y_pred_reg-yield_all[validate])**2))
# # # 	# print "The RMSE of ridge regression is", rmse
# # # 	RMSE[day-9] = rmse
# # # 	if day == 32:
# # # 		print 'NDVI+weather',rmse,np.mean((Y_pred_reg-yield_all[validate]))
# # # plt.plot(np.arange(9,33)*8+49,RMSE)


# # # print 'NDVI+soil+fullweather'
# # # RMSE  = np.zeros([24])
# # # for day in range(32,8,-1):
# # # 	# sklearn linear regression

# # # 	feature = np.concatenate((image_NDVI[:,0:day],soil_all,weather_all),axis=1)

# # # 	lr = linear_model.Ridge(10)
# # # 	lr.fit(feature[train],yield_all[train])
# # # 	Y_pred_reg = lr.predict(feature[validate])
# # # 	rmse = np.sqrt(np.mean((Y_pred_reg-yield_all[validate])**2))
# # # 	# print "The RMSE of ridge regression is", rmse
# # # 	RMSE[day-9] = rmse
# # # # plt.plot(range(24),RMSE)

# # # plt.legend(['NDVI','NDVI+weather','NDVI+soil','NDVI+soil+weather'])
# # # plt.xlabel('Day of year')
# # # plt.ylabel('RMSE')
# # # plt.show()

