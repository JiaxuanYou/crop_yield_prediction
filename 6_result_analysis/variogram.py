import scipy.io as io
import numpy as np

save_path = '/atlas/u/jiaxuan/data/train_results/final/monthly/'
path_current = save_path+str(0)+str(30)+str(2013)+'result_prediction.npz'
data = np.load(path_current)
year = data['year_out']
real = data['real_out']
pred = data['pred_out']
locations = data['locations_out']

err = pred-real
print(err.shape,year.shape,locations.shape)

result = np.concatenate((year[:,np.newaxis], locations, err[:,np.newaxis]),axis=1)

io.savemat('variogram_data.mat', {'result':result})
print('saved')