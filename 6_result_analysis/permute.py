import numpy as np
import matplotlib.pyplot as plt

# result = np.load('paper_result.npy')
# result = result[1:,:]
# result_mean = np.mean(result,axis=0,keepdims=True)
# result = np.concatenate((result, result_mean),axis=0)

# print np.round(result,2)

permute_band = np.load('permute_band.npy')
permute_band_plot_temp = permute_band[0:9]-permute_band[9]
print(permute_band_plot_temp.shape)
permute_band_plot = np.zeros([permute_band_plot_temp.shape[0],permute_band_plot_temp.shape[1],permute_band_plot_temp.shape[2],3])
permute_band_plot[:,:,:,0] = (permute_band_plot_temp[:,:,:,0]+permute_band_plot_temp[:,:,:,1])/2
permute_band_plot[:,:,:,1] = (permute_band_plot_temp[:,:,:,2]+permute_band_plot_temp[:,:,:,3])/2
permute_band_plot[:,:,:,2] = (permute_band_plot_temp[:,:,:,4]+permute_band_plot_temp[:,:,:,5])/2

# plt.plot(range(10),permute_band_mean[:,0])
# plt.plot(range(10),permute_band_mean[:,1])
# plt.plot(range(10),permute_band_mean[:,2])
# plt.plot(range(10),permute_band_mean[:,3])
# plt.plot(range(10),permute_band_mean[:,4])
# plt.plot(range(10),permute_band_mean[:,5])
# plt.legend(['5','6','7','8','9','10'])
# plt.show()

# bar plot
n_groups = 9
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.22
opacity = 0.6
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, np.mean(permute_band_plot,axis=(1,2))[:,0], bar_width,
                 alpha=opacity,
                 color='b',
                 yerr=0,
                 error_kw=error_config,
                 label='May&Jun')
rects2 = plt.bar(index + bar_width, np.mean(permute_band_plot,axis=(1,2))[:,1], bar_width,
                 alpha=opacity,
                 color='g',
                 yerr=0,
                 error_kw=error_config,
                 label='Jul&Aug')
rects3 = plt.bar(index + bar_width*2, np.mean(permute_band_plot,axis=(1,2))[:,2], bar_width,
                 alpha=opacity,
                 color='r',
                 yerr=0,
                 error_kw=error_config,
                 label='Sept&Oct')
# rects4 = plt.bar(index + bar_width*3, np.mean(permute_band_plot,axis=(1,2))[:,3], bar_width,
#                  alpha=opacity,
#                  color='c',
#                  yerr=np.std(permute_band_plot,axis=(1,2))[:,3],
#                  error_kw=error_config,
#                  label='Aug')
# rects5 = plt.bar(index + bar_width*4, np.mean(permute_band_plot,axis=(1,2))[:,4], bar_width,
#                  alpha=opacity,
#                  color='m',
#                  yerr=np.std(permute_band_plot,axis=(1,2))[:,4],
#                  error_kw=error_config,
#                  label='Sept')
# rects6 = plt.bar(index + bar_width*5, np.mean(permute_band_plot,axis=(1,2))[:,5], bar_width,
#                  alpha=opacity,
#                  color='y',
#                  yerr=np.std(permute_band_plot,axis=(1,2))[:,5],
#                  error_kw=error_config,
#                  label='Oct')
plt.xlabel('Spectral bands in remote sensing image',fontsize=16)
plt.ylabel('Increase of RMSE',fontsize=16)
# plt.title('Root Mean Square Error')
plt.xticks(index + bar_width*1.5, ('1', '2', '3', '4', '5', '6','7','8','9'))
plt.legend(fontsize=14,loc=2)

axes = plt.gca()
axes.set_ylim([0,3.5])

plt.tight_layout()
plt.show()




permute_time = np.load('permute_time.npy')
permute_time_plot = permute_time[0:30]-permute_time[30]

x = list(range(49,282,8))
y = np.mean(permute_time_plot,axis=(1,2))
# example error bar values that vary with x-position
error = 0

plt.errorbar(x, y, yerr=error, fmt='-o',ecolor='0.3',linewidth=1,color='b')
plt.xlabel('Day of year',fontsize=16)
plt.ylabel('Increase of RMSE',fontsize=16)

plt.show()

