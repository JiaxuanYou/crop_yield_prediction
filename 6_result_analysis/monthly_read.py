import numpy as np
import matplotlib.pyplot as plt

save_path = '/atlas/u/jiaxuan/data/train_results/final/monthly/'


for loop in range(0,1):
	RMSE_all = np.zeros([6])
	ME_all = np.zeros([6])
	for predict_year in range(2009,2016):
		RMSE = np.zeros([6])
		ME = np.zeros([6])
		for i,time in enumerate(range(10,31,4)):
			data = np.load(save_path+str(loop)+str(time)+str(predict_year)+'result_prediction.npz')
			year_all = data['year_out']
			real = data['real_out']
			pred = data['pred_out']

			validate = np.nonzero(year_all == predict_year)[0]
			train = np.nonzero(year_all < predict_year)[0]
			
			rmse=np.sqrt(np.mean((real[validate]-pred[validate])**2))
			me = np.mean(pred[validate]-real[validate])
			RMSE[i]=rmse
			ME[i]=me
		RMSE_all+=RMSE
		ME_all+=np.absolute(ME)
	RMSE_all/=7
	ME_all/=7




	plt.plot(range(6),RMSE_all)
	plt.title(str(predict_year))
	plt.show()
	plt.plot(range(6),ME_all)
	plt.title(str(predict_year))
	plt.show()
