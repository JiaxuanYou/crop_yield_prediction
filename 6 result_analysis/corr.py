import numpy as np
import os
import matplotlib.pyplot as plt

def preprocess_save_data(file):
    path = "/atlas/u/jiaxuan/data/google_drive/img_output/"
    if file.endswith(".npy"):
        path_current=os.path.join(path, file)
        image_temp = np.load(path_current)

        image_temp=np.reshape(image_temp,(image_temp.shape[0]*image_temp.shape[1],image_temp.shape[2]))
        image_temp=np.reshape(image_temp,(-1,46,9))
        image_temp=np.reshape(image_temp,(-1,9))

        f_0=image_temp>0
        f_5000=image_temp<5000
        f=f_0*f_5000
        f=np.squeeze(np.prod(f,1).nonzero())

        # print image_temp.shape
        image_temp=image_temp[f,:]
        print image_temp.shape

        corr = np.corrcoef(np.transpose(image_temp))

        # print np.absolute(corr)
        # plt.imshow(np.absolute(corr),cmap='Greys_r',interpolation='none')
        # plt.show()

        return np.absolute(corr)

if __name__ == "__main__":
    # # save data
    corr = np.zeros([9,9])
    path = "/atlas/u/jiaxuan/data/google_drive/img_output/"
    count=0
    try:
        for _, _, files in os.walk(path):
            for file in files:
                try:
                    corr += preprocess_save_data(file)
                    count+=1
                except:
                    continue
    except:
        print 'break'
    np.save('corr.npy', corr)
    corr = np.load('corr.npy')
    fig, ax = plt.subplots()
    img = plt.imshow(corr/count,cmap='Greys_r',interpolation='none',vmin=0,vmax=1)
    cbar = fig.colorbar(img, ticks=[0,0.5,1])
    cbar.ax.set_yticklabels(['0','0.5','1'])
    plt.show()
        