from nnet_for_hist_dropout_stride import *
from GP_crop_v3 import *


if __name__ == "__main__":
    config = Config()
    summary_train_loss = []
    summary_eval_loss = []
    summary_RMSE = []
    summary_ME = []


    # load data to memory
    filename = 'histogram_all' + '.npz'
    # filename = 'histogram_all_soilweather' + '.npz'
    content = np.load(config.load_path + filename)
    image_all = content['output_image']
    yield_all = content['output_yield']
    year_all = content['output_year']
    locations_all = content['output_locations']
    index_all = content['output_index']
    
    # delete broken image
    list_delete=[]
    for i in range(image_all.shape[0]):
        if np.sum(image_all[i,:,:,:])<=287:
            if year_all[i]<2016:
                list_delete.append(i)
    image_all=np.delete(image_all,list_delete,0)
    yield_all=np.delete(yield_all,list_delete,0)
    year_all = np.delete(year_all,list_delete, 0)
    locations_all = np.delete(locations_all, list_delete, 0)
    index_all = np.delete(index_all, list_delete, 0)


    # keep major counties
    list_keep=[]
    for i in range(image_all.shape[0]):
        if (index_all[i,0]==5)or(index_all[i,0]==17)or(index_all[i,0]==18)or(index_all[i,0]==19)or(index_all[i,0]==20)or(index_all[i,0]==27)or(index_all[i,0]==29)or(index_all[i,0]==31)or(index_all[i,0]==38)or(index_all[i,0]==39)or(index_all[i,0]==46):
            list_keep.append(i)
    image_all=image_all[list_keep,:,:,:]
    yield_all=yield_all[list_keep]
    year_all = year_all[list_keep]
    locations_all = locations_all[list_keep,:]
    index_all = index_all[list_keep,:]

    
    ME_test_mean=np.zeros([6])
    ME_val_mean=np.zeros([6])

    count = 0
    for predict_year in range(2014,2011,-1):
        # split into train and validate
        index_train = np.nonzero(year_all < predict_year)[0]
        index_validate = np.nonzero(year_all == predict_year)[0]
        index_test = np.nonzero(year_all == predict_year)[0]
        print 'train size',index_train.shape[0]
        print 'validate size',index_validate.shape[0]
        print 'test size',index_test.shape[0]

        # calc train image mean (for each band), and then detract (broadcast)
        image_mean=np.mean(image_all[index_train],(0,1,2))
        image_all = image_all - image_mean
        year_mean = np.mean(year_all)
        print 'year_mean',year_mean

        image_validate=image_all[index_validate]
        yield_validate=yield_all[index_validate]
        image_test=image_all[index_test]
        yield_test=yield_all[index_test]

        for loop in range(0,1):
            RMSE_test_all=[]
            ME_test_all=[]
            RMSE_val_all=[]
            ME_val_all=[]
            for time in  range(10,31,4):
                g = tf.Graph()
                with g.as_default():
                    print 'year',predict_year,'loop',loop,'time',time
                    # modify config
                    config = Config()
                    config.H=time

                    model= NeuralModel(config,'net')

                    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
                    # Launch the graph.
                    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                    sess.run(tf.initialize_all_variables())
                    saver=tf.train.Saver()
     
                    saver.restore(sess, config.save_path+str(loop)+str(time)+str(predict_year)+"CNN_model.ckpt")
                    # Restore log results
                    # npzfile = np.load(config.save_path+str(loop)+str(time) + str(predict_year)+'result.npz')
                    # summary_train_loss = npzfile['summary_train_loss'].tolist()
                    # summary_eval_loss = npzfile['summary_eval_loss'].tolist()
                    # summary_RMSE = npzfile['summary_RMSE'].tolist()
                    # summary_ME = npzfile['summary_ME'].tolist()
                    # print("Model restored.")

                    # do test
                    pred = []
                    real = []
                    for j in range(image_test.shape[0] / config.B):
                        real_temp = yield_test[j * config.B:(j + 1) * config.B]
                        pred_temp= sess.run(model.logits, feed_dict={
                            model.x: image_test[j * config.B:(j + 1) * config.B,:,0:config.H,:],
                            model.y: yield_test[j * config.B:(j + 1) * config.B],
                            model.keep_prob: 1,
                            model.year: year_all[j * config.B:(j + 1) * config.B,np.newaxis]-year_mean
                            })
                        pred.append(pred_temp)
                        real.append(real_temp)
                    pred=np.concatenate(pred)
                    real=np.concatenate(real)
                    RMSE_test=np.sqrt(np.mean((pred-real)**2))
                    ME_test=np.mean(pred-real)/np.mean(real)*100
                    RMSE_test_all.append(RMSE_test)
                    ME_test_all.append(ME_test)

                    print 'Test set','RMSE',RMSE_test,'ME',ME_test

                    # do validation
                    pred = []
                    real = []
                    for j in range(image_validate.shape[0] / config.B):
                        real_temp = yield_validate[j * config.B:(j + 1) * config.B]
                        pred_temp= sess.run(model.logits, feed_dict={
                            model.x: image_validate[j * config.B:(j + 1) * config.B,:,0:config.H,:],
                            model.y: yield_validate[j * config.B:(j + 1) * config.B],
                            model.keep_prob: 1,
                            model.year: year_all[j * config.B:(j + 1) * config.B,np.newaxis]-year_mean
                            })
                        pred.append(pred_temp)
                        real.append(real_temp)
                    pred=np.concatenate(pred)
                    real=np.concatenate(real)
                    RMSE_val=np.sqrt(np.mean((pred-real)**2))
                    ME_val=np.mean(pred-real)/np.mean(real)*100
                    RMSE_val_all.append(RMSE_val)
                    ME_val_all.append(ME_val)
                    print 'Validation set','RMSE',RMSE_val,'ME',ME_val

                    # save result
                    pred_out = []
                    real_out = []
                    feature_out = []
                    year_out = []
                    locations_out =[]
                    index_out = []
                    for i in range(image_all.shape[0] / config.B):
                        feature,pred = sess.run(
                            [model.fc6,model.logits], feed_dict={
                            model.x: image_all[i * config.B:(i + 1) * config.B,:,0:config.H,:],
                            model.y: yield_all[i * config.B:(i + 1) * config.B],
                            model.keep_prob: config.drop_out,
                            model.year: year_all[i * config.B:(i + 1) * config.B,np.newaxis]-year_mean
                        })
                        real = yield_all[i * config.B:(i + 1) * config.B]

                        pred_out.append(pred)
                        real_out.append(real)
                        feature_out.append(feature)
                        year_out.append(year_all[i * config.B:(i + 1) * config.B])
                        locations_out.append(locations_all[i * config.B:(i + 1) * config.B])
                        index_out.append(index_all[i * config.B:(i + 1) * config.B])
                        # print i
                    weight_out, b_out = sess.run(
                        [model.dense_W, model.dense_B], feed_dict={
                            model.x: image_all[0 * config.B:(0 + 1) * config.B, :, 0:config.H, :],
                            model.y: yield_all[0 * config.B:(0 + 1) * config.B],
                            model.keep_prob: config.drop_out,
                            model.year: year_all[i * config.B:(i + 1) * config.B,np.newaxis]-year_mean
                        })
                    pred_out=np.concatenate(pred_out)
                    real_out=np.concatenate(real_out)
                    feature_out=np.concatenate(feature_out)
                    year_out=np.concatenate(year_out)
                    locations_out=np.concatenate(locations_out)
                    index_out=np.concatenate(index_out)
                    
                    path = config.save_path+str(loop)+str(time)+str(predict_year)+'result_prediction.npz'
                    np.savez(path,
                        pred_out=pred_out,real_out=real_out,feature_out=feature_out,
                        year_out=year_out,locations_out=locations_out,weight_out=weight_out,b_out=b_out,index_out=index_out)
                    RMSE_GP,ME_GP,Average_GP=GaussianProcess(predict_year,path)
                    print 'RMSE_GP',RMSE_GP
                    print 'ME_GP',ME_GP
                    print 'Average_GP',Average_GP

            ME_test_mean+=np.absolute(np.array(ME_test_all))
            ME_val_mean+=np.absolute(np.array(ME_val_all))
            count += 1
            print count
    print 'theoretical count', 32
    ME_test_mean/=count
    ME_val_mean/=count

    plt.plot(range(len(ME_val_mean)),ME_val_mean)
    plt.plot(range(len(ME_test_mean)),ME_test_mean)
    plt.legend(['val','test'])
    plt.show()

    # plt.bar(range(len(ME_val_mean)),ME_val_mean)
    plt.bar(range(len(ME_test_mean)),ME_test_mean)
    # plt.legend(['val','test'])
    plt.show()

