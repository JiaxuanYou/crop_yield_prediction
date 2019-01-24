from nnet_for_hist_dropout_stride import *
import logging



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

    image_all_save = np.copy(image_all)


    # result_band = np.zeros([10,2,7,6])
    # for p in range(10):
    #     for loop in range(0,2):
    #         for predict_year in range(2009,2016):
    #             image_all = np.copy(image_all_save)
    #             if p!=9:
    #                 np.take(image_all[:,:,:,p],np.random.permutation(image_all.shape[0]),axis=0,out=image_all[:,:,:,p])
    #             index_train = np.nonzero(year_all < predict_year)[0]
    #             index_validate = np.nonzero(year_all == predict_year)[0]


    #             # calc train image mean (for each band), and then detract (broadcast)
    #             image_mean=np.mean(image_all[index_train],(0,1,2))
    #             image_all = image_all - image_mean

    #             image_train=image_all[index_train]
    #             yield_train=yield_all[index_train]

    #             for count,time in enumerate(range(10,31,4)):
    #                 g = tf.Graph()
    #                 with g.as_default():
    #                     # modify config
    #                     config = Config()
    #                     config.H=time

    #                     model= NeuralModel(config,'net')

    #                     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
    #                     # Launch the graph.
    #                     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #                     sess.run(tf.initialize_all_variables())
    #                     saver=tf.train.Saver()
    #                     if predict_year==2012:
    #                         saver.restore(sess, config.save_path+str(loop+2)+str(time) + str(predict_year)+'CNN_model.ckpt')
    #                     else:
    #                         saver.restore(sess, config.save_path+str(loop)+str(time) + str(predict_year)+'CNN_model.ckpt')

    #                     # save result
    #                     pred_out = []
    #                     real_out = []
    #                     feature_out = []
    #                     for i in range(image_train.shape[0] / config.B):
    #                         feature,pred = sess.run(
    #                             [model.fc6,model.logits], feed_dict={
    #                             model.x: image_train[i * config.B:(i + 1) * config.B,:,0:config.H,:],
    #                             model.y: yield_train[i * config.B:(i + 1) * config.B],
    #                             model.keep_prob:1
    #                         })
    #                         real = yield_train[i * config.B:(i + 1) * config.B]

    #                         pred_out.append(pred)
    #                         real_out.append(real)
    #                         feature_out.append(feature)
    #                     pred_out=np.concatenate(pred_out)
    #                     real_out=np.concatenate(real_out)
    #                     feature_out=np.concatenate(feature_out)

    #                     rmse = np.sqrt(np.mean((pred_out-real_out)**2))
    #                     print 'p',p
    #                     print rmse
    #                     result_band[p,loop,predict_year-2009,count]=rmse
    # np.save('permute_band.npy', result_band)

    result_time = np.zeros([31,2,7])
    for p in range(31):
        for loop in range(0,2):
            for predict_year in range(2009,2016):
                image_all = np.copy(image_all_save)
                if p!=30:
                    np.take(image_all[:,:,p,:],np.random.permutation(image_all.shape[0]),axis=0,out=image_all[:,:,p,:])
                index_train = np.nonzero(year_all < predict_year)[0]
                index_validate = np.nonzero(year_all == predict_year)[0]

                # calc train image mean (for each band), and then detract (broadcast)
                image_mean=np.mean(image_all[index_train],(0,1,2))
                image_all = image_all - image_mean

                image_train=image_all[index_train]
                yield_train=yield_all[index_train]

                for time in range(30,31):
                    g = tf.Graph()
                    with g.as_default():
                        # modify config
                        config = Config()
                        config.H=time

                        model= NeuralModel(config,'net')

                        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
                        # Launch the graph.
                        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                        sess.run(tf.initialize_all_variables())
                        saver=tf.train.Saver()
                        if predict_year==2012:
                            saver.restore(sess, config.save_path+str(loop+2)+str(time) + str(predict_year)+'CNN_model.ckpt')
                        else:
                            saver.restore(sess, config.save_path+str(loop)+str(time) + str(predict_year)+'CNN_model.ckpt')

                        # save result
                        pred_out = []
                        real_out = []
                        feature_out = []
                        for i in range(image_train.shape[0] / config.B):
                            feature,pred = sess.run(
                                [model.fc6,model.logits], feed_dict={
                                model.x: image_train[i * config.B:(i + 1) * config.B,:,0:config.H,:],
                                model.y: yield_train[i * config.B:(i + 1) * config.B],
                                model.keep_prob:1
                            })
                            real = yield_train[i * config.B:(i + 1) * config.B]

                            pred_out.append(pred)
                            real_out.append(real)
                            feature_out.append(feature)
                        pred_out=np.concatenate(pred_out)
                        real_out=np.concatenate(real_out)
                        feature_out=np.concatenate(feature_out)

                        rmse = np.sqrt(np.mean((pred_out-real_out)**2))
                        print('p',p)
                        print(rmse)
                        result_time[p,loop,predict_year-2009]=rmse
    np.save('permute_time.npy', result_time)
