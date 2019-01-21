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

    # match corn yield
    # load corn yield
    corn_yield = np.genfromtxt('/atlas/u/jiaxuan/git/crop-forecasting/corn_yield.csv', delimiter=',')
    # keep data with corn yield
    list_delete=[]
    for i in range(image_all.shape[0]):
        key = np.array([year_all[i],index_all[i,0],index_all[i,1]])
        index = np.where(np.all(corn_yield[:,0:3] == key, axis=1))
        # print index[0].shape
        if index[0].shape[0] != 0:
            yield_all[i]=corn_yield[index,3]
        else:
            # print 'del'
            list_delete.append(i)
    image_all=np.delete(image_all,list_delete,0)
    yield_all=np.delete(yield_all,list_delete,0)
    year_all = np.delete(year_all,list_delete, 0)
    locations_all = np.delete(locations_all, list_delete, 0)
    index_all = np.delete(index_all, list_delete, 0)



    for loop in range(0,1):
        for predict_year in range(2009,2016):
            logging.basicConfig(filename=config.save_path+'log/train_for_hist_alldata_loop'+str(predict_year)+str(loop)+'.log',level=logging.DEBUG)
            # # split into train and validate
            # index_train = np.nonzero(year_all < predict_year)[0]
            # index_validate = np.nonzero(year_all == predict_year)[0]
            # index_test = np.nonzero(year_all == predict_year+1)[0]

            # random choose validation set
            index_train = np.nonzero(year_all < predict_year)[0]
            index_validate = np.nonzero(year_all == predict_year)[0]
            print('train size',index_train.shape[0])
            print('validate size',index_validate.shape[0])
            logging.info('train size %d',index_train.shape[0])
            logging.info('validate size %d',index_validate.shape[0])


            # # calc train image mean (for each band), and then detract (broadcast)
            # image_mean=np.mean(image_all[index_train],(0,1,2))
            # image_all = image_all - image_mean

            image_validate=image_all[index_validate]
            yield_validate=yield_all[index_validate]

            for time in range(30,31):
                RMSE_min = 100
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
                    for i in range(config.train_step):
                        if i==4000:
                            config.lr/=10

                        if i==20000:
                            config.lr/=10
                       
                        # index_train_batch = np.random.choice(index_train,size=config.B)
                        index_validate_batch = np.random.choice(index_validate, size=config.B)

                        # try data augmentation while training
                        shift = 1
                        index_train_batch_1 = np.random.choice(index_train,size=config.B+shift*2)
                        index_train_batch_2 = np.random.choice(index_train,size=config.B+shift*2)
                        image_train_batch = (image_all[index_train_batch_1,:,0:config.H,:]+image_all[index_train_batch_1,:,0:config.H,:])/2
                        yield_train_batch = (yield_all[index_train_batch_1]+yield_all[index_train_batch_1])/2

                        arg_index = np.argsort(yield_train_batch)
                        yield_train_batch = yield_train_batch[arg_index][shift:-shift]
                        image_train_batch = image_train_batch[arg_index][shift:-shift]

                        _, train_loss, train_loss_reg = sess.run([model.train_op, model.loss_err, model.loss_reg], feed_dict={
                            model.x:image_train_batch,
                            model.y:yield_train_batch,
                            model.lr:config.lr,
                            model.keep_prob: config.keep_prob
                            })

                        if i%500 == 0:
                            val_loss,val_loss_reg = sess.run([model.loss_err,model.loss_reg], feed_dict={
                                model.x: image_all[index_validate_batch, :, 0:config.H, :],
                                model.y: yield_all[index_validate_batch],
                                model.keep_prob: 1
                            })

                            print(str(loop)+str(time)+'predict year'+str(predict_year)+'step'+str(i),train_loss,train_loss_reg,val_loss,val_loss_reg,config.lr)
                            logging.info('%d %d %d step %d %f %f %f %f %f',loop,time,predict_year,i,train_loss,train_loss_reg,val_loss,val_loss_reg,config.lr)
                        if i%500 == 0:
                            # do validation
                            pred = []
                            real = []
                            for j in range(image_validate.shape[0] / config.B):
                                real_temp = yield_validate[j * config.B:(j + 1) * config.B]
                                pred_temp= sess.run(model.logits, feed_dict={
                                    model.x: image_validate[j * config.B:(j + 1) * config.B,:,0:config.H,:],
                                    model.y: yield_validate[j * config.B:(j + 1) * config.B],
                                    model.keep_prob: 1
                                    })
                                pred.append(pred_temp)
                                real.append(real_temp)
                            pred=np.concatenate(pred)
                            real=np.concatenate(real)
                            RMSE=np.sqrt(np.mean((pred-real)**2))
                            ME=np.mean(pred-real)
                            RMSE_ideal = np.sqrt(np.mean((pred-ME-real)**2))
                            arg_index = np.argsort(pred)
                            pred = pred[arg_index][50:-50]
                            real = real[arg_index][50:-50]
                            ME_part = np.mean(pred-real)

                            if RMSE<RMSE_min:
                                RMSE_min=RMSE
                               

                            # print 'Validation set','RMSE',RMSE,'ME',ME,'RMSE_min',RMSE_min
                            # logging.info('Validation set RMSE %f ME %f RMSE_min %f',RMSE,ME,RMSE_min)
                            print('Validation set','RMSE',RMSE,'RMSE_ideal',RMSE_ideal,'ME',ME,'ME_part',ME_part,'RMSE_min',RMSE_min)
                            logging.info('Validation set RMSE %f RMSE_ideal %f ME %f ME_part %f RMSE_min %f',RMSE,RMSE_ideal,ME,ME_part,RMSE_min)
            
                            summary_train_loss.append(train_loss)
                            summary_eval_loss.append(val_loss)
                            summary_RMSE.append(RMSE)
                            summary_ME.append(ME)
                    # save
                    save_path = saver.save(sess, config.save_path+str(loop)+str(time) + str(predict_year)+'CNN_model.ckpt')
                    print(('save in file: %s' % save_path))
                    logging.info('save in file: %s' % save_path)

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
                            model.keep_prob:1
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
                            model.keep_prob: 1
                        })
                    pred_out=np.concatenate(pred_out)
                    real_out=np.concatenate(real_out)
                    feature_out=np.concatenate(feature_out)
                    year_out=np.concatenate(year_out)
                    locations_out=np.concatenate(locations_out)
                    index_out=np.concatenate(index_out)
                    
                    np.savez(config.save_path+str(loop)+str(time)+str(predict_year)+'result_prediction.npz',
                        pred_out=pred_out,real_out=real_out,feature_out=feature_out,
                        year_out=year_out,locations_out=locations_out,weight_out=weight_out,b_out=b_out,index_out=index_out)
                    np.savez(config.save_path+str(loop)+str(time)+str(predict_year)+'result.npz',
                        summary_train_loss=summary_train_loss,summary_eval_loss=summary_eval_loss,
                        summary_RMSE=summary_RMSE,summary_ME=summary_RMSE)