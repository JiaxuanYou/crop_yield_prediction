from nnet_semi import *
from GP_crop_v3 import *
import logging
import time

predict_year = 2015

def load_data(filename,config):
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

    # split into train and validate
    index_train = np.nonzero(year_all < predict_year)[0]
    index_validate = np.nonzero(year_all == predict_year)[0]
    index_train_validate = np.nonzero(year_all <= predict_year)[0]
    print 'train size',index_train.shape[0]
    print 'validate size',index_validate.shape[0]

    # calc train image mean (for each band), and then detract (broadcast)
    image_mean=np.mean(image_all[index_train],(0,1,2))
    image_all = image_all - image_mean

    return image_all,yield_all,year_all,locations_all,index_all,index_train,index_validate,index_train_validate


if __name__ == "__main__":
    # logging.basicConfig(filename='logging_semi/'+str(predict_year)+'.log',level=logging.DEBUG)
    # Create a coordinator
    config = Config()

    filename = 'histogram_all' + '.npz'
    # filename = 'histogram_all_soilweather' + '.npz'
    time1 = time.time()
    image_all,yield_all,year_all,locations_all,index_all,index_train,index_validate,_ = load_data(filename, config)
    print("load time: %ss" % (time.time() - time1))
    image_validate=image_all[index_validate]
    yield_validate=yield_all[index_validate]

    model= NeuralModel(config,'net')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.48)
    # Launch the graph.
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.initialize_all_variables())

    # summary_train_loss = []
    # summary_eval_loss = []
    # summary_RMSE = []
    # summary_ME = []

    train_loss=0
    val_loss=0
    val_prediction = 0
    val_deviation = np.zeros([config.B])
    # #########################
    # block when test
    # add saver
    saver=tf.train.Saver()
    # Restore variables from disk.
    try:
        saver.restore(sess, config.save_path+str(predict_year)+"CNN_model.ckpt")
    # Restore log results
        # npzfile = np.load(config.save_path + str(predict_year)+'result.npz')
        # summary_train_loss = npzfile['summary_train_loss'].tolist()
        # summary_eval_loss = npzfile['summary_eval_loss'].tolist()
        # summary_RMSE = npzfile['summary_RMSE'].tolist()
        # summary_ME = npzfile['summary_ME'].tolist()
        print("Model restored.")
    except:
        print 'No history model found'
    # #########################
    
    RMSE_min = 100
    chkpoint_loop = 500
    try:
        for i in range(config.train_step):

            # # load extra unlabel data
            # if i%chkpoint_loop ==0:
            #     chkpoint = i/chkpoint_loop + 1
            #     # load unsupervised data
            #     filename = 'histogram_semi_rand_200_20000'+str(chkpoint)+'.npz'
            #     time1 = time.time()
            #     image_all_ulab,_,_,_,_,_,index_validate_ulab,index_ulab = load_data(filename, config)
            #     print("load time: %ss" % (time.time() - time1))

            # No augmentation
            index_train_batch = np.random.choice(index_train,size=config.B)
            image_train_batch = image_all[index_train_batch,:,0:config.H,:]
            # index_train_batch_ulab = np.random.choice(index_ulab,size=config.B)
            # image_train_batch_ulab = image_all_ulab[index_train_batch_ulab,:,0:config.H,:]
            # image_train_batch = np.concatenate((image_train_batch, image_train_batch_ulab),axis=0)
            image_train_batch = np.concatenate((image_train_batch, image_train_batch),axis=0)

            yield_train_batch = yield_all[index_train_batch,np.newaxis]

            # # try data augmentation while training
            # index_train_batch_1 = np.random.choice(index_train,size=config.B)
            # index_train_batch_2 = np.random.choice(index_train,size=config.B)
            # image_train_batch = (image_all[index_train_batch_1,:,0:config.H,:]+image_all[index_train_batch_1,:,0:config.H,:])/2
            # yield_train_batch = (yield_all[index_train_batch_1]+yield_all[index_train_batch_1])/2
            # # year_train_batch = (year_all[index_train_batch_1,np.newaxis]+year_all[index_train_batch_2,np.newaxis])/2

            index_validate_batch = np.random.choice(index_validate, size=config.B)
            image_validate_batch = image_all[index_validate_batch,:,0:config.H,:]
            # index_validate_batch_ulab = np.random.choice(index_validate_ulab,size=config.B)
            # image_validate_batch_ulab = image_all_ulab[index_validate_batch_ulab,:,0:config.H,:]
            # image_validate_batch = np.concatenate((image_validate_batch, image_validate_batch_ulab),axis=0)
            image_validate_batch = np.concatenate((image_validate_batch, image_validate_batch),axis=0)

            yield_validate_batch = yield_all[index_validate_batch,np.newaxis]

            _,t_L,t_C,t_U,t_R,t_loss,t_pred,t_real,t_err = sess.run(
                [model.train_op,model.L,model.C,model.U,model.R,model.loss,model.pred,model.real,model.pred_err], feed_dict={
                model.x:image_train_batch,
                model.y_lab:yield_train_batch,
                model.lr:config.lr,
                model.keep_prob:config.keep_prob
                })

            if i%10 == 0:
                v_L,v_C,v_U,v_R,v_loss,v_pred,v_real,v_err = sess.run(
                    [model.L,model.C,model.U,model.R,model.loss,model.pred,model.real,model.pred_err], feed_dict={
                    model.x: image_validate_batch,
                    model.y_lab: yield_validate_batch,
                    model.keep_prob:1
                })

                print 'predict year'+str(predict_year)+'step'+str(i),config.lr
                print t_L,t_C,t_U,t_R,t_loss,np.mean(t_pred),np.mean(t_real),np.mean(t_pred-t_real),t_err
                print v_L,v_C,v_U,v_R,v_loss,np.mean(v_pred),np.mean(v_real),np.mean(v_pred-v_real),v_err
                # logging.info('predict year %d step %d %f %f %f' % predict_year,i,train_loss,val_loss,config.lr)
                # logging.info('predict year %d step %d lr %d' % predict_year,i,config.lr)
                # logging.info('%d %d %d %d %d %d %d %d %d' % t_L,t_C,t_U,t_R,t_loss,np.mean(t_pred),np.mean(t_real),np.mean(t_pred-t_real),t_err)
                # logging.info('%d %d %d %d %d %d %d %d %d' % v_L,v_C,v_U,v_R,v_loss,np.mean(v_pred),np.mean(v_real),np.mean(v_pred-v_real),v_err)
            if i%100 == 0:
                # do validation
                pred = []
                real = []
                for j in range(image_validate.shape[0] / config.B):
                    real_temp = yield_validate[j * config.B:(j + 1) * config.B]
                    image_batch = image_validate[j * config.B:(j + 1) * config.B,:,0:config.H,:]
                    # index_batch_ulab = np.random.choice(index_validate_ulab,size=config.B)
                    # image_batch_ulab = image_all_ulab[index_batch_ulab,:,0:config.H,:]
                    # image_batch = np.concatenate((image_batch, image_batch_ulab),axis=0)
                    image_batch = np.concatenate((image_batch, image_batch),axis=0)
                    yield_batch = yield_validate[j * config.B:(j + 1) * config.B,np.newaxis]
                    pred_temp= sess.run(model.y_lab_pred, feed_dict={
                        model.x: image_batch,
                        model.y_lab: yield_batch,
                        model.keep_prob: 1
                        })
                    pred.append(pred_temp)
                    real.append(real_temp)
                pred=np.concatenate(pred,axis=0)
                real=np.concatenate(real,axis=0)
                RMSE=np.sqrt(np.mean((pred-real)**2))
                ME=np.mean(pred-real)

                if RMSE<RMSE_min:
                    RMSE_min=RMSE

                print 'Validation set','RMSE',RMSE,'ME',ME,'RMSE_min',RMSE_min
                # logging.info('Validation set RMSE %f ME %f RMSE_min %f' % RMSE,ME,RMSE_min)
            
                # summary_train_loss.append(train_loss)
                # summary_eval_loss.append(val_loss)
                # summary_RMSE.append(RMSE)
                # summary_ME.append(ME)

    except KeyboardInterrupt:
        print 'stopped'

    finally:
        # save
        save_path = saver.save(sess, config.save_path + str(predict_year)+'CNN_model.ckpt')
        print('save in file: %s' % save_path)
        # logging.info('save in file: %s' % save_path)

        # save result
        # pred_out = []
        # real_out = []
        # feature_out = []
        # year_out = []
        # locations_out =[]
        # index_out = []
        # for i in range(image_all.shape[0] / config.B):
        #     feature,pred = sess.run(
        #         [model.fc6,model.logits], feed_dict={
        #         model.x: image_all[i * config.B:(i + 1) * config.B,:,0:config.H,:],
        #         model.y: yield_all[i * config.B:(i + 1) * config.B],
        #         model.keep_prob:1
        #     })
        #     real = yield_all[i * config.B:(i + 1) * config.B]

        #     pred_out.append(pred)
        #     real_out.append(real)
        #     feature_out.append(feature)
        #     year_out.append(year_all[i * config.B:(i + 1) * config.B])
        #     locations_out.append(locations_all[i * config.B:(i + 1) * config.B])
        #     index_out.append(index_all[i * config.B:(i + 1) * config.B])
        #     # print i
        # weight_out, b_out = sess.run(
        #     [model.dense_W, model.dense_B], feed_dict={
        #         model.x: image_all[0 * config.B:(0 + 1) * config.B, :, 0:config.H, :],
        #         model.y: yield_all[0 * config.B:(0 + 1) * config.B],
        #         model.keep_prob: 1
        #     })
        # pred_out=np.concatenate(pred_out)
        # real_out=np.concatenate(real_out)
        # feature_out=np.concatenate(feature_out)
        # year_out=np.concatenate(year_out)
        # locations_out=np.concatenate(locations_out)
        # index_out=np.concatenate(index_out)
        
        # path = config.save_path + str(predict_year)+'result_prediction.npz'
        # np.savez(path,
        #     pred_out=pred_out,real_out=real_out,feature_out=feature_out,
        #     year_out=year_out,locations_out=locations_out,weight_out=weight_out,b_out=b_out,index_out=index_out)


        # np.savez(config.save_path+str(predict_year)+'result.npz',
        #                 summary_train_loss=summary_train_loss,summary_eval_loss=summary_eval_loss,
        #                 summary_RMSE=summary_RMSE,summary_ME=summary_ME)
        # # plot results
        # npzfile = np.load(config.save_path+str(predict_year)+'result.npz')
        # summary_train_loss=npzfile['summary_train_loss']
        # summary_eval_loss=npzfile['summary_eval_loss']
        # summary_RMSE = npzfile['summary_RMSE']
        # summary_ME = npzfile['summary_ME']

        # # Plot the points using matplotlib
        # plt.plot(range(len(summary_train_loss)), summary_train_loss)
        # plt.plot(range(len(summary_eval_loss)), summary_eval_loss)
        # plt.xlabel('Training steps')
        # plt.ylabel('L2 loss')
        # plt.title('Loss curve')
        # plt.legend(['Train', 'Validate'])
        # plt.show()

        # plt.plot(range(len(summary_RMSE)), summary_RMSE)
        # # plt.plot(range(len(summary_ME)), summary_ME)
        # plt.xlabel('Training steps')
        # plt.ylabel('Error')
        # plt.title('RMSE')
        # # plt.legend(['RMSE', 'ME'])
        # plt.show()

        # # plt.plot(range(len(summary_RMSE)), summary_RMSE)
        # plt.plot(range(len(summary_ME)), summary_ME)
        # plt.xlabel('Training steps')
        # plt.ylabel('Error')
        # plt.title('ME')
        # # plt.legend(['RMSE', 'ME'])
        # plt.show()
