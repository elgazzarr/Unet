from __future__ import print_function

#import cv2
import os
import numpy as np
from keras.metrics import fscore,fmeasure,recall,fbeta_score,binary_accuracy,precision
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras import backend as K
import tensorflow as tf
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import csv
from keras.utils.np_utils import to_categorical
import math
import pandas



def load_data_all(imgs_name, gt_name):
    path = os.getcwd()
    cwd_data_path = "/data/"
    x = {}
    #--------------------------------------------------------------------------
    try:

        x['FLAIR'] = np.load(path + cwd_data_path + imgs_name + "FLAIR.npy")
        print('found and loading FLAIR image stack')
    except IOError:
        print('No FLAIR data in /data/')
        pass
    #--------------------------------------------------------------------------
    try:

        x['T1'] = np.load(path + cwd_data_path + imgs_name + "T1.npy")
        print('found and loading T1 image stack')
    except IOError:
        print('No T1 data in /data/')
        pass
    #--------------------------------------------------------------------------
    try:

        x['T2'] = np.load(path + cwd_data_path + imgs_name + "T2.npy")
        print('found and loading T2 image stack')
    except IOError:
        print('No T2 data in /data/')
        pass
    #--------------------------------------------------------------------------
    y = np.load(path + "/data/" + gt_name)
    print ('NUM train datasets:', len(x))
    print ('Shape of train ground truth:', y.shape)
    return x, y


#Evaluation metrics
def acc(y_true, y_pred):

    labels = tf.to_int64(y_true)
    labels = K.flatten(labels)
    logits = tf.reshape(y_pred, (-1, 2))

    predicted_annots = K.flatten(tf.argmax(logits, 1))
    correct_predictions = tf.equal(predicted_annots, labels)
    segmentation_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return segmentation_accuracy

def prec(y_true, y_pred):

    labels = tf.to_int64(y_true)
    labels = K.flatten(labels)
    logits = tf.reshape(y_pred, (-1, 2))
    logits = K.flatten(tf.argmax(logits, 1))
    labels = tf.to_float(labels)
    logits = tf.to_float(logits)

    return precision(labels,logits)



def rec(y_true, y_pred):

    labels = tf.to_int64(y_true)
    labels = K.flatten(labels)
    logits = tf.reshape(y_pred, (-1, 2))
    logits = K.flatten(tf.argmax(logits, 1))
    labels = tf.to_float(labels)
    logits = tf.to_float(logits)

    return  recall(labels,logits)


def fm(y_true, y_pred):
    labels = tf.to_int64(y_true)
    labels = K.flatten(labels)
    logits = tf.reshape(y_pred, (-1, 2))
    logits = K.flatten(tf.argmax(logits, 1))
    labels = tf.to_float(labels)
    logits = tf.to_float(logits)

    return fmeasure(labels,logits)

#loss function
def weighted_softmax(y_true,y_pred):

    class_weight = tf.constant(np.array([0.3,0.7], dtype='f'))
    #label = K.flatten(y_true)
    labels = tf.to_int64(y_true)
    labels = tf.one_hot(labels,2)
    labels = tf.reshape(labels, [-1, 2])
    logits = tf.reshape(y_pred, (-1, 2))

    weight_map = tf.mul(labels, class_weight)

    weight_map = tf.reduce_sum(weight_map, 1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits, labels)

    weighted_loss = tf.mul(loss_map, weight_map)

    loss = tf.reduce_mean(weighted_loss)


    return loss*4


# Numpy version of the abstraction layer
def get_abstractionLayerNP(*args):
    # INPUT: (Number of classes to segment, backend1, backend2, ..., backend_END)
    # OUTPUT:
    # Description: computes the MEAN and VARIANCE feature maps from all backend models
    # weights them by the total number of classes which are possible
    # output data format is int16
    NUM_classes = np.float32(len(args))
    devisor = np.float32(1/(np.abs(NUM_classes)))
    # Compute element wise average
    arg_Asum = np.zeros(args[0].shape, np.float32)
    for arg in args:
        arg_Asum = np.add(arg_Asum,arg)
    arg_avg = np.multiply(devisor, arg_Asum)


    # Compute element wise variance
    #VARIANCE =  (1/ |NUM_classes|-1) * sum-over-each-backend-input( backend_input[i] - AVERAGE_of_backend_inputs)
    # if NUM_classes == 1 set VARIANCE to zeros
    # return -- the average_mask and the variance_mask concatonated along the Z axis
    if (np.abs(NUM_classes) == 1):
        arg_variance = np.zeros(args[0].shape, dtype=np.float32)
        out_concat = np.concatenate([arg_avg,arg_variance], axis=0)
        return out_concat

    elif (np.abs(NUM_classes) > 1):
        arg_Vsum = np.zeros(args[0].shape, dtype=np.float32)
        P_v = 1/(np.abs(NUM_classes) - 1)
        for arg in args:
            arg_Vsum = np.add( np.square( np.subtract(arg, arg_avg) ), arg_Vsum)
        arg_variance = np.multiply(P_v, arg_Vsum)
        out_concat = np.concatenate([arg_avg,arg_variance], axis=0)
        return out_concat


# Get the cropped U-net architecture to load pretrained weights into
def get_unet_cropped(verbose=1):
    img_rows = 256
    img_cols = 256

    inputs = Input(( img_rows, img_cols, 1),name='input_1')

    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',dim_ordering="tf",name='convolution2d_1')(inputs)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',dim_ordering="tf",name='convolution2d_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),dim_ordering="tf",name='maxpooling2d_1')(conv1)

    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',dim_ordering="tf",name='convolution2d_3')(pool1)
    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',dim_ordering="tf",name='convolution2d_4')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),dim_ordering="tf",name='maxpooling2d_2')(conv2)

    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',dim_ordering="tf",name='convolution2d_5')(pool2)
    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',dim_ordering="tf",name='convolution2d_6')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),dim_ordering="tf",name='maxpooling2d_3')(conv3)

    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same',dim_ordering="tf",name='convolution2d_7')(pool3)
    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same',dim_ordering="tf",name='convolution2d_8')(conv4)

    model = Model(input=inputs, output=conv4)
    if verbose==1:
        model.summary()
    model.compile(optimizer=Adam(lr=10e-5,decay=0.001), loss = weighted_softmax, metrics=[acc,prec,rec,fm])

    return model

# Get the front architecture, feed in the mean and variance of inputs
def get_frontNet():
    # For numpy input
    mean_var = Input(shape=(32,32,512), name="mean_var_input")

    conv1_MeanVar1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same',dim_ordering="tf", name='conv1_MeanVar1')(mean_var)
    conv1_MeanVar2 = Convolution2D(512, 3, 3, activation='relu', border_mode='same',dim_ordering="tf", name='conv1_MeanVar2')(conv1_MeanVar1)

    # when adding skip connections, uncomment
    #up_conv1_MeanVar = merge([Convolution2D(256, 2, 2,activation='relu', border_mode='same',dim_ordering="tf", name ='up_conv1_MeanVar')(UpSampling2D(size=(2, 2),dim_ordering="tf", name='up1_MeanVar')(conv1_MeanVar2)), conv3], mode='concat', concat_axis=3)
    up_conv1_MeanVar = Convolution2D(256, 2, 2,activation='relu', border_mode='same',dim_ordering="tf", name ='up_conv1_MeanVar')(UpSampling2D(size=(2, 2),dim_ordering="tf", name='up1_MeanVar')(conv1_MeanVar2))

    conv2_MeanVar1 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',dim_ordering="tf", name='conv2_MeanVar1')(up_conv1_MeanVar)
    conv2_MeanVar2 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',dim_ordering="tf", name='conv2_MeanVar2')(conv2_MeanVar1)

    # when adding skip connections, uncomment
    #up_conv2_MeanVar = merge([Convolution2D(128, 2, 2,activation='relu', border_mode='same',dim_ordering="tf", name='up_conv2_MeanVar')(UpSampling2D(size=(2, 2),dim_ordering="tf", name='up2_MeanVar')(conv2_MeanVar2)), conv2], mode='concat', concat_axis=3)
    up_conv2_MeanVar = Convolution2D(128, 2, 2,activation='relu', border_mode='same',dim_ordering="tf", name='up_conv2_MeanVar')(UpSampling2D(size=(2, 2),dim_ordering="tf", name='up2_MeanVar')(conv2_MeanVar2))

    conv3_MeanVar1 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',dim_ordering="tf", name='conv3_MeanVar1')(up_conv2_MeanVar)
    conv3_MeanVar2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',dim_ordering="tf", name='conv3_MeanVar2')(conv3_MeanVar1)

    # when adding skip connections, uncomment
    #up_conv3_MeanVar = merge([Convolution2D(64, 2, 2,activation='relu', border_mode='same',dim_ordering="tf", name='up_conv3_MeanVar')(UpSampling2D(size=(2, 2),dim_ordering="tf", name='up3_Meanvar')(conv3_MeanVar2)), conv1], mode='concat', concat_axis=3)
    up_conv3_MeanVar = Convolution2D(64, 2, 2,activation='relu', border_mode='same',dim_ordering="tf", name='up_conv3_MeanVar')(UpSampling2D(size=(2, 2),dim_ordering="tf", name='up3_Meanvar')(conv3_MeanVar2))

    conv4_MeanVar1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',dim_ordering="tf", name='conv4_MeanVar1')(up_conv3_MeanVar)
    conv4_MeanVar2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',dim_ordering="tf", name='conv4_MeanVar2')(conv4_MeanVar1)

    conv5_fc = Convolution2D(2, 1, 1, dim_ordering="tf", name='conv5_fc')(conv4_MeanVar2)

    model = Model(input=mean_var, output=conv5_fc)
    model.summary()

    model.compile(optimizer=Adam(lr=10e-5,decay=0.001), loss = weighted_softmax, metrics=[acc,prec,rec,fm])
    return model


def plt_history(history):
    # summarize history for accuracy
    plt.plot(history.accs)
    plt.plot(history.val_accs)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    # summarize history for loss
    plt.plot(history.losses)
    plt.plot(history.val_losses)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

class Histories(Callback):

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []
        self.prec = []
        self.recall =[]
        self.fscore =[]

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('accuracy'))
        self.prec.append(logs.get('prec1'))
        self.recall.append(logs.get('rec1'))
        self.fscore.append(logs.get('frec1'))
        print('/n')
        return

def train_front(imgs_train,gt_train,model):

    # print(np.histogram(imgs_train))
    # print(np.histogram(imgs_mask_train))
    history = Histories()
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model_checkpoint = ModelCheckpoint('unet_front_checkpoint.hdf5', monitor='loss',verbose=1, save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    pandas.DataFrame( model.fit(imgs_train, gt_train, batch_size=2, nb_epoch=1,
                                verbose=1, shuffle=True,callbacks=[model_checkpoint, history],
                                validation_split=0.2).history).to_csv("HEMIShistory.csv")

    print('-'*30)
    print('saving weights and model as .h5')
    model.save('front_model.h5')
    model.save_weights('front_model_weights.h5')
    print( history)
    plt_history(history)
    #np.savetxt('model_history.txt', np.array(history).reshape(1,),  delimiter=" ", fmt="%s")

    print('-'*30)

    print('-'*30)
    print('Training Done')
    print('-'*30)

def visualize(img,label,logit,index):

    fig = plt.figure()
    fig2 = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax4 = fig2.add_subplot(131)
    ax5 = fig2.add_subplot(132)
    ax6 = fig2.add_subplot(133)

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')

    ax1.title.set_text('Image_FLAIR')
    ax1.imshow(img['FLAIR'][index,:,:,index], cmap='gray')
    ax2.title.set_text('Image_T1')
    ax2.imshow(img['T1'][index,:,:,index], cmap='gray')
    ax3.title.set_text('Image_T2')
    ax3.imshow(img['T2'][index,:,:,index], cmap='gray')

    ax5.title.set_text('Ground Truth')
    ax5.imshow(label, cmap='gray')
    ax6.title.set_text('Predictions')
    ax6.imshow(logit, cmap='gray')

    plt.show()

def load_backendModels(verbose = 1):
    weights_path = os.getcwd() + "/Models/v1/"

    # T1 --- load the weights for the croped model, compute a prediction to feed into the front end
    print('Loading T1 model')
    model_T1 = get_unet_cropped(verbose)
    T1_weightsPath = weights_path + "T1v1_weights.h5"
    model_T1.load_weights(T1_weightsPath, by_name=True)

    # T2 --- load the weights for the croped model, compute a prediction to feed into the front end
    print('Loading T2 model')
    model_T2 = get_unet_cropped(verbose)
    T2_weightsPath = weights_path + "T2v1_weights.h5"
    model_T2.load_weights(T2_weightsPath, by_name=True)

    # FLAIR --- load the weights for the croped model, compute a prediction to feed into the front end
    print('Loading FLAIR model')
    model_flair = get_unet_cropped(verbose)
    flair_weightsPath = weights_path + "FLAIRv1_weights.h5"
    model_flair.load_weights(flair_weightsPath, by_name=True)
    return  model_T1, model_T2, model_flair

def get_backendPredictions(x, model_T1, model_T2, model_flair):
    print('Predicting T1')
    model_T1.compile(optimizer=Adam(lr=10e-5,decay=0.001), loss = weighted_softmax)
    T1_prediction = model_T1.predict(x['T1'])

    print('Predicting T2')
    model_T2.compile(optimizer=Adam(lr=10e-5,decay=0.001), loss = weighted_softmax)
    T2_prediction = model_T2.predict(x['T2'])

    print('Predicting FLAIR')
    model_flair.compile(optimizer=Adam(lr=10e-5,decay=0.001), loss = weighted_softmax)
    flair_prediction = model_flair.predict(x['FLAIR'])

    return  T1_prediction, T2_prediction, flair_prediction

def get_Unet_HeMIS_predictions(img):
    model_T1, model_T2, model_flair = load_backendModels(verbose = 0)
    T1_prediction, T2_prediction, flair_prediction = get_backendPredictions(img, model_T1, model_T2, model_flair)

    model_front = get_frontNet()
    model_front.load_weights(os.getcwd()+"/Models/Model1/front_model_weights.h5")
    abstract =  get_abstractionLayerNP(flair_prediction, T1_prediction, T2_prediction)
    UHeMIS_prediction = model_front.predict(abstract, verbose=1)

    return UHeMIS_prediction


def maintest(slice = 0):
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    x_test, y_test = load_data_all(imgs_name = "Validate/imgs_validate_256_15_", gt_name = "Validate/gt_validate_256_15.npy")

    ns= slice+1
    img = {}
    img['T1']    = x_test['T1'][slice:ns,:,:,:]
    img['T2']    = x_test['T1'][slice:ns,:,:,:]
    img['FLAIR'] = x_test['FLAIR'][slice:ns,:,:,:]
    label = y_test[slice,:,:,0]
    print("img shape: ", img['T1'].shape)

    logits = get_Unet_HeMIS_predictions(img)
    logits = logits[0,:,:,:]

    print ("Logits:", logits)
    logits = np.argmax(logits, axis=2)
    print (logits.shape)
    visualize(img,label,logits, index=0)

def main():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    # load all test and train models in /data folder
    x_train, y_train = load_data_all(imgs_name = "/imgs_train_256_40_", gt_name = "/gt_train_256_40.npy")
    model_T1, model_T2, model_flair = load_backendModels(verbose = 1)
    T1_prediction, T2_prediction, flair_prediction = get_backendPredictions(x_train, model_T1, model_T2, model_flair)


    # Compute the mean and varance of the input masks, and output the concatnation Mean-Variance
    abstract =  get_abstractionLayerNP(flair_prediction, T1_prediction, T2_prediction)
    # keras requires same number of inputs as outputs, so concatonate y_train to same size as abstract
    y_train = np.concatenate([y_train,y_train], axis=0)
    model_front = get_frontNet()
    train_front(abstract, y_train, model_front)

if __name__ == '__main__':
    main()
    ##for testing
    #maintest()

