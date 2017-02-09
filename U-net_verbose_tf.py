from __future__ import print_function

import cv2
import os
import numpy as np
from keras.metrics import fscore, precision,fmeasure,recall,fbeta_score
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras import backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import tensorflow as tf
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import csv
from keras.utils.np_utils import to_categorical
import math
import pandas

def load_traindata():
    path = os.getcwd()
    x = np.load(path + "/data/imgs_train_256_15_T1.npy")
    y = np.load(path + "/data/gt_train_256_15.npy")
    print (x.shape)
    print (y.shape)
    #imgs_train = imgs_train.astype('float32')
    #imgs_mask_train = imgs_mask_train.astype('float32')
    return  x, y


def load_testdata():
    path = os.getcwd()
    imgs_test=np.load(path + "/data/imgs_train_128_3FLAIR.npy")
    imgs_mask_test=np.load(path + "/data/gt_train_128_3.npy")
    #imgs_test = imgs_test.astype('float32')
    #imgs_mask_test = imgs_mask_test.astype('float32')
    return  imgs_test , imgs_mask_test

#Loss Function
def dice_coef(y_true, y_pred):

    labels = y_true
    labels = tf.to_int64(labels)
    labels = tf.one_hot(labels,2)
    flat_labels= tf.reshape(labels, [-1, 2])
    flat_logits = tf.reshape(y_pred, (-1, 2))
    intersection = tf.reduce_sum(flat_logits * flat_labels, 1, keep_dims=True)
    union = tf.reduce_sum(tf.mul(flat_logits, flat_logits), 1, keep_dims=True) \
                    + tf.reduce_sum(tf.mul(flat_labels, flat_labels), keep_dims=True)
    return 2 * intersection/ (union)



def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def sigmoid_cross_weighted(y_true,y_pred):
    y_p = tf.to_int64(y_true)
    y_p = tf.reshape(y_pred, (-1, 2))
    y_t = tf.to_int64(y_true)
    y_t = K.flatten(y_t)
    y_t = tf.one_hot(y_t,2)
    cross_entropy =  tf.nn.weighted_cross_entropy_with_logits(y_p, y_t, 0.001, name=None)
    return tf.reduce_mean(cross_entropy, name = 'class_cross_entropy')


def softmax_cross(y_true,y_pred):

    print ("labels Shape: ", y_true.get_shape()) # shape [batch_size*length*width, 1]
    ratio = 1 / (100)
    class_weight = tf.constant([ratio*1.0,1.0-ratio])
    logits = tf.reshape(y_pred, (-1, 2))# shape [batch_size*length*width, 2]
    print ("logits: ", tf.Print(logits,[logits])) # shape [batch_size,length,width, 2]
    weighted_logits = tf.mul(logits, class_weight) # shape [batch_size*length*width, 2]
    print ("Weighted logits: ", tf.Print(weighted_logits,[weighted_logits])) # shape [batch_size,length,width, 2]
    labels = tf.to_int64(y_true)
    labels = K.flatten(labels)# shape [batch_size*length*width]   Flattening
    print ("labels: ", tf.Print(labels,[labels]))
    labels = tf.one_hot(labels,2)# shape [batch_size*length*width, 2]   One hot encoding
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      weighted_logits, labels, name="None")

    return tf.reduce_mean(cross_entropy, name = 'class_cross_entropy')

def softmax_cross2(y_true,y_pred):

    print ("labels Shape: ", y_true.get_shape()) # shape [batch_size*length*width, 1]
    #ratio = 1 / (100)
    class_weight = tf.constant([[1.0, 1000]])
    logits = tf.reshape(y_pred, (-1, 2))# shape [batch_size*length*width, 2]
    print ("logits: ", tf.Print(logits,[logits])) # shape [batch_size,length,width, 2]



    labels = tf.to_int64(y_true)
    labels = K.flatten(labels)# shape [batch_size*length*width]   Flattening
    print ("labels: ", tf.Print(labels,[labels]))
    labels = tf.one_hot(labels,2)# shape [batch_size*length*width, 2]   One hot encoding
    weight_per_label = tf.transpose( tf.matmul(labels  , tf.transpose(class_weight)) ) #shape [1, batch_size]
    crossentropy = tf.mul(weight_per_label , tf.nn.softmax_cross_entropy_with_logits(logits, labels, name="xent_raw"))

    return tf.reduce_mean(crossentropy, name = 'class_cross_entropy')

def softmax_cross3(y_true,y_pred):
    class_weight = tf.constant(np.array([1, 1000], dtype='f'))


    labels = K.flatten(y_true)
    labels = tf.to_int64(labels)
    labels = tf.one_hot(labels,2)
    labels = tf.reshape(labels, [-1, 2])
    logits = tf.reshape(y_pred, (-1, 2))

    weight_map = tf.mul(labels, class_weight)

    weight_map = tf.reduce_sum(weight_map, 1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    weighted_loss = tf.mul(loss_map, weight_map)

    loss = tf.reduce_mean(weighted_loss)

    return loss







#Network Architecture
def get_unet():


    img_rows = 256
    img_cols = 256
    #inputs = Input(( 1, img_rows, img_cols))
    inputs = Input(( img_rows, img_cols, 1))
    print('inputs:', inputs)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(inputs)
    print('conv1_1:', conv1)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(conv1)
    print('conv1_2:', conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),dim_ordering="tf")(conv1)
    print('pool1:', pool1)

    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(pool1)
    print('conv2_1:', conv2)
    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(conv2)
    print('conv2_2:', conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),dim_ordering="tf")(conv2)
    print('pool2:', pool2)
	
    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(pool2)
    print('conv3_1:', conv3)
    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(conv3)
    print('conv3_2:', conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),dim_ordering="tf")(conv3)
    print('pool3:', pool3)

    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(pool3)
    print('conv4:', conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(conv4)
    print('conv4_2:', conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),dim_ordering="tf")(conv4)
    print('pool4:', pool4)

    convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(pool4)
    print('convdeep_1:', convdeep)
    convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(convdeep)
    print('convdeep_2:', convdeep)
    upConv_mid = UpSampling2D(size=(2, 2),dim_ordering="tf")(convdeep)
    print('upConv_mid_1:', upConv_mid)
    upConv_mid = Convolution2D(512, 2, 2, border_mode='same',dim_ordering="tf")(upConv_mid)
    print('upConv_mid_2:', upConv_mid)
    upConv_mid = merge([upConv_mid, conv4], mode='concat', concat_axis=3)
    print('upConv_mid_3:', upConv_mid)
    convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(upConv_mid)
    print('convmid_1:', convmid)
    convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(convmid)
    print('convmid_1:', convmid)

    up6 = merge([Convolution2D(256, 2, 2,activation='relu', border_mode='same',dim_ordering="tf")(UpSampling2D(size=(2, 2),dim_ordering="tf")(convmid)), conv3], mode='concat', concat_axis=3)
    print('up6:', up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(up6)
    print('conv6_1:', conv6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(conv6)
    print('conv6:_2', conv6)

    up7 = merge([Convolution2D(128, 2, 2,activation='relu', border_mode='same',dim_ordering="tf")(UpSampling2D(size=(2, 2),dim_ordering="tf")(conv6)), conv2], mode='concat', concat_axis=3)
    print('up7:', up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(up7)
    print('conv7_1:', conv7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(conv7)
    print('conv7_2:', conv7)

    up8 = merge([Convolution2D(64, 2, 2,activation='relu', border_mode='same',dim_ordering="tf")(UpSampling2D(size=(2, 2),dim_ordering="tf")(conv7)), conv1], mode='concat', concat_axis=3)
    print('up8:', up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(up8)
    print('conv8_1', conv8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(conv8)
    print('conv8_2', conv8)


    conv9 = Convolution2D(2, 1, 1, activation='relu',dim_ordering="tf")(conv8)


    print('conv9:', conv9)

    print('inputs: ', inputs)
    model = Model(input=inputs, output=conv9)

    #model.load_weights(os.getcwd()+'/u2.hdf5')

    model.compile(optimizer=Adam(lr=10e-5,decay=0.001), loss = softmax_cross3, metrics=["accuracy","precision", "recall", "fmeasure",])

    #model.compile(optimizer=Adam(lr=10e-5), loss="binary_crossentropy", metrics=["accuracy","precision", "recall", "fscore"])

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
    # summarize precision vs recall
    #plt.plot(history.recall, history.prec)
    #plt.title('precision recall')
    #plt.ylabel('precision')
    #plt.xlabel('recall')
    #plt.show()
    # summarize history of fscore
    #plt.plot(history.fscore)
    #plt.title('history of fscore')
    #plt.ylabel('fmeasure')
    #plt.xlabel('epoch')
    #plt.show()



 
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
        self.dice = []
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.val_accs.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))
        self.prec.append(logs.get('precision'))
        self.recall.append(logs.get('recall'))
        self.recall.append(logs.get('fscore'))

        return

    def on_batch_end(self, batch, logs={}):
        print ('\n')


        return
 

        
def train(imgs_train,gt_train,model):

    # print(np.histogram(imgs_train))
    # print(np.histogram(imgs_mask_train))
    history = Histories()
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    pandas.DataFrame( model.fit(imgs_train, gt_train, batch_size=4, nb_epoch=70, verbose=1, shuffle=True,callbacks=[model_checkpoint, history], validation_split=0.2).history).to_csv("history.csv")

    print('-'*30)
    print('saving weights and model as .h5')
    model.save('backend_model_70epochs.h5')
    model.save_weights('backend_model_weights_70epochs.h5')
    print( history)
    plt_history(history)
    np.savetxt('model_history.txt', np.array(history).reshape(1,),  delimiter=" ", fmt="%s")

    print('-'*30)

    print('-'*30)
    print('Training Done')
    print('-'*30)

#Compute a predicted label------------------------------------------------------------------------------------------
def test(imgs_test,model):

    print('-'*30)
    print('Predicting...')
    print('-'*30)
    model_checkpoint = ModelCheckpoint(model, monitor='loss',verbose=1, save_best_only=True)
    gt_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_predicted.npy', gt_test)
    return gt_test


def main():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    x_train,y_train = load_traindata()

    model = get_unet()



    train(x_train,y_train,model)






if __name__ == '__main__':
    main()
    ##for debuging the data
    #maintest()

