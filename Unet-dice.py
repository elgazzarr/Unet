from __future__ import print_function

import cv2
import os
import numpy as np
from keras.metrics import fscore,fmeasure,recall,fbeta_score,binary_accuracy,precision
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, EarlyStopping, CSVLogger
from keras import backend as K
import matplotlib as plt
import tensorflow as tf
import pandas

def load_traindata():
    path = os.getcwd()
    x = np.load(path + "/data/imgs_train_256_40_FLAIR.npy")
    y = np.load(path + "/data/gt_train_256_40.npy")
    print (x.shape)
    print (y.shape)
    #imgs_train = imgs_train.astype('float32')
    #imgs_mask_train = imgs_mask_train.astype('float32')
    return x,y


def process_data(x,y):
     x/=255.
     mean = x.mean()# (0)[np.newaxis,:]  # mean for data centering
     std = np.std(x)  # std for data normalization
     x -= mean
     x /= std
     y /= 255.
     return x,y

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



#Network Architecture
def get_unet():


    img_rows = 256
    img_cols = 256
    #inputs = Input(( 1, img_rows, img_cols))
    inputs = Input(( img_rows, img_cols, 1))
    print('inputs:', inputs)
    conv1 = Convolution2D(64, 3, 3, activation='relu',border_mode='same',dim_ordering="tf")(inputs)
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
    conv7 = Convolution2D(128, 3, 3, activation='relu',border_mode='same',dim_ordering="tf")(conv7)
    print('conv7_2:', conv7)

    up8 = merge([Convolution2D(64, 2, 2,activation='relu', border_mode='same',dim_ordering="tf")(UpSampling2D(size=(2, 2),dim_ordering="tf")(conv7)), conv1], mode='concat', concat_axis=3)
    print('up8:', up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu',border_mode='same',dim_ordering="tf")(up8)
    print('conv8_1', conv8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same',dim_ordering="tf")(conv8)
    print('conv8_2', conv8)


    conv9 = Convolution2D(1, 1, 1, activation="sigmoid", dim_ordering="tf")(conv8)


    print('conv9:', conv9)

    print('inputs: ', inputs)
    model = Model(input=inputs, output=conv9)

    #model.load_weights(os.getcwd()+'/u2.hdf5')

    model.compile(optimizer=Adam(lr=10e-5,decay=0.001), loss = dice_coef_loss, metrics=[dice_coef])

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

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.losses.append(logs.get('loss'))
        self.fscore.append(logs.get('fm'))
        print('/n')
        return




def train(imgs_train,gt_train,model):
    mod = "FLAIR"
    version = "dice"
    mod = mod + version


    # print(np.histogram(imgs_train))
    # print(np.histogram(imgs_mask_train))
    history = Histories()
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    c = CSVLogger(mod+'.log',separator=',')
    model_checkpoint = ModelCheckpoint((mod+'.hdf5'), monitor='loss',verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor = "val_loss",patience=3)
    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    pandas.DataFrame( model.fit(imgs_train, gt_train, batch_size=8, nb_epoch=50, verbose=1, shuffle=True,callbacks=[model_checkpoint, history,c], validation_split=0.2).history).to_csv(mod + "_history.csv")
    #h = model.fit(imgs_train, gt_train, batch_size=4, nb_epoch = 30 ,verbose=1, shuffle= False ,callbacks=[model_checkpoint,history,earlystopping,c] ,validation_split=0.2)
    print('-'*30)
    print('saving weights and model as .h5')
    model.save(mod + '.h5')
    model.save_weights(mod +'_weights.h5')

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

def visualize(img,label,logit):

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax1.title.set_text('Image')
    ax1.imshow(img, cmap='gray')
    ax2.title.set_text('Ground Truth')
    ax2.imshow(label, cmap='gray')
    ax3.title.set_text('FLAIR')
    ax3.imshow(logit, cmap='gray')

    plt.show()



def main():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    x_train,y_train = load_traindata()
    x_train,y_train = process_data(x_train,y_train)

    model = get_unet()



    train(x_train,y_train,model)


def maintest():
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    x,y = load_traindata()

    slice = 6
    ns= slice+1
    img = x[slice:ns,:,:,:]
    label = y[slice,:,:,0]

    print("img shape: ", img.shape)

    model = get_unet()
    model.load_weights(os.getcwd()+"/Models/v1/FLAIRv1_weights.h5")
    model.compile(optimizer=Adam(lr=10e-5,decay=0.001), loss = dice_coef_loss, metrics=[dice_coef])
    logits = model.predict(img)
    logits = logits[0,:,:,:]

    print ("Logits:", logits)
    logits = np.argmax(logits, 2)
    visualize(img[0,:,:,0],label,logits)







if __name__ == '__main__':
        #main()
    ##for testing
        maintest()
