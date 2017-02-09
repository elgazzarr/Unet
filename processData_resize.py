import numpy as np
import nrrd
import glob
import os
import cv2
from scipy.misc import imresize
from cv2 import resize


# Desired Modalility to produce data eg. FLAIR,T1,T2
modality = "FLAIR"
#modality = "T1"
#modality = "T2"
test_WD  = "/Data/Train/Train_small_2/" # only the first two images
train_WD = "/Data/Train/Train_small/"
path = os.getcwd()
Data_path =  path + train_WD + modality
Gt_path   =  path + train_WD + "LESION"
volumes = []
GroundTruth = []

# Number of desired volumes to be sliced and produced (max = 5 for ms_lesion Dataset)
n = 2
scale = 0.25
width  = int(512 * scale)
length = int(512 * scale)
size = (width,length)
nchannels = 1
height = 512
nsamples = n*height
slices = np.ones((nsamples,size[0],size[1],nchannels), dtype = np.int16)
slices_gt = np.ones((nsamples,size[0],size[1],nchannels), dtype = np.int16)

#slices = np.ones((nsamples,size[0],size[1]), dtype = np.int16)
#slices_gt = np.ones((nsamples,size[0],size[1]), dtype = np.int16)

for filename in sorted(glob.glob(Gt_path + "/*.nhdr")):
    gt,header=nrrd.read(filename)
    print(filename)
    GroundTruth.append(gt)
GroundTruth = np.asarray(GroundTruth,dtype=np.int16)

for filename in sorted(glob.glob(Data_path + "/*.nhdr")):
    volume,header=nrrd.read(filename)
    print(filename)
    volumes.append(volume)
volumes = np.asarray(volumes,dtype=np.int16)


n = 0
for volume in volumes:
    for i in range(height):
        sl_vl = cv2.resize(volume[:,:,i],size,interpolation = cv2.INTER_LINEAR)
        #sl_vl = volume[:,:,i]
        slices[i+n*height,:,:,0]= sl_vl
        #print(i+n*height)
        #slices[n,:,:,0]= sl_vl
    n+=1
    
n = 0
for gt in GroundTruth:
    for i in range(height):
        sl_gt = cv2.resize(gt[:,:,i],size,interpolation = cv2.INTER_NEAREST)
        #sl_gt = gt[:,:,i]
        slices_gt[i+n*height,:,:,0]= sl_gt
    n+=1

dim = int(512*scale)

print (slices.shape)
print  (slices_gt.shape)
print('Loading done.')

data_save = "imgs_train_" +str(dim)+"_12"+ modality
gt_save   = "gt_train_" +str(dim)+"_12"
np.save(data_save, slices)
np.save(gt_save, slices_gt)

print('Saving to .npy files done.')
