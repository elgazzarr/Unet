import numpy as np
import nrrd
import glob
import os
import cv2
import sys

# Desired Modalility to produce data eg. FLAIR,T1,T2
#modality = "FLAIR"
#modality = "T1"
modality = "T2"
test_WD  = "/Data/Test/Test_all/"
train_WD = "/Data/Train/Train_all/"
path = os.getcwd()
Data_path =  path + train_WD + modality
Gt_path   =  path + train_WD + "LESION"

# if = true: find slices for each n volume which contain lesion
# NOT WORKING -- if = 2: load presaved lesion_index from tex file -- use if already ran on another modality
# if = false: use all images in each volume ( don't search for lesion )
get_lesion_index = 1

# if = true  : randomly subsample from lesion indexes output_height number of images to use
# if = 2     : load from file
# if = false : use all images in each volume ( don't subsample, and disregard output_height)
get_samples = 2

# Number of samples to take from each volume
output_height = 15

# Number of desired volumes to be sliced and produced (max = 5 for ms_lesion Dataset)
n = 5

# ammount to down sample the volumes by
scale = 0.5

volumes = []
GroundTruth = []
width  = int(512 * scale)
length = int(512 * scale)
size = (width,length)
nchannels = 1
height = 512

# Load data from file
for filename in sorted(glob.glob(Data_path + "/*.nhdr")):
    volume,header=nrrd.read(filename)
    print(filename)
    volumes.append(volume)
volumes = np.asarray(volumes,dtype=np.int16)

for filename in sorted(glob.glob(Gt_path + "/*.nhdr")):
    gt,header=nrrd.read(filename)
    print(filename)
    GroundTruth.append(gt)
GroundTruth = np.asarray(GroundTruth,dtype=np.int16)

# find image slices in each volume which contain a MS lesion and return the indexes
if get_lesion_index == 1:
    lesion_index = []
    n = 0
    for gt in GroundTruth:
        tmp = []
        for i in range(height):
            sl = cv2.resize(gt[:,:,i],size,interpolation = cv2.INTER_NEAREST)
            if np.amax(sl, axis=None) == 1:
                tmp.append(i)
        lesion_index.append(tmp)
        n+=1
    #index_save_name = "lesion_index"+str(width)
    #lesion_index = np.asarray(lesion_index,dtype=np.int16)
    #np.savetxt(index_save_name, lesion_index, fmt = '%d')
elif get_lesion_index == 2:
    index_path = 'lesion_index'+str(width)+'.txt'
    lesion_index = np.loadtxt(index_path, dtype = int)
else:
    lesion_index = []
    tmp = np.arange(height)
    for i in range(n):
        lesion_index.append(tmp)
    lesion_index = np.asarray(lesion_index,dtype=np.int16)
    
# randomly sample the indexes of lesion index and return ouput_height # of them
if get_samples == 1:
    samples = []
    for i in range(n):
        tmp = []
        high = len(lesion_index[i])-1
        if  output_height > high:
            print('ERROR - output height > high for volume:',i)
            sys.exit()
        select = np.arange(high)
        np.random.shuffle(select)
        for j in range(output_height):
            tmp.append(lesion_index[i][ select[j] ])
        samples.append(tmp)
        
    samples_save_name = 'samples_'+str(width)+'_'+str(output_height)+'.txt'
    samples = np.asarray(samples,dtype=np.int16)
    np.savetxt(samples_save_name, samples, fmt = '%d')
    
elif get_samples == 2:
    samples_load_name = 'samples_'+str(width)+'_'+str(output_height)+'.txt'
    samples = np.loadtxt(samples_load_name , dtype = int)
else:
    output_height = height
    samples = []
    tmp = np.arange(height)
    for i in range(n):
        samples.append(tmp)
    samples = np.asarray(samples,dtype=np.int16)

nsamples = n*output_height    
slices = np.ones((nsamples,size[0],size[1],nchannels), dtype = np.int16)
slices_gt = np.ones((nsamples,size[0],size[1],nchannels), dtype = np.int16)

n = 0
for gt in GroundTruth:
    for i in range(output_height):
        j = samples[n][i]
        sl = cv2.resize(gt[:,:,j],size,interpolation = cv2.INTER_NEAREST)
        slices_gt[i+n*output_height,:,:,0]= sl
    n+=1
    
n = 0
for volume in volumes:
    for i in range(output_height):
        j = samples[n][i]
        #print('image slice = ',j)
        sl = cv2.resize(volume[:,:,j],size,interpolation = cv2.INTER_LINEAR)
        slices[i+n*output_height,:,:,0]= sl
    n+=1
    

print (slices.shape)
print  (slices_gt.shape)
print('Loading done.')


data_save = "imgs_train_" +str(width)+"_"+str(output_height)+"_"+modality
gt_save   = "gt_train_" +str(width)+"_"+str(output_height)
np.save(data_save, slices)
np.save(gt_save, slices_gt)

print('Saving to .npy files done.')
