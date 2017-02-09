
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt





path = os.getcwd()
x = np.load(path + "/data/gt_train_256_15.npy")


print(x.flatten().shape)

print x.shape

#plt.imshow(x[3,:,:,0], cmap='gray')

plt.show()
