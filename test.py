import numpy as np
import keras as k

imgs_train = np.load('/home/x/lzd/UNetPlusPlus-master/data/ISBI2016/npy256/imgs_train.npy')
imgs_mask_train = np.load('/home/x/lzd/UNetPlusPlus-master/data/ISBI2016/npy256/imgs_mask_train.npy')
imgs_test = np.load('/home/x/lzd/UNetPlusPlus-master/data/ISBI2016/npy256/imgs_test.npy')
imgs_mask_test = np.load('/home/x/lzd/UNetPlusPlus-master/data/ISBI2016/npy256/imgs_mask_test.npy')
#imgs_validation = np.load('/home/x/lzd/UNetPlusPlus-master/data/ISBI2017/npy_512/imgs_validation.npy')
#imgs_mask_validation = np.load('/home/x/lzd/UNetPlusPlus-master/data/ISBI2017/npy_512/imgs_mask_validation.npy')

print(imgs_train.shape,imgs_mask_train.shape,imgs_test.shape,imgs_mask_test.shape)
#print(imgs_train)
