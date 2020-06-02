from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import cv2
from libtiff import TIFF
import skimage.transform as trans

class myAugmentation(object):
	
	"""
	一个用于图像增强的类：
	首先：分别读取训练的图片和标签，然后将图片和标签合并用于下一个阶段使用
	然后：使用Keras的预处理来增强图像
	最后：将增强后的图片分解开，分为训练图片和训练标签
	"""
	def __init__(self, train_path="/home/x/lzd/UNetPlusPlus-master/new/Aug_image", label_path="/home/x/lzd/UNetPlusPlus-master/new/Aug_label", merge_path="/home/x/lzd/UNetPlusPlus-master/ISBI2017/merge", aug_merge_path="/home/x/lzd/UNetPlusPlus-master/ISBI2017/aug_merge", aug_train_path="/home/x/lzd/UNetPlusPlus-master/ISBI2017/aug_train", aug_label_path="/home/x/lzd/UNetPlusPlus-master/ISBI2017/aug_label", img_type="tif"):
		"""
		使用glob从路径中得到所有的“.img_type”文件，初始化类：__init__()
		"""
		self.train_imgs = glob.glob(train_path+"/*."+img_type)
		self.label_imgs = glob.glob(label_path+"/*."+img_type)
		self.train_path = train_path
		self.label_path = label_path
		self.merge_path = merge_path
		self.img_type = img_type
		self.aug_merge_path = aug_merge_path
		self.aug_train_path = aug_train_path
		self.aug_label_path = aug_label_path
		self.slices = len(self.train_imgs)
		self.datagen = ImageDataGenerator(
							        rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        shear_range=0.05,
							        zoom_range=0.05,
							        horizontal_flip=True,
							        fill_mode='nearest')

	def Augmentation(self):

		"""
		Start augmentation.....
		"""
		trains = self.train_imgs
		labels = self.label_imgs
		path_train = self.train_path
		path_label = self.label_path
		path_merge = self.merge_path
		imgtype = self.img_type
		path_aug_merge = self.aug_merge_path
		if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
			print ("trains can't match labels")
			return 0
		for i in range(len(trains)):
			img_t = load_img(path_train+"/"+str(i)+"."+imgtype)
			img_l = load_img(path_label+"/"+str(i)+"."+imgtype)
			x_t = img_to_array(img_t)
			x_l = img_to_array(img_l)
			x_t[:,:,2] = x_l[:,:,0]
			img_tmp = array_to_img(x_t)
			img_tmp.save(path_merge+"/"+str(i)+"."+imgtype)
			img = x_t
			img = img.reshape((1,) + img.shape)
			savedir = path_aug_merge + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			self.doAugmentate(img, savedir, str(i))

	def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=900):
        # 增强一张图片的方法
		"""
		augmentate one image
		"""
		datagen = self.datagen
		i = 0
		for batch in datagen.flow(img,
                          batch_size=batch_size,
                          save_to_dir=save_to_dir,
                          save_prefix=save_prefix,
                          save_format=save_format):
		    i += 1
		    if i > imgnum:
		        break

	def splitMerge(self):
        # 将合在一起的图片分开
		"""
		split merged image apart
		"""
		path_merge = self.aug_merge_path
		path_train = self.aug_train_path
		path_label = self.aug_label_path

		for i in range(self.slices):
			path = path_merge + "/" + str(i)
			train_imgs = glob.glob(path+"/*."+self.img_type)
			savedir = path_train + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			savedir = path_label + "/" + str(i)

			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			for imgname in train_imgs:
				midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
				img = cv2.imread(imgname)
				img_train = img[:,:,2] #cv2 read image rgb->bgr
				img_label = img[:,:,0]
				cv2.imwrite(path_train+"/"+str(i)+"/"+midname+"_train"+"."+self.img_type,img_train)
				cv2.imwrite(path_label+"/"+str(i)+"/"+midname+"_label"+"."+self.img_type,img_label)

	def splitTransform(self):
        # 拆分透视变换后的图像
		"""
		split perspective transform images
		"""
		#path_merge = "transform"
		#path_train = "transform/data/"
		#path_label = "transform/label/"

		path_merge = "deform/deform_norm2"
		path_train = "deform/train/"
		path_label = "deform/label/"

		train_imgs = glob.glob(path_merge+"/*."+self.img_type)
		for imgname in train_imgs:
			midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
			img = cv2.imread(imgname)
			img_train = img[:,:,2]#cv2 read image rgb->bgr
			img_label = img[:,:,0]
			cv2.imwrite(path_train+midname+"."+self.img_type,img_train)
			cv2.imwrite(path_label+midname+"."+self.img_type,img_label)

class dataProcess(object):
	def __init__(self, out_rows, out_cols,
				 data_path = "/media/x/lzd/UNet++/256X192/au_image",
				 label_path = "/media/x/lzd/UNet++/256X192/au_label",
				 validation_path = "/home/x/lzd/UNetPlusPlus-master/ISIC2017/validation_image",
				 validation_label_path = "/home/x/lzd/UNetPlusPlus-master/ISIC2017/validation_label",
				 test_path = "/media/x/lzd/UNet++/256X192/test_image",
				 mask_path = "/media/x/lzd/UNet++/256X192/test_label",
				 npy_path = "/home/x/lzd/UNetPlusPlus-master/data/ISBI2017/10000",
				 img_type = "tif"):
        # 数据处理类，初始化
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.mask_path = mask_path
		self.npy_path = npy_path
		self.validation_path = validation_path
		self.validation_label_path = validation_label_path
# 创建训练数据
	def create_train_data(self):
		i = 0
		print('-'*100)
		print('Creating training images...')
		print('-'*100)
		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		print(len(imgs))
		#crop_size= (256,256)
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.data_path + "/" + midname,grayscale = False)
			#img =img.resize((256,256))
			label = load_img(self.label_path + "/" + midname,grayscale = True)
			#label = label.resize((256,256))
			img = img_to_array(img)
			label = img_to_array(label)
			#img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = cv2.resize(img, crop_size, interpolation = cv2.INTER_CUBIC)
			#label = cv2.resize(label, crop_size, interpolation = cv2.INTER_CUBIC)
			#img = np.array([img])
			#label = np.array([label])
			imgdatas[i] = img[:,:,:]
			imglabels[i] = label[:,:,:]
			if i % 100 == 0:
				print('Done: {0}/{1} train_images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to .npy files done.')

# 创建验证数据
	def create_validation_data(self):
		i = 0
		print('-'*150)
		print('Creating test images...')
		print('-'*150)
		imgs = glob.glob(self.validation_path+"/*."+self.img_type)
		print(len(imgs))
		#crop_size= (256,256)
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.validation_path + "/" + midname,grayscale = False)
			img = img.resize((256,256))
			label = load_img(self.validation_label_path + "/" + midname,grayscale = True)
			label = label.resize((256,256))
			img = img_to_array(img)
			label = img_to_array(label)
			#img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = cv2.resize(img, crop_size, interpolation = cv2.INTER_CUBIC)
			#img = np.array([img])
			imgdatas[i] = img[:,:,:]
			imglabels[i] = label[:,:,:]
			if i % 10 == 0:
				print('Done: {0}/{1} test_images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_validation.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_validation.npy', imglabels)
		print('Saving to imgs_validation.npy and imgs_mask_validation.npy files done.')
# 创建测试数据
	def create_test_data(self):
		i = 0
		print('-'*379)
		print('Creating test images...')
		print('-'*379)
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		print(len(imgs))
		#crop_size= (256,256)
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.test_path + "/" + midname,grayscale = False)
			#img = img.resize((512,512))
			label = load_img(self.mask_path + "/" + midname,grayscale = True)
			#label = label.resize((512,512))
			img = img_to_array(img)
			label = img_to_array(label)
			#img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = cv2.resize(img, crop_size, interpolation = cv2.INTER_CUBIC)
			#img = np.array([img])
			imgdatas[i] = img[:,:,:]
			imglabels[i] = label[:,:,:]
			if i % 100 == 0:
				print('Done: {0}/{1} test_images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_test.npy', imglabels)
		print('Saving to imgs_test.npy and imgs_mask_test.npy files done.')

# 加载训练图片与mask
	def load_train_data(self):
		print('-'*8000)
		print('load train images...')
		print('-'*8000)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		mean = imgs_train.mean(axis = 0)
		imgs_train -= mean
		imgs_mask_train /= 255
        # 做一个阈值处理，输出的概率值大于0.5的就认为是对象，否则认为是背景
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train,imgs_mask_train

# 加载验证图片与mask
	def load_validation_data(self):
		print('-'*150)
		print('load test images...')
		print('-'*150)
		imgs_validation = np.load(self.npy_path+"/imgs_validation.npy")
		imgs_mask_validation = np.load(self.npy_path+"/imgs_mask_validation.npy")
		imgs_validation = imgs_validation.astype('float32')
		imgs_mask_validation = imgs_mask_validation.astype('float32')
		imgs_validation /= 255
		mean = imgs_validation.mean(axis = 0)
		imgs_validation -= mean
		imgs_mask_validation /= 255
	# 做一个阈值处理，输出的概率值大于0.5的就认为是对象，否则认为是背景
		imgs_mask_validation[imgs_mask_validation > 0.5] = 1
		imgs_mask_validation[imgs_mask_validation <= 0.5] = 0
		return imgs_validation,imgs_mask_validation

# 加载测试图片与mask
	def load_test_data(self):
		print('-'*600)
		print('load test images...')
		print('-'*600)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_mask_test = np.load(self.npy_path+"/imgs_mask_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_mask_test = imgs_mask_test.astype('float32')
		imgs_test /= 255
		mean = imgs_test.mean(axis = 0)
		imgs_test -= mean
		imgs_mask_test /= 255
	# 做一个阈值处理，输出的概率值大于0.5的就认为是对象，否则认为是背景
		imgs_mask_test[imgs_mask_test > 0.5] = 1
		imgs_mask_test[imgs_mask_test <= 0.5] = 0
		return imgs_test,imgs_mask_test


if __name__ == "__main__":
# 以下注释掉的部分为数据增强代码，通过他们可以将数据进行增强

	#aug = myAugmentation()
	#aug.Augmentation()
	#aug.splitMerge()
	#aug.splitTransform()

	mydata = dataProcess(192,256)
	#mydata.create_train_data()
	#mydata.create_validation_data()
	mydata.create_test_data()
	#imgs_train,imgs_mask_train = mydata.load_train_data()
	#imgs_test,imgs_mask_test = mydata.load_test_data()
#print (imgs_train.shape,imgs_mask_train.shape,imgs_test.shape,imgs_mask_test.shape)
