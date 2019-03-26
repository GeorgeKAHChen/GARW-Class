from copy import deepcopy
import GARW
from libpy import Init
import numpy as np
from PIL import Image
from scipy import misc


RWOutput = True
SignImg = 1


def GetFiles(FileDir, height, width): 
	import os

	RootFolder = ""
	Catagory = []
	XLs = []
	XUs = []
	XLSign = []
	XUSign = []
	ttl = -1
	
	for root, dirs, files in os.walk(FileDir):
		#Get direction and location
		if ttl == -1:
			RootFolder = root
			Catagory = deepcopy(dirs)
			ttl += 1
			continue

		for i in range(0, len(files)):
			FileName = RootFolder + "/" + Catagory[ttl] + "/" + files[i]
			img = GARW.RGBList2Table(misc.imresize(np.array(Image.open(FileName)), (height, width, 3)))
			#feature = model.predict([img])[len(feature) - 1]
			feature = img[0][0]			#Test Program without A3MCNN feature
			#print(feature)
			#Part XL, XU		
			if i >= 0 and i < SignImg:
				XLs.append(feature)
				XLSign.append(ttl)
			else:
				XUs.append(feature)
				XUSign.append(ttl)

		ttl += 1

	if len(XLs) == 0 or len(XUs) == 0:
		print("Input Parameter Error: SignImg")
		return
	#RW Classification	
	XURWSign = GARW.RandomLayer([XLs, XUs], kernel = "Gaussian", distance = "Euclid", para = [], method = "")
	
	Succeed = 0
	Output = []
	for i in range(0, len(XURWSign)):
		if XLSign[XURWSign[i]] == XUSign[i]:
			Succeed += 1
	
		if RWOutput == True:
			Output.append([XURWSign[i], XLSign[XURWSign[i]], XUSign[i], 1 if XLSign[XURWSign[i]] == XUSign[i] else 0])

	if RWOutput == True:
		Init.ArrOutput([Catagory])
		Init.ArrOutput([XLSign])
		Init.ArrOutput(Output)
		Init.ArrOutput([[Succeed, len(XURWSign), Succeed/len(XURWSign)]])

	print("\n\nRandom Classification finished, Total Image:" + str(len(XURWSign)) + ", Succeed: " + str(Succeed) + ".\n Accuracy: " + str(Succeed/len(XURWSign)))
	


def A3MLoadModel(net, batch_size, nb_epoch, dropout, lambdas, attr_equal, 
		region_equal, nb_classes, nb_attributes, img_rows, img_cols, L):
	#Import Files from keras
	import sys
	sys.path.append("..")
	sys.setrecursionlimit(10000)
	import numpy as np
	np.random.seed(2208)  # for reproducibility
	
	from keras.layers import Input, Dense, Permute, BatchNormalization, Lambda, Dense
	from keras.layers import Dropout, Activation, Reshape, dot, average, concatenate
	from keras.layers import Conv1D, GlobalAveragePooling1D

	from keras.models import Model
	from keras.optimizers import SGD
	from keras.utils import np_utils
	from keras import backend as K
	from keras.models import load_model
	from keras.applications.vgg16 import VGG16
	from keras.applications.resnet50 import ResNet50

	#Define Layer Block
	def LayersBlock(input_fea_map, dim_channel, nb_class, name=None):
		# conv
		fea_map = Conv1D(dim_channel, 1, padding='same')(share_fea_map)
		fea_map = BatchNormalization(axis=2)(fea_map)
		fea_map = Activation('relu')(fea_map)

		# pool
		pool = GlobalAveragePooling1D(name=name+'_avg_pool')(fea_map)
		pool = BatchNormalization()(pool)
		pool = Activation('relu')(pool)

		# classification
		prob = Dropout(dropout)(pool)
		prob = Dense(nb_class)(prob)
		prob = Activation(activation='softmax',name=name)(prob)
		return prob, pool, fea_map


	#Initial model paremeters	
	final_dim = 512 if net=='VGG16' else 2048
	emb_dim = 512
	shared_layer_name = 'block5_pool' if net=='VGG16' else 'activation_49'

	#Model define
	alphas = [lambdas[1]*1.0/len(nb_attributes)]*len(nb_attributes)
	loss_dict = {}
	weight_dict = {}

	#Input and output define
	inputs = Input(shape=(3, img_rows, img_cols))
	out_list = []

	#Build Shared CNN
	model_raw = eval(net)(input_tensor=inputs, include_top=False, weights='imagenet')
	share_fea_map = model_raw.get_layer(shared_layer_name).output
	share_fea_map = Reshape((final_dim, L), name='reshape_layer')(share_fea_map)        
	share_fea_map = Permute((2, 1))(share_fea_map) 

	#Loss-1: identity classification
	id_prob,id_pool,id_fea_map = LayersBlock(share_fea_map, emb_dim, nb_classes, name='p0')
	out_list.append(id_prob)
	loss_dict['p0'] = 'categorical_crossentropy'
	weight_dict['p0'] = lambdas[0]

	#Loss-2: attribute classification
	attr_fea_list = []
	for i in range(len(nb_attributes)):
		name ='attr'+str(i)
		attr_prob,attr_pool,_ = LayersBlock(share_fea_map, emb_dim, nb_attributes[i], name)
		out_list.append(attr_prob)
		attr_fea_list.append(attr_pool)
		loss_dict[name] = 'categorical_crossentropy'
		weight_dict[name] = alphas[i]

	#Attention generation
	region_score_map_list = []
	attr_score_list = []
	for i in range(len(nb_attributes)):
		attn1 = dot([id_fea_map,attr_fea_list[i]], axes=(2,1)) 
		fea_score = dot([id_pool,attr_fea_list[i]], axes=(1,1))
		region_score_map_list.append(attn1)
		attr_score_list.append(fea_score)

	#Regional feature fusion
	region_score_map = average(region_score_map_list, name='attn')
	region_score_map = BatchNormalization()(region_score_map)
	region_score_map = Activation('sigmoid', name='region_attention')(region_score_map)
	region_fea = dot([id_fea_map,region_score_map], axes=(1,1))
	region_fea = Lambda(lambda x: x*(1.0/L))(region_fea)
	region_fea = BatchNormalization()(region_fea)

	#Attribute feature fusion
	attr_scores = concatenate(attr_score_list)
	attr_scores = BatchNormalization()(attr_scores)
	attr_scores = Activation('sigmoid')(attr_scores)
	attr_fea = concatenate(attr_fea_list)
	attr_fea = Reshape((emb_dim, len(nb_attributes)))(attr_fea) 
	equal_attr_fea = GlobalAveragePooling1D()(attr_fea)
	attr_fea = dot([attr_fea,attr_scores], axes=(2,1))
	attr_fea = Lambda(lambda x: x*(1.0/len(nb_attributes)))(attr_fea)
	attr_fea = BatchNormalization()(attr_fea)

	#Loss-3: final classification
	if(attr_equal):
		attr_fea = equal_attr_fea
	if(region_equal):
		region_fea = id_pool

	final_fea = concatenate([attr_fea,region_fea])
	final_fea = Activation('relu', name='final_fea')(final_fea)
	final_fea = Dropout(dropout)(final_fea)
	final_prob = Dense(nb_classes)(final_fea)
	final_prob = Activation(activation='softmax',name='p')(final_prob)
	out_list.append(final_prob)
	loss_dict['p'] = 'categorical_crossentropy'
	weight_dict['p'] = lambdas[2]

	model = Model(inputs, out_list)
	model.summary()

	return model, loss_dict, weight_dict


def A3MTrain(data_folder, dataset, net, batch_size, nb_epoch, dropout, lambdas, 
	attr_equal, region_equal, nb_classes, nb_attributes, img_rows, img_cols, L, lr_list, model_weight_path):
	
	from keras.optimizers import SGD
	from keras.preprocessing.image import ImageDataGenerator
	import numpy as np
	import CUB

	model, loss_dict, weight_dict = A3MLoadModel(net, batch_size, nb_epoch, dropout, 
		lambdas, attr_equal, region_equal, nb_classes, nb_attributes, img_rows, img_cols, L)

	# the data, shuffled and split between train and test sets
	(X_train, y_train),(X_test, y_test),(A_train,A_test,C_A)=eval(dataset).load_data(
		data_folder, target_size=(img_rows, img_cols), bounding_box=True)

	print(X_train[100][1][50:60,100:110])
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)

	yA_train = np.concatenate((np.expand_dims(y_train,1), A_train), axis=1)
	yA_test = np.concatenate((np.expand_dims(y_test,1), A_test), axis=1)
	print('yA_train shape:', yA_train.shape)
	print('yA_test shape:', yA_test.shape)


	for lr in lr_list:
		if(lr==0.011):
			for layer in model.layers:
				if(layer.name=='reshape_layer'):
					break
				layer.trainable=False
		else:
			for layer in model.layers:
				layer.trainable=True
		opt = SGD(lr=lr, decay=5e-4, momentum=0.9, nesterov=True)
		model.compile(loss=loss_dict,
					  loss_weights=weight_dict,
					  optimizer=opt, metrics=['accuracy'])
		
		# data augment this will do preprocessing and realtime data augmentation
		datagen = ImageDataGenerator(
				featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=False,  # apply ZCA whitening
				rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
				width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
				zoom_range=[0.75,1.33],
				horizontal_flip=True,  # randomly flip images
				vertical_flip=False)  # randomly flip images
		
		# train for nb_epoch epoches
		for e in range(nb_epoch):
			batches = 1
			ave_loss = np.zeros(1+2*len(loss_dict))
			for X_batch, yA_batch in datagen.flow(X_train, yA_train, batch_size=batch_size):
				y_batch = yA_batch[:,:1]
				attr_batch = yA_batch[:,1:]
				label_batch_list = []
				label_batch_list.append(np_utils.to_categorical(y_batch, nb_classes))
				for i in range(len(nb_attributes)):
					label_batch_list.append(np_utils.to_categorical(attr_batch[:,i], nb_attributes[i]))
				label_batch_list.append(np_utils.to_categorical(y_batch, nb_classes))
				loss = model.train_on_batch(X_batch, label_batch_list)
			# print
			ave_loss = ave_loss*(batches-1)/batches + np.array(loss)/batches
			show_idx = [0,len(loss_dict)+1,len(loss_dict)+2,2*len(loss_dict)]
			sys.stdout.write('\rtrain-loss: %.4f, train-acc: %.4f %.4f %.4f'
						% tuple(ave_loss[show_idx].tolist()))
			sys.stdout.flush()
			batches += 1
			if batches > len(X_train)/batch_size:
				sys.stdout.write("\r  \r\n")
				break
		# test
		label_test_list = []
		label_test_list.append(np_utils.to_categorical(y_test, nb_classes))
		for i in range(len(nb_attributes)):
			label_test_list.append(np_utils.to_categorical(A_test[:,i], nb_attributes[i]))
		label_test_list.append(np_utils.to_categorical(y_test, nb_classes))
		scores = model.evaluate(X_test, label_test_list, verbose=0)
		print('\nval-loss: ',scores[:1+len(loss_dict)], '\nval-acc: ', scores[1+len(loss_dict):])
		print('Main acc: %f' %(scores[-1]))

	# save model
	model.save_weights(model_weight_path + net + str(lr)+'.h5')
	print('train stage:',lr,' sgd done!')

	return


def A3MTest(data_folder, dataset, net, batch_size, nb_epoch, dropout, lambdas, 
	attr_equal, region_equal, nb_classes, nb_attributes, img_rows, img_cols, L, lr_list, model_weight_path, savefile = "false"):
	
	import os
	import Init
	from copy import deepcopy
	import GARW

	model = model, loss_dict, weight_dict= A3MLoadModel(net, batch_size, nb_epoch, dropout, lambdas, attr_equal, 
		region_equal, nb_classes, nb_attributes, img_rows, img_cols, L, lr_list)

	model.load_weights(model_weight_path)

	ImageList = []
	RootFolder = ""
	ttl = -1
	imgs = 0

	for root, dirs, files in os.walk(FileDir):
		#Get direction and location
		if ttl == -1:
			RootFolder = root
			print(root)
			Catagory = deepcopy(dirs)
			ttl += 1
			continue

		for i in range(0, len(files)):
			FileName = RootFolder + "/" + Catagory[ttl] + "/" + files[i]
			#print(FileName)
			img = GARW.RGBList2Table( misc.imresize( np.array(Image.open(FileName)), (img_rows, img_cols, 3 )) )

			if img[0][0][0][0] == -1:
				continue

			imgs += 1
			if imgs % 1000 == 0 and imgs != 1:
				print("import image: " + str(imgs))

			ImageList.append(img)

		ttl += 1

	if savefile:
		Init.ArrOutput(model.predict(ImageList))

	return model.predict(ImageList)


def A3Model(Mode = "Train",
		data_folder = "./CUB_200_2011",
		dataset = "CUB",
		net = "VGG16",
		model_weight_path = './model/weights_resnet50_86.1.h5',
		batch_size = 10,
		nb_epoch = 10,
		dropout = 0.5,
		lambdas = [0.2,0.5,1.0],
		attr_equal = False,
		region_equal = False,
		nb_classes = 200,
		nb_attributes = [10, 16, 16, 16, 5, 16, 7, 16, 12, 16, 16, 15, 4, 16, 16, 16, 16, 6, 6, 15, 5, 5, 5, 16, 16, 16, 16, 5],
		img_rows = 448,
		img_cols = 448,
		L = 14*14,
		lr_list = [0.001,0.003,0.001,0.001,0.001,0.001,0.001,0.0001]):

	import os

	if Mode != "Train" or Mode != "Test":
		print("InputError: Input model must be detemined as 'Train' for model training or 'Test' for testing model")
	if Mode == "Train":
		if not os.path.exists(dataset + ".py"):
			print("InputError: Cannot find the file called " + dataset + ".py for input signs")
		#model_weight_path = "./Output"

		A3MTrain(data_folder, dataset, net, batch_size, nb_epoch, dropout, lambdas, 
			attr_equal, region_equal, nb_classes, nb_attributes, img_rows, img_cols, L, lr_list, model_weight_path)

	if Mode == "Test":
		A3MTest(data_folder, dataset, net, batch_size, nb_epoch, dropout, lambdas, 
			attr_equal, region_equal, nb_classes, nb_attributes, img_rows, img_cols, L, lr_list, model_weight_path)


if __name__ == '__main__':
	#GetFiles(FileDir = "./Input", height = 448, width = 448)
	A3Model(model_weight_path = "./Output")



