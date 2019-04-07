import numpy as np

from keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras import backend as K
from keras.layers import Layer
from keras import backend as K
from keras.engine.topology import Layer

"""
class MyLayer(Layer):

	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(MyLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		# 为该层创建一个可训练的权重
		self.kernel = self.add_weight(name='kernel', 
										shape=(input_shape[1], self.output_dim),
										initializer='uniform',
										trainable=True)
		super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

	def call(self, x):
		return K.dot(x, self.kernel)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)
"""


class RandomWalkLayer(Layer):
	def __init__(self, output_dim, Method = ["average", [0, ]], Parameter = [[0.01], "Gaussian", "Euclid"], **kwargs):
		self.method = Method[0]
		self.groups = Method[1]
		self.parameter = Parameter[0]
		self.output_dim = output_dim
		self.DistantKernel = Parameter[1]
		self.Distance = Parameter[1]
		super(RandomWalkLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(name='RWKernel', 
									  shape=(input_shape[1], self.output_dim), 
									  initializer='uniform',
									  trainable=True)
		self.InpSize = input_shape[1]
		super(RandomWalkLayer, self).build(input_shape)

	def call(self, x):
		#Get Euclid distance for the classification
		results = []
		for j in range(0, self.output_dim):
			kernels = []
			for i in range(0, self.InpSize):
				kernels.append(self.kernel[i][j])
			return K.sqrt(K.sum(K.square(kernels - x), axis=-1))
		
		return results

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)



def TestModel():
	InpLay = Input(shape=(3, 32, 32))
	Block1 = Conv2D(32, kernel_size=(3, 3),activation='linear', input_shape=(3,32,32), padding='same')(InpLay)
	Block1 = LeakyReLU(alpha=0.1)(Block1)
	Block1 = MaxPooling2D((4, 4),padding='same')(Block1)
	
	Block2 = Conv2D(64, (3, 3), activation='linear',padding='same')(Block1)
	Block2 = LeakyReLU(alpha=0.1)(Block2)
	Block2 = MaxPooling2D(pool_size=(4, 4),padding='same')(Block2)
	
	Block3 = Conv2D(128, (3, 3), activation='linear',padding='same')(Block2)
	Block3 = LeakyReLU(alpha=0.1)(Block3)
	Block3 = MaxPooling2D(pool_size=(4, 4),padding='same')(Block3)
	
	Finals = Flatten()(Block3)
	Finals = Dense(64, activation='linear')(Finals)
	Finals = LeakyReLU(alpha=0.1)(Finals)
	Finals = RandomWalkLayer(output_dim = 2)(Finals)
	#Finals = Dense(2, activation='softmax')(Finals)
	#Here we used random walk for classification

	model = Model(inputs = InpLay, outputs = Finals)
	model.summary()

	return model, InpLay, Finals


if __name__ == '__main__':
	#Import CNN model
	model, Input, Output =TestModel()

	#Import MNIST dataset
	from keras.datasets import cifar10
	(x_train, y_train),(x_test, y_test) = cifar10.load_data()
	X_Used = []
	Y_Used = []
	X_Sign = []
	Y_Sign = []

	for i in range(0, len(x_train)):
		if y_train[i][0] == 0 :
			X_Used.append(x_train[i])
			Y_Used.append([0])
		if y_train[i][0] == 1:
			X_Used.append(x_train[i])
			Y_Used.append([1])
	
	for i in range(0, len(x_test)):
		if y_test[i][0] == 0:
			X_Sign.append(x_test[i])
			Y_Sign.append([0])
		if y_test[i][0] == 1:
			X_Sign.append(x_test[i])
			Y_Sign.append([1])

	X_Used = np.array(X_Used)
	Y_Used = np.array(Y_Used)
	X_Sign = np.array(X_Sign)
	Y_Sign = np.array(Y_Sign)

	print(len(X_Used), len(X_Sign))

	model, Input, Output = TestModel()
	model.compile(loss=Output, metrics=['accuracy'])
	model.fit(x = [X_Used, Y_Used], y = [X_Sign, Y_Sign])
