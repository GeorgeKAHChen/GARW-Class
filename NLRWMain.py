import numpy as np

from keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Flatten, Dense
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from keras.layers import Layer
from keras import backend as K
from keras.engine.topology import Layer
from keras.datasets import mnist
from keras import optimizers

import numpy as np
np.random.seed(9621)

class NLDense(Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(NLDense, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(name = 'NLKernel', 
									  shape = (self.output_dim, input_shape[-1]), 
									  initializer = 'uniform',
									  trainable = True)
		super(NLDense, self).build(input_shape)

	def call(self, inputs):
		import tensorflow as tf
		with tf.variable_scope('pairwise_dist'):
			# squared norms of each row in A and B
			na = tf.reduce_sum(tf.square(inputs), 1)
			nb = tf.reduce_sum(tf.square(self.kernel), 1)

			# na as a row and nb as a column vectors
			na = tf.reshape(na, [-1, 1])
			nb = tf.reshape(nb, [1, -1])

			# return pairwise euclidead difference matrix
			D = tf.exp(-0.05 * tf.sqrt(tf.maximum(na - 2*tf.matmul(inputs, self.kernel, False, True) + nb, 0.0)))
			SumD = tf.reduce_sum(D, 1)
			outputs = tf.divide(D, SumD)
		return D

	def compute_output_shape(self, input_shape):
		output_shape = list(input_shape)
		output_shape[-1] = self.output_dim
		print(output_shape)
		return tuple(output_shape)



class NLRWDense(Layer):
	def __init__(self, 
				output_dim, 
				distant_parameter = 0.05, 
				**kwargs):
		self.output_dim = output_dim
		self.distant_parameter = distant_parameter
		super(NLRWDense, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(name = 'NLKernel', 
									  shape = (self.output_dim, input_shape[-1]), 
									  initializer = 'uniform',
									  trainable = True)
		super(NLRWDense, self).build(input_shape)

	def call(self, inputs):
		import tensorflow as tf
		test = True
		with tf.variable_scope('pairwise_dist'):
			# squared norms of each row in A and B
			na = tf.reduce_sum(tf.square(inputs), 1)
			nb = tf.reduce_sum(tf.square(self.kernel), 1)

			# na as a row and nb as a column vectors
			na = tf.reshape(na, [-1, 1])
			nb = tf.reshape(nb, [1, -1])
		
			# return pairwise euclidead difference matrix
			Tul = tf.exp(self.distant_parameter * tf.sqrt(tf.maximum(na - 2*tf.matmul(inputs, self.kernel, False, True) + nb, 0.0)))
			SumTul = tf.reduce_sum(Tul, 1)
			SumTul = tf.reshape(SumTul, [-1, 1])
			Tuu = tf.exp(self.distant_parameter * tf.sqrt(tf.maximum(na - 2*tf.matmul(inputs, inputs, False, True) + na, 0.0)))
			SumTuu = tf.reduce_sum(Tuu, 1)
			SumTuu = tf.reshape(SumTuu, [-1, 1])
			SumMatrix = tf.add(SumTul, SumTuu)
			Pul = tf.divide(Tul, SumMatrix)
			Puu = tf.divide(Tuu, SumMatrix)
			I = tf.eye(tf.shape(inputs)[0])
			outputs = tf.matmul(tf.linalg.inv(I - Puu), Pul, False, False)
			#outputs = tf.matmul(I - Puu, Pul, False, False)
		return outputs

	def compute_output_shape(self, input_shape):
		output_shape = list(input_shape)
		output_shape[-1] = self.output_dim
		print(output_shape)
		return tuple(output_shape)




def TestModel():
	InpLay = Input(shape=(3, 28, 28))
	Block1 = Conv2D(7, kernel_size=(3, 3),activation='linear', input_shape=(3, 28, 28), padding='same')(InpLay)
	Block1 = LeakyReLU(alpha=0.1)(Block1)
	Block1 = MaxPooling2D((4, 4),padding='same')(Block1)
	
	Block2 = Conv2D(14, (3, 3), activation='linear',padding='same')(Block1)
	Block2 = LeakyReLU(alpha=0.1)(Block2)
	Block2 = MaxPooling2D(pool_size=(4, 4),padding='same')(Block2)
	
	Block3 = Conv2D(28, (3, 3), activation='linear',padding='same')(Block2)
	Block3 = LeakyReLU(alpha=0.1)(Block3)
	Block3 = MaxPooling2D(pool_size=(4, 4),padding='same')(Block3)
	
	Finals = Flatten()(Block3)
	Finals = Dense(64, activation='linear')(Finals)
	Finals = LeakyReLU(alpha=0.1)(Finals)
	Finals = NLRWDense(10)(Finals)
	#Finals = NLDense(10)(Finals)
	#Finals = Dense(10, activation='softmax')(Finals)
	#Here we used random walk for classification

	model = Model(inputs = InpLay, outputs = Finals)
	model.summary()

	return model, Finals


def RWTrain():
	#Import MNIST dataset
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_new_train = []
	y_new_train = [[0 for n in range(10)] for n in range(len(y_train))]
	for i in range(0, len(x_train)):
		x_new_train.append([x_train[i], x_train[i], x_train[i]])
		y_new_train[i][y_train[i]] = 1
	x_new_train = np.array(x_new_train)
	y_new_train = np.array(y_new_train)
	
	#Import Model
	model, Output = TestModel()
	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#Using Stochastic gradient descent(SGD) for optimizer
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer = sgd)

	#Training model
	model.fit(x = x_new_train, y = y_new_train, validation_split=0.1, epochs = 100)
	model.save_weights("./Output/Model.h5")


if __name__ == '__main__':
	RWTrain()





