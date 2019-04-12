import numpy as np

from keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Flatten, Dense
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from keras.layers import Layer
from keras import backend as K
from keras.engine.topology import Layer
from keras.datasets import cifar10
from keras import optimizers
"""
from kerasref.keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Flatten, Dense
from kerasref.keras.layers import Input
from kerasref.keras.models import Model
from kerasref.keras import backend as K
from kerasref.keras.layers import Layer
from kerasref.keras import backend as K
from kerasref.keras.engine.topology import Layer
from kerasref.keras.datasets import cifar10
from kerasref.keras import optimizers
"""
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
									  shape=(self.output_dim, input_shape[1]), 
									  initializer='uniform',
									  trainable=True)
		self.InpSize = input_shape[1]
		super(RandomWalkLayer, self).build(input_shape)

	def call(self, inputs):
		#Get Euclid distance for the classification
		if self.InpSize == 1:
			#Testing Model with Distance Classifier
			for j in range(0, self.output_dim):
				#print(inputs.shape[0], inputs.shape[1], len(kernels))
				Partial.append(K.sqrt(K.sum(K.square(self.kernel[j] - x_elem), axis=-1)))
			results.append(Partial)
			return [results]
			#print(len(results), len(results[0]))

		elif self.InpSize <= 0:
			#Error Print
			ValueError("Input Training/Testing need one and more than one input")
		
		else:
			#Training Model with Fully Random Walk Classifier
			import RandomWalk			#Import Random Walk Functions for Training
		
			results = [[] for n in range(self.InpSize)]
			i = 0
			for x_elem in to_list(inputs):
				Partial = []
				for j in range(0, self.output_dim):
					#print(inputs.shape[0], inputs.shape[1], len(kernels))
					Partial.append(K.sqrt(K.sum(K.square(self.kernel[j] - x_elem), axis=-1)))
				results.append(Partial)
				print(len(results), len(results[0]))#, results[0].shape[0])
			return results
			#return K.square(self.kernel[0] - x_elem)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1], self.output_dim)

def WeightMain(i, inputs, kernel):
	import tensorflow as tf
	i += 1
	output = []
	for j in range(0, kernel.shape[0]):
		output.append(tf.sqrt(tf.reduce_sum(tf.square(inputs[i] - kernel[j]), axis = 0)))

	return i, tf.Variable(output, tf.float32)


def cond(i, inputs, kernel):
	print("qujimabi")
	print(inputs.shape[0])
	if i < inputs.shape[0]:
		return True
	else:
		return False
"""




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
		"""
		InputShape = inputs.get_shape().as_list()
		KernelShape = self.kernel.get_shape().as_list()
		outputs = []

		sess = tf.Session()

		for i in range(0, InputShape[0]):
			GroupDis = []
			for j in range(0, KernelShape[0]):
				euclidean_dist = (tf.sqrt(tf.reduce_sum(tf.square(input_tensor[i]-tensor_iter[j]), 1)))
				sess.run(euclidean_dist)
				euclidean_row = euclidean_dist.eval(session=sess)
				GroupDis.append(euclidean_row)
			outputs.append(GroupDis)
	
		return tf.convert_to_tensor(outputs)
		"""
		
		with tf.variable_scope('pairwise_dist'):
			# squared norms of each row in A and B
			na = tf.reduce_sum(tf.square(inputs), 1)
			nb = tf.reduce_sum(tf.square(self.kernel), 1)

			# na as a row and nb as a co"lumn vectors
			na = tf.reshape(na, [-1, 1])
			nb = tf.reshape(nb, [1, -1])

			# return pairwise euclidead difference matrix
			D = tf.sqrt(tf.maximum(na - 2*tf.matmul(inputs, self.kernel, False, True) + nb, 0.0))
		return D




	def compute_output_shape(self, input_shape):
		output_shape = list(input_shape)
		output_shape[-1] = self.output_dim - 1
		print(output_shape)
		return tuple(output_shape)




def TestModel():
	InpLay = Input(shape=(32, 32, 3))
	Block1 = Conv2D(32, kernel_size=(3, 3),activation='linear', input_shape=(32, 32, 3), padding='same')(InpLay)
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
	Finals = NLDense(2)(Finals)
	#Finals = Dense(2, activation='softmax')(Finals)
	#Here we used random walk for classification

	model = Model(inputs = InpLay, outputs = Finals)
	model.summary()

	return model, Finals


def RWTrain():
	#Import MNIST dataset
	(x_train, y_train),(x_test, y_test) = cifar10.load_data()
	X_Used = []
	Y_Used = []
	X_Sign = []
	Y_Sign = []

	for i in range(0, len(x_train)):
		if y_train[i][0] == 0 :
			X_Used.append(np.array(x_train[i]))
			Y_Used.append(0)
		if y_train[i][0] == 1:
			X_Used.append(np.array(x_train[i]))
			Y_Used.append(1)
	
	for i in range(0, len(x_test)):
		if y_test[i][0] == 0:
			X_Sign.append(np.array(x_test[i]))
			Y_Sign.append(0)
		if y_test[i][0] == 1:
			X_Sign.append(np.array(x_test[i]))
			Y_Sign.append(1)

	X_Used = np.array(X_Used)
	Y_Used = np.array(Y_Used)
	X_Sign = np.array(X_Sign)
	Y_Sign = np.array(Y_Sign)
	print(len(X_Used), len(X_Sign))


	#Import Model
	model, Output = TestModel()
	sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	#Using Stochastic gradient descent(SGD) for optimizer
	model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer = sgd)

	#Training model
	model.fit(x = X_Used, y = Y_Used, validation_split=0.2, epochs = 20)
	model.save_weights("./Output/Model.h5")


if __name__ == '__main__':
	RWTrain()





