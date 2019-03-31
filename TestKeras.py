from keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Flatten, Dense
from keras.models import Model

def TestModel():
	InpLay = Input(shape=(3, 448, 448))
	Block1 = Conv2D(32, kernel_size=(3, 3),activation='linear', input_shape=(1,28,28), padding='same')(InpLay)
	Block1 = LeakyReLU(alpha=0.1)(Block1)
	Block1 = MaxPooling2D((4, 4),padding='same')(Block1)
	
	Block2 = Conv2D(64, (3, 3), activation='linear',padding='same')(Block1)
	Block2 = LeakyReLU(alpha=0.1)(Block2)
	Block2 = MaxPooling2D(pool_size=(4, 4),padding='same')(Block2)
	
	Block3 = Conv2D(128, (3, 3), activation='linear',padding='same')(Block2)
	Block3 = LeakyReLU(alpha=0.1)(Block3)
	Block3 = MaxPooling2D(pool_size=(4, 4),padding='same')(Block3)
	
	Finals = Flatten()(Block3)
	Finals = Dense(1024, activation='linear')(Finals)
	Finals = LeakyReLU(alpha=0.1)(Finals)
	Finals = Dense(200, activation='softmax')(Finals)

	model = Model(inputs = InpLay, outputs = Finals)
	model.summary()

	return model, InpLay, Finals

TestModel()
