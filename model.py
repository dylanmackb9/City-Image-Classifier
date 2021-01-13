# Convolutional neural netork in tensorflow2

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



#print(tf.__version__)


def cnn(x_train, y_train, activ, lr, epo):
	cNN = keras.Sequential(
		[
		keras.Input(shape=(400,700,3)),  # 1642 by 2880 pixel images with RGBA 
		layers.Conv2D(32, (4,7), padding="valid", activation=activ),  # 32 out channels, 3x3 filter kernel size, valid padding allows shrinking
		layers.MaxPool2D(pool_size=(2,2)),  # x pooling layer with pool size 2
		layers.Conv2D(64, (4,7), padding="valid", activation=activ),
		layers.MaxPool2D(pool_size=(2,2)),
		layers.Conv2D(128, (4,7), padding="valid", activation=activ),
		layers.MaxPool2D(pool_size=(2,2)),
		layers.Flatten(),
		layers.Dense(50, activation=activ),  # fully connected
		layers.Dense(25, activation=activ),
		layers.Dense(2), 
		]	

	)

	cNN.compile(

	loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # True from logits giving softmax on output layer
	optimizer = keras.optimizers.Adam(learning_rate=lr),
	metrics = ['accuracy'],
	)

	cNN.fit(x_train, y_train, batch_size=64, epochs=epo, verbose=2)

	return cNN






	



