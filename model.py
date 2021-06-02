import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# CNN
import model as model

#print(tf.__version__)



def cnn(x_train, y_train, activ, lr, epo):
	'''
	Convolutional neural network model pipeline. Layers are conv, pooling, and fc:
		3 2D convolution layers each with 5x5 filters, (2,2) stride movement, and valid (none) padding
		3 max pooling layers with (2,2) pool size
		3 fully connected layers 
	All with variable dependent activation functions

	Input:
		x_train -- m x 640 x 640 x 3 ndarray, training features.
		y_train -- m x 1 ndarray, training labels.
		activ -- str, activation function for model.
		lr -- Learning rate for model optimizer.
		epo -- Number of epochs model trains on.

	Output:
		cNN -- TensorFlow model object, trained model.

	'''

	cNN = keras.Sequential(
		[
		keras.Input(shape=(640,640,3)),  # 640 by 640 pixel images with RGB  
		layers.Conv2D(32, (5,5), strides=2, padding="valid", activation=activ),  # 32 out channels, 3x3 filter kernel size, valid padding allows shrinking
		layers.MaxPool2D(pool_size=(2,2)),  # x pooling layer with pool size 2
		layers.Conv2D(64, (5,5), strides=2, padding="valid", activation=activ),
		layers.MaxPool2D(pool_size=(2,2)),
		layers.Conv2D(128, (5,5), strides=2, padding="valid", activation=activ),
		layers.MaxPool2D(pool_size=(2,2)),
		layers.Flatten(),
		layers.Dense(1000, activation=activ),  # fully connected
		layers.Dense(25, activation=activ),
		layers.Dense(2), 
		]	

	)

	cNN.compile(

	loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # True from logits giving softmax on output layer
	optimizer = keras.optimizers.Adam(learning_rate=lr),  # Using adam because it's the best 
	metrics = ['accuracy'],
	)

	cNN.fit(x_train, y_train, batch_size=64, epochs=epo, verbose=1)  # fitting over variable epoch numbers

	return cNN