import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
from PIL import ImageShow

# Takes a pretrained model and gives model predictions on a set of test images. 

modelpath = '/Users/Dylan/Desktop/+/Academia/Cornell/CS4701/proj/finalmodel'
optmod = keras.models.load_model(modelpath)  # loading in given model

evalimagepath = "/Users/Dylan/Desktop/+/Academia/Cornell/CS4701/proj/evalimages"
eval_names = os.listdir(evalimagepath)  # loading in test images


m = 1  # number of images from evalimages you want to test

evalimagestensor = np.ones((m,640,640,3))

for i in range(m):
	if eval_names[i][-3:] == 'jpg':
		evalimagestensor[i] = np.asarray(Image.open(evalimagepath+"/"+eval_names[i])) 


#optmod.summary()

print(optmod.predict(evalimagestensor))







