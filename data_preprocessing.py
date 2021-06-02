import numpy as np
from PIL import Image
import os
import sys
from sklearn.utils import shuffle


'''
Taking 2m (number of images) 640x640x3 jpg RGB represente images, and returns a numpy 
feature and label file representing the feature images and their respective labels. 
The features.npy is a (m,640,640,3) tensor, an the labels.npy is a (m,) veector of labels. Equals
amounts of tokyo and ny data no necesssary, grabbing ~2000 images from each class standard for a total of >4000 images.
'''

# Native paths to ny and tokyo image data
nyPath = r"C:\Users\Administrator\Desktop\copy\finalny"  
tokPath = r"C:\Users\Administrator\Desktop\copy\finaltok"

nyPics_names = os.listdir(nyPath)
tokPics_names = os.listdir(tokPath)

m = 2000 # setting number of training examples to grab per class

# notifying if grabbing unequal numbers of data
if len(nyPics_names)!=len(tokPics_names):  # fixed by capping at m 
	print("You are not grabbing equal numbers of tokyo and NYC data.")


# Instantiating numpy tensors 
train_ny = np.ones((m,640,640,3))  
train_tok = np.ones((m,640,640,3))
print("Ready to write")

counter=0
for i in range(m):  # starting at 2nd element because 0 is the imageProcessor file and 1 was used to instantiate
	counter=counter+1
	if nyPics_names[i][-3:]=="jpg":
		train_ny[i] = np.asarray(Image.open(nyPath+"/"+nyPics_names[i])) # Setting ith element on axis 0 of train to ith image in listdir as array shape 640x640x3
	if tokPics_names[i][-3:]=="jpg":
		print(counter)
		train_tok[i] = np.asarray(Image.open(tokPath+"/"+tokPics_names[i]))


# Should both return (m, 640, 640, 3)
print(train_ny.shape) 
print(train_tok.shape)
	
# Creating 2 (m,) label vectors
label_ny = np.zeros((train_ny.shape[0],))  # LABELING NYC 0
label_tok = np.ones((train_tok.shape[0],))  # LABELING TOKYO 1

print("Combining")
features_dataset = np.concatenate((train_ny, train_tok),axis=0)  # (4000, 640, 640, 3) feature tensor
labels_dataset = np.concatenate((label_ny, label_tok))  # (4000,) label vector


print("Shuffling")
features_dataset, labels_dataset = shuffle(features_dataset, labels_dataset, random_state=0)  # shuffling training data on 0 axis


print("Saving")
np.save("features", features_dataset)
np.save("labels", labels_dataset)












