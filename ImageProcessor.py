
import numpy as np
from PIL import Image
import os
import sys
from sklearn.utils import shuffle


# Taking m training images and creating dataset as (m,1642,2880,4) training tensor, saving as .npy
 # RGBA values for each training image


nycPath = "/Users/Dylan/Documents/python/project_drafts/GeoG/NYC_pics/nyc_2880x1642"
tokyoPath = "/Users/Dylan/Documents/python/project_drafts/GeoG/Tokyo_pics/tokyo_2880x1642"

nycPics_names = os.listdir(nycPath)
tokyoPics_names = os.listdir(tokyoPath)


m = len(nycPics_names)  # setting number of training examples

if len(nycPics_names)!=len(tokyoPics_names): 
	print("You are not grabbing equal numbers of tokyo and NYC data.")


nyc_img1 = np.asarray(Image.open(nycPath+"/"+nycPics_names[0]))  # first picture in nyc train data
tokyo_img1 = np.asarray(Image.open(tokyoPath+"/"+tokyoPics_names[0]))  # first picture in nyc train data

train_nyc = nyc_img1.reshape((1,1642,2880,4))  # instantiating nyc training data tensor
train_tokyo = tokyo_img1.reshape((1,1642,2880,4))  # instantiating tokyo training data tensor



counter=0

for i in range(1,m):  # starting at 2nd element because 0 is the imageProcessor file and 1 was used to instantiate
	counter=counter+1
	if nycPics_names[i][-3:]=="png":
		new_nyc = np.asarray(Image.open(nycPath+"/"+nycPics_names[i]))
		new_nyc = new_nyc.reshape((1,1642,2880,4))  # adding a dimension to each new picture data
		train_nyc= np.concatenate((train_nyc, new_nyc), axis=0)  # adding new picture to tensor of picture data
	if tokyoPics_names[i][-3:]=="png":
		new_tokyo = np.asarray(Image.open(tokyoPath+"/"+tokyoPics_names[i]))
		print(new_tokyo.shape)
		print(tokyoPics_names[i])
		print("")
		new_tokyo = new_tokyo.reshape((1,1642,2880,4))
		train_tokyo = np.concatenate((train_tokyo, new_tokyo),axis=0)

	print(counter)
	

#print(train_nyc.shape) 
#print(train_tokyo.shape)
	
# gives (148, 1642, 2880, 4)

label_nyc = np.zeros((train_nyc.shape[0],))  # LABELING NYC 0
label_tokyo = np.ones((train_tokyo.shape[0],))  # LABELING TOKYO 1

features_dataset = np.concatenate((train_nyc, train_tokyo),axis=0)  # (296,1642,2880,4) tensor
labels_dataset = np.concatenate((label_nyc, label_tokyo))  # (296,) labels


features_dataset, labels_dataset = shuffle(features_dataset, labels_dataset, random_state=0)  # shuffling training data

np.save("features_dataset", features_dataset)
np.save("labels_dataset", labels_dataset)




























