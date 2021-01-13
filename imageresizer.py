
import importlib  # used to import modules without addressing their name directly 
import numpy 
import os
import cv2  # what opencv-python is called!


# resizing images in bulk to 400x700


nycPath = "/Users/Dylan/Documents/python/project_drafts/GeoG/NYC_pics/nyc_2880x1642"  # path to grab old pics
tokyoPath = "/Users/Dylan/Documents/python/project_drafts/GeoG/Tokyo_pics/tokyo_2880x1642"

nycsavePath = "/Users/Dylan/Documents/python/project_drafts/GeoG/NYC_pics/nyc_400x700"  # path to save new pics to
tokyosavePath = "/Users/Dylan/Documents/python/project_drafts/GeoG/Tokyo_pics/tokyo_2880x1642"

nycPics_names = os.listdir(nycPath)
tokyoPics_names = os.listdir(tokyoPath)


dim = (700,400)  # resize dim, for some reason column is first  # resize dimension
m = len(nycPics_names)  # setting number of training examples
counter = 0

for i in range(1,m):

	counter = counter+1
	if nycPics_names[i][-3:]=="png":  
		os.chdir(nycPath)  # grab path 
		img_n = cv2.imread(nycPath+"/"+nycPics_names[i])  # grabbing big images
		resize_n = cv2.resize(img_n, dim, interpolation=cv2.INTER_AREA)  # resizing image
		os.chdir(nycsavePath)  # save path  
		cv2.imwrite(nycPics_names[i], resize_n)  # saving new image
	if tokyoPics_names[i][-3:]=="png":
		os.chdir(tokyoPath)
		img_t = cv2.imread(tokyoPath+"/"+tokyoPics_names[i])
		resize_t = cv2.resize(img_t, dim, interpolation=cv2.INTER_AREA)
		os.chdir(tokyosavePath)
		cv2.imwrite(tokyoPics_names[i], resize_t)











