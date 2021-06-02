
import requests as req
import shutil
import numpy as np
import imageio
import cv2
import PIL

'''
Grabbing data in the form of jpg images from Google Street View at lat/long specified locations. Using Google Cloud Street View 
API key to access browser jpg images, and saving them to native file. 
'''


## New York

# Brooklyn: 
brooklyn_lat = [40.694430, 40.610379]  
brooklyn_long =[-73.995971, -73.922416]

# South Manhattan: 
southman_lat = [40.749026, 40.712422] 
southman_long = [-74.007073, -73.979857]

# Mid Manhattan: 
midman_lat = [40.790413, 40.758200] 
midman_long = [-73.979563, -73.963570]

# Queens: 
queens_lat = [40.750155, 40.692347] 
queens_long = [-73.866800, -73.774465]

## Tokyo

# West sprawl: 
westsprawl_lat = [35.754961, 35.638379] 
westsprawl_long = [139.440165, 139.642680]

# West industrial: 
westind_lat = [35.721740, 35.649794]  
westind_long = [139.708284, 139.750425]

# East industrial 
eastind_lat = [35.714633, 35.667822] 
eastind_long = [139.780991, 139.840999]


KEY = "AIzaSyARmhxH3_xGwG0xbWYMK0VtkjAoCKkV4ec"
#url = "https://maps.googleapis.com/maps/api/streetview?location="+str(latitude)+","+str(longitude)+"&size=640x640&key="+KEY
#img = np.asarray(imageio.imread(imageio.core.urlopen(url).read(), 'jpg'))



##TESTING ARGS
#r = req.get(url)
#f = open("Pls", 'wb')
#f.write(r.content)
#f.close()
#print(type(img))
#print(img.shape)
#cv2.imshow("sure", img)
#cv2.waitKey()


city = "tok_"  # variable for file labeling
num_pics = 100  # grabbing in each basis direction

lat_range = np.arange(eastind_lat[0], eastind_lat[-1], -abs(eastind_lat[0]-eastind_lat[-1])/num_pics)
long_range = np.arange(eastind_long[0], eastind_long[-1], abs(eastind_long[0]-eastind_long[-1])/num_pics)


# Sweep over defined space
for latitude in lat_range:
	for longitude in long_range:
		url = "https://maps.googleapis.com/maps/api/streetview?location="+str(latitude)+","+str(longitude)+"&size=640x640&return_error_codes=true&key="+KEY
		r = req.get(url)
		f = open("tok_data/eastind/"+city+str(latitude)+"_"+str(longitude)+".jpg", 'wb')  # writing new file in binary mode 
		f.write(r.content)
		f.close()











