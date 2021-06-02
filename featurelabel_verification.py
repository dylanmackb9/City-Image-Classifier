import numpy as np
from PIL import Image
from PIL import ImageShow
import math
import random



# Verifying that features.npy and labels.npy are correctly associating labels to images


featurePath = r"C:\Users\Administrator\Desktop\copy\features.npy"
labelPath = r"C:\Users\Administrator\Desktop\copy\labels.npy"

features = np.load(featurePath) 
labels = np.load(labelPath)    


print(features.shape)
print(labels.shape)


mi = math.ceil(random.random()*4000)

print(labels[mi])  # printing label of mi image

t = Image.fromarray((features[mi]).astype(np.uint8))  # representing image as tensor
ImageShow.show(t)