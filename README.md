# CityPictureClassifier
Using machine learning to source and classify pictures of Tokyo and New York City

# Data:

ImageSource uses selenium webdriver to take screen shots of google street view locations across New York City and Tokyo. As a start I sourced 150 NYC and 150 tokyo images, which takes ~4 seconds per image to grab. 

ImageProcessor turns these images into 4 dimension tensors for training, creates labels and saves a feature and label data set as an npy file: (296, 1642, 2880, 4), (296,)


