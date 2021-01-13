# CityPictureClassifier
Using machine learning to source and classify pictures of Tokyo and New York City

# Data:

ImageSource uses selenium webdriver to take screen shots of google street view locations across New York City and Tokyo. As a start I sourced 150 NYC and 150 tokyo images, which takes ~4 seconds per image to grab. 

ImageProcessor turns these images into 4 dimension tensors for training, creates labels and saves a feature and label data set as an npy file. we have 293 final training images, each in RGBA mode with 4 channels : (293, 1642, 2880, 4), (293,)

Imageresizer takes each image and resizes it to a (293,400,700,3) image with channels RGB mode and 293 training examples. 

The model uses a convolutional neural network for feature extraction to train the classifier on. This in includes 3 repetitions of 2D convolution followed by a max pooling layer. The 2D convolutions have 32, 64, and 128 filters respectively, each with a (3,3) filter size. The max pooling each have a (2,2) pool size. Finally, there is a flatten layer followed by 2 fully connected layers. The ouput has a softmax activation. 


