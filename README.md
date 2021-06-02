# Classifying Google Street View images of New York City and Tokyo using Deep Learning.


# Objective 

Our objective is to train a model which will correctly classify Google Street View images from New York City and Tokyo into their respective classes. This is done using Deep Learning and Optical Character Recognition including data grabbing, feature processing and augmentation, model training, Hyperparameter optimization, and evaluation. 

# Data

Data is sourced through Google Cloud's Street View API, grabbing RBG images sprawled over the New York and Tokyo metropolitan areas. 4000> images were grabbed and stored with their respective labels. Images were sifted through to remove glitches and poor city representations including intense blur, images taken inside, no view of street, etc.

# Feature Processing

Images are resized and converted to create a (4000, 640, 640, 3) ndarray of feature data. These training tensors were a total of ~40GB of data. A test set of 500 data examples were set aside, never to be seen by the model. 

# Deep Learning

A Convolutional Neural Network (CNN) is used for the model written in TensorFlow 2.0 with  convolutional, pooling, and fully connected layers. Model parameters including optimizer, learning rate, training epochs, and activation functions were left variable. 

# Training and Hyperparameter Optimization 

An Amazon Web Services (AWS) EC2 Instance is used for training to handle the extremely large computation complexity, using 128GB of memory and 64 vCPUs. 

 K-fold Cross-validation is used to estimate training accuracies, and an exhaustive Grid Search is used with a static parameter space guide by the k-fold Cross-validation.
 
 Training was done on a random 2000 images to limit computation times, were were running at ~1 minute per epoch. 
 
 # Evaluation

Final training was done on 3500 data examples with optimized hyperparameters. A final training accuracy of ~\%97 and a final test accuracy on the test set of ~\%82 was achieved.