import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
from PIL import ImageShow

import model 



# Taking feature and label paths
featurePath = r"C:\Users\Administrator\Desktop\copy\features.npy"
labelPath = r"C:\Users\Administrator\Desktop\copy\labels.npy"

# loading features and labels to memory
features = np.load(featurePath) 
labels = np.load(labelPath)    

print("Finished loading")


# Splitting data set into training and testing data. Splits are made based on computation ability
# Largest size for training set is 3500. 2000 used for lighter train.

split = 3500 # number trained out of 3500, higher means more data trained on

xtrain = features[0:split]
ytrain = labels[0:split]

xtest = features[split:]
ytest = labels[split:]




## MODEL HYPERPARAMETER TUNING

def kcross_cnn(k_xtrain, k_ytrain, k, lr, epo, activ):
    '''
    Implementing k-fold Cross-validation for Convolutional Neural Network. 

    Using Cross-validation techniques to evaluate models over k validation sets, reporting average accuracy
    from k validation predictions for a give model. Returns average accuracy of a model with given lr, epo, and activ over 
    given training data and set k.
	
	Input: 
		k_xtrain -- m x 640 x 640 x 3 ndarray, training features.
		k_ytrain -- m x 1 ndarray,  training labels.
		k -- int, number of splits for Cross validation.
		lr -- int, learning rate for model's optimizer.
		epo -- int, number of traning epochs for model.
		activ -- str: Activation function for model's nonlinear layers.

	Output:
		average_kacc -- int, average model accuracy over k different prediction accuracies on k validation sets.

    '''

    m = int(k_xtrain.shape[0])  # number of examples
    kaccuracy_list = []
    
    for i in range(k):  # cross val training 
        x_val = k_xtrain[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        y_val = k_ytrain[i*m//k:i*m//k+m//k]  # finding k m/k sized portions
        
        x_train = np.vstack((k_xtrain[0:i*m//k],k_xtrain[(i*m//k+m//k):]))  # setting training to be everything but val
        y_train = np.concatenate((k_ytrain[0:i*m//k],k_ytrain[(i*m//k+m//k):]))
        
        curmod = model.cnn(x_train, y_train, activ, lr, epo)  # training nn with created training set above
       	loss, accuracy = curmod.evaluate(x_val, y_val, batch_size=64, verbose=0)
        print("finished training")
        kaccuracy_list.append(accuracy)  # appending one of the k accuracies to a list
        
    average_kacc = np.mean(kaccuracy_list)  # averaging all k accuracies 
    
    return average_kacc

#Hyperparameters for NN
lr_range = [.0001, .01]
epochs = [5, 10, 15]
activation = ['relu']

def gridsearch(k):
    '''
    Implementing an exhaustive Grid Search for Hyperparameter optimization on CNN model. Tuning over set parameter space.
    Guided by k-fold Cross-validation performance metric. 

    Input: 
    	k -- int, value for performance metric used to guide the sweep search.

    '''

    for activ in activation:
        for lr in lr_range:
            for e in epochs:
                accuracy = kcross_cnn(xtrain, ytrain, k, lr, e, activ)
                print("For "+activ+" activation and "+str(lr)+" learning rate and "+str(e)+" epochs: "+str(accuracy))


k = 2 # for crossval based on 2000 size training set


#Calling for Hyperparameter optimization
gridsearch(k)



# Optimal hyperparameters observed 

# 	Activation function -- ReLU
#	Learning Rate -- 0.0001
#	Epochs -- 8


# Optimized training
finalmod = model.cnn(xtrain, ytrain, 'relu', .0001, 8)

loss, accuracy = finalmod.evaluate(xtest, ytest, batch_size=64, verbose=0)

print("Final loss: " + str(loss))
print("Final accuracy: " + str(accuracy))

tf.keras.models.save_model(finalmod, r"C:\Users\Administrator\Desktop\copy")
model.save_weights(checkpoint_path.format(epoch=0))




                



