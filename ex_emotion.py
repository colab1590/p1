import sys, os
"""sys — System-specific parameters and functions. This module provides access to some variables used or maintained by the interpreter 
and to functions that interact strongly with the interpreter. It is always available. sys.argv. The list of command line arguments 
passed to a Python script.
The OS module in python provides functions for interacting with the operating system. OS, comes under 
Python's standard utility modules. This module provides a portable way of using operating system dependent functionality.


"""
import pandas as pd
import numpy as np
""" pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, 
it offers data structures and operations for manipulating numerical tables and time series. 

NumPy is the most basic yet a powerful package for scientific computing and data manipulation in Python. It is an open source
library available in Python. It helps as to do the mathematical
and scientific operation and used extensively in data science. It allows us to work with multi-dimensional arrays and matrices.

"""
from keras.models import Sequential
"""
The core data structure of Keras is a model, a way to organize layers. The simplest type of model is the Sequential model, 
a linear stack of layers. For more complex architectures, you should use the Keras functional API, which allows to build arbitrary 
graphs of layers.

As with the Sequential API, the model is the thing you can summarize, fit, evaluate, and use to make predictions. 
Keras provides a Model class that you can use to create a model from your created layers. 
It requires that you only specify the input and output layers

There are two main types of models available in Keras: 
>>> the Sequential model, and 
>>> the Model class used with the functional API.


"""
from keras.layers import Dense, Dropout, Activation, Flatten
"""
dense: A Dense layer feeds all outputs from the previous layer to all its neurons, 
each neuron providing one output to the next layer. It's the most basic layer in neural networks. A Dense(10) has ten neurons

Dropout: Dropout may be implemented on any or all hidden layers in the network as well as the visible or input layer. It is not
used on the output layer. The term “dropout” refers to dropping out units (hidden and visible) in a neural network. — Dropout: A Simple Way to Prevent Neural Networks from Overfitting
Image result for Dropout in keras
Dropout Neural Network Layer In Keras Explained. ... Dropout is a technique used to prevent a model from overfitting. 
Dropout works by randomly setting the outgoing edges of hidden units (neurons that make up hidden layers) to 0 at each update of the training phase.

activation: Activation functions are really important for a Artificial Neural Network to learn and make sense of something really complicated and Non-linear complex functional mappings between the inputs and response variable.They introduce non-linear properties to our Network.Their main purpose is to convert a input signal of a node in a A-NN to an output signal. That output signal now is used as a input in the next layer in the stack.
Specifically in A-NN we do the sum of products of inputs(X) and their corresponding Weights(W) and apply a Activation function f(x) to it to get the output of that layer and feed it as an input to the next
layer.

Flatten: The role of the Flatten layer in Keras is super simple: A flatten operation on a tensor reshapes the tensor to have the 
shape that is equal to the number of elements contained in tensor non including the batch dimension. Note: I used the model. 
summary() method to provide the output shape and parameter details.Jan 15, 2019



"""
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D

""" 
Conv2D: 2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None,
it is applied to the outputs as well. When using this layer as the first layer in a model, provide the keyword argument 
input_shape (tuple of integers, does not include the batch axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in 
data_format="channels_last".

MaxPooling2D: MaxPooling2D
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
Max pooling operation for spatial data.

Arguments

pool_size: integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the 
input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions.
strides: Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.

padding: One of "valid" or "same" (case-insensitive).
data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.
 channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs
 with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file 
 at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
Input shape

If data_format='channels_last': 4D tensor with shape: (batch_size, rows, cols, channels)
If data_format='channels_first': 4D tensor with shape: (batch_size, channels, rows, cols)
Output shape

If data_format='channels_last': 4D tensor with shape: (batch_size, pooled_rows, pooled_cols, channels)
If data_format='channels_first': 4D tensor with shape: (batch_size, channels, pooled_rows, pooled_cols)
https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c

https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D

"""
from keras.losses import categorical_crossentropy
#Categorical crossentropy. Categorical crossentropy is a loss function that is used for single label categorization.
#This is when only one category is applicable for each data point. In other words, an example can belong to one class only.
from keras.optimizers import Adam
#An optimizer is one of the two arguments required for compiling a Keras model
#Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update 
#network weights iterative based in training data. ... The algorithm is called Adam
from keras.regularizers import l2
"""Regularizers allow to apply penalties on layer parameters or layer activity during optimization. 
These penalties are incorporated in the loss function that the network optimizes.

The penalties are applied on a per-layer basis. The exact API will depend on the layer, but the layers Dense, Conv1D, Conv2D and 
Conv3D have a unified API.

These layers expose 3 keyword arguments:

kernel_regularizer: instance of keras.regularizers.Regularizer
bias_regularizer: instance of keras.regularizers.Regularizer
activity_regularizer: instance of keras.regularizers.Regularizer
"""
from keras.utils import np_utils
#np.utils.to_categorical is used to convert array of labeled data(from 0 to nb_classes-1) to one-hot vector.


# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

df=pd.read_csv('fer2013.csv')

#read_csv is an important pandas function to read csv files and do operations on it.
#Opening a CSV file through this is easy. But there are many others thing one can do through this function only to 
#change the returned object completely. For instance, one can read a csv file not only locally, but from a URL through read_csv
# or one can choose what columns needed to export so that we don’t have to edit the array later.

# print(df.info())
# print(df["Usage"].value_counts())

# print(df.head())
X_train,train_y,X_test,test_y=[],[],[],[]
"""The training set is a subset of the data set used to train a model.

x_train is the training data set.
y_train is the set of labels to all the data in x_train.
The test set is a subset of the data set that you use to test your model after the model has gone through initial vetting 
by the validation set.

x_test is the test data set.
y_test is the set of labels to all the data in x_test.
The validation set is a subset of the data set (separate from the training set) that you use to adjust hyperparameters.

The example you listed doesn't mention the validation set.
I've made a Deep Learning with Keras playlist on Youtube. It contains the basics for getting started with Keras, and a 
couple of the videos demo how to organize images into train/valid/test sets, as well as how to get Keras to create a validation
 set for you. Seeing this implementation may help you get a firmer grasp on how these different data sets are used in practice.


"""
for index, row in df.iterrows():#.iterrows Iterate over DataFrame rows as (index, Series) pairs.
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")


num_features = 64
num_labels = 7
batch_size = 64
epochs = 30
width, height = 48, 48
#In the neural network terminology: one epoch = one forward pass and one backward pass of all the training examples.
# batch size = the number of training examples in one forward/backward pass. The higher the batch size,
# the more memory space you'll need.

X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)

#cannot produce
#normalizing data between o and 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# print(f"shape:{X_train.shape}")
##designing the cnn
#1st convolution layer
model = Sequential()
#https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'
#Rectified Linear Unit(relu).

With default values, it returns element-wise max(x, 0).
#https://keras.io/activations/
#https://keras.io/layers/convolutional/
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
#https://keras.io/layers/pooling/
model.add(Dropout(0.5))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))

# model.summary()

#Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

#Training the model
model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, test_y),
          shuffle=True)


#Saving the  model to  use it later on
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
#https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/


