from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # To remove TF warning

"""All VGG blocks are defined here 
	C1: 64-channels 
	C2: 128-channels 
	C3: 256-channels 
	C4: 512-channels
	FC: Fully Connected 
	Note:-This VGG implementation only uses (3,3) filters, even though
	the original paper consist of (1,1) filters too but we focus
	just on (3,3)
	Note: I didn't include (1,1) Convolutional Layers as mentioned in the original paper"""
# C1 layer
def C1(num_conv=0,input_shape=None,activation=None):
	model = Sequential()
	model.add(layers.Conv2D(64, kernel_size=(3,3),input_shape=input_shape, activation=activation, padding="same"))
	if num_conv:
   	    model.add(layers.Conv2D(64, kernel_size=(3,3),padding="same"))
	model.add(layers.MaxPool2D((2,2), strides=(2,2)))
	return model

# C2 layer
def C2(num_conv=1, activation=None):
    model = Sequential()
    for _ in range(num_conv):
        model.add(layers.Conv2D(128, kernel_size=(3,3), activation=activation, padding="same"))
    model.add(layers.MaxPool2D((2,2), strides=(2,2)))
    return model

# C3 layer
def C3(num_conv=2, activation=None):
    model = Sequential()
    for _ in range(num_conv):
        model.add(layers.Conv2D(256, kernel_size=(3,3), activation=activation, padding="same"))
    model.add(layers.MaxPool2D((2,2), strides=(2,2)))
    return model

# C4 layer
def C4(num_conv=2, activation=None):
    model = Sequential()
    for _ in range(num_conv):
        model.add(layers.Conv2D(512, kernel_size=(3,3), activation=activation, padding="same"))
    model.add(layers.MaxPool2D((2,2), strides=(2,2)))
    return model

# C5 layer
def C5(num_conv=2, activation=None):
    model = Sequential()
    for _ in range(num_conv):
        model.add(layers.Conv2D(512, kernel_size=(3,3), activation=activation, padding="same"))
    model.add(layers.MaxPool2D((2,2), strides=(2,2)))
    return model
# Fully Connected Layer
def FC(classes=1000):
    model = Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dense(classes, activation="softmax"))
    return model	
