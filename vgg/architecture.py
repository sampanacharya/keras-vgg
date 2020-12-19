from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from blocks import C1, C2, C3, C4, C5, FC
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # To remove TF warning

def vgg11(input_shape=None, classes=1000):
	"""8 Convolutional Layers and 3 Fully Connected Layers"""
	model = Sequential([
		C1(num_conv=0, input_shape=input_shape, activation="relu"),
		C2(num_conv=1, activation="relu"),
		C3(num_conv=2, activation="relu"),
		C4(num_conv=2, activation="relu"),
		C5(num_conv=2, activation="relu"),
		FC(classes=classes)
		])
	return model 

def vgg13(input_shape=None, classes=1000):
	"""10 Convolutional Layers and 3 Fully Connected Layers"""
	model = Sequential([
		C1(num_conv=1, input_shape=input_shape, activation="relu"),
		C2(num_conv=2, activation="relu"),
		C3(num_conv=2, activation="relu"),
		C4(num_conv=2, activation="relu"),
		C5(num_conv=2, activation="relu"),
		FC(classes=classes)
		])
	return model 

def vgg16(input_shape=None, classes=1000):
	"""13 Convolutional Layers and 3 Fully Connected Layers"""
	model = Sequential([
		C1(num_conv=1, input_shape=input_shape, activation="relu"),
		C2(num_conv=2, activation="relu"),
		C3(num_conv=3, activation="relu"),
		C4(num_conv=3, activation="relu"),
		C5(num_conv=3, activation="relu"),
		FC(classes=classes)
		])
	return model	


def vgg19(input_shape=None, classes=1000):
	"""16 Convolutional Layers and 3 Fully Connected Layers"""
	model = Sequential([
		C1(num_conv=1, input_shape=input_shape, activation="relu"),
		C2(num_conv=2, activation="relu"),
		C3(num_conv=4, activation="relu"),
		C4(num_conv=4, activation="relu"),
		C5(num_conv=4, activation="relu"),
		FC(classes=classes)
		])
	return model	

