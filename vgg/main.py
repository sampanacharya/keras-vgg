from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from architecture import vgg11, vgg13, vgg16, vgg19
import sys,os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

"""def switcher(idx):
	vggs = {
			1:"VGG11",
			2:"VGG13",
			3:"VGG16",
			4:"VGG19",
		}
	return vggs.get(idx, None)"""	

def main():
	os.system("cls")
	print("Write the index of a vgg model")
	print("1.VGG11\n2.VGG13\n3.VGG16\n4.VGG19")
	# User inputs
	idx = int(input("Enter the index:\n"))
	inputs = input("Enter comma seperated input_shape\n")
	input_shape = [int(i) for i in inputs.split(',')]
	input_shape = tuple(input_shape)
	classes = 1000
	os.system("cls")
	os.system("cls") 
	if idx == 1:
		vgg = vgg11(input_shape=input_shape, classes=classes)
		print("Summary of the VGG11 Layer:")
		vgg.summary()
	elif idx == 2:
		vgg = vgg13(input_shape=input_shape, classes=classes)
		print("Summary of the VGG13 Layer:")
		vgg.summary()
	elif idx == 3:
		vgg = vgg16(input_shape=input_shape, classes=classes)
		print("Summary of the VGG16 Layer:")
		vgg.summary()
	elif idx == 4:
		vgg = vgg19(input_shape=input_shape, classes=classes)
		print("Summary of the VGG19 Layer:")
		vgg.summary()
	return 1
		
if __name__=="__main__":
	main()