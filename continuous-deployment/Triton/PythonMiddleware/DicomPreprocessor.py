# Imports
import tensorflow as tf
import numpy as np
import pathlib

NORMALIZE_MEAN_PATH=pathlib.Path(pathlib.Path(__file__).parent.resolve().as_posix(),"means/dataset_mean.txt")

# Function that loads image data as array from Dicom
def loadDataFromDicom(dicom):
	# Mean and std reading
	f = open(NORMALIZE_MEAN_PATH, "r")
	mean = float(f.readline())
	std = float(f.readline())
	# Read and normalize the dicom
	array = get_imageArray_from_dicom(dicom)
	array = normalize_image(array, mean, std)

	# Expand dim 0 since that is what the model need
	array = np.expand_dims(array, axis=0)
	return array

# Function that returns the array of a dicom
def get_imageArray_from_dicom(ds):

	# We need the array(pixel_array), color_scheme(inverse or normal/identity) and bits_stored(8,12,16,...)
	color_scheme = ds.get(0x20500020).value
	bits_stored = ds.get(0x00280101).value

	# Convert to range 0-255
	array = np.uint8(ds.pixel_array // (2**(bits_stored-8)))

	if color_scheme == "INVERSE":
		array = 255-array

	# Add channel dimension
	array = np.expand_dims(array, -1)

	# Transform to img
	img = tf.keras.preprocessing.image.array_to_img(array)
	# Resize to input size of the model
	img = img.resize((1024, 1024))
	# Save the image
	#img.save("radiography.png")

	# Get the image back to array and return
	array = tf.keras.preprocessing.image.img_to_array(img)
	return array

# Function that normalizes the image
def normalize_image(array, mean, std):
	array = tf.cast(array,tf.float32) # Convert to float
	array = tf.math.multiply(array, 1./255) # Convert to range 0-1
	array = tf.math.subtract(array, mean) # Normalize((x-mean)/std)
	array = tf.math.multiply(array, 1./std)
	return array
