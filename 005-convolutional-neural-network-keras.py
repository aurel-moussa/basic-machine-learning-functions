### CNNs use image inputs, good at detecting objects, classifying images, and other computer vision tasks ###
# conventional neural networks take (n x 1) vector as input, but...
# input to a convolutional neural network mostly an (n x m x 1) for grayscale images 
# or (n x m x 3) for colored images, where the number 3 represents the red, green, and blue components of each pixel in the image

### Packages ###
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from keras.layers.convolutional import Conv2D # to add convolutional layers
from keras.layers.convolutional import MaxPooling2D # to add pooling layers (average pooling is another popular option)
from keras.layers import Flatten # to flatten data for fully connected layers

### Build model with one set of convolutional and pooling layer ###
# import data
from keras.datasets import mnist

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

#normalize
X_train = X_train / 255 # normalize training 
X_test = X_test / 255 # normalize test 
