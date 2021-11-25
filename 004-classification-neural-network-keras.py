#PyTorch and TensorFlow are more powerful, but Keras is easier to use for beginners
#Simple syntax, high-level API
#This presupposes you have downloaded + installed Keras framework & underlying framework such as PyTorch

#Get required libraries
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt #for dealing with the images

### DATA WRANGLING ###
#We will use MNIST database, Modified National Institute of Standards and Technology database, 
#large database of handwritten digits that is commonly used for training various image processing systems
#contains 60,000 training images and 10,000 testing images of digits written by high school students and employees of the United States Census Bureau

# import the data
from keras.datasets import mnist

# read the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#have a look at the dataset shape
X_train.shape
y_train.shape
X_test.shape
y_test.shape

#and an example image
plt.imshow(X_train[0])

#conventional neural networks cannot just take images as inputs
#we need to flatten the images into one-dimensional vectors, each of size 1 x (28 pixels x 28 pixels) = 1 x 784.

# flatten images into one-dimensional vector

num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# for classification we need to divide our target variable into categories using to_categorical function
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)

#### BUILDING THE MODEL ####
# define classification model

def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) #softmax because we are dealing with a category
    
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
