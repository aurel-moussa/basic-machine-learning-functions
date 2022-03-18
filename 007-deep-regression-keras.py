#PyTorch and TensorFlow are more powerful, but Keras is easier to use for beginners
#Simple syntax, high-level API
#This presupposes you have downloaded + installed Keras framework & underlying framework such as PyTorch
#Also important that scikit-learn is installed previously

#Get required libraries
import pandas as pd #we will require this for putting data into nice dataframes
import numpy as np 
import keras
import sklearn
from keras.models import Sequential
from keras.layers import Dense
frm sklearn.model_selection import train_test_split 

## DATA WRANGLING ##

#get dataset and...
#Putting it into a nice Panda dataframe (who doesn't like Pandas?)
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv') #https://cocl.us/concrete_data 
concrete_data.head()
concrete_data.shape #Note: Only 1000 samples, so overfitting can be an issue here

#Checking for missing values
concrete_data.describe()
concrete_data.isnull().sum()

#Splitting data into training and test sets should be done here, but for now, I believe Keras actually does that for us....
#We will just split into X set and Y set

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

#checking whether split worked
predictors.head()
target.head()

#normalize data
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

#save number of predictors into a variable, because we will need it later
n_cols = predictors_norm.shape[1] # number of predictors

#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


### BUILDING THE MODEL ###

# define regression model (sequential (i.e. that is normal), dense (i.e. all nodes connected to all nodes of the next layer))
# output layer with ONE node, and 50 nodes in the layers before that. Using ReLu, not sigmoid as activation function, because ReLu is better
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error') #adam optimzer instead of typical gradient descent, loss function is the typical mean squared error
    return model
  
### TRAINING & TESTING MODEL ###
# build the model
model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2) #nice one, the validation is being split already!

#FIN#

