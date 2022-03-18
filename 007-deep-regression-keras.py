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
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error

## DATA WRANGLING ##

#get dataset and...
#Putting it into a nice Panda dataframe (who doesn't like Pandas?)
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv') #https://cocl.us/concrete_data 
concrete_data.head()
concrete_data.shape #Note: Only 1000 samples, so overfitting can be an issue here

#Checking for missing values
concrete_data.describe()
concrete_data.isnull().sum()

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

#checking whether split worked
predictors.head()
target.head()

#normalize data
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

#splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=42) #random_state is set for control purposes, in real application it should not be set
n_cols = X_train.shape[1] # number of predictors
print(n_cols)


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
#model.fit(X_train, y_train, validation_split=0.3, epochs=100, verbose=2) #here, we can alternatively use Keras train-test split function, but I have split before already
model.fit(X_train, y_train, epochs=100, verbose=2)

# evaluate the model using the Keras functions can be done like this, will also show you the error:
#scores = model.evaluate(X_test, y_test, verbose=0)
#print(scores)

#However, we will do it using scikitlearns's mean_squared_error function, like this:

# create predictions using our model
y_pred = model.predict(X_test)
#print(y_pred)

#compare our models predictions with the actual y values
print ("Mean Squared Error equals")
mean_squared_error(y_test, y_pred)

#Now we will run this process multiple times, to ensrue that our one-time run is not just a statistical fluke

#we will just store the results in a list
results_table_mean_squared_error = []

for i in range(3):
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3)
    n_cols = X_train.shape[1] # number of predictors
    #print(n_cols)
    model.fit(X_train, y_train, epochs=100, verbose=2)
    y_pred = model.predict(X_test)
    error_of_this_run = mean_squared_error(y_test, y_pred)
    results_table_mean_squared_error.append(error_of_this_run)
    
#Evaluation after multiple runs
#What does our error table look like?
print(results_table_mean_squared_error)

#Mean error
print(np.mean(results_table_mean_squared_error))

#SD error
print(np.std(results_table_mean_squared_error))
    
#FIN#
