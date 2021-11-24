#PyTorch and TensorFlow are more powerful, but Keras is easier to use for beginners
#Simple syntax, high-level API
#This presupposes you have downloaded + installed Keras framework & underlying framework such as PyTorch

## DATA WRANGLING ##
#Get datasets
import pandas as pd
import numpy as np

#Putting it into a nice Panda dataframe (who doesn't like Pandas?)
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()
concrete_data.shape #Note: Only 1000 samples, so overfitting can be an issue here

#Checking for missing values
concrete_data.describe()
concrete_data.isnull().sum()

#Splitting data into training and test sets should be done here, but for now....
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

