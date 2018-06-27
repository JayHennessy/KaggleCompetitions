#-----------------------------------------
#
# Gradient boosted decision tree with XGBoost
#
#-----------------------------------------------


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


data = pd.read_csv('C:/Users/JAY/Desktop/Machine_Learning_Course/KaggleCompetions/ML_tutorial1/data/melb_data.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)


from xgboost import XGBRegressor

# now build and fit a model
my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

# make predictions
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))


# Now tune the model

# n_estimators says how many boost cycles to do and early_stopping rounds will autimatically choose the best number 
# n_estimators and stop after the validation score deteriorates that number of times in a row
my_model = XGBRegressor(n_estimators=1000)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

# you can also add a learning rate that will slow down the process but will give more accurate results and less likely to overfit
# the learning rate will multiply the prediction values by a small number before adding them.
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

# for big jobs, n_job can also be set to parallelize the process. set it to the number of cores on your machine