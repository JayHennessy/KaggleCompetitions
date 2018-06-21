#------------------------------------
#
# First intro to ML with Kaggle
#
#-------------------------------

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left

# print the data
#print(data.describe())

# view columns
#print(data.columns)
#----------------------------
#
#
#-------------------------------------
# get the sales price column
priceCol = data.SalePrice
#print(priceCol.head())

#4) get 2 columns
two_columns = data[['YrSold', 'Id']]
#print(two_columns.describe()) 

# select target variable 
y = data.SalePrice

# select the predictor headers
predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = data[predictors]

#define the model
iowa_model = DecisionTreeRegressor()

# fit the model
iowa_model.fit(X,y)

#predict the first 5
print(X.head())
print('The prediction is :')
print(iowa_model.predict(X.head()))

#----------------------------------------
#
# what is model validation
#
#--------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#get validation and test data
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)

#train and predict using the decision tree
my_model = DecisionTreeRegressor()
my_model.fit(train_X,train_y)
val_predictions = my_model.predict(val_X)

print("The decision tree error is : ", mean_absolute_error(val_y, val_predictions))

# -----------------------------------------
#
# Experimenting with different models
#
#---------------------------------------------

# this function gets the mean absolute error like above
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#-----------------------------------------
#
# Random Forests
#
# -----------------------------------------

from sklearn.ensemble import RandomForestRegressor

# make a random forest model and predict with it
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print("The forest model error is : ", mean_absolute_error(val_y, melb_preds))

