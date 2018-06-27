#----------------------------
#
#  Cross validation tutorial
#
#-------------------------------

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score


# read the data
data = pd.read_csv('C:/Users/JAY/Desktop/Machine_Learning_Course/KaggleCompetions/ML_tutorial1/data/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price

# use a pipeline to model the data
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

# get the cross validation score
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)

# average across each validation piece
print('Mean Absolute Error %2f' %(-1 * scores.mean()))