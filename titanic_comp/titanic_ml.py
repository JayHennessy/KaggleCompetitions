import pandas as pd
import sklearn.preprocessing as Imputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import Imputer


def get_feature_to_drop(data, input_thresh):
	nan_count = data.isnull().sum()
	#find the number of nan's in a feature that warrants dropping of the feature
	thresh = input_thresh*len(data.index)
	#find the indexs of the features to drop
	drop_index = nan_count[nan_count > thresh]
	#print('drop_indexs were', drop_index.index)
	return drop_index.index

def get_rows_to_remove(data, input_thresh):
	nan_count = data.isnull().sum()
	#find the number of nan's in a feature that warrants removal of the data
	thresh = input_thresh*len(data.index)
	#find the indexs of the examples to remove
	remove_index = nan_count[(0 < nan_count) & (nan_count <= thresh)]

	#print('remove_index', remove_index.index)
	return remove_index.index

# first get the data
test_raw = pd.read_csv('C:/Users/JAY/Desktop/Machine_Learning_Course/KaggleCompetions/titanic_comp/data/test.csv')
train_raw = pd.read_csv('C:/Users/JAY/Desktop/Machine_Learning_Course/KaggleCompetions/titanic_comp/data/train.csv')

data = train_raw.copy()
data_test = test_raw.copy()

# see data info
#print(data_test.info())
#print(data_test.isnull().sum())

# find features and rows to remove due to nans
train_drop_features = get_feature_to_drop(data, 0.5)
train_remove_rows	= get_rows_to_remove(data, 0.5)
test_drop_features = get_feature_to_drop(data_test, 0.5)
test_remove_rows	= get_rows_to_remove(data_test, 0.5)

# process the data, drop data points that are missing data in age or fare but drop the cabin column
data_no_nan = data.drop(train_drop_features, axis=1)
test_no_nan = data_test.drop(test_drop_features, axis=1)

data_no_nan_2 = data_no_nan.dropna(axis=0, subset=train_remove_rows)
test_no_nan_2 = test_no_nan.dropna(axis=0, subset=test_remove_rows)


# change categorical data
data_clean = pd.get_dummies(data_no_nan_2, columns=['Embarked','Sex'])
test_clean = pd.get_dummies(test_no_nan_2, columns=['Embarked','Sex'])

# select the useful columns
col = ['Survived', 'PassengerId', 'Ticket', 'Name']
X = data_clean.drop(col, axis=1)
y = data_clean.Survived
col_test = ['PassengerId', 'Ticket', 'Name']
X_test = test_clean.drop(col_test, axis=1)

# split data test/val
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# normalize the data
# code here !!!

# train the linear regression model
model = SVC(kernel='poly')

# fit the model
model.fit(train_X, train_y)

# get predictions for the validation data
val_predict = model.predict(val_X)

# validate model with mean absolute error
print(' Accuracy score is: ', model.score(val_X, val_y))
print('The Mean Absolute Error is : ', mae(val_y, val_predict))
print(val_predict[1:10])

# # Now predict on the test set to get the answer to the competition
# test_predictions = model.predict(X_test)

# # To submit, save like this
# # The lines below shows you how to save your data in the format needed to score it in the competition
# output = pd.DataFrame({'PassengerId': test_clean.PassengerId,
#                        'Survived': test_predictions})

# output.to_csv('submission.csv', index=False)


