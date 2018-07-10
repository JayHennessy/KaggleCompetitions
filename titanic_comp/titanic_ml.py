import pandas as pd
import sklearn.preprocessing as Imputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing




# first get the data
test_raw = pd.read_csv('C:/Users/JAY/Desktop/Machine_Learning_Course/KaggleCompetions/titanic_comp/data/test.csv')
train_raw = pd.read_csv('C:/Users/JAY/Desktop/Machine_Learning_Course/KaggleCompetions/titanic_comp/data/train.csv')

data = train_raw.copy()
data_test = test_raw.copy()

# see data info
#print(data_test.head())
#print(data.columns)

# process the data, drop data points that are missing data in age or fare but drop the cabin column
#print('number of y NANs : ', y.isnull().sum())
#print('number of X NANs : ', X.isnull().sum())
data_no_nan = data.drop('Cabin', axis=1)
test_no_nan = data_test.drop('Cabin', axis=1)

data_no_nan_2 = data_no_nan.dropna(axis=0, subset=['Age','Fare'])

# change categorical data
data_clean = pd.get_dummies(data_no_nan_2, columns=['Embarked','Sex'])
test_clean = pd.get_dummies(test_no_nan, columns=['Embarked','Sex'])

# select the useful columns
col = ['Survived', 'PassengerId', 'Ticket', 'Name']
X = data_clean.drop(col, axis=1)
y = data_clean.Survived
col_test = ['PassengerId', 'Ticket', 'Name']
X_test = test_clean.drop(col_test, axis=1)

# split data test/val
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# normalize the data
print(train_X.describe())
# code here !!!

# train the linear regression model
model = LinearRegression()

# fit the model
model.fit(train_X, train_y)

# get predictions for the validation data
val_predict = model.predict(val_X)

# validate model with mean absolute error
print('The Mean Absolute Error is : ', mae(val_y, val_predict))


# Now predict on the test set to get the answer to the competition
test_precitions = model.predict(X_test)
print(test_precitions.head())
print(test_precitions.info())


