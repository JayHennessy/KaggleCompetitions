import pandas as pd
import sklearn

# first get the data
test_raw = pd.read_csv('C:/Users/JAY/Desktop/Machine_Learning_Course/KaggleCompetions/titanic_comp/data/test.csv')
train_raw = pd.read_csv('C:/Users/JAY/Desktop/Machine_Learning_Course/KaggleCompetions/titanic_comp/data/train.csv')

# see data info
print(train_raw.head())
print(train_raw.columns)

# select the useful columns
X = test_raw.drop('Survived', 'PassengerId')


# process the data

# train the model

# test the model