import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
titanic_data = sns.load_dataset('titanic')

# Display basic information about the dataset
print("Titanic Data")
print(titanic_data.columns)
print(titanic_data[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch',
                    'fare', 'embark_town', 'alone']])

# Preprocess the data
td = titanic_data.copy()
td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'],
        axis=1, inplace=True)
td.dropna(inplace=True)
td['sex'] = td['sex'].apply(lambda x: True if x == 'male' else False)
td['alone'] = td['alone'].apply(lambda x: True if x else False)

# Encode categorical variables
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(td[['embarked']])
onehot = enc.transform(td[['embarked']]).toarray()
cols = ['embarked_' + val for val in enc.categories_[0]]
td[cols] = pd.DataFrame(onehot)
td.drop(['embarked'], axis=1, inplace=True)
td.dropna(inplace=True)

# Display preprocessed data
print(td.columns)
print(td)

# Display statistics
print(titanic_data.median())
print(titanic_data.query("survived == 0").mean())
print(td.query("survived == 1").mean())
print("Maximums for survivors")
print(td.query("survived == 1").max())
print()
print("Minimums for survivors")
print(td.query("survived == 1").min())

# Split data into train and test sets
X = td.drop('survived', axis=1)
y = td['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# Train and test a decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('DecisionTreeClassifier Accuracy: {:.2%}'.format(accuracy))

# Train and test a logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('LogisticRegression Accuracy: {:.2%}'.format(accuracy))

# Define a new passenger
passenger = pd.DataFrame({
    'name': ['Nikki Hekmat'],
    'pclass': [2],
    'sex': ['female'],
    'age': [18],
    'sibsp': [1],
    'parch': [2],
    'fare': [16.00],
    'embarked': ['S'],
    'alone': [False]
})

print(passenger)
new_passenger = passenger.copy()

# Preprocess the new passenger data
new_passenger['sex'] = new_passenger['sex'].apply(lambda x: True if x == 'male'
                                                  else False)
new_passenger['alone'] = new_passenger['alone'].apply(lambda x: True if x else
                                                      False)
onehot = enc.transform(new_passenger[['embarked']]).toarray()
cols = ['embarked_' + val for val in enc.categories_[0]]
new_passenger[cols] = pd.DataFrame(onehot, index=new_passenger.index)
new_passenger.drop(['name', 'embarked'], axis=1, inplace=True)

print(new_passenger)

# Predict the survival probability for the new passenger
dead_proba, alive_proba = np.squeeze(logreg.predict_proba(new_passenger))

# Print the survival probability
print('Death probability: {:.2%}'.format(dead_proba))
print('Survival probability: {:.2%}'.format(alive_proba))
