import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from __init__ import app, db
from flask_sqlalchemy import SQLAlchemy
from flask import Blueprint, jsonify, request  # Import the 'request' object
from flask_restful import Api, Resource, reqparse
from sqlalchemy import Column, Integer, String
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Titanic(db.Model):
    id = Column(Integer, primary_key=True)  # Defines 'id' as a primary key
    name = Column(String)
    # add other columns as needed

class Titanic(db.Model):  
    __tablename__ = 'titanic'  

db = SQLAlchemy()

class TitanicPassenger(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.String(120), nullable=False)
    siblings_spouse = db.Column(db.String(120), nullable=False)
    parents_children = db.Column(db.String(120), nullable=False)
    fare = db.Column(db.Integer, nullable=False)
    # Add other fields and methods as necessary

# Then, in your post method:
def post(self):
    parser = reqparse.RequestParser()
    parser.add_argument("_age", type=str, required=True, help="age is required")
    # Add other arguments similarly
    args = parser.parse_args()

    new_passenger = TitanicPassenger(age=args["_age"], siblings_spouse=args["_siblings/spouse"], parents_children=args["_parents/children"], fare=args["_fare"])
    db.session.add(new_passenger)
    db.session.commit()

    return jsonify(new_passenger.id), 201  # Or return any other appropriate response


# Load the titanic dataset
titanic_data = sns.load_dataset('titanic')

# display(titanic_data[['survived','pclass', 'sex', 'age', 'sibsp', 'parch', 'class', 'fare', 'embark_town', 'alone']]) # look at selected columns
td = titanic_data
td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
td.dropna(inplace=True) # drop rows with at least one missing value, after dropping unuseful columns
td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
td['alone'] = td['alone'].apply(lambda x: 1 if x == True else 0)

# Encode categorical variables
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(td[['embarked']])
onehot = enc.transform(td[['embarked']]).toarray()
cols = ['embarked_' + val for val in enc.categories_[0]]
td[cols] = pd.DataFrame(onehot)
td.drop(['embarked'], axis=1, inplace=True)
td.dropna(inplace=True)

X = td.drop('survived', axis=1) # all except 'survived'
y = td['survived'] # only 'survived'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced')
dt.fit(X_train, y_train)

# Test the model
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('DecisionTreeClassifier Accuracy: {:.2%}'.format(accuracy))  

# Train a logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Test the model
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('LogisticRegression Accuracy: {:.2%}'.format(accuracy)) 