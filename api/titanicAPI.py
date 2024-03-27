# Import necessary libraries for the web API, data manipulation, and machine learning
from flask import Flask, request, jsonify, Blueprint
import pandas as pd
from flask_restful import Api, Resource, reqparse
from __init__ import db
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from model.titanic import dt, logreg, enc, cols, titanic_data


# Create a Blueprint for the Titanic API to structure your endpoints
titanic_api = Blueprint('titanic_api', __name__, url_prefix='/api/titanic')

api = Api(titanic_api)

CORS(titanic_api)

class titanicAPI(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("_age", type=str, required=True, help="age is required")
        parser.add_argument("_siblings/spouse", type=str, required=True, help="siblings/spouse is required")
        parser.add_argument("_parents/children", type=str, required=True, help="parents/childrens is required")
        parser.add_argument("_fare", type=int, required=True, help="fare is required")
    
        args = parser.parse_args()

        new_titanic = titanicAPI(_age=args["_age"], _siblings=args["_siblings/spouse"], _parent=args["_parents/children"], _fare=args["_fare"])
        db.session.add(new_titanic)
        db.session.commit()

        return (new_titanic.to_dict()), 201
    
    def get(self, titanic_id=None):  # Add song_id as a parameter with a default value of None
        if titanic_id is not None:
            titanic = titanic.query.get(titanic_id)
            if titanic:
                return (titanic.to_dict())
            return {"error": "surival chance not found"}, 404

# Define a route for predicting survival on the Titanic using a POST request
@titanic_api.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the features from the JSON body of the request
        data = request.get_json(force=True)
        features = pd.DataFrame(data, index=[0])

        # Apply one-hot encoding to the 'embarked' column as done during model training
        # This converts categorical 'embarked' values into a format the model can understand
        onehot = enc.transform(features[['embarked']]).toarray()
        features = features.join(pd.DataFrame(onehot, columns=cols))
        features.drop(['embarked'], axis=1, inplace=True)

        # Fill in any missing values to avoid "Input contains NaN" errors.
        # Replace this with your specific strategy used during model training (mean, median, mode, or a constant value)
        # Example: features.fillna(0, inplace=True)

        # Predict with both models and get the survival probability
        prediction_dt_proba = dt.predict_proba(features)
        prediction_logreg_proba = logreg.predict_proba(features)

        survival_probability_dt = prediction_dt_proba[0][1]  # Decision Tree survival probability
        survival_probability_logreg = prediction_logreg_proba[0][1]  # Logistic Regression survival probability

        # Return both predictions as percentages in JSON format
        return jsonify({
            'DecisionTreeClassifier Survival Probability': f"{survival_probability_dt:.2%}",
            'LogisticRegression Survival Probability': f"{survival_probability_logreg:.2%}"
        })
    except Exception as e:
        # If an error occurs, return the error message in JSON format
        return jsonify({'error': str(e)})

# This code to register the blueprint with the main Flask app is commented out because it should be uncommented and used
# when you're integrating this blueprint into your main Flask application.
# app.register_blueprint(titanic_api)

# This part ensures that the Flask app runs only if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)  # Run the application with debug mode on

api.add_resource(titanicAPI, '/', '/<int:titanic_id>', '/all') 