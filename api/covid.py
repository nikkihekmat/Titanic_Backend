from flask import Blueprint, jsonify, Flask, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from flask_restful import Api, Resource
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
covid_api = Blueprint('covid_api', __name__, url_prefix='/api/covid')
api = Api(covid_api)

# Assuming we have preprocessed COVID data similar to the Titanic dataset preparation
# For this example, let's say we have the following features: new_cases, total_cases, recovery_rate, vaccination_rate
# The target variable is risk_level (High, Medium, Low) encoded as 2, 1, 0 respectively

# Mock dataset - in a real scenario, you would load this from a CSV or database
covid_data = pd.DataFrame({
    'new_cases': [100, 500, 1000],
    'total_cases': [1000, 5000, 10000],
    'recovery_rate': [0.8, 0.5, 0.3],
    'vaccination_rate': [0.7, 0.6, 0.4],
    'risk_level': [0, 1, 2]  # 0=Low, 1=Medium, 2=High
})

# Split the data into features and target
X = covid_data.drop('risk_level', axis=1)
y = covid_data['risk_level']

# Train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X, y)

class PredictRiskLevel(Resource):
    def post(self):
        try:
            data = request.get_json()
            country_data = pd.DataFrame([data])
            
            # In a real scenario, preprocessing would be required similar to the Titanic example
            # For simplicity, we assume data comes preprocessed and directly usable
            
            # Predict the risk level for the provided country data
            risk_level_pred = logreg.predict(country_data)
            risk_level_proba = logreg.predict_proba(country_data)
            
            # Decode the risk level
            risk_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
            risk_level = risk_levels[risk_level_pred[0]]
            
            return {
                'risk_level': risk_level,
                'probabilities': {
                    'Low': risk_level_proba[0][0] * 100,
                    'Medium': risk_level_proba[0][1] * 100,
                    'High': risk_level_proba[0][2] * 100
                }
            }, 200
        except Exception as e:
            return {'error': str(e)}, 400

api.add_resource(PredictRiskLevel, '/predict_risk_level')

if __name__ == "__main__":
    app.run(debug=True)
