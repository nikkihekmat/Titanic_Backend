from flask import Blueprint, jsonify, Flask
from flask_restful import Api, Resource
import requests
import time

# Initialize Flask app
app = Flask(__name__)

# Create Blueprint for COVID API
covid_api = Blueprint('covid_api', __name__, url_prefix='/api/covid')
api = Api(covid_api)

# Global variables
last_run = None
covid_data = None

# Time Keeper function


def update_time():
    global last_run
    if last_run is None or time.time() - last_run > 86400:
        last_run = time.time()
        return True
    return False

# Function to fetch COVID-19 data from the API


def get_covid_data():
    global covid_data
    if update_time():
        url = "https://corona-virus-world-and-india-data.p.rapidapi.com/api"
        headers = {
            'x-rapidapi-key':
                "dec069b877msh0d9d0827664078cp1a18fajsn2afac35ae063",
            'x-rapidapi-host':
                "corona-virus-world-and-india-data.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers)
        covid_data = response.json()
    return covid_data

# Function to filter COVID-19 data by country


def get_country_data(country_filter):
    data = get_covid_data()
    if data:
        countries = data.get('countries_stat')
        for country in countries:
            if country["country_name"].lower() == country_filter.lower():
                return country
    return {"message": country_filter + " not found"}

# API Resources


class CovidData(Resource):
    def get(self):
        return get_covid_data()


class CovidCountryData(Resource):
    def get(self, country_filter):
        return jsonify(get_country_data(country_filter))

# Add resources to the API


api.add_resource(CovidData, '/')
api.add_resource(CovidCountryData, '/<string:country_filter>')

# Register Blueprint
app.register_blueprint(covid_api)

# Main condition to run the app
if __name__ == "__main__":
    app.run(debug=True)
