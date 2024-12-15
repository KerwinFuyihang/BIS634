from flask import Flask, request, jsonify, render_template
import joblib
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os


app = Flask(__name__)

# Load your trained model
model = joblib.load('/Users/yihang/Desktop/BIS_634/finalproject/AQI-prediction-Using-Flask-Web-App-main/.ipynb_checkpoints/airquality.joblib')
data = pd.read_csv('/Users/yihang/Desktop/BIS_634/finalproject/AQI-prediction-Using-Flask-Web-App-main/.ipynb_checkpoints/data.csv')
#data['Date'] = pd.to_datetime(data['Date'])  # Ensure Date is in datetime format
# OpenWeather API Key
API_KEY = 'bd5e378503939ddaee76f12ad7a97608'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/heatmap')
def heatmap():
    return render_template('heatmap.html')

@app.route('/predict_manually', methods=['POST','GET'])
def predict_manually():
    if request.method == 'POST':
        # Extract data from form
        pm25 = float(request.form['PM2.5'])
        pm10 = float(request.form['PM10'])
        o3 = float(request.form['O3'])
        no2 = float(request.form['NO2'])
        co = float(request.form['CO'])
        so2 = float(request.form['SO2'])

        # Prepare data for prediction
        sample = [[pm25, pm10, o3, no2, co, so2]]
        prediction = model.predict(sample)[0]

        # Determine Air Quality Index based on prediction
        result, conclusion = determine_air_quality(prediction)

        # Return the result to the user
        return render_template('manual_results.html', prediction=prediction, result=result, conclusion=conclusion)
    else:
        return render_template('index.html')
    
    
@app.route('/predict_automatically', methods=['GET', 'POST'])
def predict_automatically():
    if request.method == 'POST':
        city_name = request.form.get('city_name')
        if not city_name:
            error_message = "Missing city name parameter"
            error_code = 400
            return render_template('error.html', error=error_message, error_code=error_code), 400

        # Fetch air quality data for the city
        geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={API_KEY}"
        geocode_response = requests.get(geocode_url)
        if geocode_response.status_code != 200:
            error_message = "Failed to fetch location data"
            error_code = 500
            return render_template('error.html', error=error_message, error_code=error_code), 500

        geocode_data = geocode_response.json()
        if not geocode_data:
            error_message = "City not found"
            error_code = 404
            return render_template('error.html', error=error_message, error_code=error_code), 404

        lat = geocode_data[0]['lat']
        lon = geocode_data[0]['lon']

        air_quality_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        air_quality_response = requests.get(air_quality_url)
        if air_quality_response.status_code != 200:
            error_message = "Failed to fetch Air Quality Index data"
            error_code = 500
            return render_template('error.html', error=error_message, error_code=error_code), 500

        air_quality_data = air_quality_response.json()['list'][0]['components']
        sample = [
            [air_quality_data['pm2_5'], air_quality_data['pm10'], air_quality_data['o3'],
             air_quality_data['no2'], air_quality_data['co'], air_quality_data['so2']]
        ]
        prediction = round(model.predict(sample)[0], 2)

        # Generate a line plot for historical AQI of the city
        city_data = data[data['City'].str.lower() == city_name.lower()]
        if city_data.empty:
            error_message = "No historical data available for this city."
            error_code = 404
            return render_template('error.html', error=error_message, error_code=error_code), 404

        plt.figure(figsize=(10, 6))
        plt.plot(city_data['Date'], city_data['AQI'], label='Historical AQI', color='blue')
        plt.scatter([city_data['Date'].iloc[-1]], [prediction], color='red', label=f'Predicted AQI: {prediction}', zorder=5)
        plt.title(f'AQI Trends for {city_name}', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('AQI', fontsize=14)
        plt.legend()
        plt.grid(True)

        # Save the plot to the static folder
        plot_path = os.path.join(app.root_path, 'static', 'aqi_plot.png')
        print(f"Saving plot to: {plot_path}")  # Debug statement
        plt.savefig(plot_path)
        plt.close()

        # Render results template with the plot
        result, conclusion = determine_air_quality(prediction)
        return render_template(
            'results.html',
            prediction=prediction,
            result=result,
            conclusion=conclusion,
            plot_url=plot_path
        )
    else:
        return render_template('city.html')
    
@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        # Get form data
        city1 = request.form['city1']
        city2 = request.form['city2']
        time_range = int(request.form['time_range'])  # Get the selected time range in days

        # Filter AQI data for the selected cities
        city1_data = data[data['City'].str.lower() == city1.lower()]
        city2_data = data[data['City'].str.lower() == city2.lower()]

        # Check if data exists for both cities
        if city1_data.empty or city2_data.empty:
            error_message = "One or both cities not found in the dataset. Please try again."
            return render_template('compare.html', error_message=error_message)

        # Convert the 'Date' column to datetime (with day-first format)
        city1_data['Date'] = pd.to_datetime(city1_data['Date'], dayfirst=True)
        city2_data['Date'] = pd.to_datetime(city2_data['Date'], dayfirst=True)

        # Filter data for the selected time range
        end_date = city1_data['Date'].max()  # Get the latest date in the data
        start_date = end_date - pd.Timedelta(days=time_range)

        city1_data = city1_data[(city1_data['Date'] >= start_date) & (city1_data['Date'] <= end_date)]
        city2_data = city2_data[(city2_data['Date'] >= start_date) & (city2_data['Date'] <= end_date)]

        # Check if both datasets have data in the filtered range
        if city1_data.empty or city2_data.empty:
            error_message = f"One or both cities do not have data for the past {time_range} days. Please try another range or cities."
            return render_template('compare.html', error_message=error_message)

        # Calculate average AQI
        city1_avg_aqi = city1_data['AQI'].mean()
        city2_avg_aqi = city2_data['AQI'].mean()

        # Generate text-based conclusion
        if city1_avg_aqi < city2_avg_aqi:
            conclusion = (
                f"{city1} has a better air quality on average with an AQI of {city1_avg_aqi:.2f}, "
                f"compared to {city2}, which has an average AQI of {city2_avg_aqi:.2f}."
            )
        elif city1_avg_aqi > city2_avg_aqi:
            conclusion = (
                f"{city2} has a better air quality on average with an AQI of {city2_avg_aqi:.2f}, "
                f"compared to {city1}, which has an average AQI of {city1_avg_aqi:.2f}."
            )
        else:
            conclusion = f"Both {city1} and {city2} have similar air quality with an average AQI of {city1_avg_aqi:.2f}."

        # Create the comparison plot
        plt.figure(figsize=(12, 6))
        plt.plot(city1_data['Date'], city1_data['AQI'], label=f"{city1} AQI", color='blue', linewidth=2)
        plt.plot(city2_data['Date'], city2_data['AQI'], label=f"{city2} AQI", color='red', linewidth=2)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Air Quality Index (AQI)', fontsize=14)
        plt.title(f"AQI Comparison: {city1} vs {city2} (Last {time_range} Days)", fontsize=16)
        plt.legend()

        # Adjust x-axis ticks for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(app.root_path, 'static', 'comparison_plot.png')
        plt.savefig(plot_path)
        plt.close()

        # Render the results page with the plot and conclusion
        return render_template(
            'comparison_results.html',
            city1=city1,
            city2=city2,
            plot_url='/static/comparison_plot.png',
            conclusion=conclusion
        )

    return render_template('compare.html')







def determine_air_quality(prediction):
    if prediction < 50:
        return 'Air Quality Index is Good', 'The Air Quality Index is excellent. It poses little or no risk to human health.'
    elif 51 <= prediction < 100:
        return 'Air Quality Index is Satisfactory', 'The Air Quality Index is satisfactory, but there may be a risk for sensitive individuals.'
    elif 101 <= prediction < 200:
        return 'Air Quality Index is Moderately Polluted', 'Moderate health risk for sensitive individuals.'
    elif 201 <= prediction < 300:
        return 'Air Quality Index is Poor', 'Health warnings of emergency conditions.'
    elif 301 <= prediction < 400:
        return 'Air Quality Index is Very Poor', 'Health alert: everyone may experience more serious health effects.'
    else:
        return 'Air Quality Index is Severe', 'Health warnings of emergency conditions. The entire population is more likely to be affected.'

if __name__ == '__main__':
    app.run(debug=True)
