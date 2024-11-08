from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
from sklearn.linear_model import LinearRegression
import numpy as np

# Use non-GUI backend for Matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load the data once when the server starts
try:
    data = pd.read_csv('time_series_covid19_confirmed_US.csv')
except Exception as e:
    print("Error loading data:", e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    state_name = request.form['state']
    
    # Filter the data for the specified state
    state_data = data[data['Province_State'].str.lower() == state_name.lower()]

    if state_data.empty:
        result_text = f"State '{state_name}' not found in the dataset."
        return render_template('result.html', result=result_text)

    # Sum across all counties to get state-level data
    state_timeseries = state_data.iloc[:, 11:].sum(axis=0)
    state_timeseries.index = pd.to_datetime(state_timeseries.index, format='%m/%d/%y')

    # Prepare data for linear regression
    days = np.arange(len(state_timeseries)).reshape(-1, 1)  # Day indices as X
    cases = state_timeseries.values  # Case counts as y

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(days, cases)

    # Predict the next day's case count
    next_day = np.array([[len(days)]])  # The day after the last day in the dataset
    next_day_prediction = model.predict(next_day)[0]  # Predicted case count for the next day

    # Get the peak information
    peak_cases = state_timeseries.max()
    peak_date = state_timeseries.idxmax()

    # Create a result message
    result_text = (
        f"The peak for {state_name} was on {peak_date.date()} with {peak_cases} cases. "
        f"The predicted cases for the ({(state_timeseries.index[-1] + pd.Timedelta(days=1)).date()}) "
        f"is approximately {int(next_day_prediction)} cases."
    )

    # Plot the data with prediction
    fig, ax = plt.subplots()
    ax.plot(state_timeseries.index, cases, label='Actual Cases')
    ax.plot(state_timeseries.index[-1] + pd.Timedelta(days=1), next_day_prediction, 'ro', label='Predicted Next Day')
    ax.set_title(f"COVID-19 Cases Time Series for {state_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Confirmed Cases")
    ax.legend()

    # Save the plot to a BytesIO object and encode it in base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    
    # Render the template with the result text, plot data, and location data
    return render_template('result.html', result=result_text, plot_url=plot_url, state_name=state_name)

if __name__ == '__main__':
    app.run(debug=True)
