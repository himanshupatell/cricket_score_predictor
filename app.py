from flask import Flask, render_template, request, send_file
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Teams and Cities
teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi',
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff',
    'Christchurch', 'Trinidad'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    success_message = None

    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        city = request.form['city']
        current_score = int(request.form['current_score'])
        overs = float(request.form['overs'])
        wickets = int(request.form['wickets'])
        last_five = int(request.form['last_five'])

        balls_left = 120 - (overs * 6)
        wicket_left = 10 - wickets
        crr = current_score / overs

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [current_score],
            'balls_left': [balls_left],
            'wicket_left': [wicket_left],
            'current_run_rate': [crr],
            'last_five': [last_five]
        })

        result = pipe.predict(input_df)
        prediction = int(result[0])  # Only single score now ✅

        # Save prediction to CSV
        history_file = 'prediction_history.csv'
        new_prediction = pd.DataFrame({
            'Batting Team': [batting_team],
            'Bowling Team': [bowling_team],
            'City': [city],
            'Current Score': [current_score],
            'Overs Done': [overs],
            'Wickets': [wickets],
            'Runs Last 5 Overs': [last_five],
            'Predicted Final Score': [prediction]
        })

        if os.path.exists(history_file):
            new_prediction.to_csv(history_file, mode='a', header=False, index=False)
        else:
            new_prediction.to_csv(history_file, index=False)

        success_message = "✅ Prediction saved successfully!"

    return render_template('index.html', teams=sorted(teams), cities=sorted(cities), prediction=prediction, success_message=success_message)

# Route to download prediction history
@app.route('/download')
def download_file():
    path = "prediction_history.csv"
    return send_file(path, as_attachment=True)

# Route to clear prediction history
@app.route('/clear')
def clear_history():
    open('prediction_history.csv', 'w').close()
    return "<h2 style='text-align:center;'>✅ Prediction History Cleared Successfully!</h2><br><br><div style='text-align:center;'><a href='/'>Go Back</a></div>"

if __name__ == '__main__':
    app.run(debug=True)
