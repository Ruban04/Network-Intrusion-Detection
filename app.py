import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
loaded_label_encoder = joblib.load('label_encoder.pkl')
loaded_model = joblib.load('xgboost_model.pkl')

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/index', methods=['POST'])
def index():
    # Check login credentials
    # Example: if request.form['username'] == 'admin' and request.form['password'] == 'password':
    #             return render_template('index.html')
    #         else:
    #             return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    new_data = pd.DataFrame({
        'flow_duration': [float(request.form['flow_duration'])],
        'Header_Length': [float(request.form['Header_Length'])],
        'Protocol Type': [float(request.form['Protocol_Type'])],
        'Duration': [float(request.form['Duration'])],
        'Rate': [float(request.form['Rate'])],
        'Srate': [float(request.form['Srate'])],
        'fin_flag_number': [float(request.form['fin_flag_number'])],
        'Std': [float(request.form['Std'])],
        'Tot size': [float(request.form['Tot_size'])],
        'IAT': [float(request.form['IAT'])],
        'Magnitue': [float(request.form['Magnitue'])],
        'Radius': [float(request.form['Radius'])],
        'Weight': [float(request.form['Weight'])]
    })

    # Make predictions
    predictions = loaded_model.predict(new_data)
    
    # Decode predicted labels
    decoded_predictions = loaded_label_encoder.inverse_transform(predictions)
    
    return render_template('result.html', predictions=decoded_predictions)

if __name__ == '__main__':
    app.run(debug=True)
