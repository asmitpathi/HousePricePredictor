import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app= Flask(__name__)

#load model
regmodel= pickle.load(open('regmodel.pkl', 'rb'))
#load scaler
scaler= pickle.load(open('scaler.pkl', 'rb'))

feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
    'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
    'B', 'LSTAT'
]

#home route
@app.route('/')
def home():
    return render_template('home.html')

#Prediction route
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data= request.json['data']   #capture the input data which is in json format
    print(data)
    input_df = pd.DataFrame(
        [list(data.values())],
        columns=feature_names
    )
    scaled_data = scaler.transform(input_df)
    output= regmodel.predict(scaled_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    if len(data) != 13:
        return render_template(
            "home.html",
            prediction_text="Please enter all 13 feature values."
        )
    input_df = pd.DataFrame(
        [data],
        columns=feature_names
    )
    scaled_data = scaler.transform(input_df)
    output = regmodel.predict(scaled_data)[0]
    return render_template(
        "home.html",
        prediction_text=f"The house price prediction is {output}"
    )

if __name__=='__main__':
    app.run(debug=True)