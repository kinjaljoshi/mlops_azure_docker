import h2o
import pandas as pd
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from flask import Flask, render_template, request

# Initialize H2O outside the app launch so it's done only once
h2o.init()

# Load the pre-trained model once when the app starts
model_path = "./predictions/best_diabetes_prediction"
model = h2o.load_model(model_path)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        pregnancies = request.form.get('pregnancies')
        glucose = request.form.get('glucose')
        bloodPressure = int(request.form['bloodPressure'])
        skinThickness = int(request.form['skinThickness'])
        insulin = request.form.get('insulin')
        bmi = int(request.form['bmi'])
        diabetesPedigreeFunction = request.form['diabetesPedigreeFunction']
        age = request.form.get('age')
        
    # Step 2: Organize the input into a dictionary (each key is a feature)
    data = {
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [bloodPressure],
        'SkinThickness': [skinThickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetesPedigreeFunction],
        'Age': [age]
    }

    # Step 3: Convert the dictionary into a pandas DataFrame
    input_df = pd.DataFrame(data)

    # Step 4: Convert the pandas DataFrame into an H2OFrame
    # h2o.init()
    h2o_input = h2o.H2OFrame(input_df)
    # # Load model
    # model_path = "./predictions/best_diabetes_prediction"
    # model = h2o.load_model(model_path)

    prediction = model.predict(h2o_input)

    # Step 5: Display the prediction result
    print(prediction)

    # Option 1: Convert the prediction H2OFrame to a pandas DataFrame to extract individual values
    prediction_df = prediction.as_data_frame()

    print(prediction_df.head())

    # Extract individual values
    predicted_class = prediction_df['predict'][0]  # Predicted class

    return render_template('result.html', prediction=predicted_class)     

if __name__ == '__main__':
	app.run(debug=False, host="0.0.0.0",port=8000)
     


# # Step 1: Initialize H2O
# h2o.init()

# # Step 2: Load the pretrained model
# # Replace 'path_to_saved_model' with the actual path where your H2O model is saved
# model_path = "./predictions/best_diabetes_prediction"
# model = h2o.load_model(model_path)

# # Step 3: Prepare test data
# # Sample test data, replace with actual data
# sample_data = {
#     'Pregnancies': 2,
#     'Glucose': 140,
#     'BloodPressure': 80,
#     'SkinThickness': 35,
#     'Insulin': 120,
#     'BMI': 30.0,
#     'DiabetesPedigreeFunction': 0.5,
#     'Age': 45
# }

# # Convert sample data into a pandas DataFrame
# sample_df = pd.DataFrame([sample_data])

# # Convert pandas DataFrame to H2OFrame (H2O's data structure)
# test_data = h2o.H2OFrame(sample_df)

# # Step 4: Perform a prediction using the loaded model
# prediction = model.predict(test_data)

# # Step 5: Display the prediction result
# print(prediction)

# # Option 1: Convert the prediction H2OFrame to a pandas DataFrame to extract individual values
# prediction_df = prediction.as_data_frame()

# print(prediction_df.head())

# # Extract individual values
# predicted_class = prediction_df['predict'][0]  # Predicted class
# probability_no = prediction_df['no'][0]  # Probability of class "no"
# probability_yes = prediction_df['yes'][0]  # Probability of class "yes"

# # Print the extracted values
# print(f"Predicted Class: {predicted_class}")
# print(f"Probability of No: {probability_no}")
# print(f"Probability of Yes: {probability_yes}")

# # Get individual values from the pandas DataFrame
# # predicted_class = prediction_df['predict'][0]  # Predicted class (0 or 1)
# # probability_no_diabetes = prediction_df['p0'][0]  # Probability of class 0 (no diabetes)
# # probability_diabetes = prediction_df['p1'][0]  # Probability of class 1 (diabetes)

# # # Print the values
# # print(f"Predicted Class: {predicted_class}")
# # print(f"Probability of No Diabetes (p0): {probability_no_diabetes}")
# # print(f"Probability of Diabetes (p1): {probability_diabetes}")