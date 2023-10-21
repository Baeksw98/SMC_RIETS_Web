import os
from flask import Flask, render_template, request
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import dynamo
import pickle
import joblib

app = Flask(__name__)

# Load the saved RIETS_scaler
RIETS_scaler = joblib.load('models/RIETS_scaler.pkl')

# Load the saved RIETS model
RIETS_model = load_model("models/RIETS_model.h5")

# Load the severity measures for the 4787 patients
severity_data = pd.read_csv('models/severity_results.csv')
severity = severity_data['severity_proba'].values

# Load the saved DDR_Tree model necessary outputs from the pickle file
W = joblib.load('models/W.pkl')
Y_orig = joblib.load('models/Y_orig.pkl')


@app.route("/") 
def home():
    return render_template('home.html')

def project_to_ddr_space(W, new_data_point):
    projected_point = np.dot(new_data_point, W)
    return projected_point

def plot_DDR_single_point(Y_orig, new_point_location, severity, point_color='red', point_size=30):
    # Transpose Y for easier plotting
    Y = Y_orig.copy().T  

    color_list = ['#B6DA4D', '#8C362A']
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", color_list, N=256)

    fig, ax = plt.subplots(figsize=(4,4))

    # Plotting all the patients
    ax.scatter(Y[:, 0], Y[:, 1], c=severity, alpha=0.7, cmap=custom_cmap, s=20)

    # Plotting the single patient's point
    ax.scatter(new_point_location[0, 0], new_point_location[0, 1], c=point_color, s=point_size, marker='*')

    ax.set_xlabel('Dimension 1', fontsize=12) 
    ax.set_ylabel('Dimension 2', fontsize=12) 

    plt.tight_layout()
    plt.savefig("static/Patient_DDR_Tree.png", dpi=600)
    plt.close()
    
@app.route("/predict", methods=['POST'])
def predict():
    Age = float(request.form['Age'])
    Dyspnea = float(request.form['dyspnea'])
    DM = float(request.form['DM'])
    SPO2 = float(request.form['SPO2'])
    RR = float(request.form['RR'])
    CRP = float(request.form['CRP'])
    LDH = float(request.form['LDH'])
    ANC = float(request.form['ANC'])
    WBC = float(request.form['WBC'])
    ALC = float(request.form['ALC'])
    PLT = float(request.form['PLT'])

    # Create an array with the inputs in the correct order
    input_results = np.array([[CRP, LDH, ALC, ANC, RR, Dyspnea, WBC, Age, SPO2, PLT, DM]])

    # Scale the specific columns using the loaded RIETS_scaler
    results_scaled = RIETS_scaler.transform(input_results)

    # Make a prediction using the RIETS model
    prediction = RIETS_model.predict(results_scaled)
    prediction_percentage = round(prediction[0][0] * 100, 1)

    # Project the patient's data to DDR Tree space
    new_point_location = project_to_ddr_space(W, results_scaled)

    # Use the plot function above to visualize the patient's point on DDR Tree
    plot_DDR_single_point(Y_orig, new_point_location, severity)

    return render_template('result.html', prediction=prediction_percentage)

if __name__ == "__main__":
    app.run(debug=True)