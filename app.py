import os
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import pyodbc

app = Flask(__name__)
app.secret_key = 'riets-key'  

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

def plot_DDR_single_point(Y_orig, new_point_location, severity, point_color='red', point_size=50):
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
    try:
        # Extract form data
        Age = request.form['Age']
        Dyspnea = request.form['dyspnea']
        DM = request.form['DM']
        SPO2 = request.form['SPO2']
        RR = request.form['RR']
        CRP = request.form['CRP']
        LDH = request.form['LDH']
        ANC = request.form['ANC']
        WBC = request.form['WBC']
        ALC = request.form['ALC']
        PLT = request.form['PLT']

        error_message = None

        # Validate Age
        if not Age.isdigit() or not 0 <= int(Age) <= 120:
            error_message = "Invalid Age entered. Please enter an age between 0 and 120."

        # Validate Dyspnea
        elif not Dyspnea.isdigit() or not 0 <= int(Dyspnea) <= 1:
            error_message = "Invalid value for Dyspnea. Please enter 0 or 1."

        # Validate DM
        elif not DM.isdigit() or not 0 <= int(DM) <= 1:
            error_message = "Invalid value for DM. Please enter 0 or 1."

        # Validate SPO2
        elif not SPO2.isdigit() or not 0 <= int(SPO2) <= 100:
            error_message = "Invalid value for SPO2. Please enter a value between 0 and 100."

        # Validate RR
        elif not RR.isdigit() or not 0 <= int(RR) <= 100:
            error_message = "Invalid value for RR. Please enter a value between 0 and 100."

        # Validate CRP
        elif not CRP.isdigit() or not 0 <= int(CRP) <= 1000:
            error_message = "Invalid value for CRP. Please enter a value between 0 and 1000."

        # Validate LDH
        elif not LDH.isdigit() or not 0 <= int(LDH) <= 10000:
            error_message = "Invalid value for LDH. Please enter a value between 0 and 10000."

        # Validate ANC
        elif not ANC.isdigit() or not 0 <= int(ANC) <= 100000:
            error_message = "Invalid value for ANC. Please enter a value between 0 and 100000."

        # Validate WBC
        elif not WBC.isdigit() or not 0 <= int(WBC) <= 100000:
            error_message = "Invalid value for WBC. Please enter a value between 0 and 100000."

        # Validate ALC
        elif not ALC.isdigit() or not 0 <= int(ALC) <= 100000:
            error_message = "Invalid value for ALC. Please enter a value between 0 and 100000."

        # Validate PLT
        elif not PLT.isdigit() or not 0 <= int(PLT) <= 1000000:
            error_message = "Invalid value for PLT. Please enter a value between 0 and 1000000."

        # If there's an error message, render the home page with the error and retain the previous data
        if error_message:
            return render_template('home.html', error_message=error_message, previous_data=request.form)

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

        # Store results and data in the session
        session['prediction'] = prediction_percentage
        session['Age'] = Age
        session['Dyspnea'] = Dyspnea
        session['DM'] = DM
        session['SPO2'] = SPO2
        session['RR'] = RR
        session['CRP'] = CRP
        session['LDH'] = LDH
        session['ANC'] = ANC
        session['WBC'] = WBC
        session['ALC'] = ALC
        session['PLT'] = PLT
        
        return redirect(url_for('result_page'))

    except Exception as e:
        print(f"Error encountered: {e}")  # This will print the error to your terminal/console
        return render_template('home.html', error_message="An error occurred while processing your request. Please try again.")

@app.route("/result", methods=['GET'])
def result_page():
    if 'prediction' in session:
        return render_template('result.html', 
                               prediction=session['prediction'], 
                               Age=session['Age'], 
                               Dyspnea=session['Dyspnea'], 
                               DM=session['DM'], 
                               SPO2=session['SPO2'], 
                               RR=session['RR'], 
                               CRP=session['CRP'], 
                               LDH=session['LDH'], 
                               ANC=session['ANC'], 
                               WBC=session['WBC'], 
                               ALC=session['ALC'], 
                               PLT=session['PLT'])
    else:
        return redirect(url_for('home'))

def create_connection():
    conn_str = 'Server=tcp:riets-web-server.database.windows.net,1433;Initial Catalog=riets-web-data;Persist Security Info=False;User ID=baeksw98;Password=Qortkddnjs1!;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;'
    conn = pyodbc.connect(conn_str)
    return conn

def save_to_cloud_database(data):
    try:
        with create_connection() as conn:
            cursor = conn.cursor()
            
            query = """INSERT INTO Records (country, race, hospital, dyspnea, dm, spo2, rr, crp, ldh, anc, wbc, alc, plt, prediction_proba, actual_outcome) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            
            cursor.execute(query, (data['country'], data['race'], data['hospital'], data['Dyspnea'], data['DM'], data['SPO2'], data['RR'], data['CRP'], data['LDH'], data['ANC'], data['WBC'], data['ALC'], data['PLT'], data['prediction'], data['actual_outcome']))
            
            conn.commit()
            return True

    except Exception as e:
        print(f"Error: {e}")
        return False

@app.route("/submit_survey", methods=['POST'])
def submit_survey():
    # Extract data from the form
    data = {
        'country': request.form['country'],
        'race': request.form['race'],
        'hospital': request.form['hospital'],
        'Dyspnea': session['Dyspnea'],
        'DM': session['DM'],
        'SPO2': session['SPO2'],
        'RR': session['RR'],
        'CRP': session['CRP'],
        'LDH': session['LDH'],
        'ANC': session['ANC'],
        'WBC': session['WBC'],
        'ALC': session['ALC'],
        'PLT': session['PLT'],
        'prediction': session['prediction'],
        'actual_outcome': request.form['actual_outcome']
    }

    # Call the function to save data to the database
    save_to_cloud_database(data)

    success = save_to_cloud_database(data)
    if success:
        return redirect(url_for('thank_you'))
    else:
        return redirect(url_for('error_page'))

@app.route("/error")
def error_page():
    return render_template('error.html')

@app.route("/thank_you", methods=['GET'])
def thank_you():
    return render_template('thank_you.html')

@app.route('/survey')
def survey():
    return render_template('survey.html', 
                           Age=session['Age'], 
                           Dyspnea=session['Dyspnea'], 
                           DM=session['DM'], 
                           SPO2=session['SPO2'], 
                           RR=session['RR'], 
                           CRP=session['CRP'], 
                           LDH=session['LDH'], 
                           ANC=session['ANC'], 
                           WBC=session['WBC'], 
                           ALC=session['ALC'], 
                           PLT=session['PLT'])


if __name__ == "__main__":
    app.run(debug=True)