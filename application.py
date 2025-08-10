import pickle
from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app=application

# Load the pre-trained model
ridge_model = pickle.load(open('models/ridge_cv_model.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        pass
    else:
        return render_template('home.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
