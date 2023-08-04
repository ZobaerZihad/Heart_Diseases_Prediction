from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('report.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/submit', methods=['POST','GET'])
def submit():
    age = int(request.form['age'])
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = request.form['fbs']
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = request.form['exang']
    oldpeak = float(request.form['oldpeak'])
    slope = request.form['slope']
    ca = int(request.form['ca'])
    thal = request.form['thal']
    
    data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    with open('model_RFC.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(data)
    
    
    result = "You are diagnosed with Heart Disease. Consult with a Heart-specialist" if prediction == 1 else "No Heart Disease has been detected"
    
    
    return render_template('submit.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
