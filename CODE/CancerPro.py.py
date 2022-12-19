from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import csv 
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/SignUp')
def SignUp():
    return render_template('SignUp.html')
@app.route('/SignIn') 
def SignIn():
   return render_template('SignIn.html')
@app.route('/contact')
def contact():
   return render_template('about.html')
@app.route('/NextInput')
def NextInput():
    return render_template('NextInput.html')
@app.route('/parkForm')
def parkForm():
    return render_template('parkForm.html')
@app.route('/men')
def men():
    return render_template('men.html')
@app.route('/result')
def result():
   return render_template('result.html')
@app.route('/predict', methods=['POST'])
def parkPredict():
    df = pd.read_csv("CancerPP.csv")
#the line of code above invokes the csv file needed for the simulation of the code.
    X = df.drop(['Age', 'AALMC', 'AAFP','DOBF', 'WEIGHT','FHS', 'ERE', 'UOBCD', 'PHT','AINTKE', 'Smoking'], axis = 1 )
    y = df['Age']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)
    estimator = RandomForestClassifier(n_estimators = 100, random_state = 40, n_jobs = -1)
    selectFeatures = RFE(estimator, n_features_to_select = 14)
    selectFeatures.fit(X_train, y_train)
#the data training is done on line of code bellow.
    Xtrain = selectFeatures.transform(X_train)
    Xtest = selectFeatures.transform(X_test)
   #Here the csv file used by random forest for the mathmatical simulation of the code_info(x)
    model = SVC( C = 10.0, kernel = 'rbf', gamma = 'auto', shrinking = True, max_iter = -1)
    model.fit(Xtrain, y_train)
    if request.method == 'POST':
        #Age = request.form['Age']
        AOLMC = request.form['AAFMC']
        AALMC = request.form['AALMC']
        AAFP = request.form['AAFP']
        DOBF = request.form['DOBF']
        WEIGHT = request.form['WEIGHT']
        #FHS = request.form['FHS']
        #ERE = request.form['ERE']
        HIM = request.form['HIM']
       # UOBCD = request.form['UOBCD']
        #PHT = request.form['PHT']
        #AINTKE = request.form['AINTKE']
        #Smoking = request.form['Smoking']       
#THE averge number requiored for the code is that of infinite decimal
        test = np.array([AOLMC, AALMC], dtype = 'float64')
        test = test.reshape(1, -1)
        test = test.astype(float)
        my_prediction = model.predict(test)     
    return render_template('result.html', prediction = my_prediction)
if __name__ == "__main__":
    app.run(debug=True)

