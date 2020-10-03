from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import joblib

app = Flask(__name__)

#Vectorizer
vectorizer = open('vectorizer.pkl','rb')
cv = joblib.load(vectorizer)

#Naive Bayes Classifier
NB_model = open('NB_model.pkl','rb')
clf = joblib.load(NB_model)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
