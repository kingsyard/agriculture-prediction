import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    int_features2 = np.array(int_features)

    int_features1 = int_features2.reshape(1, -1)
    prediction = model.predict(int_features1)

    output = round(prediction[0])

    return render_template('index.html', prediction_text='production will be    {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)