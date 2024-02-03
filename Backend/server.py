from flask import Flask, jsonify,request
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from pandas import read_csv
import pandas as pd
import sys
app = Flask(__name__)
CORS(app)

df = pd.read_csv('crop.csv')
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

x = df.iloc[:, df.columns != 'label']
y = df.iloc[:, df.columns == 'label']

model = RandomForestClassifier()
model.fit(x, y.values.ravel())

@app.route('/')
def hello_world():
    data = {
        'message': 'Hello, World!',
        'status': 'success'
    }
    return jsonify(data)
@app.route('/predict', methods=['OPTIONS', 'POST'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        print('Received data:', data)

        input_data = pd.DataFrame(data, index=[0])
        print('Input data:', input_data)

        predicted_output = model.predict(input_data)
        predicted_crop_name = le.inverse_transform(predicted_output)[0]

        print('Predicted crop name:', predicted_crop_name)

        return jsonify({'predicted_crop_name': predicted_crop_name})

    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': str(e)})
df = pd.read_csv('fert.csv')
l1 = LabelEncoder()
df['Fertilizer_Name'] = l1.fit_transform(df['Fertilizer_Name'])
l2 = LabelEncoder()
df['Crop_Type'] = l2.fit_transform(df['Crop_Type'])
l3 = LabelEncoder()
df['Soil_Type'] = l3.fit_transform(df['Soil_Type'])
x = df.iloc[:, df.columns != 'Fertilizer_Name']
y = df.iloc[:, df.columns == 'Fertilizer_Name']
model = RandomForestClassifier()
# print(x,y.values, file=sys.stderr)
model.fit(x, y.values.ravel())
@app.route('/predict-fert', methods=['OPTIONS', 'POST'])
def predic():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        print('Received data:', data, file=sys.stderr)

        input_data = pd.DataFrame(data, index=[0])
        print('Input data:', input_data)
        #label encoding
        input_data['Crop_Type'] = l2.transform(input_data['Crop_Type'])
        input_data['Soil_Type'] = l3.transform(input_data['Soil_Type'])

        predicted_output = model.predict(input_data)
        predicted_fertilizer = l1.inverse_transform(predicted_output)[0]

        print('Predicted Fertilizer:', predicted_fertilizer)

        return jsonify({'predicted_fertilizer': predicted_fertilizer})

    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)