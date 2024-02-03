from flask import Flask, jsonify,request
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from pandas import read_csv
import pandas as pd
import sys
app = Flask(__name__)
CORS(app)
df12= pd.read_csv('fert.csv')
l12 = LabelEncoder()
df12['Fertilizer_Name'] = l12.fit_transform(df12['Fertilizer_Name'])
l22 = LabelEncoder()
df12['Crop_Type'] = l22.fit_transform(df12['Crop_Type'])
l32 = LabelEncoder()
df12['Soil_Type'] = l32.fit_transform(df12['Soil_Type'])
x2 = df12.iloc[:, df12.columns != 'Fertilizer_Name']
y2 = df12.iloc[:, df12.columns == 'Fertilizer_Name']
model2 = RandomForestClassifier()
# print(x,y.values, file=sys.stderr)
model2.fit(x2, y2.values.ravel())

df = pd.read_csv('crop.csv')
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

x = df.iloc[:, df.columns != 'label']
y = df.iloc[:, df.columns == 'label']

model = RandomForestClassifier()
model.fit(x, y.values.ravel())
# print(x, y.values, file=sys.stderr)
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
        #change the firt 3 column name to N P K
        # print('Input data:', input_data, file=sys.stderr)
        
        print('Input data:', input_data,file=sys.stderr)

        predicted_output = model.predict(input_data)
        predicted_crop_name = le.inverse_transform(predicted_output)[0]

        print('Predicted crop name:', predicted_crop_name)

        return jsonify({'predicted_crop_name': predicted_crop_name})

    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': str(e)})
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
        input_data['Crop_Type'] = l22.transform(input_data['Crop_Type'])
        input_data['Soil_Type'] = l32.transform(input_data['Soil_Type'])
    
        predicted_output = model2.predict(input_data)
        predicted_fertilizer = l12.inverse_transform(predicted_output)[0]

        print('Predicted Fertilizer:', predicted_fertilizer)

        return jsonify({'predicted_fertilizer': predicted_fertilizer})

    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)