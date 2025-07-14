from flask import Flask, render_template, request, send_from_directory, jsonify, redirect
import pandas as pd
import numpy as np
import pickle
import os
import requests
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

# Google Drive file links
pkl_url = "https://drive.google.com/uc?export=download&id=1qw-CLrfc3UaWKhj1imj35Qia1hBKVnMe"
xlsx_url = "https://drive.google.com/uc?export=download&id=1KLJYrbHoshYW6ETUzI9iRKX64HJgjKqs"

model_path = 'model/nifty_processed.pkl'
data_path = 'data/processed_data.xlsx'
prediction_log = []

def load_model():
    response = requests.get(pkl_url)
    return pickle.loads(response.content)

def load_data():
    response = requests.get(xlsx_url)
    df = pd.read_excel(BytesIO(response.content))
    df.dropna(inplace=True)
    df['Lag1'] = df['Return'].shift(1)
    df['Lag2'] = df['Return'].shift(2)
    df['Lag3'] = df['Return'].shift(3)
    df.dropna(inplace=True)
    return df

def calculate_metrics(model, df):
    X = df[['Lag1', 'Lag2', 'Lag3']]
    y = df['Return']
    y_pred = model.predict(X)
    return round(r2_score(y, y_pred), 4), round(mean_squared_error(y, y_pred), 6)

model = load_model()
df = load_data()
model_r2, model_mse = calculate_metrics(model, df)

@app.route('/')
def home():
    return render_template('index.html', prediction_text='', prediction_log=prediction_log,
                           model_r2=model_r2, model_mse=model_mse)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        lag1 = float(request.form['lag1'])
        lag2 = float(request.form['lag2'])
        lag3 = float(request.form['lag3'])
        input_data = np.array([[lag1, lag2, lag3]])
        prediction = model.predict(input_data)[0]
        entry = {'Lag1': lag1, 'Lag2': lag2, 'Lag3': lag3, 'Prediction': round(prediction, 4)}
        prediction_log.append(entry)
        pd.DataFrame(prediction_log).to_excel('data/prediction_log.xlsx', index=False)
        return render_template('index.html', prediction_text=f"📈 Predicted Return: {prediction:.4f}",
                               prediction_log=prediction_log, model_r2=model_r2, model_mse=model_mse)
    except Exception:
        return render_template('index.html', prediction_text='⚠️ Invalid input!',
                               prediction_log=prediction_log, model_r2=model_r2, model_mse=model_mse)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['csv_file']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            df['Return'] = df['Close'].pct_change()
            df.dropna(inplace=True)
            df.to_excel(data_path, index=False)
            return redirect('/')
        return "Invalid file format. Only .csv accepted."
    return render_template('upload.html')

@app.route('/retrain', methods=['POST'])
def retrain():
    df = pd.read_excel(data_path)
    df['Lag1'] = df['Return'].shift(1)
    df['Lag2'] = df['Return'].shift(2)
    df['Lag3'] = df['Return'].shift(3)
    df.dropna(inplace=True)
    X = df[['Lag1', 'Lag2', 'Lag3']]
    y = df['Return']
    new_model = LinearRegression()
    new_model.fit(X, y)
    pickle.dump(new_model, open(model_path, 'wb'))

    global model, model_r2, model_mse
    model = new_model
    model_r2, model_mse = calculate_metrics(model, df)
    return redirect('/')

@app.route('/download')
def download_data():
    return send_from_directory('data', 'processed_data.xlsx', as_attachment=True)

@app.route('/download-log')
def download_log():
    return send_from_directory('data', 'prediction_log.xlsx', as_attachment=True)

@app.route('/chart-data')
def chart_data():
    df = load_data()
    return jsonify({
        'dates': df['Date'].astype(str).tolist(),
        'close': df['Close'].tolist(),
        'high': df['High'].tolist(),
        'low': df['Low'].tolist()
    })

@app.route('/about')
def about():
    return render_template('about.html')

# Swagger UI
SWAGGER_URL = '/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "Nifty Predictor"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    app.run(debug=True)
