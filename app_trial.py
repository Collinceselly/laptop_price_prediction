from flask import Flask, request, render_template,jsonify
from flask_cors import CORS,cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import pandas as pd

app = Flask(__name__)

app = app

@app.route('/')
@cross_origin()
def home_page():
    return render_template('index_trial.html')

@app.route('/get_features', methods=['GET'])
def get_features():
    try:
        # Load the dataset
        df = pd.read_csv('C:/Users/user/MACHINE_LEARNING/laptop_price_prediction/notebook/data/laptop_price.csv')
        
        # Extract unique features from the 'Features' column
        unique_features = df['Product'].dropna().unique().tolist()
        
        return jsonify(unique_features)
    except FileNotFoundError:
        return jsonify({"error": "File not found. Ensure 'laptop_data.csv' exists in the same directory."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
