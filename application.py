from flask import Flask, request, render_template,jsonify
from flask_cors import CORS,cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import pandas as pd

application = Flask(__name__)

app = application

@app.route('/')
@cross_origin()
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            Inches = float(request.form.get('Inches')),
            CPU_Frequency = float(request.form.get('CPU_Frequency')),
            RAM = float(request.form.get('RAM')),
            Weight = float(request.form.get('Weight')),
            Company = request.form.get('Companyt'),
            Product= request.form.get('Product'),
            TypeName = request.form.get('TypeName'),
            ScreenResolution = request.form.get('ScreenResolution'),
            CPU_Company = request.form.get('CPU_Company'),
            CPU_Type = request.form.get('CPU_Type'),
            Memory = request.form.get('Memory'),
            GPU_Company = request.form.get('GPU_Company'),
            GPU_Type = request.form.get('GPU_Type'),
            OpSys = request.form.get('OpSys')
        )

        pred_df = data.get_data_as_dataframe()
        
        print(pred_df)

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        results = round(pred[0],2)
        return render_template('index.html',results=results,pred_df = pred_df)
    
@app.route('/predictAPI',methods=['POST'])
@cross_origin()
def predict_api():
    if request.method=='POST':
        data = CustomData(
            Inches = float(request.json['Inches']),
            CPU_Frequency = float(request.json['CPU_Frequency']),
            RAM = float(request.json['RAM']),
            Weight = float(request.json['Weight']),
            Company = request.json['Company'],
            Product = request.json['Product'],
            TypeName = request.json['TypeName'],
            ScreenResolution = request.json['ScreenResolution'],
            CPU_Company = request.json['CPU_Company'],
            CPU_Type = request.json['CPU_Type'],
            Memory = request.json['Memory'],
            GPU_Company = request.json['GPU_Company'],
            GPU_Type = request.json['GPU_Type'],
            OpSys = request.json['OpSys']
        )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)

        dct = {'Price':round(pred[0],2)}
        return jsonify(dct)

# Add a new route for dynamic dropdown values for products
@app.route('/get_products', methods=['GET'])
@cross_origin()
def get_products():
    try:
        # Load your data
        df = pd.read_csv('C:/Users/user/MACHINE_LEARNING/laptop_price_prediction/notebook/data/laptop_price.csv')  # Replace with your actual dataset path
        
        # Extract unique products from the 'Product' column
        unique_products = df['Product'].dropna().unique().tolist()
        
        # Return the unique product list as JSON
        return jsonify(unique_products)
    except FileNotFoundError:
        return jsonify({"error": "File not found. Ensure 'laptop_data.csv' exists."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Add a new route for dynamic dropdown values for GPU TYPE
@app.route('/get_gpu_type', methods=['GET'])
@cross_origin()
def get_gpu_type():
    try:
        # Load your data
        df = pd.read_csv('C:/Users/user/MACHINE_LEARNING/laptop_price_prediction/notebook/data/laptop_price.csv')  # Replace with your actual dataset path
        
        # Extract unique products from the 'GPU_Type' column
        unique_gpu_type = df['GPU_Type'].dropna().unique().tolist()
        
        # Return the unique GPU_Type list as JSON
        return jsonify(unique_gpu_type)
    except FileNotFoundError:
        return jsonify({"error": "File not found. Ensure 'laptop_data.csv' exists."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Add a new route for dynamic dropdown values for CPU TYPE
@app.route('/get_cpu_type', methods=['GET'])
@cross_origin()
def get_cpu_type():
    try:
        # Load your data
        df = pd.read_csv('C:/Users/user/MACHINE_LEARNING/laptop_price_prediction/notebook/data/laptop_price.csv')  # Replace with your actual dataset path
        
        # Extract unique products from the 'GPU_Type' column
        unique_cpu_type = df['CPU_Type'].dropna().unique().tolist()
        
        # Return the unique GPU_Type list as JSON
        return jsonify(unique_cpu_type)
    except FileNotFoundError:
        return jsonify({"error": "File not found. Ensure 'laptop_data.csv' exists."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)