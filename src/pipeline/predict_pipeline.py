import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 Inches:float,
                 CPU_Frequency:float,
                 RAM:float,
                 Weight:float,
                 Company:str,
                 Product:str,
                 TypeName:str,
                 ScreenResolution:str,
                 CPU_Company:str,
                 CPU_Type:str,
                 Memory:str,
                 GPU_Company:str,
                 GPU_Type:str,
                 OpSys:str):
        
        self.Inches = Inches
        self.CPU_Frequency = CPU_Frequency
        self.RAM = RAM
        self.Weight = Weight
        self.Company = Company
        self.Product = Product
        self.TypeName = TypeName
        self.ScreenResolution = ScreenResolution
        self.CPU_Company = CPU_Company
        self.CPU_Type = CPU_Type
        self.Memory = Memory
        self.GPU_Company = GPU_Company
        self.GPU_Type = GPU_Type
        self.OpSys = OpSys

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Inches':[self.Inches],
                'CPU_Frequency':[self.CPU_Frequency],
                'RAM':[self.RAM],
                'Weight':[self.Weight],
                'Company':[self.Company],
                'Product':[self.Product],
                'TypeName':[self.TypeName],
                'ScreenResolution':[self.ScreenResolution],
                'CPU_Company':[self.CPU_Company],
                'CPU_Type':[self.CPU_Type],
                'Memory':[self.Memory],
                'GPU_Company':[self.GPU_Company],
                'GPU_Type':[self.GPU_Type],
                'OpSys':[self.OpSys]


            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
            