import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.frequency_maps = {}

    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)

        # Create frequency maps for all specified columns
        for col in self.columns:
            if col not in X.columns:
                raise ValueError(f"Column {col} not found in input data.")
            self.frequency_maps[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)

        # Apply the frequency maps
        X = X.copy(deep=True)
        for col in self.columns:
            if col not in X.columns:
                raise ValueError(f"Column {col} not found in input data.")
            X[col] = X[col].map(self.frequency_maps[col]).fillna(0)
        return X
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Company','Product','TypeName','ScreenResolution','CPU_Company','CPU_Type','Memory','GPU_Company','GPU_Type','OpSys']
            numerical_cols = ['Inches','CPU_Frequency','RAM','Weight']

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())                
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('frequecy_encoder',FrequencyEncoder(columns=categorical_cols)),
                ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical Columns : {categorical_cols}')
            logging.info(f'Numerical Columns   : {numerical_cols}')

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.info('Exception occured in Data Transformation Phase')
            raise CustomException(e,sys)
        
    def initate_data_transformation(self,train_path,test_path):

        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Price'
            # drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]


            input_feature_test_df=test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(f"Transformed data type: {type(input_feature_train_df)}")
            logging.info(f"Transformed data shape: {input_feature_train_df.shape}")
            logging.info(f'The head of the train df: \n{input_feature_train_df.head()}')

            logging.info(f"Transformed data type: {type(input_feature_test_df)}")
            logging.info(f"Transformed data shape: {input_feature_test_df.shape}")
            logging.info(f'The head of the test df: \n{input_feature_test_df.head()}')

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
          

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            logging.info('Exception occured in initiate_data_transformation function')
            raise CustomException(e,sys)