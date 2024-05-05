# use sklearn to develop a pipeline to preprocess the data and train the model
# the pipeline should load the csv datasets, merge them, do all the transformations as necessary, do the feature engineering we did before (date features, lag features, rolling features, expanding features)
# then it should preprocess the data (one hot encoding, standard scaling), split the data into training and test data, train the model and save the model

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

class LoadData(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # load the data
        train_df = pd.read_csv('train.csv')
        store_df = pd.read_csv('stores.csv')
        feature_df = pd.read_csv('features.csv')

        # merge the data
        train_df = train_df.merge(store_df, how='left', on='Store').merge(feature_df, how='left', on=['Store', 'Date', 'IsHoliday'])

        return train_df
    
class FeatureEngineering(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # extract month
        X['month'] = pd.to_datetime(X['Date']).dt.month

        # extract week number
        X['week'] = pd.to_datetime(X['Date']).dt.isocalendar().week.astype("int32")

        # extract year
        X['year'] = pd.to_datetime(X['Date']).dt.year

        # sort the data
        X = X.sort_values(by=['Store', 'Dept', 'Date'])

        # lag features
        X['Weekly_Sales_lag1'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
        X['Weekly_Sales_lag2'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(2)
        X['Weekly_Sales_lag3'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(3)

        # rolling mean and std features
        X['Weekly_Sales_roll_mean_3'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=3).mean())
        X['Weekly_Sales_roll_mean_5'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=5).mean())
        X['Weekly_Sales_roll_mean_7'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=7).mean())
        X['Weekly_Sales_roll_mean_14'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=14).mean())

        X['Weekly_Sales_roll_std_3'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=3).std())
        X['Weekly_Sales_roll_std_5'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=5).std())
        X['Weekly_Sales_roll_std_7'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=7).std())
        X['Weekly_Sales_roll_std_14'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=14).std())

        # historical avg and std features
        X['Weekly_Sales_avg_store_dept_week'] = X.groupby(['Store', 'Dept', 'week'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().mean())
        X['Weekly_Sales_std_store_dept_week'] = X.groupby(['Store', 'Dept', 'week'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().std())

        X['Weekly_Sales_avg_store_week'] = X.groupby(['Store', 'week'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().mean())
        X['Weekly_Sales_std_store_week'] = X.groupby(['Store', 'week'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().std())

        X['Weekly_Sales_avg_dept_week'] = X.groupby(['Dept', 'week'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().mean())
        X['Weekly_Sales_std_dept_week'] = X.groupby(['Dept', 'week'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().std())

        X['Weekly_Sales_avg_store_dept'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().mean())
        X['Weekly_Sales_std_store_dept'] = X.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().std())

        X['Weekly_Sales_avg_store'] = X.groupby(['Store'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().mean())
        X['Weekly_Sales_std_store'] = X.groupby(['Store'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().std())

        X['Weekly_Sales_avg_dept'] = X.groupby(['Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().mean())
        X['Weekly_Sales_std_dept'] = X.groupby(['Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().std())

        X['Weekly_Sales_avg_week'] = X.groupby(['week'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().mean())
        X['Weekly_Sales_std_week'] = X.groupby(['week'])['Weekly_Sales'].transform(lambda x: x.shift(1).expanding().std())

        return X
    
class PreprocessData(BaseEstimator, TransformerMixin):
    
        def fit(self, X, y=None):
            return self
    
        def transform(self, X, y=None):
            # encode categorical variables
            encoder = OneHotEncoder()
            X_encoded = encoder.fit_transform(X[['Store', 'Dept', 'Type']])
            X_encoded = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(['Store', 'Dept', 'Type']))
    
            # merge the encoded data with the original data
            X = pd.concat([X, X_encoded], axis=1)
    
            # drop the original columns
            X.drop(['Store', 'Dept', 'Type'], axis=1, inplace=True)
    
            # list of all numerical columns
            num_features = ['Size', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 
                'CPI', 'Unemployment', 'IsHoliday', 'month', 'week', 'year', 'Weekly_Sales_lag1', 
                'Weekly_Sales_lag2', 'Weekly_Sales_lag3', 'Weekly_Sales_roll_mean_3', 'Weekly_Sales_roll_mean_5', 
                'Weekly_Sales_roll_mean_7', 'Weekly_Sales_roll_mean_14', 'Weekly_Sales_roll_std_3', 'Weekly_Sales_roll_std_5', 
                'Weekly_Sales_roll_std_7', 'Weekly_Sales_roll_std_14', 'Weekly_Sales_avg_store_dept_week', 'Weekly_Sales_std_store_dept_week', 
                'Weekly_Sales_avg_store_week', 'Weekly_Sales_std_store_week', 'Weekly_Sales_avg_dept_week', 'Weekly_Sales_std_dept_week', 
                'Weekly_Sales_avg_store_dept', 'Weekly_Sales_std_store_dept', 'Weekly_Sales_avg_store', 'Weekly_Sales_std_store', 
                'Weekly_Sales_avg_dept', 'Weekly_Sales_std_dept', 'Weekly_Sales_avg_week', 'Weekly_Sales_std_week']
    
            # list of all categorical columns 
            cat_features = X.columns[~X.columns.isin(num_features)].tolist()
    
            # create an instance of standard scaler
            scaler = StandardScaler()
    
            # normalise the numerical columns
            X[num_features] = scaler.fit_transform(X[num_features])

            return X
        
class TrainModel(BaseEstimator, TransformerMixin):
        
        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            # split the data into training and test data
            X_train = X[X['Date'] < '2012-09-01']
    
            # split the data into features and target
            y_train = X_train['Weekly_Sales']
            X_train = X_train.drop(['Date', 'Weekly_Sales'], axis=1)
    
            # split the data into training and test data
            X_test = X[X['Date'] >= '2012-09-01']
    
            # split the data into features and target
            y_test = X_test['Weekly_Sales']
            X_test = X_test.drop(['Date', 'Weekly_Sales'], axis=1)
    
            # create an instance of the xgboost regressor
            model = XGBRegressor()
    
            # fit the model
            model.fit(X_train, y_train)
    
            # make predictions
            train_preds = model.predict(X_train)
    
            # calculate the rmse
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    
            # make predictions
            test_preds = model.predict(X_test)
    
            # calculate the rmse
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
            # save the model
            joblib.dump(model, 'model.pkl')
    
            return train_rmse, test_rmse
        
# create a pipeline
pipeline = Pipeline([
    ('load_data', LoadData()),
    ('feature_engineering', FeatureEngineering()),
    ('preprocess_data', PreprocessData()),
    ('train_model', TrainModel())
])

if __name__ == "__main__":
    # Execute the pipeline
    pipeline.fit(None)