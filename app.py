from flask import Flask, request, jsonify
import pandas as pd
import joblib
import xgboost

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Load train data
train_df = pd.read_csv('train.csv')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get parameters from the request
        store = int(request.json.get('Store'))
        dept = int(request.json.get('Dept'))
        date = pd.to_datetime(request.json.get('Date'))

        # Merge input parameters with train data to get the latest features
        merged_df = pd.merge(train_df[(train_df['Store'] == store) & (train_df['Dept'] == dept)], 
                             pd.DataFrame({'Date': [date]}), 
                             how='left', 
                             on=['Store', 'Dept'])

        # Drop 'Date' column
        merged_df.drop('Date', axis=1, inplace=True)

        # Make predictions
        prediction = model.predict(merged_df.iloc[[-1]])

        # Return prediction
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
