import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras


from scikeras.wrappers import KerasClassifier

def load_model():
    return keras.models.load_model("csat_model.h5")


def load_scaler():
    scaler_path = 'C:/Users/HP/Downloads/E-Commerce-Customer-Satisfaction-Score-Prediction-DL-Model/scaler .pkl'
    return joblib.load(scaler_path)

def load_feature_list():
    features_path = 'C:/Users/HP/Downloads/E-Commerce-Customer-Satisfaction-Score-Prediction-DL-Model/features_new.pkl'
    return joblib.load(features_path)
    
    
# Load the encoders and scaler
#with open('label_encoder_gender.pkl', 'rb') as file:
    #label_encoder_gender = pickle.load(file)

#with open('features_new.pkl', 'rb') as file:
    #onehot_encoder_geo = pickle.load(file)

#with open('scaler.pkl', 'rb') as file:
    #scaler = pickle.load(file)    

def preprocess_new_data(data, features, numerical_features):
    empty_df = pd.DataFrame()
    for col in features:
        
        if col not in data.columns:
            empty_df[col] = 0
        elif col in numerical_features:
            empty_df[col] = data[col]
        else:
            empty_df[col] = 1
    return empty_df

st.set_page_config(page_title="CSAT Prediction APP")
st.header("Customer Satisfaction Prediction System")
st.subheader("Input Features for Prediction")

channel_name = st.text_input("Channel Name")
category = st.text_input("Category")
sub_category = st.text_input("Sub-category")
order_date_time = st.text_input("Order Date Time (YYYY-MM-DD HH:MM:SS)")
issue_reported_at = st.text_input("Issue Reported At (YYYY-MM-DD HH:MM:SS)")
issue_responded = st.text_input("Issue Responded (YYYY-MM-DD HH:MM:SS)")
customer_city = st.text_input("Customer City")
product_category = st.text_input("Product Category")
item_price = st.number_input("Item Price", min_value=0.0, step=0.01)
connected_handling_time = st.number_input("Connected Handling Time (seconds)", min_value=0.0, step=0.01)
agent_name = st.text_input("Agent Name")
supervisor = st.text_input("Supervisor")
manager = st.text_input("Manager")
tenure_bucket = st.text_input("Tenure Bucket")
agent_shift = st.text_input("Agent Shift")
survey_response_date = st.text_input("Survey Response Date (01-Aug-23)")

if st.button("Predict CSAT Score"):
    new_data = pd.DataFrame({
        'channel_name': [channel_name],
        'category': [category],
        'Sub-category': [sub_category],
        'order_date_time': [order_date_time],
        'Issue_reported at': [issue_reported_at],
        'issue_responded': [issue_responded],
        'Customer_City': [customer_city],
        'Product_category': [product_category],
        'Item_price': [item_price],
        'connected_handling_time': [connected_handling_time],
        'Agent_name': [agent_name],
        'Supervisor': [supervisor],
        'Manager': [manager],
        'Tenure Bucket': [tenure_bucket],
        'Agent Shift': [agent_shift],
        'Survey_response_Date': [survey_response_date]
    })

    new_data['Issue_reported at'] = pd.to_datetime(new_data['Issue_reported at'], format='%d/%m/%Y %H:%M')
    new_data['issue_responded'] = pd.to_datetime(new_data['issue_responded'], format='%d/%m/%Y %H:%M')
    new_data['Response_Time_seconds'] = (new_data['issue_responded'] - new_data['Issue_reported at']).dt.total_seconds()
    new_data['order_date_time'] = pd.to_datetime(new_data['order_date_time'], format='%d/%m/%Y %H:%M')
    new_data['day_number_order_date'] = new_data['order_date_time'].dt.day
    new_data['Survey_response_Date'] = pd.to_datetime(new_data['Survey_response_Date'], format='%d-%b-%y')
    new_data['day_number_response_date'] = new_data['Survey_response_Date'].dt.day
    new_data['weekday_num_response_date'] = new_data['Survey_response_Date'].dt.weekday + 1
    new_data = new_data.drop(columns=['order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date'])
    
    def rename_column(df,numerical_col):
        y=""
        for cols in df.columns.to_list():
            if cols not in numerical_col:
                y=df[cols]
                df.rename(columns={cols: cols+"_"+y[0]}, inplace=True)
        return df

    numerical_features = ['Item_price', 'connected_handling_time', 'Response_Time_seconds',
       'day_number_order_date', 'day_number_response_date',
       'weekday_num_response_date']
    
    new_data=rename_column(new_data,numerical_features)
    
    scaler = load_scaler()
    
    sorted_features = load_feature_list()
    
    
    new_data1 = preprocess_new_data(new_data, sorted_features, numerical_features)
    new_data1[numerical_features] = scaler.transform(new_data1[numerical_features])
    
    X_test_array = new_data1.values.astype(np.float32)
    
    
    # Load the model
    keras_model = load_model()
    
    # Make predictions
    predictions = keras_model.predict(X_test_array)
    pred_classes = np.argmax(predictions, axis=1)
    
    st.write("Prediction Results")
    st.write(f"The Predicted Customer Satisfaction Score is {int(pred_classes)+1}")
    
    
    
    #print(keras_model.summary())
    
    
            
    st.write("All Predictions:")
    st.write(predictions)
    

    
   
    

    
   
