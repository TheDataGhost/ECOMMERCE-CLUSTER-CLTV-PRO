import streamlit as st
import pandas as pd
import pickle as pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Customer Segmentation E-commerce",
    page_icon=":female_doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_clean_data():

    data = pd.read_csv("/Users/Dataghost/Machine Learning/The Ghostmode.csv")

    print(data.head())

    data.drop(['CustomerID','RFM_SCORE'],axis =1,inplace=True)
  
    data['Gender']=data['Gender'].map({'Male':1,'Female':0})

    for i in data.columns:

      if i != 'Segment' and data[i].dtype == 'object':

         label_encoder = LabelEncoder()

         data[i] = label_encoder.fit_transform(data[i])

   
    return data

def add_sidebar():
      
    st.sidebar.header("Customer Data")

    data = get_clean_data()

    slider_labels = [
        ("Gender","Gender"),
        ("Location", "Location"),
        ("TransactionAmount", "TransactionAmount_mean"),
        ("Account Balance", "AccountBalance_mean"),
        ("Age", "CustomerAge"),
        ("Unique Purchase", "Unique_purchase"),
        ("Total Spend", "Total_spend"),
        ("Recency", "Recency"),
        ("Frequency", "Frequency"),
        ("Monetary", "Monetary")
    ]

    input_dict = {}


    for label, key in slider_labels:
        min_value = 0
        max_value = int(data[key].max())
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            key=f"{key}_slider",
            value=input_dict.get(key, min_value),
            step=1
        )
    return input_dict

def get_scaled_values(input_dict):

    data = get_clean_data()

    x = data.drop(['Segment'],axis=1)

    scaled_dict = {}

    for key ,value in input_dict.items():

        max_value = x[key].max()

        min_value = x[key].min()

        scaled_value = (value - min_value) / (max_value- min_value)

        scaled_dict[key]= scaled_value

    return scaled_dict

def get_radar_chart(input_data):

    input_data  = get_scaled_values(input_data)
   
       
    categories = ["Gender",'Location','TransactionAmount_mean','AccountBalance_mean','CustomerAge',
                  
    'Unique_purchase','Total_spend','Recency','Frequency','Monetary']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
      r=[
          input_data['Gender'],input_data['Location'], input_data['TransactionAmount_mean'],input_data['AccountBalance_mean'],
          input_data['CustomerAge'], input_data['Unique_purchase'], input_data['Total_spend'],input_data['Recency'],
          input_data['Frequency'], input_data['Monetary'], 
        ],
      theta=categories,
      fill='toself',
      name='Max values'
    ))
    fig.add_trace(go.Scatterpolar(
        
      r=[
          
        input_data['Gender'],input_data['Location'], input_data['TransactionAmount_mean'],input_data['AccountBalance_mean'],
          input_data['CustomerAge'], input_data['Unique_purchase'], input_data['Total_spend'],input_data['Recency'],
          input_data['Frequency'], input_data['Monetary']], 

      theta=categories,
      fill='toself',
      name='Average values'
    ))

    fig.update_layout(
      polar=dict(
      radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True
  )

    return fig

def add_predictions(input_data):


    model = pickle.load(open("model/model.pkl","rb"))

    scaler = pickle.load(open("model/scaler.pkl", "rb"))
     
    input_array = np.array(list(input_data.values())).reshape(1,-1)

    input_array_scaled = scaler.transform(input_array)
   
    st.write(input_array_scaled)

    prediction = model.predict(input_array_scaled)



    st.subheader("Customer Segmentation")

    st.write("PREDICTION :")

    prediction_html = f'<div style="width: 300px; border: 1px solid white; padding: 10px">{prediction}</div>'
    st.markdown(prediction_html, unsafe_allow_html=True)   


def main():

   input_data = add_sidebar()

   with st.container():

    st.title("Customer Segmentation E-commerce")
    st.write("Hello Jarvis! This is really fun")


    col1, col2 = st.columns([4, 1])

    with col1:

      radar_chart = get_radar_chart(input_data)

      st.plotly_chart(radar_chart)

    with col2:
           
      add_predictions(input_data)


if __name__ == '__main__':
    main()