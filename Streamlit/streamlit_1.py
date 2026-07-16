import streamlit as st
import pandas as pd
import numpy as np

st.title('Streamlit Text Input')

name=st.text_input('Enter your name')

if name:
    st.write(f"Hello {name}")

#Slider
age=st.slider('Select your age',0,100,18)
st.write(f"Your age is {age}")

# Select Box (Drop down)
options=['Python','C','C++','Java']

choice=st.selectbox("Choose yout favourite language:",options)
st.write(f"You Selected {choice}")

# for uploading -> st.file_upload

# create the line chart
chart_data=pd.DataFrame(
    np.random.rand(20,3),columns=['A',"B","C"]
)
st.line_chart(chart_data)