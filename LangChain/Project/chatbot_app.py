from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

##Langsmith for Tracing and Debugging
os.environ["LANGSMITH_TRACING"]='true'
os.environ['LANGSMITH_PROJECT']='Simple Q&A Chatbot'


## Prompt Template
prompt=ChatPromptTemplate.from_messages([
    ('system',"You are a helpful assistant. Please response to the user queries"),
    ('user',"Question:{question}")
])

def generate_response(question,temperature,max_tokens):
    model=ChatGroq(model="llama-3.3-70b-versatile")
    parser=StrOutputParser()
    chain= prompt | model | parser
    answer=chain.invoke({'question':question})
    return answer


# Title of the app
st.title("Enhanced Q&A Chatbot")

#Sidebar for params
st.sidebar.title("Parameters")

temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.3)
max_tokes=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)


# Main Interface for user Input
st.write("Go ahead and ask any Question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,temperature,max_tokes)
    st.write(response)
else:
    st.write("Please provide the Query") 