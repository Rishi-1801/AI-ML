import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile')

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the more accurate response based on the question.
    In case the context is insufficient to answer, say it I don't know
    <context>
    {context}
    <context>
    Question:{input}
    """
)


def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        st.session_state.loader=PyPDFDirectoryLoader(r"D:\AI\LangChain\Project\research_papers") # Data Ingestion
        st.session_state.docs=st.session_state.loader.load()  # Docs loadings
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)

st.title("RAG Documnent Q&A with Lama3")
user_input=st.text_input("Enter your query from the research paper")

if st.button("Documnet Embedding"):
    create_vector_embedding()
    st.write("Vector Database is Ready")


if user_input:
    document_chain=create_stuff_documents_chain(model,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    response=retrieval_chain.invoke({'input':user_input})

    st.write(response['answer'])

    # To see retrived Docs
    with st.expander("Documnet similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------------')