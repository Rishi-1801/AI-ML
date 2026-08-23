import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

model=ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./PDFS")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50)
        st.session_state.chunks=st.session_state.splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.chunks,st.session_state.embeddings)



st.title("NVIDIA NIM")
prompt=ChatPromptTemplate.from_template(
    """
Answer the question based on the provided context only.PLease provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}
"""
)

prompt1=st.text_input("Enter your Question from Documnets")

if st.button("Document Embedding"):
    vector_embeddings()
    st.write("Your VectorStore is Ready")

if prompt1:
    document_chain=create_stuff_documents_chain(model,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({"input":prompt1})
    st.write(response['answer'])

# the llama model also gives context also
with st.expander("Document Similarity Search"):
    for i,doc in enumerate(response['context']):
        st.write(doc.page_content)
        st.write("---------------------------")