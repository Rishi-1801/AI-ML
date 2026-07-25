## RAG Q&A Conversation with PDF INcluding Chat History
import streamlit as st
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
load_dotenv()

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Settinh up the Streamlit
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload Pdf's and chat with their content")

#Input the Groq Api key
api_key=st.text_input("Enter your Groq API Key:",type="password")

# Check if groq api key is provided
if api_key:
    model=ChatGroq(api_key=api_key,model="llama-3.3-70b-versatile")
    session_id=st.text_input("Session_ID",value='default_session')

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose a Pdf file",type='pdf',accept_multiple_files=True)

    ## Process the uploaded files
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

        # Split and create the embeddings
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(splits,embeddings)
        retriever=vectorstore.as_retriever()

        contextualize_q_system_prompt=(
                "Given a chat history and the latest user question which might reference context in the chat history,formulate a standalone question which can be understood without the chat history.Do not answer the question,just reformulate it  if needed and otherwise return it as is."
            )

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',contextualize_q_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}')
            ])

        history_aware_retriever=create_history_aware_retriever(model,retriever,contextualize_q_prompt)


        system_prompt=(
                "You are an assistant for question-answering tasks.USe the following pieces of retrieved context to answerthe question.If you don't know the answer,say that you don't know .USe threes sentences maximum and keep the answer concise.\n\n{context}"
            )
        qa_prompt=ChatPromptTemplate.from_messages([
                ('system',system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}')
            ])

        question_answer_chain=create_stuff_documents_chain(model,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
        )

        user_input=st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke({"input":user_input},config={'configurable':{"session_id":session_id}})


            st.write(st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write("Chat History:",session_history.messages)

else:
    st.warning("Please Enter your Groq API key")

        