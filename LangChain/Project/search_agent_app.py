## IMPORTANT: Even the RAG can be used as tool by using the create_retriever_tool

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent,AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

# sidebar for api key
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type='password')

## Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title("Langchain - Chat with search")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I am chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine Larning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    model=ChatGroq(api_key=api_key,model="llama-3.3-70b-versatile",streaming=True)
    tools=[arxiv,wiki,search]

    search_agent=initialize_agent(tools,model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):                   # it creates the assistant container and everthing under the code is didplayed in that container
        st_callback=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)  # used for display the events poerformed by the agent (in the st.container)
        response=search_agent.invoke({"input":prompt},config={"callbacks":[st_callback]})
        result=response["output"]
        st.session_state.messages.append({'role':'assistant',"content":result})
        st.write(response)


