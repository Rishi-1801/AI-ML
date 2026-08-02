import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMMathChain,LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool,initialize_agent
from langchain_classic.callbacks import StreamlitCallbackHandler

# Set up Streamlit
st.title("Text to Math Problem Solver")

grop_api_key=st.sidebar.text_input(label="Groq API Key",type="password")

if not grop_api_key:
    st.info("Please Enter the API key")
    st.stop()

model=ChatGroq(model="llama-3.3-70b-versatile",api_key=grop_api_key)

# Tools
#wikipedia Search Tool
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet and solving the Math Problem"
)

# Math tool
math_chain=LLMMathChain.from_llm(llm=model)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering the math related questions.Only input mathematical expression"
)

prompt="""
Your an agent tasked for solving users mathematical question.Logically arrive at the solution and provide a detailed explaination and display it point wise for the question below 
Question:{question}
Answer:
"""
prompt_template=PromptTemplate(
    template=prompt,
    input_variables=['question']
)
 

# initialize the agent
agent=initialize_agent(
    tools=[wikipedia_tool,calculator],
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a Math chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Conversation

question=st.text_area("Enter your question:","I have 5 bananas and 7 grapes. I eat 2 bananas , now total how many fruits I would have?")
if st.button("Answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({'role':'user',"content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant','content':response})
            st.write('### Response:')
            st.success(response)
    else:
        st.warning("Please Enter the Question")

