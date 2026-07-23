from typing import Annotated,TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END,add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os
load_dotenv()

os.environ['LANGSMITH_TRACING']='true'
os.environ['LANGSMITH_PROJECT']='Test'

model=ChatGroq(model='llama-3.3-70b-versatile')

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def make_tool_graph():
    #  Graph with tool call
    @tool
    def add(a:float,b:float):
        """Add two numbers"""
        return a+b;

    tool_node=ToolNode([add]) 

    model_with_tools=model.bind_tools([add])

    # node functionality
    def tool_calling_llm(state:State):
        return{'messages':[model_with_tools.invoke(state['messages'])]}

    #Graph
    builder=StateGraph(State)
    builder.add_node('tool_calling_llm',tool_calling_llm)
    builder.add_node('tools',tool_node)

    #Add Edges
    builder.add_edge(START,'tool_calling_llm')
    builder.add_conditional_edges('tool_calling_llm',tools_condition)
    builder.add_edge('tools','tool_calling_llm')

    #compile the graph
    graph=builder.compile()
    return graph

tool_agent=make_tool_graph() 


# here we cannot run "langgraph dev" command becoz to run this on langgraph studio u need 3.11>= so we cant run this