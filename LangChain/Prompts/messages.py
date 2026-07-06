from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv

llm=HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task='text-generation'
)
model=ChatHuggingFace(llm=llm)
messages=[
    SystemMessage(content='You are ahelpful assisstant'),
    HumanMessage(content='tell me about Virat Kohli')
]

result=model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)