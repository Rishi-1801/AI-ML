from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# Here Problem is that the chat is saved just like the messages used not filling the place holder

#chat_template=ChatPromptTemplate([
#   SystemMessage(content='You are an helpful {domain} Assistant'),
#    HumanMessage(content='Explain me in simple terms , what is {topic}')
#])


# This will work here
chat_template=ChatPromptTemplate([
   ('system','You are an helpful {domain} Assistant'),
   ('human','Explain me in simple terms , what is {topic}')
])

prompt=chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)