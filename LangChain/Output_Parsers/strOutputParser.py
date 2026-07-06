from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text generation'
)

model=ChatHuggingFace(llm=llm)

# 1st prompt ->detailed document
template1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt --> summary
template2=PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

# Here string output parser may not be required but in chains it is required

#prompt1=template1.invoke({'topic':'black hole'})
#result1=model.invoke(prompt1)
#prompt2=template2.invoke({'text':result1.content})
#result2=model.invoke(prompt2)
#print(result2.content)

parser=StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result=chain.invoke({'topic':'black hole'})

print(result)