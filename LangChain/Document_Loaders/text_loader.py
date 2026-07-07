from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text generation'
)

prompt=PromptTemplate(
    template='Write a Summary for the following poem\n{poem}',
    input_variables=['poem']
)

parser=StrOutputParser()

model=ChatHuggingFace(llm=llm)

loader=TextLoader('Document_Loaders\cricket.txt',encoding='utf-8')

docs=loader.load()

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0]}))