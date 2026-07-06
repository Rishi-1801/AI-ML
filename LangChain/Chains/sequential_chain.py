from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text generation'
)

parser=StrOutputParser()

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='genearte the detailed report on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate the 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({'topic':'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()