from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text generation'
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template='Answer the following question\n{question} from the following text-\n{text}',
    input_variables=['question','text']
)

parser=StrOutputParser()

url= 'https://www.amazon.in/Dell-G15-5530-Gaming-i5-13450HX-Processor/dp/B0CRKX3JRT/ref=sr_1_3?sr=8-3'

loader=WebBaseLoader(url)

docs=loader.load()


# print(docs[0].page_content)  if u hit 1 url u only get one document

chain= prompt | model | parser

print(chain.invoke({'question':'What is the peak brightness of this product','text':docs[0].page_content[:1000]}))