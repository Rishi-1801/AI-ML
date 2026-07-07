from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text generation'
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='Tell me a joke on the {topic}',
    input_variables=['topic']
)


prompt2=PromptTemplate(
    template='Explain the following joke\n{joke}',
    input_variables=['joke']
)

parser=StrOutputParser()

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)

print(chain.invoke({'topic':'AI'}))