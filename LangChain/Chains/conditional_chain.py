from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_classic.output_parsers import PydanticOutputParser,OutputFixingParser
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text generation'
)

parser1=StrOutputParser()
model=ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']= Field(description='Return the sentiment of the given feedback')

parser2=PydanticOutputParser(pydantic_object=Feedback)

fixing_parser = OutputFixingParser.from_llm(
    parser=parser2,
    llm=model
)
prompt1=PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback}\n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

prompt2=PromptTemplate(
    template='Write an appropriate response for the positive feedback\n{feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='Write an appropriate response for the negative feedback\n{feedback}',
    input_variables=['feedback']
)

classifier_chain= prompt1 | model | fixing_parser

branch_chain=RunnableBranch(
    (lambda x: x.sentiment=='positive',prompt2|model|parser1),
    (lambda x: x.sentiment=='negative',prompt3|model|parser1),
    RunnableLambda(lambda x:"could not find sentiment")
)

chain = classifier_chain|branch_chain

print(chain.invoke({'feedback':"This is a beautiful phone"}))

chain.get_graph().print_ascii()