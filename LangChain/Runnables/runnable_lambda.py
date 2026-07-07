from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda

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


parser=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'words':RunnableLambda(lambda x:len(x.split()))
})

chain= RunnableSequence(joke_gen_chain,parallel_chain)

print(chain.invoke({'topic':'AI'}))
