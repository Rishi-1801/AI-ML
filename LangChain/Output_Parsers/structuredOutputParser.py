from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text generation'
)

model=ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_1',description='Fact 1 about the topic')
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template=' Give me 3 facts about the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

'''   Normal 

prompt=template.invoke({'topic':'black hole'})
result=model.invoke(prompt)
final_result=parser.parse(result.content)
print(final_result)
'''

chain=template|model|parser
result=chain.invoke({"topic":'balck hole'})
print(result) 