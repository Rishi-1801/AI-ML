from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#text='Delhi is the Capital of India'
documents=[
    'Delhi is Capital of India',
    'kolkata is Capital of West Bengal',
    'Paris is Capital of France'
]
#vector=embedding.embed_query(text)
vector=embedding.embed_documents(documents)
print(len(vector),len(vector[0]))
