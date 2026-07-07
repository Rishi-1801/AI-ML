from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('Document_Loaders\dl-curriculum.pdf')

docs=loader.load()

print(len(docs))