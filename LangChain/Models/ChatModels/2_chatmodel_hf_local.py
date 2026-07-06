import os
os.environ["HF_HOME"] = "D:/huggingface_cache"

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# Create Hugging Face pipeline
pipe = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=0,   # RTX 3050
)

# Wrap the transformers pipeline
llm = HuggingFacePipeline(
    pipeline=pipe
)

# Convert it to a ChatModel
model = ChatHuggingFace(llm=llm)

# Invoke
result = model.invoke("What is the capital of India?")

print(result.content)