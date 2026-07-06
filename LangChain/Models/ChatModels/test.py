import os
os.environ["HF_HOME"] = "D:/huggingface_cache"

import torch
from transformers import pipeline

print("CUDA:", torch.cuda.is_available())
print("Current Device:", torch.cuda.current_device())
print("GPU:", torch.cuda.get_device_name(0))

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=0,
)

print(pipe.device)

result = pipe(
    "What is the capital of India?",
    max_new_tokens=100,
    temperature=0.5
)

print(result)