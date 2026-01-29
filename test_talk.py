import torch
from core import PlasticCortex
from senses import update_telemetry
import os

brain = PlasticCortex()
brain.load_cortex()

query_text = "Hello Organism"
query_bytes = query_text.encode('utf-8')
data = torch.tensor(list(query_bytes), dtype=torch.long).unsqueeze(0)

response_bytes = brain.associate(data)
response_text = response_bytes.decode('utf-8', errors='ignore')

with open("RESPONSE.txt", "w", encoding='utf-8') as f:
    f.write(f"PROMPT: {query_text}\n")
    f.write(f"ORGANISM THOUGHT: {response_text}")

print(f"DECODED THOUGHT: {response_text}")
