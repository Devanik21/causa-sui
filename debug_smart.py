import torch
from core import PlasticCortex
import os

brain = PlasticCortex()
brain.load_cortex()

query = "ChatGPT"
data = torch.tensor(list(query.encode()), dtype=torch.long).unsqueeze(0)

print(f"--- TESTING ASSOCIATION FOR: {query} ---")

# Test 1: Pure Association (No recursion)
with torch.no_grad():
    activation, _ = brain.forward(data)
    signal = torch.matmul(activation, brain.synapse.t())
    all_bytes = torch.arange(256)
    all_embeds = brain.byte_embed(all_bytes)
    scores = torch.matmul(signal, all_embeds.t())
    top_indices = torch.topk(scores, k=32).indices[0]
    print(f"PURE THOUGHT: {bytes(top_indices.tolist()).decode('utf-8', errors='ignore')}")

# Test 2: Recursive Association (3 steps)
with torch.no_grad():
    activation, _ = brain.forward(data)
    for i in range(3):
        cv = torch.matmul(activation, brain.synapse.t())
        res = torch.matmul(cv, brain.synapse)
        activation = torch.tanh(res)
    
    signal = torch.matmul(activation, brain.synapse.t())
    scores = torch.matmul(signal, all_embeds.t())
    top_indices = torch.topk(scores, k=32).indices[0]
    print(f"RECURSIVE THOUGHT (3x): {bytes(top_indices.tolist()).decode('utf-8', errors='ignore')}")
