import torch
from core import PlasticCortex
import os

print("--- TESTING HYPER-INTELLIGENCE FEATURES ---")
brain = PlasticCortex(hidden_dim=128) # Start small for test
print(f"Initial Synapse Shape: {brain.synapse.shape}")

# 1. Test Mitosis
brain.grow(64)
assert brain.synapse.shape[1] == 192
print("✅ Mitosis Successful: Expanded to 192 neurons.")

# 2. Test Consolidation
# Fill buffer
for i in range(5):
    data = torch.randint(0, 256, (1, 10))
    brain.forward(data)

print(f"Buffer size: {len(brain.experience_buffer)}")
brain.consolidate()
assert len(brain.experience_buffer) == 0
print("✅ Consolidation Successful: Memories solidified and buffer cleared.")

# 3. Test Persistence of expanded brain
brain.save_cortex("test_growth.pth")
brain2 = PlasticCortex(hidden_dim=128) # Small init
brain2.load_cortex("test_growth.pth")
assert brain2.synapse.shape[1] == 192
print("✅ Resilience Successful: Auto-expanded on load.")

print("--- ALL HYPER-INTELLIGENCE TESTS PASSED ---")
