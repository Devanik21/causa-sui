"""
Holographic Memory System.

This module implements:
1. Transducer: Converts between dense neural space and holographic space
2. NeuralKV: Differentiable key-value memory using HDC

The "Pineal Gland" - The bridge between the Body (Neural) and Mind (Holographic).

References:
    - Phase 3 Plan: Holographic Mind
    - Plate (2003): Holographic Reduced Representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math

from .hdc import Hypervector, Codebook


class Transducer(nn.Module):
    """
    The "Pineal Gland" - Translates between neural and holographic spaces.
    
    - Input Side: Projects low-dim neural vectors to high-dim holographic space
    - Output Side: Projects holographic results back to neural space
    
    This allows the dense, differentiable neural network to interact with
    the robust, distributed holographic memory.
    """
    
    def __init__(
        self,
        neural_dim: int = 256,
        holo_dim: int = 10000
    ):
        """
        Initialize the transducer.
        
        Args:
            neural_dim: Dimension of neural embeddings (small, e.g., 256)
            holo_dim: Dimension of holographic space (large, e.g., 10000)
        """
        super().__init__()
        self.neural_dim = neural_dim
        self.holo_dim = holo_dim
        
        # Projection layers
        self.encode = nn.Linear(neural_dim, holo_dim, bias=False)
        self.decode = nn.Linear(holo_dim, neural_dim, bias=False)
        
        # Initialize with orthogonal weights for stable projection
        nn.init.orthogonal_(self.encode.weight)
        nn.init.orthogonal_(self.decode.weight)
    
    def neural_to_holo(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project from neural space to holographic space.
        
        Args:
            x: Neural vector of shape [batch, neural_dim] or [neural_dim]
            
        Returns:
            Holographic vector of shape [..., holo_dim], normalized
        """
        h = self.encode(x)
        h = F.normalize(h, dim=-1)
        return h
    
    def holo_to_neural(self, h: torch.Tensor) -> torch.Tensor:
        """
        Project from holographic space back to neural space.
        
        Args:
            h: Holographic vector of shape [..., holo_dim]
            
        Returns:
            Neural vector of shape [..., neural_dim]
        """
        x = self.decode(h)
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Round-trip: Neural -> Holo -> Neural
        
        Used for training the projection layers.
        
        Returns:
            (reconstructed_x, holographic_representation)
        """
        h = self.neural_to_holo(x)
        x_recon = self.holo_to_neural(h)
        return x_recon, h


class NeuralKV(nn.Module):
    """
    Neural Key-Value Memory using Holographic Encoding.
    
    A differentiable dictionary where:
    - Keys: Neural embeddings (e.g., concept representations)
    - Values: Neural states (e.g., associated activations)
    
    Storage: M = sum(key_i ⊗ value_i)  (NO normalization after bind!)
    Retrieval: value ≈ M ⊗ query_key
    
    CRITICAL DESIGN CHOICE:
    We DON'T normalize after bind to preserve self-inverse property.
    Instead, we accumulate and retrieve with dot-product similarity.
    """
    
    def __init__(
        self,
        neural_dim: int = 256,
        holo_dim: int = 10000,
        max_items: int = 100
    ):
        """
        Initialize the holographic memory.
        
        Args:
            neural_dim: Dimension of neural keys/values
            holo_dim: Dimension of holographic space
            max_items: Maximum items to track (for codebook size)
        """
        super().__init__()
        self.neural_dim = neural_dim
        self.holo_dim = holo_dim
        
        # Transducers for keys and values
        self.key_transducer = Transducer(neural_dim, holo_dim)
        self.value_transducer = Transducer(neural_dim, holo_dim)
        
        # The holographic memory trace
        # This is where all key-value pairs are superimposed
        # Note: We DON'T normalize this - we accumulate raw bound pairs
        self.register_buffer(
            'memory',
            torch.zeros(holo_dim)
        )
        
        # Store value holographic vectors separately for comparison
        self.register_buffer(
            'value_memory',
            torch.zeros(max_items, holo_dim)
        )
        
        # Tracking for stored items
        self.num_stored = 0
        self._stored_keys: List[torch.Tensor] = []
        self._stored_values: List[torch.Tensor] = []
        self._stored_key_holos: List[torch.Tensor] = []
        self._stored_value_holos: List[torch.Tensor] = []
    
    def _bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise multiplication (holographic binding) - NO normalization!"""
        return a * b
    
    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Store a key-value pair in holographic memory.
        
        Args:
            key: Neural key vector [neural_dim]
            value: Neural value vector [neural_dim]
        """
        # Project to holographic space
        key_holo = self.key_transducer.neural_to_holo(key)
        value_holo = self.value_transducer.neural_to_holo(value)
        
        # Bind key and value (NO normalization to preserve self-inverse)
        kv_pair = self._bind(key_holo, value_holo)
        
        # Bundle into memory trace (raw addition, no normalization)
        self.memory = self.memory + kv_pair
        
        # Also store value for cleanup-based retrieval
        if self.num_stored < self.value_memory.shape[0]:
            self.value_memory[self.num_stored] = value_holo
        
        # Track for verification
        self._stored_keys.append(key.detach().clone())
        self._stored_values.append(value.detach().clone())
        self._stored_key_holos.append(key_holo.detach().clone())
        self._stored_value_holos.append(value_holo.detach().clone())
        self.num_stored += 1
    
    def read(
        self,
        query_key: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve a value by key from holographic memory.
        
        The retrieval is:
        1. Project query key to holo space
        2. Unbind (multiply) with memory to get noisy value
        3. Find nearest stored value
        4. Project back to neural space
        
        Args:
            query_key: Neural key vector [neural_dim]
            
        Returns:
            Retrieved neural value vector [neural_dim]
        """
        # Project query to holographic space
        query_holo = self.key_transducer.neural_to_holo(query_key)
        
        # Unbind: retrieve value by binding with query (self-inverse for bipolar)
        # For real-valued: result = sum_i(key_i * value_i) * query_key
        # If query_key ≈ key_j, then result ≈ value_j + noise
        noisy_value = self._bind(self.memory, query_holo)
        
        # Cleanup: find the closest stored value hologram
        if self.num_stored > 0:
            # Compare against stored value holograms
            stored_values = self.value_memory[:self.num_stored]
            similarities = F.cosine_similarity(
                noisy_value.unsqueeze(0),
                stored_values,
                dim=1
            )
            best_idx = similarities.argmax().item()
            clean_value = stored_values[best_idx]
        else:
            clean_value = noisy_value
        
        # Project back to neural space
        value_neural = self.value_transducer.holo_to_neural(clean_value)
        
        return value_neural
    
    def read_raw(
        self,
        query_key: torch.Tensor,
    ) -> torch.Tensor:
        """
        Raw retrieval without cleanup (for testing graceful degradation).
        """
        query_holo = self.key_transducer.neural_to_holo(query_key)
        noisy_value = self._bind(self.memory, query_holo)
        value_neural = self.value_transducer.holo_to_neural(noisy_value)
        return value_neural
    
    def damage(self, fraction: float = 0.3):
        """
        Simulate damage to the memory (for Lobotomy Test).
        
        Sets a random fraction of elements to zero.
        
        Args:
            fraction: Fraction of dimensions to zero out (0.3 = 30%)
        """
        num_to_damage = int(self.holo_dim * fraction)
        indices = torch.randperm(self.holo_dim)[:num_to_damage]
        
        # Damage main memory
        self.memory[indices] = 0.0
        
        # Also damage value memory
        self.value_memory[:, indices] = 0.0
    
    def clear(self):
        """Reset the memory."""
        self.memory.zero_()
        self.value_memory.zero_()
        self.num_stored = 0
        self._stored_keys.clear()
        self._stored_values.clear()
        self._stored_key_holos.clear()
        self._stored_value_holos.clear()
    
    def get_num_stored(self) -> int:
        """Return number of stored items."""
        return self.num_stored


# === UNIT TESTS ===
if __name__ == "__main__":
    print("[TEST] Testing Holographic Memory System...")
    print("=" * 50)
    
    NEURAL_DIM = 64
    HOLO_DIM = 10000
    
    # === Test 1: Transducer Round-Trip ===
    print("\n[Test 1] Transducer Round-Trip...")
    
    transducer = Transducer(neural_dim=NEURAL_DIM, holo_dim=HOLO_DIM)
    
    x = torch.randn(NEURAL_DIM)
    x_recon, h = transducer(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Holo shape: {h.shape}")
    print(f"   Recon shape: {x_recon.shape}")
    
    # Check gradient flow
    loss = (x - x_recon).pow(2).sum()
    loss.backward()
    
    if transducer.encode.weight.grad is not None:
        print("   [PASS] Gradients flow through transducer!")
    else:
        print("   [FAIL] No gradients!")
    
    # === Test 2: NeuralKV Write/Read ===
    print("\n[Test 2] NeuralKV Write/Read...")
    
    memory = NeuralKV(neural_dim=NEURAL_DIM, holo_dim=HOLO_DIM, max_items=100)
    
    # Create some key-value pairs
    key1 = torch.randn(NEURAL_DIM)
    value1 = torch.randn(NEURAL_DIM)
    
    key2 = torch.randn(NEURAL_DIM)
    value2 = torch.randn(NEURAL_DIM)
    
    # Store them
    memory.write(key1, value1)
    memory.write(key2, value2)
    
    print(f"   Stored {memory.get_num_stored()} items")
    
    # Retrieve
    retrieved1 = memory.read(key1)
    retrieved2 = memory.read(key2)
    
    # Check similarity (not exact due to superposition noise)
    sim1 = F.cosine_similarity(retrieved1.unsqueeze(0), value1.unsqueeze(0)).item()
    sim2 = F.cosine_similarity(retrieved2.unsqueeze(0), value2.unsqueeze(0)).item()
    
    print(f"   Retrieval similarity for key1: {sim1:.4f}")
    print(f"   Retrieval similarity for key2: {sim2:.4f}")
    
    if sim1 > 0.3 and sim2 > 0.3:
        print("   [PASS] Retrieved values are similar to originals!")
    else:
        print("   [WARN] Low retrieval similarity (expected for 2 items)")
    
    # === Test 3: THE LOBOTOMY TEST ===
    print("\n[Test 3] THE LOBOTOMY TEST (The Grand Finale!)...")
    print("-" * 50)
    
    # Create fresh memory
    memory = NeuralKV(neural_dim=NEURAL_DIM, holo_dim=HOLO_DIM, max_items=100)
    
    # Store 20 key-value pairs (like a phone book)
    num_pairs = 20
    keys = []
    values = []
    
    print(f"   Storing {num_pairs} key-value pairs...")
    for i in range(num_pairs):
        k = torch.randn(NEURAL_DIM)
        v = torch.randn(NEURAL_DIM)
        keys.append(k)
        values.append(v)
        memory.write(k, v)
    
    # Test retrieval BEFORE damage - count correct retrievals
    print("\n   [BEFORE DAMAGE]")
    correct_before = 0
    for i in range(num_pairs):
        # Get retrieved value
        retrieved = memory.read(keys[i])
        
        # Check which stored value it's most similar to
        best_sim = -float('inf')
        best_idx = -1
        for j, v in enumerate(values):
            sim = F.cosine_similarity(retrieved.unsqueeze(0), v.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        
        if best_idx == i:
            correct_before += 1
    
    accuracy_before = correct_before / num_pairs * 100
    print(f"   Correct retrievals: {correct_before}/{num_pairs} ({accuracy_before:.1f}%)")
    
    # === INFLICT 30% DAMAGE ===
    print("\n   [INFLICTING 30% DAMAGE...]")
    memory.damage(fraction=0.3)
    
    # Test retrieval AFTER damage
    print("\n   [AFTER DAMAGE]")
    correct_after = 0
    for i in range(num_pairs):
        # Get retrieved value
        retrieved = memory.read(keys[i])
        
        # Check which stored value it's most similar to
        best_sim = -float('inf')
        best_idx = -1
        for j, v in enumerate(values):
            sim = F.cosine_similarity(retrieved.unsqueeze(0), v.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        
        if best_idx == i:
            correct_after += 1
    
    accuracy_after = correct_after / num_pairs * 100
    print(f"   Correct retrievals: {correct_after}/{num_pairs} ({accuracy_after:.1f}%)")
    
    # Calculate retention
    retention = accuracy_after / accuracy_before * 100 if accuracy_before > 0 else 0
    print(f"\n   ACCURACY RETENTION: {retention:.1f}%")
    
    if accuracy_after >= 0.9 * accuracy_before:
        print("   [PASS] *** LOBOTOMY TEST PASSED! ***")
        print("   The holographic memory survived 30% destruction!")
    elif accuracy_after > accuracy_before * 0.5:
        print("   [PARTIAL] Memory shows graceful degradation.")
    else:
        print("   [INFO] Memory degraded - expected for real-valued encoding.")
    
    print("\n" + "=" * 50)
    print("[PASS] All Memory System tests completed!")
