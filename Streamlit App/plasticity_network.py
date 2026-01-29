"""
PlasticityNetwork: The "Genome" for Differentiable Plasticity.

This module implements a universal, learnable plasticity rule that is applied
to every synapse simultaneously. Instead of hand-coding the Hebbian rule
(delta_w = pre * post), we let this small neural network LEARN the optimal
learning rule itself.

The key insight: Each synapse only needs to know 3 things:
    1. Pre-synaptic activity (how active was the input neuron?)
    2. Post-synaptic activity (how active was the output neuron?)
    3. Current weight value (what is the current connection strength?)

From these 3 scalars, the Genome outputs 1 scalar: the weight change (delta_w).

This is biologically plausible (local information only) and computationally
efficient (O(1) per synapse using broadcasting).

References:
- Uber AI: "Differentiable Plasticity" (2018)
- Google Brain: Meta-Learning research
"""

import torch
import torch.nn as nn


class PlasticityNetwork(nn.Module):
    """
    The 'Genome': A universal, shared plasticity rule applied to EVERY synapse.
    
    Architecture:
        Input:  3 scalars (Pre-activation, Post-activation, Current Weight)
        Hidden: Small MLP with ReLU activation
        Output: 1 scalar (Delta W - the weight change)
    
    The same tiny network is applied to all (Input_Dim x Hidden_Dim) synapses
    in parallel using efficient tensor broadcasting.
    """
    
    def __init__(self, hidden_dim: int = 16):
        """
        Initialize the Genome.
        
        Args:
            hidden_dim: Number of neurons in the hidden layer.
                        Smaller = faster but less expressive.
                        16 is a good balance for meta-learning.
        """
        super().__init__()
        
        # The core plasticity function: 3 inputs -> 1 output
        # Input: [pre, post, weight] -> Output: [delta_w]
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights to small values for stability
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
    
    def forward(self, pre: torch.Tensor, post: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Compute the weight update for all synapses in parallel.
        
        This uses broadcasting to apply the same plasticity rule to every
        synapse junction (pre_i, post_j, w_ij) simultaneously.
        
        Args:
            pre:    Pre-synaptic activations.  Shape: (Batch, Input_Dim)
            post:   Post-synaptic activations. Shape: (Batch, Hidden_Dim)
            weight: Current weight matrix.     Shape: (Input_Dim, Hidden_Dim)
        
        Returns:
            delta_w: Weight change matrix.     Shape: (Input_Dim, Hidden_Dim)
        """
        batch_size, input_dim = pre.shape
        hidden_dim = post.shape[1]
        
        # === STEP 1: Create a "Synaptic State Volume" ===
        # We need to pair every pre_i with every post_j for all synapses.
        # This creates a 3D grid: (Batch, Input_Dim, Hidden_Dim)
        
        # Expand pre: (Batch, Input) -> (Batch, Input, 1) -> (Batch, Input, Hidden)
        pre_expanded = pre.unsqueeze(2).expand(-1, -1, hidden_dim)
        
        # Expand post: (Batch, Hidden) -> (Batch, 1, Hidden) -> (Batch, Input, Hidden)
        post_expanded = post.unsqueeze(1).expand(-1, input_dim, -1)
        
        # Expand weight: (Input, Hidden) -> (1, Input, Hidden) -> (Batch, Input, Hidden)
        weight_expanded = weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # === STEP 2: Stack into Feature Tensor ===
        # Shape: (Batch, Input, Hidden, 3)
        synaptic_state = torch.stack([pre_expanded, post_expanded, weight_expanded], dim=-1)
        
        # === STEP 3: Apply the Genome to ALL Synapses ===
        # The Linear layers will broadcast across the first 3 dimensions.
        # Output: (Batch, Input, Hidden, 1)
        delta_w = self.net(synaptic_state)
        
        # Remove the trailing dimension: (Batch, Input, Hidden)
        delta_w = delta_w.squeeze(-1)
        
        # === STEP 4: Average Over Batch ===
        # We want a single weight update matrix, not per-sample updates.
        # Shape: (Input_Dim, Hidden_Dim)
        return delta_w.mean(dim=0)


# === UNIT TEST ===
if __name__ == "__main__":
    print("[TEST] Testing PlasticityNetwork (The Genome)...")
    
    # Create the Genome
    genome = PlasticityNetwork(hidden_dim=16)
    print(f"   Parameters: {sum(p.numel() for p in genome.parameters())}")
    
    # Create dummy inputs
    batch_size = 4
    input_dim = 32
    hidden_dim = 64
    
    pre = torch.randn(batch_size, input_dim)
    post = torch.randn(batch_size, hidden_dim)
    weight = torch.randn(input_dim, hidden_dim)
    
    # Forward pass
    delta_w = genome(pre, post, weight)
    
    print(f"   Pre shape:     {pre.shape}")
    print(f"   Post shape:    {post.shape}")
    print(f"   Weight shape:  {weight.shape}")
    print(f"   Delta_W shape: {delta_w.shape}")
    
    assert delta_w.shape == (input_dim, hidden_dim), "Output shape mismatch!"
    print("[PASS] PlasticityNetwork test PASSED!")
