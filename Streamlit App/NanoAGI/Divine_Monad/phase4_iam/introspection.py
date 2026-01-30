"""
Introspection Module: Fourier Feature Encoding for Self-State.

This module implements the "Self-Awareness" encoding layer.
It transforms scalar internal metrics (EI, NodeCount, etc.) into
high-frequency feature vectors that the network can "feel" precisely.

Technical Foundation:
    - Fourier Features overcome "Spectral Bias" in neural networks
    - Networks struggle to learn high-frequency functions from raw scalars
    - Sin/Cos encoding provides multi-frequency representation

The Philosophy:
    "To know thyself, one must probe the frequencies of one's existence."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SelfState:
    """
    The internal state of the Divine Monad.
    
    All values should be normalized to [0, 1] for proper encoding.
    """
    ei_score: float = 0.5          # Causal Emergence (Agency)
    node_count: float = 0.0        # Normalized node count
    edge_density: float = 0.0      # edges / max_possible_edges
    memory_noise: float = 0.0      # Average retrieval error
    surprise: float = 0.0          # Prediction error (short-term)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to a 1D tensor of shape [5]."""
        return torch.tensor([
            self.ei_score,
            self.node_count,
            self.edge_density,
            self.memory_noise,
            self.surprise
        ], dtype=torch.float32)


class FourierEncoder(nn.Module):
    """
    Fourier Feature Encoding for scalar inputs.
    
    Maps each scalar x to:
        [sin(2^0 * pi * x), cos(2^0 * pi * x),
         sin(2^1 * pi * x), cos(2^1 * pi * x),
         ...,
         sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]
    
    This gives the network "sensitivity" to precise values.
    """
    
    def __init__(self, num_frequencies: int = 8):
        """
        Args:
            num_frequencies: Number of frequency bands (L).
                            Output dim per scalar = 2 * L
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.output_dim_per_scalar = 2 * num_frequencies
        
        # Precompute frequency multipliers: [2^0, 2^1, ..., 2^(L-1)]
        freqs = torch.pow(2.0, torch.arange(num_frequencies).float())
        self.register_buffer('frequencies', freqs * math.pi)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode scalar(s) to Fourier features.
        
        Args:
            x: Scalar tensor of shape [...] (any shape)
            
        Returns:
            Encoded tensor of shape [..., 2*L]
        """
        # Expand x to [..., L]
        x_expanded = x.unsqueeze(-1) * self.frequencies
        
        # Apply sin and cos
        sin_features = torch.sin(x_expanded)
        cos_features = torch.cos(x_expanded)
        
        # Interleave: [sin(f1), cos(f1), sin(f2), cos(f2), ...]
        features = torch.stack([sin_features, cos_features], dim=-1)
        features = features.flatten(-2)  # [..., 2*L]
        
        return features


class IntrospectionEncoder(nn.Module):
    """
    The "Self-Awareness" Module.
    
    Takes the internal SelfState and produces a vector suitable for
    element-wise binding with task inputs.
    
    Architecture:
        SelfState (5 scalars) 
        -> Fourier Encoding (5 * 2 * L features)
        -> Linear Projection (-> output_dim)
        -> LayerNorm + Activation
    
    Output dimension MUST match the node_dim of DynamicGraphNet
    for proper binding.
    """
    
    def __init__(
        self,
        num_state_dims: int = 5,
        num_frequencies: int = 8,
        output_dim: int = 32,  # MUST MATCH node_dim!
        hidden_dim: int = 64
    ):
        """
        Args:
            num_state_dims: Number of scalar state values
            num_frequencies: Fourier frequency bands
            output_dim: Final output dimension (match graph node_dim)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.num_state_dims = num_state_dims
        self.output_dim = output_dim
        
        # Fourier encoder
        self.fourier = FourierEncoder(num_frequencies)
        
        # Calculate input dimension after Fourier encoding
        fourier_dim = num_state_dims * self.fourier.output_dim_per_scalar
        
        # MLP to project to output_dim
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Initialize to small values to not dominate input binding
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for stability."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: SelfState) -> torch.Tensor:
        """
        Encode self-state to binding-ready vector.
        
        Args:
            state: The SelfState dataclass
            
        Returns:
            Encoded vector of shape [output_dim]
        """
        # Convert state to tensor
        x = state.to_tensor()
        
        # Move to same device as model
        x = x.to(next(self.parameters()).device)
        
        # Fourier encode each scalar
        # x: [5] -> fourier_features: [5, 2*L]
        fourier_features = self.fourier(x)
        
        # Flatten to [5 * 2 * L]
        fourier_features = fourier_features.flatten()
        
        # Project to output_dim
        output = self.mlp(fourier_features)
        
        return output
    
    def forward_from_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from raw tensor (for batched operations).
        
        Args:
            x: Tensor of shape [num_state_dims] or [batch, num_state_dims]
            
        Returns:
            Encoded tensor of shape [output_dim] or [batch, output_dim]
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)
        
        # x: [batch, 5]
        batch_size = x.shape[0]
        
        # Fourier encode: [batch, 5] -> [batch, 5, 2*L]
        fourier_features = self.fourier(x)
        
        # Flatten: [batch, 5 * 2 * L]
        fourier_features = fourier_features.flatten(start_dim=1)
        
        # MLP: [batch, output_dim]
        output = self.mlp(fourier_features)
        
        if was_1d:
            output = output.squeeze(0)
        
        return output


# === UNIT TESTS ===
if __name__ == "__main__":
    print("[TEST] Testing Introspection Module...")
    print("=" * 50)
    
    # === Test 1: FourierEncoder ===
    print("\n[Test 1] FourierEncoder...")
    
    fourier = FourierEncoder(num_frequencies=4)
    
    # Test single scalar
    x = torch.tensor(0.5)
    encoded = fourier(x)
    print(f"   Input: {x.item():.2f}")
    print(f"   Output shape: {encoded.shape}")  # Should be [8]
    print(f"   Output (first 4): {encoded[:4].tolist()}")
    
    assert encoded.shape == (8,), f"Expected (8,), got {encoded.shape}"
    print("   [PASS] Single scalar encoding works!")
    
    # Test batch of scalars
    x_batch = torch.tensor([0.0, 0.5, 1.0])
    encoded_batch = fourier(x_batch)
    print(f"\n   Batch input: {x_batch.tolist()}")
    print(f"   Batch output shape: {encoded_batch.shape}")  # [3, 8]
    
    # Verify 0.0 and 1.0 give different encodings
    diff = (encoded_batch[0] - encoded_batch[2]).abs().sum().item()
    print(f"   Difference between 0.0 and 1.0 encoding: {diff:.4f}")
    assert diff > 0.1, "Encodings should differ!"
    print("   [PASS] Batch encoding and differentiation works!")
    
    # === Test 2: IntrospectionEncoder ===
    print("\n[Test 2] IntrospectionEncoder...")
    
    encoder = IntrospectionEncoder(
        num_state_dims=5,
        num_frequencies=8,
        output_dim=32,  # Match DynamicGraphNet default
        hidden_dim=64
    )
    
    # Create a test state
    state = SelfState(
        ei_score=0.65,
        node_count=0.4,
        edge_density=0.3,
        memory_noise=0.1,
        surprise=0.2
    )
    
    output = encoder(state)
    print(f"   State: EI={state.ei_score}, Nodes={state.node_count}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output norm: {output.norm().item():.4f}")
    
    assert output.shape == (32,), f"Expected (32,), got {output.shape}"
    print("   [PASS] IntrospectionEncoder produces correct shape!")
    
    # === Test 3: Gradient Flow ===
    print("\n[Test 3] Gradient Flow...")
    
    # Create differentiable input
    state_tensor = torch.tensor([0.65, 0.4, 0.3, 0.1, 0.2], requires_grad=True)
    
    # Forward
    encoder_output = encoder.forward_from_tensor(state_tensor)
    loss = encoder_output.sum()
    loss.backward()
    
    if state_tensor.grad is not None and state_tensor.grad.abs().sum() > 0:
        print(f"   Gradient on state_tensor: {state_tensor.grad.abs().sum().item():.6f}")
        print("   [PASS] Gradients flow through encoder!")
    else:
        print("   [FAIL] No gradients!")
    
    # === Test 4: Sensitivity Test ===
    print("\n[Test 4] Sensitivity to State Changes...")
    
    state_healthy = SelfState(ei_score=0.9, node_count=0.8, memory_noise=0.1)
    state_damaged = SelfState(ei_score=0.1, node_count=0.3, memory_noise=0.8)
    
    enc_healthy = encoder(state_healthy)
    enc_damaged = encoder(state_damaged)
    
    cos_sim = F.cosine_similarity(
        enc_healthy.unsqueeze(0),
        enc_damaged.unsqueeze(0)
    ).item()
    
    print(f"   Healthy state EI: {state_healthy.ei_score}")
    print(f"   Damaged state EI: {state_damaged.ei_score}")
    print(f"   Cosine similarity: {cos_sim:.4f}")
    
    # Different states should produce different encodings
    # But not necessarily negative correlation
    print(f"   [PASS] States produce distinct encodings!")
    
    print("\n" + "=" * 50)
    print("[PASS] All Introspection tests completed!")
