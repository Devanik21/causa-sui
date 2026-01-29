"""
MicroCausalNet: A minimal stochastic neural network for Causal Emergence experiments.

This is a 2-layer network (4 input, hidden, 1 output) with learnable weights.
The output is a probability (sigmoid activation), enabling differentiable
Effective Information calculation.

Architecture (v1.1 - WITH HIDDEN LAYER):
    Input Layer: 4 binary nodes (A, B, C, D)
    Hidden Layer: N hidden nodes with ReLU activation
    Output Layer: 1 stochastic node (E)
    
    h = ReLU(W1 @ x + b1)
    E = sigmoid(W2 @ h + b2)
    
References:
    - Erik Hoel: "Causal Emergence" (2017)
    - Divine Monad Phase 1 Plan
"""

import torch
import torch.nn as nn


class MicroCausalNet(nn.Module):
    """
    A tiny 2-layer stochastic network for studying Causal Emergence.
    
    Key Design Decisions:
    1. Inputs are BINARY (0 or 1) but treated as floats for gradient flow.
    2. Hidden layer with ReLU enables learning non-linear functions (XOR).
    3. Output is a PROBABILITY P(E=1|inputs), not a hard 0/1.
    4. This allows us to compute Effective Information analytically.
    """
    
    def __init__(self, num_inputs: int = 4, hidden_dim: int = 8):
        """
        Initialize the MicroCausalNet with hidden layer.
        
        Args:
            num_inputs: Number of input nodes. Default 4 for tractable EI calculation.
            hidden_dim: Number of hidden units. More = more expressive.
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        
        # Layer 1: Input -> Hidden
        self.W1 = nn.Parameter(torch.randn(num_inputs, hidden_dim) * 0.5)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        
        # Layer 2: Hidden -> Output
        self.W2 = nn.Parameter(torch.randn(hidden_dim) * 0.5)
        self.b2 = nn.Parameter(torch.zeros(1))
        
    def forward_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute P(E=1 | inputs) for each input configuration.
        
        This is the CORE method for differentiable EI calculation.
        
        Args:
            x: Input tensor. Shape: [batch_size, num_inputs]
               Values should be 0.0 or 1.0 (binary, but float dtype).
               
        Returns:
            probs: Tensor of probabilities. Shape: [batch_size, 1]
                   Each value is P(E=1|x_i) in range (0, 1).
        """
        # Layer 1: Linear + ReLU
        hidden = torch.relu(torch.matmul(x, self.W1) + self.b1)
        
        # Layer 2: Linear + Sigmoid
        logits = torch.matmul(hidden, self.W2) + self.b2
        probs = torch.sigmoid(logits)
        
        return probs.unsqueeze(-1)  # Shape: [batch, 1]
    
    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """
        Forward pass with optional stochastic sampling.
        
        Args:
            x: Input tensor. Shape: [batch_size, num_inputs]
            sample: If True, sample binary output from Bernoulli(p).
                    If False, return the raw probability.
                    
        Returns:
            output: Either probabilities or sampled binary values.
        """
        probs = self.forward_prob(x)
        
        if sample:
            # Stochastic sampling (NOT differentiable)
            return torch.bernoulli(probs)
        else:
            return probs
    
    def get_all_input_states(self) -> torch.Tensor:
        """
        Generate all possible binary input configurations.
        
        For num_inputs=4, this returns 16 states (2^4).
        Essential for exhaustive EI calculation.
        
        Returns:
            states: Tensor of all binary states. Shape: [2^num_inputs, num_inputs]
        """
        num_states = 2 ** self.num_inputs
        states = []
        
        for i in range(num_states):
            # Convert integer to binary representation
            state = []
            for bit in range(self.num_inputs):
                state.append((i >> bit) & 1)
            states.append(state)
            
        return torch.tensor(states, dtype=torch.float32)

def create_xor_net() -> MicroCausalNet:
    """
    Create a network pre-configured to compute XOR(A, B).
    
    XOR truth table:
        A  B  | XOR
        0  0  |  0
        0  1  |  1
        1  0  |  1
        1  1  |  0
        
    With a 2-layer network, we CAN represent XOR exactly.
    
    Returns:
        net: A MicroCausalNet configured for XOR.
    """
    net = MicroCausalNet(num_inputs=2, hidden_dim=2)
    
    # XOR can be computed as: XOR = (A AND NOT B) OR (NOT A AND B)
    # Hidden unit 0: A AND NOT B (fires when A=1, B=0)
    # Hidden unit 1: NOT A AND B (fires when A=0, B=1)
    # Output: OR of hidden units
    with torch.no_grad():
        # W1: [2, 2] - two hidden units
        net.W1.data = torch.tensor([
            [10.0, -10.0],  # Hidden 0: +A, -B
            [-10.0, 10.0]   # Hidden 1: -A, +B
        ])
        net.b1.data = torch.tensor([-5.0, -5.0])  # Bias so only one-hot inputs activate
        
        # W2: [2] -> output
        net.W2.data = torch.tensor([10.0, 10.0])  # OR of hidden units
        net.b2.data = torch.tensor([-5.0])
    
    return net


# === UNIT TEST ===
if __name__ == "__main__":
    print("[TEST] Testing MicroCausalNet (2-layer)...")
    
    # Create network
    net = MicroCausalNet(num_inputs=4, hidden_dim=8)
    print(f"   Inputs: {net.num_inputs}")
    print(f"   Hidden: {net.hidden_dim}")
    print(f"   W1 shape: {net.W1.shape}")  # [4, 8]
    print(f"   W2 shape: {net.W2.shape}")  # [8]
    
    # Get all input states
    all_states = net.get_all_input_states()
    print(f"   All states shape: {all_states.shape}")  # Should be [16, 4]
    
    # Forward pass (probabilities)
    probs = net.forward_prob(all_states)
    print(f"   Probs shape: {probs.shape}")  # Should be [16, 1]
    print(f"   Probs range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    
    # Check gradient flow
    loss = probs.sum()
    loss.backward()
    print(f"   Gradient on W1: {net.W1.grad.abs().sum().item():.4f} (total magnitude)")
    
    if net.W1.grad is not None and net.W1.grad.abs().sum() > 0:
        print("[PASS] MicroCausalNet 2-layer test PASSED!")
    else:
        print("[FAIL] No gradients - check computation graph!")
    
    # Test XOR network
    print("\n[TEST] Testing XOR Network...")
    xor_net = create_xor_net()
    xor_inputs = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    xor_probs = xor_net.forward_prob(xor_inputs)
    print(f"   XOR outputs: {xor_probs.squeeze().tolist()}")
    print(f"   Expected:    [~0, ~1, ~1, ~0]")
    
    # Check if XOR is approximately correct
    expected = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    error = (xor_probs - expected).abs().max().item()
    if error < 0.1:
        print("[PASS] XOR network test PASSED!")
    else:
        print(f"[WARN] XOR error = {error:.4f}")

