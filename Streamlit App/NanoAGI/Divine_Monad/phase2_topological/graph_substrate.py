"""
Graph Substrate: The Dynamic Neural Graph using Pure PyTorch.

This module implements the DynamicGraphNet - a fluid neural network
whose topology can be rewired at runtime.

Architecture:
    - Nodes: Hold learnable state vectors (neuron activations)
    - Edges: Hold learnable weights (synapse strengths)
    - Message Passing: Dense Matrix Multiplication (Masked by Adjacency)

References:
    - Phase 2 Plan: Topological Computing
    - Removed 'torch_geometric' dependency for Streamlit Cloud compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


class DynamicGraphNet(nn.Module):
    """
    A neural network with mutable topology using Dense PyTorch operations.
    
    Key Features:
    1. Node states are learnable parameters
    2. Edge connectivity is simulated via a masked Adjacency Matrix
    3. Fully differentiable, runs on standard PyTorch (CPU/GPU)
    """
    
    def __init__(
        self,
        num_nodes: int = 16,
        node_dim: int = 32,
        num_heads: int = 4, # Unused in simple dense version, kept for API compatibility
        num_input_nodes: int = 4,
        num_output_nodes: int = 1
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        
        # === NODE STATE ===
        # Learnable node features
        self.node_features = nn.Parameter(torch.randn(num_nodes, node_dim) * 0.1)
        
        # === EDGES (DENSE ADJACENCY) ===
        # Instead of sparse edge_index, we store explicit edge weights in a dict or list 
        # and build a dense adjacency mask on the fly.
        # Actually, let's keep PyG's "edge_index" format for the Mutator, 
        # but convert to Dense Matrix for the Foward Pass.
        
        edge_list = []
        num_hidden = num_nodes - num_input_nodes - num_output_nodes
        hidden_start = num_input_nodes
        output_start = num_input_nodes + num_hidden
        
        # Input -> Hidden
        for i in range(num_input_nodes):
            for h in range(hidden_start, output_start):
                edge_list.append([i, h])
        
        # Hidden -> Output
        for h in range(hidden_start, output_start):
            for o in range(output_start, num_nodes):
                edge_list.append([h, o])
        
        # Register buffer for topology (COO format)
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
        self.register_buffer('edge_index', edge_index)
        
        # === LEARNABLE WEIGHTS ===
        num_edges = edge_index.shape[1]
        self.edge_weights = nn.Parameter(torch.randn(num_edges) * 0.1)
        
        # === LAYERS ===
        # Simple Linear layers applied to aggregated inputs
        self.linear1 = nn.Linear(node_dim, node_dim)
        self.linear2 = nn.Linear(node_dim, node_dim)
        
        self.output_proj = nn.Linear(node_dim, 1)
        
    def _build_adjacency(self) -> torch.Tensor:
        """Constructs the NxN weighted adjacency matrix from edge lists."""
        N = self.num_nodes
        device = self.node_features.device
        adj = torch.zeros((N, N), device=device)
        
        if self.edge_index.shape[1] > 0:
            # Use sophisticated indexing: adj[src, dst] = weight
            # edge_index is [2, E], first row is src, second is dst
            src, dst = self.edge_index
            adj[src, dst] = self.edge_weights
            
        return adj
        
    def forward(
        self,
        x_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using Dense Graph Convolution: A @ X @ W."""
        
        # 1. Prepare Node States
        x = self.node_features.clone()
        
        # 2. Inject Input
        if x_input is not None:
            if x_input.dim() == 1:
                x_input = x_input.unsqueeze(0)
            for i in range(min(x_input.shape[1], self.num_input_nodes)):
                x[i, 0] = x_input[0, i]
                
        # 3. Build Adjacency Matrix (Dynamic Topology)
        adj = self._build_adjacency()
        # Add self-loops to preserve own state? Let's say yes implicitly or explicitly.
        # For stability, let's normalize adjacency? 
        # For now, raw weights (Hebbian style).
        
        # 4. Layer 1: X_new = ReLU(Adj @ X @ W)
        # Transform features first
        x_trans = self.linear1(x)
        # Aggregate neighbors
        x = torch.matmul(adj.t(), x_trans) # Transpose because adj[src, dst] means flow from src to dst
        x = F.leaky_relu(x, 0.2)
        
        # 5. Layer 2
        x_trans = self.linear2(x)
        x = torch.matmul(adj.t(), x_trans)
        x = F.leaky_relu(x, 0.2)
        
        # 6. Output
        output_start = self.num_nodes - self.num_output_nodes
        output_nodes = x[output_start:]
        output = torch.sigmoid(self.output_proj(output_nodes))
        
        return output.squeeze(), x

    def get_num_edges(self) -> int:
        return self.edge_index.shape[1]
    
    def get_num_nodes(self) -> int:
        return self.num_nodes
        
    def _sync_check(self):
        """No-op for dense version, but kept for API compatibility."""
        pass

    def get_status(self) -> Dict:
        return {
            "nodes": self.num_nodes,
            "edges": self.edge_index.shape[1]
        }


# === UNIT TEST ===
if __name__ == "__main__":
    print("[TEST] Testing Pure PyTorch DynamicGraphNet...")
    net = DynamicGraphNet(num_nodes=10)
    print(f"   Nodes: {net.get_num_nodes()}")
    print(f"   Edges: {net.get_num_edges()}")
    
    x_in = torch.tensor([1.0, 0.0, 1.0, 0.0])
    out, states = net(x_in)
    print(f"   Output: {out.item():.4f}")
    
    # Grad check
    out.backward()
    print(f"   Grads: {'OK' if net.edge_weights.grad is not None else 'FAIL'}")

