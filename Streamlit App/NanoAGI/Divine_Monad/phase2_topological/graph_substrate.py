"""
Graph Substrate: The Dynamic Neural Graph.

This module implements the DynamicGraphNet - a fluid neural network
whose topology can be rewired at runtime.

Architecture:
    - Nodes: Hold learnable state vectors (neuron activations)
    - Edges: Hold learnable weights (synapse strengths)
    - Message Passing: Uses PyTorch Geometric if available, otherwise pure PyTorch fallback

References:
    - Phase 2 Plan: Topological Computing
    - PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import torch_geometric, fall back to pure PyTorch if unavailable
try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    Data = None  # Placeholder


class SimpleMessagePassing(nn.Module):
    """
    Pure PyTorch fallback for message passing.
    Implements a simple attention-like aggregation without torch_geometric.
    """
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        
        # Attention weights
        self.W_q = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.W_k = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.W_e = nn.Linear(1, num_heads, bias=False)  # Edge weight projection
        self.W_out = nn.Linear(out_channels * num_heads, out_channels, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v, self.W_e, self.W_out]:
            nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Simple message passing.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_weights: Edge weights [num_edges, 1]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]
        
        if num_edges == 0:
            # No edges, just project
            return self.W_out(self.W_v(x).view(num_nodes, self.num_heads, self.out_channels).mean(dim=1))
        
        # Source and target indices
        src = edge_index[0]  # [num_edges]
        tgt = edge_index[1]  # [num_edges]
        
        # Get node features at source and target
        x_src = x[src]  # [num_edges, in_channels]
        x_tgt = x[tgt]  # [num_edges, in_channels]
        
        # Compute attention scores
        q = self.W_q(x_tgt).view(-1, self.num_heads, self.out_channels)  # [num_edges, heads, out]
        k = self.W_k(x_src).view(-1, self.num_heads, self.out_channels)  # [num_edges, heads, out]
        v = self.W_v(x_src).view(-1, self.num_heads, self.out_channels)  # [num_edges, heads, out]
        
        # Dot product attention
        attn = (q * k).sum(dim=-1) / (self.out_channels ** 0.5)  # [num_edges, heads]
        
        # Add edge weight contribution
        e = self.W_e(edge_weights)  # [num_edges, heads]
        attn = attn + e
        
        # Softmax over incoming edges for each target node
        # Create scatter indices
        attn_weights = torch.zeros(num_nodes, num_edges, self.num_heads, device=x.device)
        for i in range(num_edges):
            attn_weights[tgt[i], i] = F.softmax(attn[tgt == tgt[i]], dim=0).mean(dim=0)
        
        # Simpler approach: normalize and aggregate
        attn = F.leaky_relu(attn, negative_slope=0.2)
        attn = F.softmax(attn, dim=0)  # Approximate normalization
        
        # Weighted values
        weighted_v = attn.unsqueeze(-1) * v  # [num_edges, heads, out]
        
        # Aggregate to target nodes
        out = torch.zeros(num_nodes, self.num_heads, self.out_channels, device=x.device)
        for i in range(num_edges):
            out[tgt[i]] += weighted_v[i]
        
        # Combine heads
        out = out.view(num_nodes, -1)  # [num_nodes, heads * out_channels]
        out = self.W_out(out)  # [num_nodes, out_channels]
        
        # Residual connection (Self-Loop equivalent for Autopoiesis)
        if x.shape == out.shape:
             out = out + x

        return out


class DynamicGraphNet(nn.Module):
    """
    A neural network with mutable topology.
    
    Key Features:
    1. Node states are learnable parameters
    2. Edge connectivity can be added/removed at runtime
    3. Uses Graph Attention for differentiable message passing
    """
    
    def __init__(
        self,
        num_nodes: int = 16,
        node_dim: int = 32,
        num_heads: int = 4,
        num_input_nodes: int = 4,
        num_output_nodes: int = 1
    ):
        """
        Initialize the dynamic graph.
        
        Args:
            num_nodes: Initial number of nodes (can grow)
            node_dim: Dimension of node state vectors
            num_heads: Number of attention heads in GATv2
            num_input_nodes: Number of input nodes (fixed)
            num_output_nodes: Number of output nodes (fixed)
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        
        # === NODE STATE ===
        # Learnable node features (the "neuron states")
        self.node_features = nn.Parameter(torch.randn(num_nodes, node_dim) * 0.1)
        
        # === EDGE CONNECTIVITY ===
        # Start with a simple feedforward-like graph
        # Input -> Hidden -> Output
        edge_list = []
        num_hidden = num_nodes - num_input_nodes - num_output_nodes
        hidden_start = num_input_nodes
        output_start = num_input_nodes + num_hidden
        
        # Input -> Hidden (fully connected)
        for i in range(num_input_nodes):
            for h in range(hidden_start, output_start):
                edge_list.append([i, h])
        
        # Hidden -> Output (fully connected)
        for h in range(hidden_start, output_start):
            for o in range(output_start, num_nodes):
                edge_list.append([h, o])
        
        # Convert to tensor (COO format)
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Register as buffer (not a parameter, but should be saved)
        self.register_buffer('edge_index', edge_index)
        
        # === EDGE WEIGHTS ===
        # Learnable edge features (synapse strengths)
        num_edges = edge_index.shape[1]
        self.edge_weights = nn.Parameter(torch.randn(num_edges, 1) * 0.1)
        
        # === MESSAGE PASSING LAYERS ===
        if PYG_AVAILABLE:
            self.conv1 = GATv2Conv(
                in_channels=node_dim,
                out_channels=node_dim,
                heads=num_heads,
                concat=False,  # Average heads instead of concat
                edge_dim=1,    # Use edge weights as features
                add_self_loops=True # Critical for Autopoiesis (Self-Causation)
            )
            
            self.conv2 = GATv2Conv(
                in_channels=node_dim,
                out_channels=node_dim,
                heads=num_heads,
                concat=False,
                edge_dim=1,
                add_self_loops=True
            )
        else:
            # Fallback to pure PyTorch implementation
            self.conv1 = SimpleMessagePassing(node_dim, node_dim, num_heads)
            self.conv2 = SimpleMessagePassing(node_dim, node_dim, num_heads)
        
        # === OUTPUT PROJECTION ===
        self.output_proj = nn.Linear(node_dim, 1)
        
    def forward(
        self,
        x_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the dynamic graph.
        
        Args:
            x_input: Optional input to inject into input nodes.
                     Shape: [batch_size, num_input_nodes] or [num_input_nodes]
                     
        Returns:
            output: Predictions from output nodes. Shape: [num_output_nodes] or [batch, num_output_nodes]
            node_states: Final node states. Shape: [num_nodes, node_dim]
        """
        # Start with learned node features
        x = self.node_features.clone()
        
        # Inject input if provided
        if x_input is not None:
            if x_input.dim() == 1:
                x_input = x_input.unsqueeze(0)  # Add batch dim
            # Project input to node_dim and set input nodes
            # For now, simple: use first dimension of node as the "activation"
            for i in range(min(x_input.shape[1], self.num_input_nodes)):
                x[i, 0] = x_input[0, i]  # Set first feature to input value
        
        # Message passing (2 hops)
        x = self.conv1(x, self.edge_index, self.edge_weights)
        x = F.relu(x)
        
        x = self.conv2(x, self.edge_index, self.edge_weights)
        x = F.relu(x)
        
        # Extract output from output nodes
        output_start = self.num_nodes - self.num_output_nodes
        output_nodes = x[output_start:]
        output = self.output_proj(output_nodes)
        output = torch.sigmoid(output)  # Probability output
        
        return output.squeeze(), x
    
    def get_graph_data(self):
        """Return current graph as a PyG Data object (or dict if PyG unavailable)."""
        if PYG_AVAILABLE and Data is not None:
            return Data(
                x=self.node_features.detach(),
                edge_index=self.edge_index,
                edge_attr=self.edge_weights.detach()
            )
        else:
            return {
                'x': self.node_features.detach(),
                'edge_index': self.edge_index,
                'edge_attr': self.edge_weights.detach()
            }
    
    def get_num_edges(self) -> int:
        """Return current number of edges."""
        return self.edge_index.shape[1]
    
    def get_num_nodes(self) -> int:
        """Return current number of nodes."""
        return self.num_nodes
    
    def _sync_check(self):
        """
        CRITICAL: Verify edge_index and edge_weights are synchronized.
        Call after any mutation.
        """
        num_edges = self.edge_index.shape[1]
        num_weights = self.edge_weights.shape[0]
        assert num_edges == num_weights, (
            f"PyG Sync Error! edge_index has {num_edges} edges "
            f"but edge_weights has {num_weights} weights."
        )


# === UNIT TEST ===
if __name__ == "__main__":
    print(f"[TEST] Testing DynamicGraphNet (PyG Available: {PYG_AVAILABLE})...")
    
    # Create network
    net = DynamicGraphNet(
        num_nodes=10,
        node_dim=16,
        num_heads=2,
        num_input_nodes=4,
        num_output_nodes=1
    )
    
    print(f"   Nodes: {net.get_num_nodes()}")
    print(f"   Edges: {net.get_num_edges()}")
    print(f"   Node features shape: {net.node_features.shape}")
    print(f"   Edge weights shape: {net.edge_weights.shape}")
    
    # Sync check
    net._sync_check()
    print("   Sync check: PASSED")
    
    # Forward pass
    x_input = torch.tensor([1.0, 0.0, 1.0, 0.0])
    output, node_states = net(x_input)
    
    print(f"   Output: {output.item():.4f}")
    print(f"   Node states shape: {node_states.shape}")
    
    # Gradient check
    loss = output.sum()
    loss.backward()
    
    if net.node_features.grad is not None and net.node_features.grad.abs().sum() > 0:
        print("   Gradient on node_features: FLOWING")
    else:
        print("   [WARN] No gradients on node_features!")
        
    if net.edge_weights.grad is not None and net.edge_weights.grad.abs().sum() > 0:
        print("   Gradient on edge_weights: FLOWING")
    else:
        print("   [WARN] No gradients on edge_weights!")
    
    print("[PASS] DynamicGraphNet test completed!")
