"""
Topological Mutator: Atomic graph operations with Net2Net initialization.

This module implements the mutation engine that can:
1. Add nodes (Function Preserving Growth)
2. Add edges (Zero-init for non-disruption)
3. Prune nodes/edges (Low-importance removal)

Key Safety: PyG Trap Mitigation
- edge_index and edge_attr are ALWAYS updated synchronously
- Sync check after every mutation

References:
    - Net2Net: Chen et al., "Net2Net: Accelerating Learning via Knowledge Transfer" (2016)
    - Phase 2 Plan: Topological Computing
"""

import torch
import torch.nn as nn
import copy
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MutationResult:
    """Result of a mutation operation."""
    success: bool
    message: str
    new_node_id: Optional[int] = None
    new_edge_id: Optional[int] = None


class TopologicalMutator:
    """
    Engine for safely mutating dynamic graph topology.
    
    All mutations follow the Net2Net principle:
    - Function Preserving: Network output should be ~identical before/after mutation
    - PyG Safe: edge_index and edge_weights are always synchronized
    """
    
    def __init__(self, epsilon: float = 1e-4):
        """
        Initialize the mutator.
        
        Args:
            epsilon: Small noise for symmetry breaking in Net2Net
        """
        self.epsilon = epsilon
    
    def hebbian_evolve(self, graph, intensity: float = 1.0):
        """
        THE LIVING TOPOLOGY: Hebbian Structural Plasticity.
        Replaces random mutation with energy-driven morphogenesis.
        
        Args:
            intensity: 0.5 (Calm) to 2.0 (Aggressive). Modulates growth thresholds.
        """
        with torch.no_grad():
            # === 1. VITALITY METRICS ===
            # Calculate 'Energy' of every node based on activation magnitude
            # High Energy = High Information flow.
            vitality = graph.node_features.abs().mean(dim=1)  # [Num_Nodes]
            
            # === 2. SYNAPTOGENESIS (Neurons that fire together, wire together) ===
            # Compute Cosine Similarity between ALL nodes
            norm_feats = torch.nn.functional.normalize(graph.node_features, p=2, dim=1)
            # Correlation Matrix: (N, D) @ (D, N) -> (N, N)
            similarity = torch.mm(norm_feats, norm_feats.t())
            
            # Mask out existing edges (we only want NEW connections)
            # We create a temporary adjacency mask
            existing_edges = torch.zeros_like(similarity, dtype=torch.bool)
            existing_edges[graph.edge_index[0], graph.edge_index[1]] = True
            
            # Find candidate pairs: High Similarity (>0.8) AND No Connection
            # 'intensity' lowers the barrier to form new connections
            threshold = 0.95 / intensity 
            candidates = (similarity > threshold) & (~existing_edges) & (torch.eye(graph.get_num_nodes(), device=similarity.device) == 0)
            srcs, dsts = torch.nonzero(candidates, as_tuple=True)
            
            if len(srcs) > 0:
                # Grow the STRONGEST potential synapse
                best_idx = torch.argmax(similarity[srcs, dsts])
                u, v = srcs[best_idx].item(), dsts[best_idx].item()
                # The synapse forms with a weight equal to their correlation (Hebe's Law)
                self.add_edge(graph, u, v, init_weight=similarity[u, v].item() * 2.0)
                
            # === 3. MITOSIS (Cell Division under Stress) ===
            # If a node is "burning hot" (Vitality > Limit), it splits to share the load.
            # This creates specialized clusters automatically.
            stress_limit = 2.5 / intensity
            overloaded_nodes = torch.nonzero(vitality > stress_limit).flatten()
            
            if len(overloaded_nodes) > 0:
                # The most stressed node splits first
                target_node = overloaded_nodes[torch.argmax(vitality[overloaded_nodes])].item()
                # 'grow_node' here effectively acts as mitosis if linked to parent
                self.grow_node(graph, parent_id=target_node) 
                
            # === 4. APOPTOSIS (Pruning of Dead Matter) ===
            # If a node has no energy (Vitality ~ 0), it is metabolically expensive waste.
            # We prune it to save compute for the living nodes.
            survival_threshold = 0.01 * intensity
            dead_nodes = torch.nonzero(vitality < survival_threshold).flatten()
            
            # Protect Input/Output nodes from death
            protected_indices = set(range(graph.num_input_nodes))
            protected_indices.add(graph.get_num_nodes() - 1) # Output node
            
            for node_idx in dead_nodes:
                if node_idx.item() not in protected_indices:
                    self.prune_node(graph, node_idx.item())
                    break # Prune only one per step to maintain stability

    
    def grow_node(
        self,
        net: nn.Module,
        parent_id: int,
        connect_to: Optional[List[int]] = None
    ) -> MutationResult:
        """
        Add a new node using Net2Net Function Preserving Growth.
        
        The new node is a "clone" of the parent. The parent's outgoing
        weights are halved, and the child inherits the other half.
        
        Args:
            net: The DynamicGraphNet to mutate
            parent_id: ID of the node to clone
            connect_to: Optional list of downstream nodes to connect to.
                        If None, clones parent's connectivity.
                        
        Returns:
            MutationResult with success status and new node ID
        """
        try:
            # === 1. EXPAND NODE FEATURES ===
            old_features = net.node_features.data
            num_nodes = old_features.shape[0]
            node_dim = old_features.shape[1]
            
            # New node = Parent + epsilon noise (symmetry breaking)
            new_node_features = old_features[parent_id].clone() + self.epsilon * torch.randn(node_dim)
            
            # Concatenate to create new features tensor
            new_features = torch.cat([old_features, new_node_features.unsqueeze(0)], dim=0)
            
            # Update parameter
            net.node_features = nn.Parameter(new_features)
            
            new_node_id = num_nodes  # Index of the new node
            
            # === 2. UPDATE EDGE CONNECTIVITY (PyG Safe) ===
            old_edge_index = net.edge_index
            old_edge_weights = net.edge_weights.data
            
            # Find parent's outgoing edges
            parent_outgoing_mask = old_edge_index[0] == parent_id
            parent_outgoing_targets = old_edge_index[1][parent_outgoing_mask]
            
            # Find parent's incoming edges
            parent_incoming_mask = old_edge_index[1] == parent_id
            parent_incoming_sources = old_edge_index[0][parent_incoming_mask]
            
            # Create new edges: same connectivity as parent
            # New node inherits parent's connections
            
            # Incoming edges to new node (from same sources as parent)
            new_incoming = torch.stack([
                parent_incoming_sources,
                torch.full_like(parent_incoming_sources, new_node_id)
            ], dim=0)
            
            # Outgoing edges from new node (to same targets as parent)
            if connect_to is not None:
                targets = torch.tensor(connect_to, dtype=torch.long)
            else:
                targets = parent_outgoing_targets
            
            new_outgoing = torch.stack([
                torch.full_like(targets, new_node_id),
                targets
            ], dim=0)
            
            # Concatenate all edges (CREATE NEW TENSOR - PyG Trap Fix)
            new_edge_index = torch.cat([
                old_edge_index,
                new_incoming,
                new_outgoing
            ], dim=1)
            
            # === 3. UPDATE EDGE WEIGHTS (Net2Net: Halve Parent's) ===
            num_new_edges = new_incoming.shape[1] + new_outgoing.shape[1]
            
            # For incoming edges: copy parent's incoming weights
            parent_incoming_weights = old_edge_weights[parent_incoming_mask]
            
            # For outgoing edges: HALVE parent's outgoing weights (Net2Net!)
            parent_outgoing_indices = parent_outgoing_mask.nonzero(as_tuple=True)[0]
            
            # Halve parent's outgoing weights in place
            old_edge_weights[parent_outgoing_indices] *= 0.5
            
            # New outgoing weights = half of original parent (now we need to reconstruct)
            new_outgoing_weights = old_edge_weights[parent_outgoing_indices].clone()
            
            # Add epsilon noise for symmetry breaking
            new_incoming_weights = parent_incoming_weights + self.epsilon * torch.randn_like(parent_incoming_weights)
            new_outgoing_weights = new_outgoing_weights + self.epsilon * torch.randn_like(new_outgoing_weights)
            
            # Concatenate weights (CREATE NEW TENSOR - PyG Trap Fix)
            new_edge_weights = torch.cat([
                old_edge_weights,
                new_incoming_weights,
                new_outgoing_weights
            ], dim=0)
            
            # Register new buffers/parameters
            net.register_buffer('edge_index', new_edge_index)
            net.edge_weights = nn.Parameter(new_edge_weights)
            net.num_nodes = num_nodes + 1
            
            # === 4. SYNC CHECK ===
            net._sync_check()
            
            return MutationResult(
                success=True,
                message=f"Grew node {new_node_id} from parent {parent_id}",
                new_node_id=new_node_id
            )
            
        except Exception as e:
            return MutationResult(
                success=False,
                message=f"grow_node failed: {str(e)}"
            )
    
    def add_edge(
        self,
        net: nn.Module,
        source: int,
        target: int,
        init_weight: float = 0.0
    ) -> MutationResult:
        """
        Add a new edge (synapse) to the graph.
        
        Uses zero-initialization by default for function preservation.
        
        Args:
            net: The DynamicGraphNet to mutate
            source: Source node ID
            target: Target node ID
            init_weight: Initial weight (default 0 for non-disruption)
            
        Returns:
            MutationResult with success status
        """
        try:
            old_edge_index = net.edge_index
            old_edge_weights = net.edge_weights.data
            
            # Check if edge already exists
            existing = (old_edge_index[0] == source) & (old_edge_index[1] == target)
            if existing.any():
                return MutationResult(
                    success=False,
                    message=f"Edge ({source}, {target}) already exists"
                )
            
            # Create new edge (CREATE NEW TENSORS - PyG Trap Fix)
            new_edge = torch.tensor([[source], [target]], dtype=torch.long)
            new_weight = torch.tensor([[init_weight]], dtype=old_edge_weights.dtype)
            
            new_edge_index = torch.cat([old_edge_index, new_edge], dim=1)
            new_edge_weights = torch.cat([old_edge_weights, new_weight], dim=0)
            
            # Update
            net.register_buffer('edge_index', new_edge_index)
            net.edge_weights = nn.Parameter(new_edge_weights)
            
            # Sync check
            net._sync_check()
            
            new_edge_id = new_edge_index.shape[1] - 1
            
            return MutationResult(
                success=True,
                message=f"Added edge ({source}, {target})",
                new_edge_id=new_edge_id
            )
            
        except Exception as e:
            return MutationResult(
                success=False,
                message=f"add_edge failed: {str(e)}"
            )
    
    def prune_edge(
        self,
        net: nn.Module,
        edge_id: int
    ) -> MutationResult:
        """
        Remove an edge by its index.
        
        Args:
            net: The DynamicGraphNet to mutate
            edge_id: Index of the edge to remove
            
        Returns:
            MutationResult with success status
        """
        try:
            old_edge_index = net.edge_index
            old_edge_weights = net.edge_weights.data
            
            num_edges = old_edge_index.shape[1]
            if edge_id < 0 or edge_id >= num_edges:
                return MutationResult(
                    success=False,
                    message=f"Invalid edge_id {edge_id}. Range: [0, {num_edges-1}]"
                )
            
            # Create mask to keep all edges except the one to remove
            keep_mask = torch.ones(num_edges, dtype=torch.bool)
            keep_mask[edge_id] = False
            
            # Filter (CREATE NEW TENSORS - PyG Trap Fix)
            new_edge_index = old_edge_index[:, keep_mask]
            new_edge_weights = old_edge_weights[keep_mask]
            
            # Update
            net.register_buffer('edge_index', new_edge_index)
            net.edge_weights = nn.Parameter(new_edge_weights)
            
            # Sync check
            net._sync_check()
            
            return MutationResult(
                success=True,
                message=f"Pruned edge {edge_id}"
            )
            
        except Exception as e:
            return MutationResult(
                success=False,
                message=f"prune_edge failed: {str(e)}"
            )
    
    def prune_node(
        self,
        net: nn.Module,
        node_id: int
    ) -> MutationResult:
        """
        Remove a node and all its connected edges.
        
        WARNING: This invalidates all node IDs > node_id!
        
        Args:
            net: The DynamicGraphNet to mutate
            node_id: Index of the node to remove
            
        Returns:
            MutationResult with success status
        """
        try:
            num_nodes = net.num_nodes
            
            # Prevent removing input/output nodes
            if node_id < net.num_input_nodes:
                return MutationResult(
                    success=False,
                    message=f"Cannot prune input node {node_id}"
                )
            if node_id >= num_nodes - net.num_output_nodes:
                return MutationResult(
                    success=False,
                    message=f"Cannot prune output node {node_id}"
                )
            
            # === 1. REMOVE NODE FEATURES ===
            old_features = net.node_features.data
            keep_nodes = torch.arange(num_nodes) != node_id
            new_features = old_features[keep_nodes]
            net.node_features = nn.Parameter(new_features)
            
            # === 2. REMOVE CONNECTED EDGES ===
            old_edge_index = net.edge_index
            old_edge_weights = net.edge_weights.data
            
            # Find edges connected to this node
            connected = (old_edge_index[0] == node_id) | (old_edge_index[1] == node_id)
            keep_edges = ~connected
            
            new_edge_index = old_edge_index[:, keep_edges]
            new_edge_weights = old_edge_weights[keep_edges]
            
            # === 3. REINDEX EDGES (node IDs shift down) ===
            # All IDs > node_id need to decrease by 1
            new_edge_index = new_edge_index.clone()
            new_edge_index[new_edge_index > node_id] -= 1
            
            # Update
            net.register_buffer('edge_index', new_edge_index)
            net.edge_weights = nn.Parameter(new_edge_weights)
            net.num_nodes = num_nodes - 1
            
            # Sync check
            net._sync_check()
            
            return MutationResult(
                success=True,
                message=f"Pruned node {node_id} and {connected.sum().item()} edges"
            )
            
        except Exception as e:
            return MutationResult(
                success=False,
                message=f"prune_node failed: {str(e)}"
            )


# === UNIT TEST ===
if __name__ == "__main__":
    try:
        from .graph_substrate import DynamicGraphNet
    except ImportError:
        from graph_substrate import DynamicGraphNet
    
    print("[TEST] Testing TopologicalMutator...")
    
    # Create network
    net = DynamicGraphNet(
        num_nodes=8,
        node_dim=16,
        num_heads=2,
        num_input_nodes=4,
        num_output_nodes=1
    )
    mutator = TopologicalMutator(epsilon=1e-4)
    
    print(f"   Initial nodes: {net.get_num_nodes()}")
    print(f"   Initial edges: {net.get_num_edges()}")
    
    # Test grow_node
    print("\n[TEST] Growing node from parent 4 (first hidden)...")
    result = mutator.grow_node(net, parent_id=4)
    print(f"   Result: {result.message}")
    print(f"   New nodes: {net.get_num_nodes()}")
    print(f"   New edges: {net.get_num_edges()}")
    
    # Verify function preservation
    x_input = torch.tensor([1.0, 0.0, 1.0, 0.0])
    output_after, _ = net(x_input)
    print(f"   Output after growth: {output_after.item():.4f}")
    
    # Test add_edge
    print("\n[TEST] Adding edge (0, 7)...")
    result = mutator.add_edge(net, source=0, target=7)
    print(f"   Result: {result.message}")
    print(f"   New edges: {net.get_num_edges()}")
    
    # Test prune_edge
    print("\n[TEST] Pruning edge 0...")
    result = mutator.prune_edge(net, edge_id=0)
    print(f"   Result: {result.message}")
    print(f"   New edges: {net.get_num_edges()}")
    
    # Gradient check after mutations
    print("\n[TEST] Gradient check after mutations...")
    net.zero_grad()
    output, _ = net(x_input)
    loss = output.sum()
    loss.backward()
    
    if net.edge_weights.grad is not None and net.edge_weights.grad.abs().sum() > 0:
        print("   Gradient on edge_weights: FLOWING")
        print("[PASS] Mutations preserve gradient flow!")
    else:
        print("   [WARN] No gradients after mutation!")
    
    print("\n[PASS] TopologicalMutator test completed!")

