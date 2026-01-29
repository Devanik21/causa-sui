"""
Hybrid Optimizer: Inner/Outer loop training for Topological Computing.

Architecture:
    - Inner Loop (Fast): SGD trains weights on fixed topology
    - Outer Loop (Slow): Evolutionary mutations explore topology

The key insight: Weight training converges, then topology mutation
unlocks new capacity, then weight training resumes.

References:
    - Phase 2 Plan: Topological Computing
    - NEAT: Stanley & Miikkulainen (2002) "Evolving Neural Networks"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import copy
import random

from .graph_substrate import DynamicGraphNet
from .mutator import TopologicalMutator, MutationResult


@dataclass
class HybridConfig:
    """Configuration for hybrid optimization."""
    
    # Inner loop (weights)
    inner_lr: float = 0.01
    inner_steps: int = 100
    
    # Outer loop (topology)
    outer_steps: int = 10
    mutation_prob: float = 0.3
    
    # Early stopping for inner loop
    inner_patience: int = 20
    inner_threshold: float = 1e-5
    
    # Logging
    verbose: bool = True


@dataclass 
class TopologyCandidate:
    """A candidate topology with its fitness."""
    net: DynamicGraphNet
    fitness: float
    num_nodes: int
    num_edges: int
    mutation_history: List[str]


class HybridOptimizer:
    """
    Hybrid optimization combining gradient descent and evolutionary search.
    
    The loop:
    1. Train weights to convergence (Inner Loop)
    2. If stuck, mutate topology (Outer Loop)
    3. Check if mutation improved fitness
    4. If yes, keep mutation and go to 1
    5. If no, revert and try different mutation
    """
    
    def __init__(
        self,
        net: DynamicGraphNet,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: Optional[HybridConfig] = None
    ):
        """
        Initialize the hybrid optimizer.
        
        Args:
            net: The DynamicGraphNet to optimize
            loss_fn: Loss function(outputs, targets) -> scalar
            config: Optimization configuration
        """
        self.net = net
        self.loss_fn = loss_fn
        self.config = config or HybridConfig()
        self.mutator = TopologicalMutator()
        
        # History
        self.history = {
            'inner_losses': [],
            'outer_fitness': [],
            'mutations': [],
            'topology_sizes': []
        }
        
        # Best found
        self.best_fitness = float('-inf')
        self.best_state = None
        
    def train_weights(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[float, bool]:
        """
        Inner loop: Train weights on fixed topology.
        
        Args:
            inputs: Training inputs
            targets: Training targets
            
        Returns:
            (final_loss, converged): Loss value and whether training converged
        """
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.inner_lr)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for step in range(self.config.inner_steps):
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.net(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_val = loss.item()
            self.history['inner_losses'].append(loss_val)
            
            # Check convergence
            if loss_val < best_loss - self.config.inner_threshold:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.inner_patience:
                return loss_val, True  # Converged
                
        return loss_val, False  # Did not converge
    
    def evaluate_fitness(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Evaluate current network fitness (negative loss).
        
        Higher fitness = better performance.
        """
        with torch.no_grad():
            outputs, _ = self.net(inputs)
            loss = self.loss_fn(outputs, targets)
        return -loss.item()  # Negate so higher = better
    
    def mutate_topology(self) -> MutationResult:
        """
        Apply a random topology mutation.
        
        Mutation types:
        1. grow_node: Add a clone of a random hidden node
        2. add_edge: Add a random new connection
        """
        mutation_type = random.choice(['grow_node', 'add_edge'])
        
        if mutation_type == 'grow_node':
            # Pick a random hidden node to clone
            num_hidden = self.net.num_nodes - self.net.num_input_nodes - self.net.num_output_nodes
            if num_hidden > 0:
                hidden_start = self.net.num_input_nodes
                parent_id = random.randint(hidden_start, hidden_start + num_hidden - 1)
                return self.mutator.grow_node(self.net, parent_id)
            else:
                # No hidden nodes yet, grow from first input
                return self.mutator.grow_node(self.net, 0)
        
        elif mutation_type == 'add_edge':
            # Pick random source and target
            source = random.randint(0, self.net.num_nodes - 2)
            target = random.randint(source + 1, self.net.num_nodes - 1)
            return self.mutator.add_edge(self.net, source, target)
        
        return MutationResult(success=False, message="Unknown mutation type")
    
    def save_state(self) -> Dict:
        """Save current network state for potential rollback."""
        return {
            'state_dict': copy.deepcopy(self.net.state_dict()),
            'edge_index': self.net.edge_index.clone(),
            'num_nodes': self.net.num_nodes
        }
    
    def load_state(self, state: Dict):
        """
        Restore network state from saved checkpoint.
        
        Handles the case where the network topology has changed
        (e.g., after a mutation we want to revert).
        """
        # The network may have grown since the checkpoint.
        # We need to resize parameters to match checkpoint before loading.
        
        saved_num_nodes = state['num_nodes']
        saved_edge_index = state['edge_index']
        saved_num_edges = saved_edge_index.shape[1]
        
        # Resize node_features to match saved state
        saved_node_features = state['state_dict']['node_features']
        self.net.node_features = nn.Parameter(saved_node_features.clone())
        
        # Resize edge_weights to match saved state
        saved_edge_weights = state['state_dict']['edge_weights']
        self.net.edge_weights = nn.Parameter(saved_edge_weights.clone())
        
        # Restore edge_index
        self.net.register_buffer('edge_index', saved_edge_index.clone())
        
        # Restore num_nodes
        self.net.num_nodes = saved_num_nodes
        
        # Load the rest of state_dict (conv layers, output_proj, etc.)
        # Use strict=False since we've already manually set node_features/edge_weights
        remaining_state = {
            k: v for k, v in state['state_dict'].items()
            if k not in ['node_features', 'edge_weights', 'edge_index']
        }
        self.net.load_state_dict(remaining_state, strict=False)
    
    def optimize(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict:
        """
        Run the full hybrid optimization loop.
        
        Args:
            inputs: Training inputs (single sample for now)
            targets: Training targets
            
        Returns:
            Dictionary with optimization results
        """
        if self.config.verbose:
            print("[HYBRID] Starting Hybrid Optimization...")
            print(f"   Initial nodes: {self.net.get_num_nodes()}")
            print(f"   Initial edges: {self.net.get_num_edges()}")
            print("=" * 50)
        
        for outer_step in range(self.config.outer_steps):
            # === INNER LOOP: TRAIN WEIGHTS ===
            loss_before, converged = self.train_weights(inputs, targets)
            fitness_before = self.evaluate_fitness(inputs, targets)
            
            if self.config.verbose:
                status = "CONVERGED" if converged else "running"
                print(f"\n[Outer {outer_step+1}/{self.config.outer_steps}] "
                      f"Loss: {-fitness_before:.4f} | Nodes: {self.net.get_num_nodes()} | "
                      f"Edges: {self.net.get_num_edges()} | {status}")
            
            # Track best
            if fitness_before > self.best_fitness:
                self.best_fitness = fitness_before
                self.best_state = self.save_state()
            
            # === OUTER LOOP: MUTATE TOPOLOGY ===
            if converged and random.random() < self.config.mutation_prob:
                # Save state for potential rollback
                saved_state = self.save_state()
                
                # Mutate
                result = self.mutate_topology()
                
                if result.success:
                    if self.config.verbose:
                        print(f"   [MUTATION] {result.message}")
                    
                    self.history['mutations'].append(result.message)
                    
                    # Train on new topology
                    loss_after, _ = self.train_weights(inputs, targets)
                    fitness_after = self.evaluate_fitness(inputs, targets)
                    
                    # Accept or reject
                    if fitness_after > fitness_before:
                        if self.config.verbose:
                            print(f"   [ACCEPT] Fitness improved: {fitness_before:.4f} -> {fitness_after:.4f}")
                    else:
                        # Rollback
                        self.load_state(saved_state)
                        if self.config.verbose:
                            print(f"   [REJECT] Rolling back mutation")
                else:
                    if self.config.verbose:
                        print(f"   [MUTATION FAILED] {result.message}")
            
            # Track topology size
            self.history['topology_sizes'].append({
                'nodes': self.net.get_num_nodes(),
                'edges': self.net.get_num_edges()
            })
            self.history['outer_fitness'].append(fitness_before)
        
        # Restore best state
        if self.best_state is not None:
            self.load_state(self.best_state)
        
        final_fitness = self.evaluate_fitness(inputs, targets)
        
        if self.config.verbose:
            print("\n" + "=" * 50)
            print("[HYBRID] OPTIMIZATION COMPLETE")
            print(f"   Final Fitness: {final_fitness:.4f}")
            print(f"   Final Nodes: {self.net.get_num_nodes()}")
            print(f"   Final Edges: {self.net.get_num_edges()}")
            print(f"   Mutations Applied: {len(self.history['mutations'])}")
        
        return {
            'final_fitness': final_fitness,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'final_nodes': self.net.get_num_nodes(),
            'final_edges': self.net.get_num_edges()
        }


# === UNIT TEST / DEMO ===
if __name__ == "__main__":
    print("[DEMO] Testing Hybrid Optimizer on XOR Problem...")
    print("=" * 50)
    
    # Create a small network
    net = DynamicGraphNet(
        num_nodes=6,  # 4 input + 1 hidden + 1 output
        node_dim=8,
        num_heads=2,
        num_input_nodes=4,
        num_output_nodes=1
    )
    
    # XOR-like dataset
    inputs = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],  # Simple pattern
    ])
    targets = torch.tensor([0.5])  # Target output
    
    # Loss function
    def mse_loss(outputs, targets):
        return ((outputs - targets) ** 2).mean()
    
    # Configure
    config = HybridConfig(
        inner_lr=0.05,
        inner_steps=50,
        outer_steps=5,
        mutation_prob=0.5,
        inner_patience=10,
        verbose=True
    )
    
    # Run optimization
    optimizer = HybridOptimizer(net, mse_loss, config)
    results = optimizer.optimize(inputs, targets)
    
    print("\n[PASS] Hybrid Optimizer demo completed!")
