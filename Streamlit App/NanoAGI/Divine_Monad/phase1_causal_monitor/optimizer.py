"""
Optimizer: The training loop for maximizing Causal Emergence.

This module implements the core optimization loop that:
1. Trains network weights to maximize emergence score (via SGD)
2. Optionally evolves the coarse-graining partition (via ES/mutations)

Key Functions:
    - train_emergence: Main training loop
    - EarlyStopMonitor: Convergence detection

References:
    - Phase 1 Plan: The Optimization Loop
"""

import torch
import torch.optim as optim
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from .micro_causal_net import MicroCausalNet
from .effective_info import calc_micro_ei, calc_macro_ei, calc_emergence_score
from .coarse_graining import CoarseGraining, SumPartition, LearnablePartition


@dataclass
class TrainingConfig:
    """Configuration for the emergence training loop."""
    
    # Network
    num_inputs: int = 4
    
    # Optimization
    lr: float = 0.01
    num_epochs: int = 1000
    
    # Partition evolution (optional)
    evolve_partition: bool = False
    partition_mutation_rate: float = 0.1
    partition_evolve_every: int = 50
    
    # Logging
    log_every: int = 100
    
    # Early stopping
    early_stop_patience: int = 200
    early_stop_threshold: float = 1e-6


class EarlyStopMonitor:
    """Monitor for detecting convergence and triggering early stopping."""
    
    def __init__(self, patience: int, threshold: float):
        self.patience = patience
        self.threshold = threshold
        self.best_score = float('-inf')
        self.epochs_without_improvement = 0
        
    def update(self, score: float) -> bool:
        """
        Update with new score. Returns True if should stop.
        """
        if score > self.best_score + self.threshold:
            self.best_score = score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            
        return self.epochs_without_improvement >= self.patience


def train_emergence(
    config: Optional[TrainingConfig] = None,
    net: Optional[MicroCausalNet] = None,
    partition: Optional[CoarseGraining] = None,
    verbose: bool = True
) -> Dict:
    """
    Train a network to maximize Causal Emergence.
    
    This is the CORE optimization loop:
    1. Forward pass: Compute P(Y|x) for all micro-states
    2. EI calculation: Compute differentiable EI_micro and EI_macro
    3. Loss: emergence_score = EI_macro - EI_micro
    4. Backward: Gradients flow to weights
    5. (Optional) Evolve partition every N epochs
    
    Args:
        config: Training configuration
        net: Pre-initialized network (or create new one)
        partition: Pre-initialized partition (or use SumPartition)
        verbose: Print progress
        
    Returns:
        results: Dictionary with training history and final metrics
    """
    if config is None:
        config = TrainingConfig()
    
    # Initialize network
    if net is None:
        net = MicroCausalNet(num_inputs=config.num_inputs)
    
    # Initialize partition
    if partition is None:
        if config.evolve_partition:
            partition = LearnablePartition(
                num_micro_states=2**config.num_inputs,
                num_macro_states=3
            )
        else:
            partition = SumPartition(num_inputs=config.num_inputs)
    
    # Get all input states once (constant)
    all_inputs = net.get_all_input_states()
    
    # Optimizer (only for network weights)
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    
    # Early stopping monitor
    early_stop = EarlyStopMonitor(
        patience=config.early_stop_patience,
        threshold=config.early_stop_threshold
    )
    
    # History tracking
    history = {
        'ei_micro': [],
        'ei_macro': [],
        'emergence_score': [],
        'epochs': []
    }
    
    best_emergence = float('-inf')
    best_weights = None
    best_partition = partition
    
    # === MAIN TRAINING LOOP ===
    with torch.enable_grad():
        for epoch in range(config.num_epochs):
            optimizer.zero_grad()
        
        
        # 1. Calculate EI values (fully differentiable)
        ei_micro = calc_micro_ei(net, all_inputs)
        ei_macro = calc_macro_ei(net, all_inputs, partition)
        
        # 2. Emergence score (what we want to MAXIMIZE)
        emergence_score = ei_macro - ei_micro
        
        # 3. Loss = negative emergence (minimize to maximize emergence)
        loss = -emergence_score
        
        # 4. Backprop with gradient clipping for stability
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 5. Track best (skip NaN values)
        score_val = emergence_score.item()
        if not torch.isnan(emergence_score) and score_val > best_emergence:
            best_emergence = score_val
            best_weights = {k: v.clone() for k, v in net.state_dict().items()}
            if isinstance(partition, LearnablePartition):
                best_partition = LearnablePartition(
                    partition.num_micro_states,
                    partition.num_macro_states
                )
                best_partition.partition_map = partition.partition_map.copy()
        
        # 6. History
        history['epochs'].append(epoch)
        history['ei_micro'].append(ei_micro.item())
        history['ei_macro'].append(ei_macro.item())
        history['emergence_score'].append(score_val)
        
        # 7. Partition evolution (if enabled)
        if config.evolve_partition and isinstance(partition, LearnablePartition):
            if (epoch + 1) % config.partition_evolve_every == 0:
                # Try a mutated partition
                candidate = partition.mutate(config.partition_mutation_rate)
                candidate_ei_macro = calc_macro_ei(net, all_inputs, candidate)
                candidate_score = (candidate_ei_macro - ei_micro).item()
                
                if candidate_score > score_val:
                    partition = candidate
                    if verbose:
                        print(f"     [Epoch {epoch+1}] Partition evolved! New score: {candidate_score:.4f}")
        
        # 8. Logging
        if verbose and (epoch + 1) % config.log_every == 0:
            print(
                f"Epoch {epoch+1}/{config.num_epochs} | "
                f"EI_micro: {ei_micro.item():.4f} | "
                f"EI_macro: {ei_macro.item():.4f} | "
                f"Emergence: {score_val:.4f}"
            )
        
        # 9. Early stopping
        if early_stop.update(score_val):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best weights
    if best_weights is not None:
        net.load_state_dict(best_weights)
    
    # === FINAL RESULTS ===
    final_ei_micro = calc_micro_ei(net, all_inputs)
    final_ei_macro = calc_macro_ei(net, all_inputs, best_partition)
    final_emergence = final_ei_macro - final_ei_micro
    
    results = {
        'history': history,
        'final_ei_micro': final_ei_micro.item(),
        'final_ei_macro': final_ei_macro.item(),
        'final_emergence': final_emergence.item(),
        'best_emergence': best_emergence,
        'net': net,
        'partition': best_partition,
        'converged_epoch': len(history['epochs'])
    }
    
    if verbose:
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(f"Final EI_micro:  {results['final_ei_micro']:.4f} bits")
        print(f"Final EI_macro:  {results['final_ei_macro']:.4f} bits")
        print(f"Final Emergence: {results['final_emergence']:.4f} bits")
        print(f"Best Emergence:  {results['best_emergence']:.4f} bits")
        
        if results['final_emergence'] > 0:
            print("\n[SUCCESS] CAUSAL EMERGENCE ACHIEVED!")
            print("The MACRO-LEVEL has more causal power than the MICRO-LEVEL.")
        else:
            print("\n[WARNING] No emergence detected. Try different partition or more epochs.")
    
    return results


# === UNIT TEST / DEMO ===
if __name__ == "__main__":
    print("[DEMO] Training for Causal Emergence (2-layer net)...")
    print("="*50 + "\n")
    
    config = TrainingConfig(
        num_inputs=4,
        lr=0.01,  # Lower LR for stability
        num_epochs=1000,
        log_every=200,
        evolve_partition=False
    )
    
    results = train_emergence(config, verbose=True)
    
    print("\n" + "="*50)
    print("Network Weights After Training:")
    net = results['net']
    print(f"   W1 norm: {net.W1.data.norm().item():.4f}")
    print(f"   W2 norm: {net.W2.data.norm().item():.4f}")


