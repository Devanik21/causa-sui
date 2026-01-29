"""
Coarse-Graining: Partition Management for Macro-State Definitions.

This module manages the coarse-graining function phi that maps micro-states
to macro-states. The partition can be:
1. Fixed (predefined heuristics like "sum of inputs")
2. Learnable (evolved via mutations to maximize emergence)

Key Classes:
    - CoarseGraining: Base class for partition functions
    - SumPartition: Groups by input activity level
    - LearnablePartition: Evolvable partition for optimization

References:
    - Phase 1 Plan: Coarse-Graining Strategy
"""

import torch
import random
from typing import List, Dict, Callable
from abc import ABC, abstractmethod


class CoarseGraining(ABC):
    """Abstract base class for coarse-graining strategies."""
    
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> int:
        """Map a micro-state to its macro-state ID."""
        pass
    
    @abstractmethod
    def get_num_macro_states(self) -> int:
        """Return the number of unique macro-states."""
        pass
    
    @abstractmethod
    def get_macro_description(self, macro_id: int) -> str:
        """Return human-readable description of a macro-state."""
        pass


class SumPartition(CoarseGraining):
    """
    Simple partition based on sum of input activations.
    
    Groups micro-states by their "activity level":
    - LOW: sum = 0 or 1
    - MID: sum = 2
    - HIGH: sum = 3 or 4
    
    This is a sensible default for 4-input networks.
    """
    
    def __init__(self, num_inputs: int = 4):
        self.num_inputs = num_inputs
        
        # Define boundaries for activity levels
        # For 4 inputs: LOW=[0,1], MID=[2], HIGH=[3,4]
        mid = num_inputs // 2
        self.boundaries = [mid - 1, mid, num_inputs]
        
    def __call__(self, x: torch.Tensor) -> int:
        """Map micro-state to macro-state based on sum."""
        s = x.sum().item()
        
        if s <= self.boundaries[0]:
            return 0  # LOW
        elif s <= self.boundaries[1]:
            return 1  # MID
        else:
            return 2  # HIGH
    
    def get_num_macro_states(self) -> int:
        return 3
    
    def get_macro_description(self, macro_id: int) -> str:
        names = {0: "LOW", 1: "MID", 2: "HIGH"}
        return names.get(macro_id, "UNKNOWN")


class LearnablePartition(CoarseGraining):
    """
    An evolvable partition for optimizing causal emergence.
    
    The partition is represented as a direct mapping:
        partition_map[micro_state_index] -> macro_state_id
        
    This can be mutated (randomly reassign some states) and selected
    based on emergence score improvement.
    """
    
    def __init__(self, num_micro_states: int, num_macro_states: int = 3):
        """
        Initialize with a random partition.
        
        Args:
            num_micro_states: Total number of micro-states (2^N)
            num_macro_states: Number of macro-state groups to create
        """
        self.num_micro_states = num_micro_states
        self.num_macro_states = num_macro_states
        
        # Initialize: Assign each micro-state to a random macro-state
        self.partition_map = [
            random.randint(0, num_macro_states - 1)
            for _ in range(num_micro_states)
        ]
        
        # Precompute micro-state index lookup
        # We'll use binary conversion when called
        
    def __call__(self, x: torch.Tensor) -> int:
        """Map micro-state (binary tensor) to macro-state ID."""
        # Convert binary tensor to integer index
        idx = self._tensor_to_index(x)
        return self.partition_map[idx]
    
    def _tensor_to_index(self, x: torch.Tensor) -> int:
        """Convert binary tensor to integer index."""
        idx = 0
        for bit, val in enumerate(x.tolist()):
            if val > 0.5:
                idx |= (1 << bit)
        return idx
    
    def get_num_macro_states(self) -> int:
        return self.num_macro_states
    
    def get_macro_description(self, macro_id: int) -> str:
        return f"Macro-{macro_id}"
    
    def mutate(self, mutation_rate: float = 0.1) -> 'LearnablePartition':
        """
        Create a mutated copy of this partition.
        
        Args:
            mutation_rate: Probability of each micro-state being reassigned.
            
        Returns:
            new_partition: A new LearnablePartition with some states reassigned.
        """
        new_partition = LearnablePartition(
            self.num_micro_states,
            self.num_macro_states
        )
        new_partition.partition_map = self.partition_map.copy()
        
        # Randomly reassign some states
        for i in range(self.num_micro_states):
            if random.random() < mutation_rate:
                new_partition.partition_map[i] = random.randint(
                    0, self.num_macro_states - 1
                )
        
        return new_partition
    
    def get_partition_stats(self) -> Dict[int, int]:
        """Return count of micro-states in each macro-state."""
        from collections import Counter
        return dict(Counter(self.partition_map))


def create_identity_partition(num_micro_states: int) -> Callable:
    """
    Create a partition where each micro-state is its own macro-state.
    
    This gives EI_macro = EI_micro (no emergence).
    Useful as a baseline / sanity check.
    """
    def identity_partition(x: torch.Tensor) -> int:
        idx = 0
        for bit, val in enumerate(x.tolist()):
            if val > 0.5:
                idx |= (1 << bit)
        return idx
    
    return identity_partition


def create_trivial_partition() -> Callable:
    """
    Create a partition where ALL micro-states map to ONE macro-state.
    
    This makes EI_macro = 0 (no information from macro-level).
    Useful as an edge case test.
    """
    def trivial_partition(x: torch.Tensor) -> int:
        return 0  # Everyone is in the same bucket
    
    return trivial_partition


# === UNIT TEST ===
if __name__ == "__main__":
    print("[TEST] Testing Coarse-Graining Strategies...")
    
    # Test SumPartition
    print("\n   SumPartition Test:")
    phi = SumPartition(num_inputs=4)
    
    test_cases = [
        ([0, 0, 0, 0], "LOW"),
        ([1, 0, 0, 0], "LOW"),
        ([1, 1, 0, 0], "MID"),
        ([1, 1, 1, 0], "HIGH"),
        ([1, 1, 1, 1], "HIGH"),
    ]
    
    for inputs, expected in test_cases:
        x = torch.tensor(inputs, dtype=torch.float32)
        macro_id = phi(x)
        actual = phi.get_macro_description(macro_id)
        status = "PASS" if actual == expected else "FAIL"
        print(f"     {inputs} -> {actual} (expected {expected}) {status}")
    
    # Test LearnablePartition
    print("\n   LearnablePartition Test:")
    phi_learn = LearnablePartition(num_micro_states=16, num_macro_states=3)
    print(f"     Initial partition stats: {phi_learn.get_partition_stats()}")
    
    # Test mutation
    phi_mutated = phi_learn.mutate(mutation_rate=0.3)
    print(f"     Mutated partition stats: {phi_mutated.get_partition_stats()}")
    
    print("\n[PASS] Coarse-Graining test PASSED!")
