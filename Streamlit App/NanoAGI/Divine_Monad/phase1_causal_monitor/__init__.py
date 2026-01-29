"""
Divine Monad - Phase 1: The Causal Monitor

This package implements the mathematical infrastructure for detecting
and optimizing Causal Emergence in neural networks.

Core Components:
    - micro_causal_net: The tiny stochastic network (4 input, 1 output)
    - effective_info: Differentiable Effective Information calculation
    - coarse_graining: Partition strategies for macro-state definitions
    - optimizer: Training loop to maximize emergence

Usage:
    from Divine_Monad.phase1_causal_monitor import train_emergence
    results = train_emergence()
"""

from .micro_causal_net import MicroCausalNet, create_xor_net
from .effective_info import (
    binary_entropy,
    calc_micro_ei,
    calc_macro_ei,
    calc_emergence_score
)
from .coarse_graining import (
    CoarseGraining,
    SumPartition,
    LearnablePartition,
    create_identity_partition,
    create_trivial_partition
)
from .optimizer import (
    TrainingConfig,
    EarlyStopMonitor,
    train_emergence
)

__version__ = "0.1.0"
__author__ = "Divine Monad Project"
