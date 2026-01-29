"""
Phase 2: Topological Computing

Modules:
    - graph_substrate: DynamicGraphNet using PyTorch Geometric
    - mutator: Net2Net function-preserving topology mutations
    - hybrid_optimizer: Inner/Outer loop optimization
"""

from .graph_substrate import DynamicGraphNet
from .mutator import TopologicalMutator, MutationResult
from .hybrid_optimizer import HybridOptimizer, HybridConfig

__all__ = [
    'DynamicGraphNet',
    'TopologicalMutator',
    'MutationResult',
    'HybridOptimizer',
    'HybridConfig'
]
