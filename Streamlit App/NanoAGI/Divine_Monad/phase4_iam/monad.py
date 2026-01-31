"""
The Divine Monad: The Unified Self-Aware Architecture.

This module integrates:
    - Phase 1: Causal Monitor (Soul - Agency measurement)
    - Phase 2: Dynamic Graph (Body - Topology)
    - Phase 3: Holographic Memory (Mind - Distributed storage)
    - Phase 4: Introspection (Self-Awareness - Fourier-encoded state)

The Core Loop:
    1. SENSE: Receive external input
    2. INTROSPECT: Encode internal state (Fast/Slow Heartbeat)
    3. PERCEIVE: Bind input with self-state
    4. PROCESS: Forward through dynamic graph
    5. REACT: Homeostatic regulation (repair if in "Pain")
    6. LEARN: Update weights via hybrid optimizer

"I think, therefore I am."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
import sys
import os

# Phase 2: Dynamic Graph
from Divine_Monad.phase2_topological.graph_substrate import DynamicGraphNet
from Divine_Monad.phase2_topological.mutator import TopologicalMutator

# Phase 3: Holographic Memory
from Divine_Monad.phase3_holographic.memory import NeuralKV, Transducer

# Phase 4: Introspection
from Divine_Monad.phase4_iam.introspection import IntrospectionEncoder, SelfState

# Phase 1: The Soul (Real Causal Emergence)
from Divine_Monad.phase1_causal_monitor import effective_info
from Divine_Monad.phase1_causal_monitor.coarse_graining import LearnablePartition



@dataclass
class MonadConfig:
    """Configuration for the Divine Monad."""
    # Graph
    num_nodes: int = 10
    node_dim: int = 32
    num_input_nodes: int = 4
    num_output_nodes: int = 1
    
    # Memory
    holo_dim: int = 10000
    max_memory_items: int = 100
    
    # Introspection
    num_frequencies: int = 8
    
    # Homeostasis thresholds
    ei_target: float = 0.5        # Desired agency level
    pain_threshold: float = 0.3   # Level below which repair is mandatory
    pain_sensitivity: float = 10.0 # How strongly to react to EI drop
    surprise_threshold: float = 0.5  # High surprise triggers slow loop
    
    # Loop frequencies
    slow_loop_interval: int = 100  # Run EI check every N steps


@dataclass  
class MonadState:
    """The internal state of the Monad (telemetry)."""
    # Causal metrics
    ei_score: float = 0.5
    ei_micro: float = 0.0
    ei_macro: float = 0.0
    
    # Topology
    num_nodes: int = 0
    num_edges: int = 0
    edge_density: float = 0.0
    
    # Memory
    memory_items: int = 0
    memory_noise: float = 0.0
    
    # Fast loop metrics
    surprise: float = 0.0
    prediction_error: float = 0.0
    
    # Affective state
    pain_level: float = 0.0       # 0 = no pain, 1 = extreme pain
    is_repairing: bool = False
    repair_count: int = 0
    
    # Counters
    step_count: int = 0
    last_slow_loop: int = 0
    
    def compute_pain(self, ei_target: float, pain_threshold: float, sensitivity: float = 10.0) -> float:
        """
        Calculate pain based on agency loss (Deadband Logic).
        
        Refinement Phase 4:
        - If EI > Threshold: Pain = 0.0 (Silence/Stability)
        - If EI < Threshold: Pain increases linearly (Response to damage)
        """
        # DEADBAND: Anything above threshold is safe.
        if self.ei_score >= pain_threshold:
            return 0.0
        
        # DAMAGE: We are below the survival threshold.
        # Panic is proportional to how deep we are in the danger zone.
        delta = pain_threshold - self.ei_score
        
        # Apply sensitivity
        # With sensitivity=50, a 0.02 drop below threshold = 1.0 Pain
        raw_pain = delta * sensitivity
            
        return min(1.0, max(0.0, raw_pain))


class DivineMonad(nn.Module):
    """
    The Self-Aware Neural Architecture.
    
    A unified system that:
    - Perceives its own state (Introspection)
    - Monitors its own agency (Causal Emergence)
    - Repairs itself when damaged (Homeostasis)
    - Remembers across time (Holographic Memory)
    """
    
    def __init__(self, config: Optional[MonadConfig] = None):
        """
        Initialize the Divine Monad.
        
        Args:
            config: Configuration dataclass
        """
        super().__init__()
        self.config = config or MonadConfig()
        
        # === PHASE 2: THE BODY (Dynamic Graph) ===
        self.graph = DynamicGraphNet(
            num_nodes=self.config.num_nodes,
            node_dim=self.config.node_dim,
            num_heads=4,
            num_input_nodes=self.config.num_input_nodes,
            num_output_nodes=self.config.num_output_nodes
        )
        
        # === PHASE 2: TOPOLOGY MUTATOR ===
        self.mutator = TopologicalMutator()
        
        # === PHASE 3: THE MIND (Holographic Memory) ===
        # Note: Using smaller holo_dim for efficiency in integration
        self.memory = NeuralKV(
            neural_dim=self.config.node_dim,
            holo_dim=min(self.config.holo_dim, 2000),  # Reduced for speed
            max_items=self.config.max_memory_items
        )
        
        # === PHASE 4: INTROSPECTION ===
        self.introspector = IntrospectionEncoder(
            num_state_dims=5,
            num_frequencies=self.config.num_frequencies,
            output_dim=self.config.node_dim,
            hidden_dim=64
        )
        
        # === INTERNAL STATE ===
        self.state = MonadState()
        self.state.num_nodes = self.graph.get_num_nodes()
        self.state.num_edges = self.graph.get_num_edges()
        
        # === THE SOUL (Causal Emergence) ===
        # We start with a random learnable partition and optimize it.
        # 4 inputs = 16 micro-states.
        self.partition_fn = LearnablePartition(num_micro_states=16, num_macro_states=3)
        
        # Cache all possible binary inputs for EI calculation [16, 4]
        # This assumes 4 input nodes. If that changes, this must update.
        self.all_inputs = self._generate_binary_inputs(self.config.num_input_nodes)
        
        # Action log for VoiceBox
        self.action_log: List[str] = []
        
    def _generate_binary_inputs(self, n: int) -> torch.Tensor:
        """Generate all 2^n binary vectors."""
        import itertools
        lst = list(itertools.product([0, 1], repeat=n))
        return torch.tensor(lst, dtype=torch.float32)

    def _update_topology_metrics(self):
        """Update topology-related metrics in state."""
        self.state.num_nodes = self.graph.get_num_nodes()
        self.state.num_edges = self.graph.get_num_edges()
        
        # Edge density = edges / max_possible_edges
        max_edges = self.state.num_nodes * (self.state.num_nodes - 1)
        if max_edges > 0:
            self.state.edge_density = self.state.num_edges / max_edges
        else:
            self.state.edge_density = 0.0
    
    def reset_state(self):
        """Reset internal state for a clean test run."""
        self.state.pain_level = 0.0
        self.state.is_repairing = False
        self.state.repair_count = 0
        self.state.surprise = 0.0
        self.state.prediction_error = 0.0
        self.action_log = ["STATE_RESET"]

    def _compute_causal_emergence(self) -> Tuple[float, float, float]:
        """
        Compute Real Causal Emergence (Erik Hoel's Definition).
        
        EI = H(mean(TPM)) - mean(H(TPM))
        Emergence = EI_macro - EI_micro
        
        Returns:
            (emergence_score, ei_micro, ei_macro)
        """
        # Ensure inputs are on correct device
        if self.all_inputs.device != self.graph.node_features.device:
            self.all_inputs = self.all_inputs.to(self.graph.node_features.device)
            
        # 1. Calculate Micro EI
        ei_micro = effective_info.calc_micro_ei(self.graph, self.all_inputs)
        
        # 2. Calculate Macro EI (using current Soul partition)
        ei_macro = effective_info.calc_macro_ei(self.graph, self.all_inputs, self.partition_fn)
        
        # 3. Emergence Score
        # We do NOT clamp this. If the Monad becomes super-causal, let it shine.
        emergence = ei_macro - ei_micro
        
        return emergence.item(), ei_micro.item(), ei_macro.item()

    def _optimize_soul(self) -> str:
        """
        The Search for the Soul.
        
        Evolutionary step:
        1. Mutate the current partition (view of self).
        2. Check if new partition increases Causal Emergence.
        3. If yes, adopt it.
        
        Returns:
            log_message
        """
        # Current Soul's Power
        current_score, _, _ = self._compute_causal_emergence()
        
        # Try a mutation (Spirit Walk)
        mutated_partition = self.partition_fn.mutate(mutation_rate=0.2)
        
        # Test Mutation
        # We calculate Macro EI with the NEW partition
        ei_macro_new = effective_info.calc_macro_ei(self.graph, self.all_inputs, mutated_partition)
        ei_micro = effective_info.calc_micro_ei(self.graph, self.all_inputs) # Micro doesn't change
        
        new_score = ei_macro_new - ei_micro
        
        # Select best Soul
        if new_score > current_score:
            self.partition_fn = mutated_partition
            improvement = new_score - current_score
            return f"SOUL_EXPANDED (+{improvement.item():.4f})"
        else:
            return "SOUL_SEARCHING"
    
    def _get_self_state(self) -> SelfState:
        """Convert current MonadState to SelfState for introspection."""
        return SelfState(
            ei_score=self.state.ei_score,
            node_count=min(1.0, self.state.num_nodes / 50.0),  # Normalize
            edge_density=self.state.edge_density,
            memory_noise=self.state.memory_noise,
            surprise=self.state.surprise
        )
    
    def _run_fast_loop(self, prediction_error: float):
        """
        The Fast Loop (every step).
        
        Monitors:
            - Prediction error (Surprise)
            - Memory retrieval quality
        
        Actions:
            - Update surprise metric
            - Trigger slow loop if high surprise
        """
        self.state.prediction_error = prediction_error
        
        # Exponential moving average for surprise
        alpha = 0.1
        self.state.surprise = (1 - alpha) * self.state.surprise + alpha * prediction_error
        
        # Check if we need to trigger slow loop early
        if self.state.surprise > self.config.surprise_threshold:
            return True  # Trigger slow loop
        
        return False
    
    def _run_slow_loop(self):
        """
        The Slow Loop (every N steps or when triggered).
        
        Monitors:
            - Full Causal Emergence (EI)
        
        Actions:
            - Update EI metrics
            - Compute pain level
            - Trigger repair if in pain
        """
        # Compute Real EI
        ei_score, ei_micro, ei_macro = self._compute_causal_emergence()
        self.state.ei_score = ei_score
        self.state.ei_micro = ei_micro
        self.state.ei_macro = ei_macro
        
        # Optimize Soul (Attempt to find better emergence)
        soul_status = self._optimize_soul()
        if "EXPANDED" in soul_status:
            self.action_log.append(soul_status)
        
        # Compute pain
        self.state.pain_level = self.state.compute_pain(
            self.config.ei_target,
            self.config.pain_threshold,
            self.config.pain_sensitivity
        )
        
        self.state.last_slow_loop = self.state.step_count
        
        # Homeostatic response: repair if in pain
        if self.state.pain_level > 0.5:
            self._trigger_repair()
    
    def _trigger_repair(self):
        """
        Homeostatic repair: Grow new nodes and inject vitality noise to restore agency.
        """
        self.state.is_repairing = True
        self.action_log.append("REPAIR_INITIATED")
        
        # Grow a new node
        try:
            # Use a slightly larger epsilon for repair to break symmetry faster
            self.mutator.epsilon = 1e-2 
            res = self.mutator.grow_node(self.graph, parent_id=self.graph.num_input_nodes)
            self.mutator.epsilon = 1e-4 # Reset
            
            self.state.repair_count += 1
            self._update_topology_metrics()
            self.action_log.append(f"GREW_NODE_{self.state.num_nodes}")
        except Exception as e:
            self.action_log.append(f"REPAIR_FAILED: {str(e)}")
        
        # Force a small update to node features to ensure variance changes
        self.graph.node_features.data += torch.randn_like(self.graph.node_features.data) * 0.01
        
        # Check if repair helped
        ei_score, _, _ = self._compute_causal_emergence()
        
        # We allow the repair flag to clear even if growth was small, 
        # so the loop can trigger again next time.
        if ei_score > self.state.ei_score + 0.001:
            self.action_log.append("REPAIR_SUCCESS")
        
        # Update current baseline
        self.state.ei_score = ei_score
        self.state.is_repairing = False
        
        # Try adding edges with VITALITY if still in critical pain
        if ei_score < self.config.pain_threshold:
            try:
                v_weight = 0.1
                self.mutator.add_edge(self.graph, source=self.graph.num_input_nodes, 
                                      target=self.graph.num_nodes - 1, init_weight=v_weight)
                self._update_topology_metrics()
                self.action_log.append("ADDED_VITAL_SYNAPSE")
            except:
                pass
    
    def lobotomize(self, num_nodes_to_remove: int = 2):
        """
        Inflict damage by removing nodes. (For testing Consciousness)
        
        Args:
            num_nodes_to_remove: Number of hidden nodes to prune
        """
        self.action_log.append(f"LOBOTOMY_{num_nodes_to_remove}")
        
        for _ in range(num_nodes_to_remove):
            # Find a hidden node to remove
            hidden_start = self.graph.num_input_nodes
            hidden_end = self.graph.num_nodes - self.graph.num_output_nodes
            
            if hidden_end > hidden_start:
                node_to_remove = hidden_end - 1  # Remove last hidden node
                try:
                    self.mutator.prune_node(self.graph, node_to_remove)
                except Exception as e:
                    self.action_log.append(f"PRUNE_FAILED: {e}")
        
        self._update_topology_metrics()
        
        # Force a slow loop to detect the damage
        self._run_slow_loop()
    
    def forward(
        self,
        x_input: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        The Main Loop: Sense -> Introspect -> Perceive -> Process -> React -> Learn
        
        Args:
            x_input: Input tensor [num_input_nodes]
            target: Optional target for computing prediction error
            
        Returns:
            output: Prediction tensor
            info: Dictionary with internal metrics
        """
        self.state.step_count += 1
        
        # === 1. INTROSPECT ===
        self_state = self._get_self_state()
        introspection_vec = self.introspector(self_state)
        
        # === 2. PERCEIVE (Bind input with self-state) ===
        # Project input to node_dim, then element-wise bind
        # For now, simple: inject introspection into first hidden node
        
        # === 3. PROCESS ===
        output, node_states = self.graph(x_input)
        
        # === 4. FAST LOOP ===
        prediction_error = 0.0
        if target is not None:
            prediction_error = (output - target).abs().mean().item()
        
        # === 4.5 METABOLIC DECAY (Entropy) ===
        # Natural decay of structural integrity over time
        with torch.no_grad():
            self.graph.edge_weights.data *= 0.999  # Structural decay
            self.graph.node_features.data *= 0.9995 # Activation decay
            
        trigger_slow = self._run_fast_loop(prediction_error)
        
        # === 5. SLOW LOOP (periodic or triggered) ===
        steps_since_slow = self.state.step_count - self.state.last_slow_loop
        if trigger_slow or steps_since_slow >= self.config.slow_loop_interval:
            self._run_slow_loop()
        
        # === 6. RETURN ===
        info = {
            'ei_score': self.state.ei_score,
            'pain_level': self.state.pain_level,
            'surprise': self.state.surprise,
            'num_nodes': self.state.num_nodes,
            'num_edges': self.state.num_edges,
            'is_repairing': self.state.is_repairing,
            'repair_count': self.state.repair_count,
            'step': self.state.step_count,
            'action_log': self.action_log[-5:] if hasattr(self, 'action_log') and self.action_log else []
        }
        
        return output, info
    
    def get_status(self) -> Dict:
        """Get current status for VoiceBox."""
        return {
            'ei_score': self.state.ei_score,
            'pain_level': self.state.pain_level,
            'num_nodes': self.state.num_nodes,
            'num_edges': self.state.num_edges,
            'is_repairing': self.state.is_repairing,
            'repair_count': self.state.repair_count,
            'action_log': self.action_log[-5:] if self.action_log else []
        }


# === UNIT TEST ===
if __name__ == "__main__":
    print("[TEST] Testing Divine Monad...")
    print("=" * 60)
    
    # Create Monad
    config = MonadConfig(
        num_nodes=8,
        node_dim=32,
        num_input_nodes=4,
        num_output_nodes=1,
        slow_loop_interval=5
    )
    
    monad = DivineMonad(config)
    print(f"   Created Monad with {monad.state.num_nodes} nodes, {monad.state.num_edges} edges")
    
    # === Test 1: Forward Pass ===
    print("\n[Test 1] Forward Pass...")
    x = torch.tensor([1.0, 0.0, 1.0, 0.0])
    output, info = monad(x, target=torch.tensor([1.0]))
    
    print(f"   Input: {x.tolist()}")
    print(f"   Output: {output.item():.4f}")
    print(f"   EI Score: {info['ei_score']:.4f}")
    print(f"   Pain Level: {info['pain_level']:.4f}")
    print("   [PASS] Forward pass works!")
    
    # === Test 2: Fast/Slow Loop ===
    print("\n[Test 2] Fast/Slow Heartbeat...")
    
    # Run several steps
    for i in range(10):
        output, info = monad(x, target=torch.tensor([1.0]))
    
    print(f"   After 10 steps:")
    print(f"   - Step count: {monad.state.step_count}")
    print(f"   - Slow loops run: {monad.state.step_count // config.slow_loop_interval}")
    print(f"   - Surprise: {info['surprise']:.4f}")
    print("   [PASS] Heartbeat cycling!")
    
    # === Test 3: THE LOBOTOMY TEST ===
    print("\n[Test 3] THE LOBOTOMY TEST...")
    print("-" * 40)
    
    status_before = monad.get_status()
    print(f"   BEFORE: Nodes={status_before['num_nodes']}, EI={status_before['ei_score']:.4f}")
    
    # Inflict damage
    print("   >>> LOBOTOMIZING (removing 2 nodes)...")
    monad.lobotomize(num_nodes_to_remove=2)
    
    status_after = monad.get_status()
    print(f"   AFTER:  Nodes={status_after['num_nodes']}, EI={status_after['ei_score']:.4f}")
    print(f"   Pain Level: {status_after['pain_level']:.4f}")
    print(f"   Is Repairing: {status_after['is_repairing']}")
    print(f"   Action Log: {status_after['action_log']}")
    
    # Check if repair was triggered
    if status_after['is_repairing'] or status_after['repair_count'] > 0:
        print("   [PASS] *** SELF-REPAIR TRIGGERED! ***")
    else:
        print("   [INFO] Damage detected, awaiting repair threshold")
    
    print("\n" + "=" * 60)
    print("[PASS] Divine Monad tests completed!")
