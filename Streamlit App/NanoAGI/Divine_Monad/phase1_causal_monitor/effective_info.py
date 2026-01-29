"""
Effective Information Calculator: Differentiable EI for Causal Emergence.

This module implements the "Soft Path" calculation of Effective Information (EI).
All computations use probabilities, not samples, ensuring gradient flow.

Key Functions:
    - binary_entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)
    - calc_micro_ei: EI at the neuron level
    - calc_macro_ei: EI at the module level

Mathematical Foundation:
    EI = I(X_do ; Y) = H(Y) - H(Y|X)
    
    Where:
    - X_do = Maximum entropy intervention (uniform distribution over inputs)
    - Y = Output distribution
    - H = Entropy
    
References:
    - Erik Hoel: "Quantifying Causal Emergence" (2017)
    - Phase 1 Plan with "Soft Path" refinement
"""

import torch
from typing import Callable, List

# Small epsilon to prevent log(0)
EPS = 1e-10


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """
    Compute binary entropy H(p) in bits.
    
    H(p) = -p * log2(p) - (1-p) * log2(1-p)
    
    Args:
        p: Probability tensor. Values should be in (0, 1).
           Can be any shape.
           
    Returns:
        entropy: Entropy values in bits. Same shape as input.
    """
    # Clamp to avoid log(0)
    p_clamped = torch.clamp(p, EPS, 1 - EPS)
    
    # Binary entropy formula
    entropy = -p_clamped * torch.log2(p_clamped) - (1 - p_clamped) * torch.log2(1 - p_clamped)
    
    return entropy


def calc_micro_ei(net, all_inputs: torch.Tensor) -> torch.Tensor:
    """
    Calculate Effective Information at the MICRO (neuron) level.
    
    This measures: "How well can I predict the output if I know
    which specific micro-state I intervened with?"
    
    Formula:
        EI_micro = H(Y) - H(Y|X)
        
    Where:
        - H(Y) = Marginal entropy of output (averaged over uniform X)
        - H(Y|X) = Mean conditional entropy (how uncertain is Y given EACH X?)
    
    Args:
        net: A MicroCausalNet instance
        all_inputs: All possible binary input states. Shape: [2^N, N]
        
    Returns:
        ei: Effective Information in bits (scalar tensor, differentiable)
    """
    # 1. Get P(Y=1|x) for all micro-states
    # Shape: [num_states, 1]
    probs = net.forward_prob(all_inputs)
    probs = probs.squeeze(-1)  # Shape: [num_states]
    
    # 2. Marginal Probability P(Y=1)
    # Under maximum entropy intervention (uniform X), just average the probs
    p_marginal = torch.mean(probs)
    
    # 3. Marginal Entropy H(Y)
    h_marginal = binary_entropy(p_marginal)
    
    # 4. Conditional Entropy H(Y|X)
    # For each specific input x, the output entropy is H(P(Y=1|x))
    # Average over all inputs (since intervention is uniform)
    h_conditional = torch.mean(binary_entropy(probs))
    
    # 5. EI = H(Y) - H(Y|X)
    # This measures how much knowing X reduces uncertainty about Y
    ei = h_marginal - h_conditional
    
    return ei


def calc_macro_ei(
    net,
    all_inputs: torch.Tensor,
    partition_fn: Callable[[torch.Tensor], int]
) -> torch.Tensor:
    """
    Calculate Effective Information at the MACRO (module) level.
    
    This measures: "How well can I predict the output if I only know
    which MACRO-STATE I intervened with?"
    
    Formula:
        EI_macro = H(Y) - H(Y|M)
        
    Where:
        - H(Y) = Same marginal entropy as micro
        - H(Y|M) = Mean conditional entropy given macro-states
        
    Key Insight:
        P(Y=1 | M_k) = (1/|M_k|) * sum_{x in M_k} P(Y=1|x)
        
        This is the weighted mixture model for macro-intervention.
    
    Args:
        net: A MicroCausalNet instance
        all_inputs: All possible binary input states. Shape: [2^N, N]
        partition_fn: A function that maps a micro-state to its macro-state ID.
        
    Returns:
        ei: Effective Information in bits (scalar tensor, differentiable)
    """
    # 1. Get P(Y=1|x) for all micro-states
    probs = net.forward_prob(all_inputs)
    probs = probs.squeeze(-1)  # Shape: [num_states]
    
    # 2. Group micro-states by their macro-state
    # partition_fn returns an integer macro-state ID for each micro-state
    macro_ids = torch.tensor([partition_fn(x) for x in all_inputs])
    unique_macros = torch.unique(macro_ids)
    
    # 3. For each macro-state, compute P(Y=1|M_k)
    macro_probs = []
    for m_k in unique_macros:
        # Indices of micro-states belonging to this macro-state
        mask = (macro_ids == m_k)
        
        # P(Y=1|M_k) = Mean of P(Y=1|x) for all x in M_k
        # This represents uniform intervention over micro-states in M_k
        p_y_given_m = torch.mean(probs[mask])
        macro_probs.append(p_y_given_m)
    
    macro_probs = torch.stack(macro_probs)
    
    # 4. H(Y|M) = Mean entropy across macro-states
    # (Each macro-state equally likely under uniform macro-intervention)
    h_conditional_macro = torch.mean(binary_entropy(macro_probs))
    
    # 5. Marginal H(Y) - same as micro case
    p_marginal = torch.mean(probs)
    h_marginal = binary_entropy(p_marginal)
    
    # 6. EI = H(Y) - H(Y|M)
    ei = h_marginal - h_conditional_macro
    
    return ei


def calc_emergence_score(
    net,
    all_inputs: torch.Tensor,
    partition_fn: Callable[[torch.Tensor], int]
) -> torch.Tensor:
    """
    Calculate the Causal Emergence Score.
    
    Emergence Score = EI_macro - EI_micro
    
    If positive: The macro-level description has MORE causal power.
    If negative: The micro-level is more informative.
    If zero: No emergence (or degenerate case).
    
    Args:
        net: A MicroCausalNet instance
        all_inputs: All possible binary input states
        partition_fn: Coarse-graining function
        
    Returns:
        score: Emergence score (positive = emergence exists)
    """
    ei_micro = calc_micro_ei(net, all_inputs)
    ei_macro = calc_macro_ei(net, all_inputs, partition_fn)
    
    return ei_macro - ei_micro


# === UNIT TEST ===
if __name__ == "__main__":
    from micro_causal_net import MicroCausalNet
    
    print("[TEST] Testing Effective Information Calculator...")
    
    # Create network
    net = MicroCausalNet(num_inputs=4)
    all_inputs = net.get_all_input_states()
    
    # Test binary entropy
    print("   Binary Entropy Tests:")
    print(f"     H(0.5) = {binary_entropy(torch.tensor(0.5)).item():.4f} (should be 1.0)")
    print(f"     H(0.0) = {binary_entropy(torch.tensor(0.01)).item():.4f} (should be ~0)")
    print(f"     H(1.0) = {binary_entropy(torch.tensor(0.99)).item():.4f} (should be ~0)")
    
    # Test Micro-EI
    ei_micro = calc_micro_ei(net, all_inputs)
    print(f"\n   Micro-EI: {ei_micro.item():.4f} bits")
    
    # Simple partition: Split by sum of inputs (LOW vs HIGH activity)
    def simple_partition(x: torch.Tensor) -> int:
        """Partition by input sum: 0-1=LOW, 2=MID, 3-4=HIGH"""
        s = x.sum().item()
        if s <= 1:
            return 0  # LOW
        elif s == 2:
            return 1  # MID
        else:
            return 2  # HIGH
    
    # Test Macro-EI
    ei_macro = calc_macro_ei(net, all_inputs, simple_partition)
    print(f"   Macro-EI: {ei_macro.item():.4f} bits")
    
    # Test Emergence Score
    emergence = calc_emergence_score(net, all_inputs, simple_partition)
    print(f"   Emergence Score: {emergence.item():.4f} bits")
    
    # Check gradient flow
    loss = -emergence
    loss.backward()
    
    if net.weights.grad is not None and net.weights.grad.abs().sum() > 0:
        print(f"\n   Gradient on weights: {net.weights.grad}")
        print("[PASS] Effective Information test PASSED!")
    else:
        print("[FAIL] No gradients - gradient flow broken!")
