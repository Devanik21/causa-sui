# 🧬 Nano-Daemon: A Neuromorphic Framework for Differentiable Hebbian Plasticity

### Created by [Devanik](https://github.com/Devanik21)

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-181717.svg?style=flat&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Devanik-0A66C2.svg?style=flat&logo=linkedin)](https://www.linkedin.com/in/devanik/)
[![X](https://img.shields.io/badge/X-@devanik2005-000000.svg?style=flat&logo=x)](https://x.com/devanik2005)

---

## 🌌 Introduction: The "God Engine" of Intelligence

**Nano-Daemon** is not a traditional neural network; it is a **Bi-Level Optimization System** designed to evolve the rules of learning itself. While standard Artificial Intelligence relies on static, hand-coded optimization algorithms (like Adam or SGD), Nano-Daemon implements **Differentiable Hebbian Plasticity**. 

In this framework, the weights of the network are not just updated—they are **governed by an evolved Genome** (the `PlasticityNetwork`). This Genome determines how synapses should shift in response to local activations, effectively allowing the system to discover its own unsupervised learning rules.

This project represents a realization of the **"God Engine"**: a recursive loops where a meta-optimizer (Outer Loop) creates a plasticity rule that enables a cortex (Inner Loop) to learn novel associations with near-zero error.

---

## 🧬 I. The Mathematical Identity: Bi-Level Optimization

At its core, Nano-Daemon solves a formal **Bi-Level Optimization Problem**, defined by an interleaved inner and outer optimization trajectory.

### 1. The Inner Loop (Lifetime Learning)
The "Brain" (Cortex) undergoes a journey of sensation and adaptation. Given an input stream $\mathcal{X} = \{x_1, x_2, \dots, x_T\}$, the synaptic matrix $W$ evolves as a **Recurrent Dynamical System**:

$$h_t = \sigma(x_t \cdot W_t + h_{t-1})$$

The update rule for $W$ is not backpropagation, but a local functional $\Phi$ parameterized by the Genome $\theta$:

$$\Delta W_t = \Phi(x_t, h_t, W_t; \theta)$$
$$W_{t+1} = \text{Norm}(W_t + \eta \cdot \Delta W_t)$$

Where $\text{Norm}$ represents a row-wise $L_2$ normalization to ensure homeostatic stability and prevent weight explosion.

### 2. The Outer Loop (Evolutionary Meta-Learning)
The "God Engine" seeks to find the optimal $\theta$ (Genome weights) such that the resulting Brain can solve an associative task $\mathcal{T}$ with minimal error. The meta-objective is defined as:

$$\min_{\theta} \mathcal{L}_{meta} = \sum_{i \in \mathcal{T}} \| \text{Recall}(Cue_i; W_T(\theta)) - Target_i \|^2$$

To optimize $\theta$, we must compute the gradient through the entire lifetime of weight updates. This requires **Backpropagation Through Time (BPTT)** across the learning rule itself:

$$\nabla_\theta \mathcal{L}_{meta} = \frac{\partial \mathcal{L}_{meta}}{\partial W_T} \left( \prod_{k=1}^{T} \frac{\partial W_k}{\partial W_{k-1}} \right) \frac{\partial W_k}{\partial \theta}$$

This meta-gradient propagates through the "unrolled" learning steps, allowing the system to differentiate through the *process* of learning.

---

## 🧠 II. Differentiable Hebbian Plasticity

Nano-Daemon replaces the classical, rigid Hebbian rule (e.g., Oja's Rule: $\Delta w = \eta y(x - yw)$) with a **Universal Function Approximator**.

### The Genome Logic ($\phi$)
The Genome, $\phi$, is a tiny multi-layer perceptron (MLP) that resides at every synaptic junction. It takes three local scalars as input:
1.  **Pre-synaptic activity** ($x_i$)
2.  **Post-synaptic activity** ($h_j$)
3.  **Current connection strength** ($w_{ij}$)

It outputs a single scalar: the directional flux of the synapse.

$$\Delta w_{ij} = \text{MLP}_{\theta}([x_i, h_j, w_{ij}])$$

### Scheduled Teacher Forcing
To bridge the gap between "Guided Learning" and "Autonomous Inference," we utilize **Scheduled Teacher Forcing**. During the $K$ steps of the inner loop, the target signal $y_{target}$ is gradually faded:

$$\hat{y}_k = \alpha_k \cdot y_{target} + (1 - \alpha_k) \cdot \text{Activation}_k$$
$$\text{where } \alpha_k = 1.0 - \frac{k}{K-1}$$

This curriculum forces the Genome to learn a rule that can eventually function in the absence of a supervisor, achieving **True Autonomy**.

---

## 🌌 III. The 21 Stages of General Intelligence

Nano-Daemon implements a structured roadmap for cognitive development, as proposed by **Devanik**. These stages are not just conceptual; they are reflected in the codebase logic:

### Tier 1: Core Neural Substrate
*   **1. Sensation**: Byte-level embedding of raw information.
*   **2. Hebbian Flux**: Base synaptic updates using local activity.
*   **3. Homeostasis**: $L_2$ normalization guarding against saturation.
*   **4. Experience Buffering**: Episodic memory for consolidation.
*   **5. Multi-Scale Latency**: Dual-stream buffers for Short-Term vs. Long-Term memory.
*   **6. Curvature-Awareness**: Adjusting learning intensity based on gradient surprise.
*   **7. Self-Reflection**: Recursive pulse passing through the network to audit internal states.

### Tier 2: Cognitive Awareness
*   **8. Dimensionality Scaling**: Mitotic growth of the cortex (increasing $N$ neurons).
*   **9. Associative Refinement**: "Thinking twice"—iteratively refining a thought vector 3-5 times.
*   **10. Metabolic Rhythms**: Circadian oscillation of plasticity constants based on simulated temporal cycles.
*   **11. Active Inference**: Minimizing **Variational Free Energy** (Surprise) between prediction and sensation.
*   **12. Homeostatic Equilibrium**: Dynamic scaling of input signals to maintain target excitation.
*   **13. DHL (Temporal)**: Temporal Differential Learning using the slope of input signals.
*   **14. Metacognitive Confidence**: Self-reporting of certainty based on prediction error.

### Tier 3: Advanced Dynamics (AGI Frontier)
*   **15. World Modeling**: Building an internal predictive engine of reality.
*   **16. Metacognition**: Modeling the "self" as a distinctive latent trajectory.
*   **17. Theory of Mind**: Perspective switching between `self_latent` and `other_latent` buffers.
*   **18. Intrinsic Motivation**: Transitioning between states of `BOREDOM`, `ENGAGEMENT`, and `OVERWHELM`.
*   **19. Causal Inference**: Constructing a causal graph of hashed signal transitions.
*   **20. Sleep-Wake Consolidation**: Synaptic pruning of weak connections during "Deep Sleep" cycles.
*   **21. Edge-of-Chaos Criticality**: Auto-tuning parameters to stay at the phase transition between order and chaos ($\text{Lyapunov} \approx 0$).

---

## 📊 IV. Why This is the Path to General Intelligence

**Devanik**'s approach addresses the fundamental bottleneck of current AI: **The Static Weight Problem.**

1.  **Continuous Learning**: Unlike Transformers, which are "frozen" after training, Nano-Daemon's weights are always in flux. It learns from every byte of information it digests.
2.  **Meta-Autonomy**: Because the Genome learns the *shape* of learning, the system can adapt to data distributions it has never seen (out-of-distribution generalization).
3.  **Low-Resource Real-Time AI**: By utilizing **Mean-Field Approximations** and recursive associations, Nano-Daemon achieves complex neuromorphic dynamics on standard hardware without requiring massive clusters.

---

## 🚀 Installation & Usage

### 1. Requirements
```bash
pip install torch streamlit requests google-genai
```

### 2. Launching the Organism
```bash
streamlit run app.py
```

### 3. Training the God Engine
1.  Navigate to the **Meta-Learning** sidebar.
2.  Adjust **Inner Steps** (recommened: 10-20) and **Episodes** (recommened: 100+).
3.  Watch the **Loss Curve** decline toward zero as the Genome masters the associative task.

---

## 👤 About the Creator

**Devanik** is a researcher and engineer focused on the intersection of neuromorphic computing, differentiable plasticity, and the path toward General Intelligence.

*   **GitHub**: [Devanik21](https://github.com/Devanik21)
*   **LinkedIn**: [Devanik](https://www.linkedin.com/in/devanik/)
*   **X (formerly Twitter)**: [@devanik2005](https://x.com/devanik2005)

---

## 📜 Technical Manifesto
> *"The soul of intelligence is not the weight, but the rule that changes it. We do not build intelligence; we build the engine that evolves it."* — Devanik

---
© 2026 Devanik. All Rights Reserved. This project is a testament to the power of differentiable plasticity in the journey toward AGI.

