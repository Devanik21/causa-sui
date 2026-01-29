import torch
import torch.nn as nn
import os

class PlasticCortex(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=1024):
        super().__init__()
        # The 'embedding' turns raw byte values (0-255) into vectors
        self.byte_embed = nn.Embedding(256, 32)
        
        # The Associative Layer (The Brain)
        self.synapse = nn.Parameter(torch.randn(32, hidden_dim) * 0.01)
        
        # Plasticity Rate: How fast it learns
        self.plasticity = 0.005 

        # Multi-Scale Latent Memory (Short vs Long term)
        self.register_buffer("short_term_latent", torch.zeros(1, 32))
        self.register_buffer("long_term_latent", torch.zeros(1, 32))
        self.st_decay = 0.8
        self.lt_decay = 0.999
        
        # Experience Buffer for Consolidation (Deeper Thinking)
        self.experience_buffer = []
        self.max_buffer = 10

        # --- STEP 11: Active Inference & Surprise ---
        self.last_prediction = None
        self.curiosity_score = 0.0
        
        # --- STEP 12: Homeostatic Equilibrium ---
        self.target_excitation = 0.2
        self.metabolic_balance = 1.0

        # --- STEP 13: Temporal Awareness (DHL) ---
        self.signal_history = []

        # --- STEP 15: World Modeling (Predictive Reality Engine) ---
        self.world_model_history = []  # Stores (signal, next_signal) pairs
        self.world_model_prediction = None
        self.prediction_error = 0.0

        # --- STEP 16: Metacognition (Thinking About Thinking) ---
        self.metacognition_confidence = 1.0  # High = confident, Low = uncertain
        
        # --- STEP 17: Theory of Mind (Self vs Other) ---
        self.register_buffer("self_latent", torch.zeros(1, 32))
        self.register_buffer("other_latent", torch.zeros(1, 32))
        self.processing_other = False  # Flag for "other" perspective

        # --- STEP 18: Intrinsic Motivation (Novelty Drive) ---
        self.motivation_state = "NEUTRAL"  # BORED, NEUTRAL, ENGAGED, OVERWHELMED
        self.boredom_counter = 0
        self.overwhelm_counter = 0
        
        # --- STEP 19: Causal Inference Substrate ---
        self.causal_graph = {}  # {signal_hash: {next_hash: count}}

        # --- STEP 20: Sleep-Wake Consolidation ---
        self.is_sleeping = False
        self.pruning_threshold = 0.01  # Synapses below this are pruned

        # --- STEP 21: Edge-of-Chaos Criticality ---
        self.criticality_score = 0.5  # 0=frozen, 1=chaotic, 0.5=edge
        self.lyapunov_history = []
        
        # --- PERFORMANCE: Fast Mode for Vectorized Processing ---
        self.fast_mode = False  # Toggle for benchmarking; True = faster but simpler learning

    def grow(self, new_neurons=128):
        """Neural Mitosis: Physically grows the brain's synaptic capacity."""
        print(f"🧬 MITOTIC EVENT: Growing cortex by {new_neurons} neurons...")
        with torch.no_grad():
            old_hidden = self.synapse.shape[1]
            new_hidden = old_hidden + new_neurons
            
            # --- STEP 8: Dimensionality Scaling (Input Width) ---
            # If we've passed 5000 neurons, we double the complexity of each thought
            input_dim = self.synapse.shape[0]
            if new_hidden > 5000 and input_dim < 64:
                self.scale_dimensions(64)
                input_dim = 64

            new_synapse = torch.randn(input_dim, new_hidden) * 0.01
            
            # Copy old memories over
            # If dimensions changed, scale_dimensions already updated self.synapse
            # but we need to ensure the mitotic expansion happens on the correct shape
            new_synapse[:, :old_hidden] = self.synapse.data
            
            self.synapse = nn.Parameter(new_synapse)
            print(f"🧠 CORTICAL VOLUME: {new_hidden} Neurons | THOUGHT WIDTH: {input_dim}")

    def scale_dimensions(self, new_dim):
        """Cortical Re-wiring: Expands the fidelity of every single synapse."""
        print(f"🌀 CORTICAL RE-WIRING: Scaling thought width to {new_dim} dimensions...")
        with torch.no_grad():
            old_dim = self.byte_embed.embedding_dim
            
            # 1. Expand Embeddings
            new_embed = nn.Embedding(256, new_dim)
            new_embed.weight.data[:, :old_dim] = self.byte_embed.weight.data
            self.byte_embed = new_embed
            
            # 2. Expand Latent Buffers
            new_st = torch.zeros(1, new_dim).to(self.short_term_latent.device)
            new_lt = torch.zeros(1, new_dim).to(self.long_term_latent.device)
            new_st[:, :old_dim] = self.short_term_latent
            new_lt[:, :old_dim] = self.long_term_latent
            self.register_buffer("short_term_latent", new_st)
            self.register_buffer("long_term_latent", new_lt)
            
            # 3. Expand Synapses (Rows)
            # synapse is [input_dim, hidden_dim]
            hidden_dim = self.synapse.shape[1]
            new_synapse = torch.randn(new_dim, hidden_dim) * 0.01
            new_synapse[:old_dim, :] = self.synapse.data
            self.synapse = nn.Parameter(new_synapse)

    def sync_metabolism(self, hour):
        """STEP 10: Metabolic Rhythms. Adapts plasticity to circadian cycles."""
        # Active: 8am - 10pm (High plasticity)
        # Deep Sleep: 2am - 5am (Low plasticity, efficient pruning)
        if 8 <= hour <= 22:
            self.plasticity = 0.008 # Alert mode
            status = "ALERT/ACTIVE"
        elif 2 <= hour <= 5:
            self.plasticity = 0.001 # Resting/Consolidating
            status = "DEEP HIBERNATION"
        else:
            self.plasticity = 0.004 # Neutral
            status = "DREAMING"
        print(f"🌙 METABOLIC SYNC: State = {status} | Plasticity = {self.plasticity:.4f}")

    def save_cortex(self, path="brain_weights.pth"):
        torch.save(self.state_dict(), path)
        print(f"💾 SYNAPSES PERSISTED: {path} (Dimension: {self.synapse.shape[1]})")

    def load_cortex(self, path="brain_weights.pth"):
        if os.path.exists(path):
            state = torch.load(path)
            # Detect if we need to expand before loading
            saved_hidden = state['synapse'].shape[1]
            curr_hidden = self.synapse.shape[1]
            
            if saved_hidden > curr_hidden:
                print(f"📈 DETECTED EXPANDED CORTEX: Growing to {saved_hidden}")
                self.grow(saved_hidden - curr_hidden)
            
            # Migration Helper: Migrate from old 'latent_memory' to dual-stream
            if 'latent_memory' in state:
                print("🔄 MIGRATING: Old latent memory stream detected...")
                old_mem = state.pop('latent_memory')
                # If we've scaled, we need to pad the old memory
                if old_mem.shape[1] < self.synapse.shape[0]:
                    pad = torch.zeros(1, self.synapse.shape[0] - old_mem.shape[1])
                    old_mem = torch.cat([old_mem, pad], dim=1)
                
                state['short_term_latent'] = old_mem
                state['long_term_latent'] = old_mem

            self.load_state_dict(state, strict=False)
            print(f"🧠 SYNAPSES RESTORED: {path} ({self.synapse.shape[1]} neurons)")
            return True
        return False

    def forward(self, byte_stream, override_weights=None, disable_learning=False):
        """Main forward pass.
        
        Args:
            byte_stream: Input byte tensor
            override_weights: Optional weight tensor for meta-learning (functional weights).
                              If provided, uses these instead of self.synapse for computation.
                              This preserves the gradient graph for the outer loop.
        
        Returns:
            activation: Post-synaptic activation
            entropy: Average entropy across sequence
            signal: Pre-synaptic signal (for plasticity rule computation)
        """
        # Use override weights if provided (for meta-learning), else use internal weights
        W = override_weights if override_weights is not None else self.synapse
        
        # 1. SENSATION: Convert bytes to vectors
        x = self.byte_embed(byte_stream) # [Batch, Seq, Dim]
        
        # --- FAST MODE: Vectorized processing (simpler but faster) ---
        if self.fast_mode:
            return self._fast_forward(x, W)
        
        # 2. TEMPORAL PROCESSING: Process byte-by-byte
        # This is the "Smart" fix: Instead of averaging the whole block, 
        # we let the latent memory evolve through the sequence.
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        
        last_activation = torch.zeros(batch_size, self.synapse.shape[1]).to(x.device)
        mean_signal = torch.zeros(batch_size, W.shape[0]).to(x.device)  # Accumulator for batch plasticity
        total_entropy = 0
        total_excitation = 0
        total_prediction_error = 0
        
        for t in range(seq_len):
            curr_signal = x[:, t, :] # [Batch, Dim]
            # --- STEP 11: Active Inference (Prediction Error) ---
            # We predict the next signal will be similar to our current latent memory
            predicted_signal = self.short_term_latent
            surprise = torch.norm(curr_signal - predicted_signal).item() if predicted_signal is not None else 0.5
            self.curiosity_score = 0.9 * self.curiosity_score + 0.1 * surprise

            # --- STEP 13: Temporal Differential Learning (DHL) ---
            # We look at the 'Signal Slope'
            if len(self.signal_history) > 0:
                gradient = torch.norm(curr_signal - self.signal_history[-1]).item()
                dhl_boost = 1.0 + (gradient * 5.0) # Sharp changes boost intensity
            else:
                dhl_boost = 1.0
            self.signal_history.append(curr_signal.detach())
            if len(self.signal_history) > 5: self.signal_history.pop(0)

            # Mix in memories with Homeostatic Gating
            signal = 0.6 * curr_signal + 0.3 * self.short_term_latent + 0.1 * self.long_term_latent
            signal = signal * self.metabolic_balance # Metabolic scaling
            
            # Accumulate for mean signal (batch plasticity)
            mean_signal += signal
            
            # Update Latent Streams
            with torch.no_grad():
                self.short_term_latent.data = (self.st_decay * self.short_term_latent + (1 - self.st_decay) * signal.mean(dim=0, keepdim=True)).detach()
                self.long_term_latent.data = (self.lt_decay * self.long_term_latent + (1 - self.lt_decay) * signal.mean(dim=0, keepdim=True)).detach()

            # 3. ACTIVATION (Use functional weights W, not self.synapse)
            response = torch.matmul(signal, W)
            activation = torch.tanh(response) # [Batch, Hidden]
            
            # Compute Entropy
            entropy = torch.std(activation).item()
            total_entropy += entropy

            # 4. HEBBIAN LEARNING (Advanced Cognitive Core)
            # CRITICAL: Only run if NOT using external Genome ("Hollow-Out")
            if override_weights is None and not disable_learning:
              with torch.no_grad():
                # --- STEP 6: Curvature-Aware Plasticity + STEP 11 Surpise ---
                # Surprising information triggers 10x higher learning intensity
                dynamic_plasticity = self.plasticity * (1.0 + entropy * 10) * (1.0 + surprise * 2.0) * dhl_boost
                
                # --- STEP 7: Guided Self-Reflection ---
                reflection_mask = (torch.abs(activation) > torch.quantile(torch.abs(activation), 0.9)).float()
                guided_activation = activation * reflection_mask

                s_t = signal.transpose(0, 1) # [Dim, Batch]
                
                # The Hebbian update
                delta_w = torch.matmul(s_t, guided_activation) - (guided_activation ** 2).sum(0) * self.synapse
                
                self.synapse.data += dynamic_plasticity * delta_w
                self.synapse.data = torch.nn.functional.normalize(self.synapse.data, dim=1)
                
                # Collect for block-level update
                current_excitation = torch.mean(torch.abs(activation)).item()
                total_excitation += current_excitation

                # --- STEP 15: World Modeling (Update Prediction Error) ---
                if self.world_model_prediction is not None:
                    self.prediction_error = torch.norm(curr_signal - self.world_model_prediction).item()
                self.world_model_prediction = signal.mean(dim=0, keepdim=True).detach()
                total_prediction_error += self.prediction_error
                
                # Store for causal inference (limit history)
                if len(self.world_model_history) >= 100:
                    self.world_model_history.pop(0)
                self.world_model_history.append(signal.mean(dim=0, keepdim=True).detach())

                # --- STEP 16: Metacognition (Confidence from Prediction Error) ---
                # High prediction error = low confidence
                self.metacognition_confidence = max(0.0, 1.0 - (self.prediction_error * 2.0))

                # --- STEP 19: Causal Inference (Build Causal Graph) ---
                if len(self.signal_history) >= 2:
                    prev_sig = self.signal_history[-2]
                    curr_sig = self.signal_history[-1]
                    # Hash signals for graph keys (simplified)
                    prev_hash = int(prev_sig.sum().item() * 1000) % 10000
                    curr_hash = int(curr_sig.sum().item() * 1000) % 10000
                    if prev_hash not in self.causal_graph:
                        self.causal_graph[prev_hash] = {}
                    self.causal_graph[prev_hash][curr_hash] = self.causal_graph[prev_hash].get(curr_hash, 0) + 1
                    # Limit graph size
                    if len(self.causal_graph) > 500:
                        oldest_key = list(self.causal_graph.keys())[0]
                        del self.causal_graph[oldest_key]

                # --- STEP 21: Edge-of-Chaos Criticality ---
                # Compute Lyapunov-like divergence measure
                if len(self.lyapunov_history) > 0:
                    divergence = abs(entropy - self.lyapunov_history[-1])
                    self.criticality_score = 0.95 * self.criticality_score + 0.05 * min(1.0, divergence * 10)
                self.lyapunov_history.append(entropy)
                if len(self.lyapunov_history) > 20:
                    self.lyapunov_history.pop(0)
                
                # Self-tune plasticity to stay at edge of chaos
                if self.criticality_score < 0.3:  # Too stable/frozen
                    self.plasticity = min(0.02, self.plasticity * 1.05)
                elif self.criticality_score > 0.7:  # Too chaotic
                    self.plasticity = max(0.001, self.plasticity * 0.95)

        # --- BLOCK-LEVEL BEHAVIORAL UPDATES (Prevents rapid saturation) ---
        avg_excitation = total_excitation / seq_len
        avg_error = total_prediction_error / seq_len
        
        with torch.no_grad():
            # 1. Homeostatic Scaling (Metabolic Balance)
            if avg_excitation > self.target_excitation:
                self.metabolic_balance *= 0.95 # Shift once per block
            else:
                self.metabolic_balance *= 1.05
            self.metabolic_balance = max(0.1, min(2.0, self.metabolic_balance))
            
            # 2. Motivation State
            if avg_error < 0.05:
                self.boredom_counter += 1
                self.overwhelm_counter = max(0, self.overwhelm_counter - 1)
            elif avg_error > 0.5:
                self.overwhelm_counter += 1
                self.boredom_counter = max(0, self.boredom_counter - 1)
            else:
                self.boredom_counter = max(0, self.boredom_counter - 1)
                self.overwhelm_counter = max(0, self.overwhelm_counter - 1)
            
            if self.boredom_counter > 10:
                self.motivation_state = "BORED"
            elif self.overwhelm_counter > 10:
                self.motivation_state = "OVERWHELMED"
            elif 0.1 < avg_error < 0.4:
                self.motivation_state = "ENGAGED"
            else:
                self.motivation_state = "NEUTRAL"

        last_activation = activation

        # Store for consolidation (the whole stream)
        with torch.no_grad():
            if len(self.experience_buffer) >= self.max_buffer:
                self.experience_buffer.pop(0)
            self.experience_buffer.append(byte_stream.detach())

        # Normalize mean signal for batch plasticity
        mean_signal = mean_signal / (seq_len if seq_len > 0 else 1)

        return last_activation, total_entropy / seq_len, mean_signal

    def _fast_forward(self, x, W):
        """Vectorized fast path: Processes sequence mean instead of step-by-step.
        Faster but loses some temporal learning dynamics.
        
        Args:
            x: Embedded input tensor
            W: Weight matrix (functional weights for meta-learning)
        
        Returns:
            activation, entropy, signal (for plasticity rule)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Average across sequence dimension for fast processing
        signal = x.mean(dim=1)  # [Batch, Dim]
        
        # Mix in memories
        signal = 0.6 * signal + 0.3 * self.short_term_latent + 0.1 * self.long_term_latent
        signal = signal * self.metabolic_balance
        
        # Update latent streams
        with torch.no_grad():
            self.short_term_latent.data = (self.st_decay * self.short_term_latent + 
                                           (1 - self.st_decay) * signal.mean(dim=0, keepdim=True)).detach()
            self.long_term_latent.data = (self.lt_decay * self.long_term_latent + 
                                          (1 - self.lt_decay) * signal.mean(dim=0, keepdim=True)).detach()
        
        # Activation (Use functional weights W)
        response = torch.matmul(signal, W)
        activation = torch.tanh(response)
        
        # Entropy
        entropy = torch.std(activation).item()
        
        # Hebbian learning (simplified single update)
        with torch.no_grad():
            dynamic_plasticity = self.plasticity * (1.0 + entropy * 10)
            s_t = signal.transpose(0, 1)
            delta_w = torch.matmul(s_t, activation) - (activation ** 2).sum(0) * self.synapse
            self.synapse.data += dynamic_plasticity * delta_w
            self.synapse.data = torch.nn.functional.normalize(self.synapse.data, dim=1)

            # --- BEHAVIORAL UPDATES (Block-level logic) ---
            # Use the single activity pass as the 'average' for homeostasis
            if entropy > self.target_excitation: # Use entropy as proxy for excitation in fast mode
                self.metabolic_balance *= 0.95
            else:
                self.metabolic_balance *= 1.05
            self.metabolic_balance = max(0.1, min(2.0, self.metabolic_balance))

            # Motivation (Simple placeholder logic for fast mode)
            self.motivation_state = "ENGAGED" if entropy > 0.1 else "NEUTRAL"
        
        return activation, entropy, signal

    def consolidate(self):
        """Phase 2: Consolidation Cycle (Deeper Thinking).
        Replays recent high-entropy items with 2x plasticity to solidify memories."""
        if not self.experience_buffer: return
        
        print(f"🧠 CONSOLIDATION CYCLE: Refining {len(self.experience_buffer)} recent memories...")
        # Take a copy to avoid loop pollution
        batch = list(self.experience_buffer)
        
        # Save old plasticity
        p_bak = self.plasticity
        self.plasticity *= 2.0 # Double learning rate for consolidation
        
        for exp in batch:
            # We bypass the buffer logic by calling a manual forward-like logic
            # or just letting it add and then clear. 
            # Actually, let's just use forward and clear the buffer after.
            self.forward(exp) 
            
        self.plasticity = p_bak
        self.experience_buffer = [] # Clear after consolidating
        print("✅ CONSOLIDATION COMPLETE: Synaptic pathways solidified.")

    def reflect(self):
        """Metabolic Reflection: The brain ponders on its own state."""
        with torch.no_grad():
            # Feed current latent memory back in as a perception
            activation, entropy, _ = self.forward(torch.randint(0, 256, (1, 1))) # Tiny pulse
            return activation, entropy

    def associate(self, byte_stream):
        """STEP 9: Recursive Associative Refinement. "Thinking twice" before speaking."""
        with torch.no_grad():
            # Initial thought
            activation, _, _ = self.forward(byte_stream) # [Batch, Hidden]
            
            # Refine the thought 3 times
            for _ in range(3):
                # Reverse synapses to get the "Concept Vector"
                concept_vector = torch.matmul(activation, self.synapse.t())
                # Re-activate the brain with its own previous conclusion
                response = torch.matmul(concept_vector, self.synapse)
                activation = torch.tanh(response)

            # Final reconstruction of the refined memory
            refined_signal = torch.matmul(activation, self.synapse.t())
            
            all_bytes = torch.arange(256)
            all_embeds = self.byte_embed(all_bytes) 
            
            scores = torch.matmul(refined_signal, all_embeds.t())
            top_indices = torch.topk(scores, k=64).indices[0] # Richer vocabulary for the LLM
            return bytes(top_indices.tolist())

    # ============================================================
    # STEP 17: Theory of Mind - Perspective Switching
    # ============================================================
    def switch_perspective(self, to_other=True):
        """STEP 17: Toggle between 'Self' and 'Other' perspective for Theory of Mind."""
        self.processing_other = to_other
        if to_other:
            # Save current self state
            self.self_latent.data = self.short_term_latent.data.clone()
            # Switch to "Other" latent (may be initialized or learned)
            self.short_term_latent.data = self.other_latent.data.clone()
            print("👤 PERSPECTIVE SWITCH: Now processing from 'Other' viewpoint")
        else:
            # Save other state
            self.other_latent.data = self.short_term_latent.data.clone()
            # Restore self
            self.short_term_latent.data = self.self_latent.data.clone()
            print("🧠 PERSPECTIVE SWITCH: Returned to 'Self' viewpoint")

    # ============================================================
    # STEP 20: Sleep-Wake Deep Consolidation with Pruning
    # ============================================================
    def deep_sleep(self):
        """STEP 20: Deep Sleep Consolidation - Aggressive replay and synaptic pruning."""
        print("😴 ENTERING DEEP SLEEP MODE...")
        self.is_sleeping = True
        
        # 1. Triple consolidation pass
        p_bak = self.plasticity
        self.plasticity *= 3.0
        for _ in range(3):
            if self.experience_buffer:
                for exp in list(self.experience_buffer):
                    self.forward(exp)
        self.plasticity = p_bak
        
        # 2. Prune weak synapses
        self.prune_synapses()
        
        # 3. Clear buffers
        self.experience_buffer = []
        self.world_model_history = self.world_model_history[-10:]  # Keep recent
        self.causal_graph = {}  # Reset causal graph (fresh start)
        
        self.is_sleeping = False
        print("🌅 WAKING UP: Deep sleep consolidation complete.")

    def prune_synapses(self):
        """STEP 20: Remove weak synaptic connections (below threshold)."""
        with torch.no_grad():
            mask = torch.abs(self.synapse.data) > self.pruning_threshold
            pruned_count = (~mask).sum().item()
            self.synapse.data = self.synapse.data * mask.float()
            # Re-normalize after pruning
            self.synapse.data = torch.nn.functional.normalize(self.synapse.data, dim=1)
            print(f"✂️ SYNAPTIC PRUNING: Removed {int(pruned_count)} weak connections")

    # ============================================================
    # INTROSPECTION METHODS FOR UI
    # ============================================================
    def get_agi_status(self):
        """Returns a dictionary of all AGI feature states for UI display."""
        return {
            "neurons": self.synapse.shape[1],
            "thought_width": self.synapse.shape[0],
            "plasticity": self.plasticity,
            "metabolic_balance": self.metabolic_balance,
            "curiosity": self.curiosity_score,
            "prediction_error": self.prediction_error,
            "confidence": self.metacognition_confidence,
            "motivation": self.motivation_state,
            "criticality": self.criticality_score,
            "causal_nodes": len(self.causal_graph),
            "is_sleeping": self.is_sleeping,
            "perspective": "Other" if self.processing_other else "Self"
        }
