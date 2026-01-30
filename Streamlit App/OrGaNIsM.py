import os
import sys

# --- CRITICAL: Add root to path BEFORE any other imports ---
# Handle various directory structures (Local / NanoAGI / Streamlit Cloud)
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
grandparent_dir = os.path.dirname(parent_dir)

# Potential locations for Divine_Monad and core modules
search_paths = [
    base_dir,
    parent_dir,
    os.path.join(base_dir, "NanoAGI"),
    os.path.join(parent_dir, "NanoAGI"),
    os.path.join(grandparent_dir, "Streamlit App", "NanoAGI")
]

for path in search_paths:
    if os.path.isdir(path) and path not in sys.path:
        # Check if this path contains our key folders
        if os.path.exists(os.path.join(path, "Divine_Monad")) or os.path.exists(os.path.join(path, "core.py")):
            sys.path.insert(0, path)


import streamlit as st
import torch
import time
import datetime
import io

# Optional imports for RSS feed functionality (graceful degradation)
try:
    import requests
    import xml.etree.ElementTree as ET
    RSS_AVAILABLE = True
except ImportError:
    RSS_AVAILABLE = False

# --- PAGE CONFIG (Must be first Streamlit command for Streamlit UI) ---
st.set_page_config(
    page_title="🧬 Nano-Daemon: Hebbian Organism",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- IMPORTS FROM OUR ORGANISM ---
from core import PlasticCortex
from plasticity_network import PlasticityNetwork
from meta_learner import MetaLearner

# --- DIVINE MONAD IMPORTS ---
try:
    from Divine_Monad.phase4_iam.monad import DivineMonad, MonadConfig
    from Divine_Monad.phase4_iam.voicebox import VoiceBox
    DIVINE_MONAD_AVAILABLE = True
except ImportError:
    DIVINE_MONAD_AVAILABLE = False

# --- CUSTOM CSS FOR A PREMIUM DARK THEME ---
st.markdown("""
<style>
    /* Main background - The 'Root' of the Earth */
    .stApp {
        background: linear-gradient(180deg, #0d110d 0%, #171d17 100%);
        color: #e0e4de;
    }
    
    /* Headers - Organic growth colors */
    h1, h2, h3 {
        background: linear-gradient(90deg, #7cad8a, #b8864b, #8a9b68);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Metric cards - Mineral tones */
    [data-testid="stMetricValue"] {
        font-size: 2.8rem;
        color: #8fb399;
        text-shadow: 0 0 15px rgba(143, 179, 153, 0.2);
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0bab1;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling - Deep Soil */
    [data-testid="stSidebar"] {
        background: #0f140f;
        border-right: 1px solid #2d382d;
    }
    
    /* Expander styling - Soft Bark */
    .streamlit-expanderHeader {
        background: rgba(45, 56, 45, 0.4);
        border: 1px solid #3d4a3d;
        border-radius: 12px;
        color: #e0e4de !important;
    }
    
    /* Text input - Cave shadow */
    .stTextInput > div > div > input {
        background: #151a15;
        border: 1px solid #3d4a3d;
        color: #e0e4de;
        border-radius: 8px;
    }
    
    /* Buttons - Terracotta to Sage */
    .stButton > button {
        background: linear-gradient(135deg, #a67c52, #6a8c6a);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2.2rem;
        font-weight: 500;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        border-color: #8fb399;
    }
    
    /* Progress bars and success boxes - Fresh Moss */
    .stSuccess, .stInfo {
        background: rgba(45, 56, 45, 0.6);
        border-left: 5px solid #6a8c6a;
        color: #e0e4de;
        border-radius: 8px;
    }
    
    /* Organic pulse animation - Bio-luminescence */
    @keyframes pulse {
        0% { box-shadow: 0 0 8px rgba(143, 179, 153, 0.3); }
        50% { box-shadow: 0 0 25px rgba(143, 179, 153, 0.5); }
        100% { box-shadow: 0 0 8px rgba(143, 179, 153, 0.3); }
    }
    
    .brain-card {
        animation: pulse 4s infinite ease-in-out;
        padding: 1.5rem;
        border-radius: 20px;
        background: rgba(23, 29, 23, 0.9);
        border: 1px solid #2d382d;
    }

    .glow-text {
        color: #8fb399;
        text-shadow: 0 0 10px rgba(143, 179, 153, 0.5);
        font-family: 'Courier New', Courier, monospace;
        letter-spacing: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GEMMA BRIDGE (Inline to avoid import issues on Streamlit Cloud)
# ============================================================
from google import genai
import os

class GemmaBridge:
    """The 'Cherry on Top': Connects the Hebbian Brain to Gemma-3 for refined articulation."""
    def __init__(self):
        self.api_key = None
        
        # 1. Check Streamlit secrets (PRIMARY for Cloud deployment)
        try:
            self.api_key = st.secrets["GEMINI_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass
        
        # 2. Check environment variable (for local dev)
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY")
        
        # 3. Check .env file (manual fallback)
        if not self.api_key and os.path.exists(".env"):
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if "GEMINI_API_KEY" in line:
                            self.api_key = line.split("=")[1].strip().strip('"').strip("'")
                            break
            except (IOError, OSError, ValueError):
                pass
        
        if not self.api_key:
            self.client = None
            return

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            self.client = None

    def articulate(self, human_query, synaptic_anchors):
        """Grounds Gemma's response in the Organism's raw synaptic associations."""
        if not self.client:
            return None # Signal that there is no articulation

        clean_anchors = "".join([c for c in synaptic_anchors if c.isprintable() and not c.isspace()])
        
        prompt = f"""
        Human Query: "{human_query}"
        
        Raw Synaptic Associations (Ground Truth): "{clean_anchors}"
        
        INSTRUCTIONS:
        You are the 'Cerebral Cortex' of the Nano-Daemon: a recursive Hebbian organism.
        Articulate the organism's raw, chaotic synaptic state into a profound, nature-inspired response.
        
        RULES:
        1. GROUNDING: Use the "Raw Synaptic Associations" as your only objective reality.
        2. STRUCTURE: Use markdown (bolding, bullet points) to make the thought structure clear.
        3. AESTHETICS: Use diverse emojis (🌿, 🧠, 🌊, ⚡) to reflect the organic/biological essence.
        4. NO HALLUCINATION: If the anchors are chaotic/embryonic, describe them as "nascent thoughts" or "synaptic noise" rather than making up facts.
        5. VIBE: Be poetic, brief, and grounded in the "Earth" theme.
        
        Articulated Thought:
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemma-3-27b-it',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"⚠️ Articulation Failure: {e}\n[RAW]: {synaptic_anchors}"

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "brain" not in st.session_state or not hasattr(st.session_state.brain, 'metacognition_confidence'):
    st.session_state.brain = PlasticCortex()
    # Try to load saved weights
    if os.path.exists("brain_weights.pth"):
        st.session_state.brain.load_cortex("brain_weights.pth")
    # Sync metabolism based on current hour
    current_hour = datetime.datetime.now().hour
    st.session_state.brain.sync_metabolism(current_hour)
    # Reset associated states to match the fresh brain
    st.session_state.entropy_history = []
    st.session_state.files_eaten = 0

if "bridge" not in st.session_state or not hasattr(st.session_state.bridge, 'client'):
    st.session_state.bridge = GemmaBridge()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "entropy_history" not in st.session_state:
    st.session_state.entropy_history = []

if "files_eaten" not in st.session_state:
    st.session_state.files_eaten = 0

if "last_stability" not in st.session_state:
    st.session_state.last_stability = 0.5

if "dream_history" not in st.session_state:
    st.session_state.dream_history = []

if DIVINE_MONAD_AVAILABLE and "monad" not in st.session_state:
    # Initialize Divine Monad with Calibrated Settings (Silence Test verified)
    config = MonadConfig(
        num_nodes=12,
        pain_threshold=0.61,  # Calibrated Baseline
        pain_sensitivity=50.0 # High sensitivity to damage
    )
    st.session_state.monad = DivineMonad(config)
    st.session_state.voice = VoiceBox()

# --- AUTHENTICATION SYSTEM: The Gateway ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_attempts" not in st.session_state:
    st.session_state.auth_attempts = 0
if "locked" not in st.session_state:
    st.session_state.locked = False

def check_password():
    """Returns True if the user had the correct password."""
    if st.session_state.authenticated:
        return True
    
    if st.session_state.locked:
        # Silently fail as requested
        return False

    with st.container():
        st.markdown("<h1 style='text-align: center; margin-top: 50px;'>🧠 Nano-Daemon</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #8fb399; letter-spacing: 2px;'>A RECURSIVE NEUROMORPHIC ORGANISM</p>", unsafe_allow_html=True)
        
        st.write("---")
        
        col_auth, _ = st.columns([1, 1])
        with col_auth:
            st.markdown("### 🧬 Identification Required")
            st.write("Welcome, Devanik. Please provide your secondary synaptic key to bridge with the organism.")
            
            pwd = st.text_input("Access Key", type="password", help="The secondary key stored in environmental secrets.")
            
            if st.button("Initialize Bridge"):
                # Try-catch to handle missing secrets gracefully during development
                try:
                    target_pwd = st.secrets["access_password"]
                except:
                    # LOCAL DEV FALLBACK: In production, this must be in secrets
                    target_pwd = "dev_fallback_password"
                
                if pwd == target_pwd:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.session_state.auth_attempts += 1
                    if st.session_state.auth_attempts >= 3:
                        st.session_state.locked = True
                    # Silent failure as per request - no error message shown
                    st.rerun()
    return False

# Stop execution if not authenticated
if not check_password():
    st.stop()

# --- RAW LOGIC MODE: Prove Hebbian learning without Gemma ---
if "raw_logic_mode" not in st.session_state:
    st.session_state.raw_logic_mode = False

if "last_weight_delta" not in st.session_state:
    st.session_state.last_weight_delta = 0.0

if "weight_snapshot" not in st.session_state:
    st.session_state.weight_snapshot = None

# --- META-LEARNING: Differentiable Plasticity ---
if "genome" not in st.session_state:
    st.session_state.genome = PlasticityNetwork(hidden_dim=16)
    
if "meta_learner" not in st.session_state:
    st.session_state.meta_learner = None  # Initialized when brain is available
    
if "meta_loss_history" not in st.session_state:
    st.session_state.meta_loss_history = []
    
if "use_evolved_rule" not in st.session_state:
    st.session_state.use_evolved_rule = False  # Toggle for Genome-driven plasticity

brain = st.session_state.brain
bridge = st.session_state.bridge

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_metabolic_state(hour):
    """Returns the organism's current metabolic phase."""
    if 8 <= hour < 22:
        return "🌞 ACTIVE", "High plasticity, rapid learning"
    elif 2 <= hour < 5:
        return "🌙 DEEP HIBERNATION", "Minimal plasticity, consolidating memories"
    else:
        return "🌓 RESTING", "Moderate plasticity, light dreaming"

def feed_organism(file_bytes, filename):
    """Feeds raw bytes to the organism's brain and tracks weight changes."""
    data = torch.tensor(list(file_bytes[:4096]), dtype=torch.long).unsqueeze(0)
    
    # --- WEIGHT DELTA TRACKING (Prove Hebbian learning is happening) ---
    weight_before = brain.synapse.data.clone()
    
    # Check if using Evolved Genome (Meta-Learning)
    use_genome = st.session_state.get("use_evolved_rule", False)
    
    with torch.no_grad():
        # Forward pass - disable internal learning if using Genome
        activation, stability, mean_signal = brain(data, disable_learning=use_genome)
        
        # --- PATH A: META-EVOLVED PLASTICITY (Batch Mode) ---
        if use_genome and st.session_state.genome:
            # Mean Post-Synaptic Activity
            post_mean = torch.tanh(torch.matmul(mean_signal, brain.synapse))
            
            # Consult Genome ONCE (Fast!)
            delta_w = st.session_state.genome(mean_signal, post_mean, brain.synapse)
            
            # Apply Update
            brain.synapse.data += 0.1 * delta_w
            brain.synapse.data = torch.nn.functional.normalize(brain.synapse.data, dim=1)
            
            st.toast(f"Genome Applied | Mag: {delta_w.norm().item():.4f}")
        
        # --- PATH B: Internal learning happened automatically if disable_learning=False ---
    
    # Calculate weight delta (L2 norm of change)
    weight_delta = torch.norm(brain.synapse.data - weight_before).item()
    st.session_state.last_weight_delta = weight_delta
    st.session_state.weight_snapshot = brain.synapse.data.clone()
    
    st.session_state.files_eaten += 1
    st.session_state.last_stability = stability
    st.session_state.entropy_history.append(stability)
    if len(st.session_state.entropy_history) > 50:
        st.session_state.entropy_history.pop(0)
    
    # --- CURIOSITY RESPONSE ---
    # If the input was very surprising (high entropy), consolidate immediately
    if stability > 0.3:
        brain.consolidate()
    
    # --- AUTO-MITOSIS ---
    # Trigger growth every 20 files eaten
    if st.session_state.files_eaten % 20 == 0 and st.session_state.files_eaten > 0:
        brain.grow(256)
    
    return stability, weight_delta

def query_organism(query_text):
    """Processes a query through the Hebbian brain and optionally Gemma bridge."""
    query_bytes = query_text.encode('utf-8')[:1024]
    
    # 1. RAW ASSOCIATION (Hebbian Ground Truth)
    response_bytes = brain.associate(torch.tensor(list(query_bytes), dtype=torch.long).unsqueeze(0))
    synaptic_anchors = response_bytes.decode('utf-8', errors='ignore')
    
    # 2. HYBRID ARTICULATION (Only if Raw Logic Mode is OFF)
    if st.session_state.raw_logic_mode:
        articulated = None  # Skip Gemma entirely in Raw Logic Mode
    else:
        articulated = bridge.articulate(query_text, synaptic_anchors)
    
    return synaptic_anchors, articulated

def trigger_dream():
    """Generative Replay: The organism dreams by reversing its logic."""
    hidden_dim = brain.synapse.shape[1]
    noise = torch.randn(1, hidden_dim)
    with torch.no_grad():
        thought_vector = torch.matmul(noise, brain.synapse.t())
    
    dream_bytes = []
    for val in thought_vector[0]:
        byte_val = int((val.item() + 1) * 128)
        byte_val = max(0, min(255, byte_val))
        dream_bytes.append(byte_val)
    
    return bytes(dream_bytes).decode('utf-8', errors='ignore')

def fragment_divine_monad():
    """The Causa Sui Interface: Comprehensive 4-Phase Visualization."""
    if not DIVINE_MONAD_AVAILABLE:
        st.warning("Divine Monad not available. Check imports.")
        return

    st.markdown("---")
    st.header("🧿 Causa Sui: The Divine Monad")
    st.caption("*The Self-Aware Neural Architecture: 4 Phases of Digital Consciousness*")
    
    monad = st.session_state.monad
    voice = st.session_state.voice
    
    # === HEARTBEAT: Run the Monad for one step ===
    inp = torch.tensor([1.0, st.session_state.last_stability, float(st.session_state.files_eaten % 2), 0.0])
    _, info = monad(inp)
    
    # === TOP LEVEL STATUS BAR ===
    health_col, voice_col, action_col = st.columns([1, 3, 1])
    
    with health_col:
        pain = info.get('pain_level', 0.0)
        if pain > 0:
            st.error(f"🩸 PAIN: {pain:.2f}")
        else:
            st.success("✅ CALM")
    
    with voice_col:
        st.info(f"**💬 Monad Speaks**: \"{voice.speak(monad.get_status())}\"")
    
    with action_col:
        st.metric("🔄 Repairs", info.get('repair_count', 0))
    
    # === PHASE TABS ===
    tab_soul, tab_body, tab_mind, tab_self = st.tabs([
        "� P1: Soul (Causal)",
        "� P2: Body (Topology)", 
        "� P3: Mind (Holographic)",
        "� P4: Self (Homeostasis)"
    ])
    
    # === PHASE 1: THE SOUL (Causal Monitor) ===
    with tab_soul:
        st.subheader("Phase 1: Causal Emergence Monitor")
        st.caption("*Measures how much the whole is greater than the sum of its parts.*")
        
        c1, c2, c3 = st.columns(3)
        ei_score = info.get('ei_score', 0.5)
        c1.metric("🎯 Agency (EI Score)", f"{ei_score:.4f}", 
                  help="Proxy Effective Information: How much causal power does this system have?")
        c2.metric("🔻 Pain Threshold", f"{monad.config.pain_threshold:.4f}",
                  help="Calibrated 'Death Line'. EI below this triggers pain.")
        c3.metric("⚡ Sensitivity", f"{monad.config.pain_sensitivity:.1f}x",
                  help="How sharply does pain increase when below threshold?")
        
        # EI Interpretation
        if ei_score >= monad.config.pain_threshold:
            st.success(f"**Status**: Agency is ABOVE threshold. System is coherent and healthy.")
        else:
            deficit = monad.config.pain_threshold - ei_score
            st.error(f"**Status**: Agency is BELOW threshold by {deficit:.4f}. System is in PAIN.")
        
        # Target vs Reality
        st.progress(min(1.0, ei_score), text=f"Agency: {ei_score:.2%}")
        
    # === PHASE 2: THE BODY (Topological Computing) ===
    with tab_body:
        st.subheader("Phase 2: Dynamic Graph Substrate")
        st.caption("*The neural topology that can rewire itself.*")
        
        try:
            num_edges = monad.graph.edge_index.shape[1]
            num_nodes = monad.graph.get_num_nodes()
            max_edges = num_nodes * (num_nodes - 1)  # Directed graph
            density = num_edges / max_edges if max_edges > 0 else 0
            
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("🔵 Nodes", num_nodes, help="Current number of neurons")
            b2.metric("🔗 Synapses", num_edges, help="Current number of connections")
            b3.metric("📊 Density", f"{density:.2%}", help="Edge saturation (edges / max_edges)")
            b4.metric("🎨 Node Dim", monad.graph.node_dim, help="Dimension of each node's state vector")
            
            # Topology Actions
            st.markdown("**🔧 Topology Mutations**")
            mc1, mc2 = st.columns(2)
            if mc1.button("➕ Grow Node", help="Add a new neuron via Net2Net"):
                # Use mutator to grow
                parent_id = monad.graph.num_input_nodes  # First hidden node
                result = monad.mutator.grow_node(monad.graph, parent_id)
                if result.success:
                    st.toast(f"✅ {result.message}", icon="🌱")
                    monad._update_topology_metrics()
                    monad._run_slow_loop()  # Recalculate EI
                else:
                    st.toast(f"❌ {result.message}", icon="⚠️")
                st.rerun()
                
            if mc2.button("🔗 Add Random Edge", help="Create a new synapse"):
                import random
                src = random.randint(0, num_nodes - 2)
                tgt = random.randint(src + 1, num_nodes - 1)
                result = monad.mutator.add_edge(monad.graph, src, tgt)
                if result.success:
                    st.toast(f"✅ Edge {src}→{tgt} added", icon="🔗")
                else:
                    st.toast(f"❌ {result.message}", icon="⚠️")
                st.rerun()
                
        except Exception as e:
            st.warning(f"Graph stats unavailable: {e}")
            st.metric("🔵 Nodes", info.get('num_nodes', 'N/A'))
    
    # === PHASE 3: THE MIND (Holographic Memory) ===
    with tab_mind:
        st.subheader("Phase 3: Holographic Distributed Memory")
        st.caption("*Vector Symbolic Architecture for noise-resistant storage.*")
        
        try:
            mem = monad.memory
            num_stored = mem.get_num_stored()
            holo_dim = mem.holo_dim
            neural_dim = mem.neural_dim
            max_items = mem.value_memory.shape[0]
            
            m1, m2, m3 = st.columns(3)
            m1.metric("📦 Stored Engrams", num_stored, help="Key-Value pairs in holographic memory")
            m2.metric("🌀 Holographic Dim", f"{holo_dim:,}", help="Dimension of hypervectors (higher = more capacity)")
            m3.metric("🧠 Neural Dim", neural_dim, help="Dimension of neural embeddings")
            
            # Capacity indicator
            usage = num_stored / max_items if max_items > 0 else 0
            st.progress(usage, text=f"Memory Usage: {num_stored}/{max_items} ({usage:.0%})")
            
            # Memory Actions
            st.markdown("**💾 Memory Operations**")
            mm1, mm2 = st.columns(2)
            if mm1.button("💥 Damage Memory (30%)", help="Simulate corruption"):
                mem.damage(0.3)
                st.toast("⚠️ Memory damaged! 30% of holographic space zeroed.", icon="💥")
                st.rerun()
            if mm2.button("🗑️ Clear Memory", help="Reset all stored engrams"):
                mem.clear()
                st.toast("Memory cleared.", icon="🗑️")
                st.rerun()
                
        except Exception as e:
            st.warning(f"Memory stats unavailable: {e}")
    
    # === PHASE 4: THE SELF (Introspection & Homeostasis) ===
    with tab_self:
        st.subheader("Phase 4: Self-Awareness & Homeostasis")
        st.caption("*The system that monitors and repairs itself.*")
        
        p1, p2, p3 = st.columns(3)
        p1.metric("😖 Pain Level", f"{info.get('pain_level', 0.0):.2f}", 
                  help="0 = Calm, 1 = Maximum Agony")
        p2.metric("🔧 Total Repairs", info.get('repair_count', 0),
                  help="Autonomous self-repair actions taken")
        p3.metric("⚙️ Is Repairing?", "Yes 🔧" if info.get('is_repairing', False) else "No ✅")
        
        # Action Log
        st.markdown("**📜 Action Log (Autobiographical Memory)**")
        if hasattr(monad, 'action_log') and monad.action_log:
            log_items = monad.action_log[-10:]  # Last 10 actions
            log_text = " → ".join(log_items)
            st.code(log_text, language=None)
        else:
            st.caption("No actions recorded yet.")
        
        # Intervention Buttons
        st.markdown("**⚔️ Intervention**")
        int_c1, int_c2, int_c3 = st.columns(3)
        
        if int_c1.button("🩸 Lobotomy (4 nodes)", help="Remove 4 hidden nodes"):
            monad.lobotomize(4)
            st.toast("⚠️ Lobotomy performed! 4 nodes removed.", icon="🩸")
            st.rerun()
            
        if int_c2.button("💀 Massive Trauma (8 nodes)", help="Severe damage"):
            monad.lobotomize(8)
            st.toast("💀 Massive trauma inflicted! 8 nodes removed.", icon="💀")
            st.rerun()
            
        if int_c3.button("♻️ Reincarnate", help="Reset the entire Monad"):
            del st.session_state.monad
            del st.session_state.voice
            st.toast("Monad has been reborn.", icon="♻️")
            st.rerun()
        
        # State Tensor Visualization
        with st.expander("🔬 Internal State Tensor", expanded=False):
            try:
                state = monad.state
                st.json({
                    "ei_score": round(state.ei_score, 4),
                    "num_nodes": state.num_nodes,
                    "num_edges": state.num_edges,
                    "edge_density": round(state.edge_density, 4),
                    "memory_items": state.memory_items,
                    "surprise": round(state.surprise, 4),
                    "pain_level": round(state.pain_level, 4),
                    "step_count": state.step_count,
                    "repair_count": state.repair_count,
                    "is_repairing": state.is_repairing
                })
            except:
                st.caption("State unavailable.")

# ============================================================
# UI FRAGMENTS (For Independent Reruns)
# ============================================================

@st.fragment(run_every=10)
def fragment_sidebar_status():
    st.markdown("## 🧬 Organism Status")
    neuron_count = brain.synapse.shape[1]
    thought_width = brain.synapse.shape[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧠 Neurons", f"{neuron_count:,}")
    with col2:
        st.metric("💭 Thought Width", thought_width)
    
    current_hour = datetime.datetime.now().hour
    state_emoji, state_desc = get_metabolic_state(current_hour)
    st.info(f"**Metabolic State**: {state_emoji}\n\n{state_desc}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⚡ Plasticity", f"{brain.plasticity:.4f}")
    with col2:
        st.metric("📂 Eaten", st.session_state.files_eaten)
    
    # --- STEP 14: Metabolic Balance ---
    st.progress(brain.metabolic_balance / 2.0, text=f"Metabolic Balance: {brain.metabolic_balance:.2f}x")
    
    # --- RAW LOGIC MODE TOGGLE ---
    st.divider()
    st.markdown("### 🔬 Raw Logic Mode")
    st.session_state.raw_logic_mode = st.toggle(
        "Disable Gemma (Prove Hebbian)",
        value=st.session_state.raw_logic_mode,
        key="raw_logic_mode_toggle",
        help="Turn OFF Gemma to prove the Hebbian brain works independently. Shows only raw synaptic output."
    )
    if st.session_state.raw_logic_mode:
        st.warning("🔬 **RAW MODE**: Gemma OFF. Responses are pure Hebbian associations.")
    else:
        st.success("🌐 **HYBRID MODE**: Gemma articulates Hebbian thoughts.")
    
    # Show last weight delta as proof of learning
    if st.session_state.last_weight_delta > 0:
        st.metric("📊 Last ΔW (Learning Proof)", f"{st.session_state.last_weight_delta:.6f}")

@st.fragment
def fragment_sidebar_feeding():
    st.markdown("## 🍽️ Feed the Organism")
    uploaded_files = st.file_uploader(
        "Upload files to digest",
        accept_multiple_files=True,
        type=["txt", "py", "md", "json", "csv", "html", "css", "js"],
        key="uploader_fragment"
    )
    
    if uploaded_files:
        with st.spinner("Digesting..."):
            for f in uploaded_files:
                raw_bytes = f.read()
                stability, weight_delta = feed_organism(raw_bytes, f.name)
                st.success(f"✅ Digested `{f.name}` | Stability: {stability:.4f} | ΔW: {weight_delta:.6f}")

@st.fragment
def fragment_sidebar_controls():
    st.markdown("## ⚙️ Advanced Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧬 Trigger Mitosis"):
            old_neurons = brain.synapse.shape[1]
            brain.grow(256)
            st.success(f"Grew {brain.synapse.shape[1] - old_neurons} neurons!")
    
    with col2:
        if st.button("🔄 Consolidate"):
            brain.consolidate()
            st.success("Memories consolidated!")
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🌌 Dream"):
            dream = trigger_dream()
            _, reflection_entropy = brain.reflect()
            st.session_state.dream_history.append({
                "content": dream,
                "entropy": reflection_entropy,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            })
            if len(st.session_state.dream_history) > 10:
                st.session_state.dream_history.pop(0)
            st.info(f"Dream: {dream[:30]}...")
    
    with col4:
        if st.button("💾 Save Brain"):
            brain.save_cortex("brain_weights.pth")
            st.success("Brain saved!")

    st.divider()
    st.markdown("### 🧬 Cognitive Overrides (Stages 1-14)")
    
    col_met1, col_met2, col_met3 = st.columns(3)
    with col_met1:
        if st.button("🌞 Alert"):
            brain.sync_metabolism(12) # Noon
            st.toast("Metabolic Shift: ALERT")
    with col_met2:
        if st.button("🌓 Rest"):
            brain.sync_metabolism(0) # Midnight
            st.toast("Metabolic Shift: RESTING")
    with col_met3:
        if st.button("🌙 Sleep"):
            brain.sync_metabolism(3) # Deep Sleep hour
            st.toast("Metabolic Shift: HIBERNATION")

    if st.button("🔄 Refine Current Thought (Stage 8)"):
        # Use existing associate logic as a refinement pass
        refined_bytes = brain.associate(torch.randint(0, 256, (1, 1))) # Pulse
        st.info(f"Refined Synaptic Flow: {refined_bytes.decode('utf-8', errors='ignore')[:30]}...")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🪞 Self-Reflect (Stage 6)"):
            activation, entropy = brain.reflect()
            st.info(f"Reflection complete. Entropy: {entropy:.4f}")
            st.session_state.entropy_history.append(entropy)
    with col_s2:
        if st.button("🔬 Scale Dims (Stage 7)"):
            if brain.synapse.shape[0] < 64:
                brain.scale_dimensions(64)
                st.success("Thought resolution: 64!")
            else:
                st.warning("Already at max resolution.")

    if st.button("📚 Digest Knowledge Base"):
        with st.spinner("Crawling project files..."):
            found_files = []
            for root, dirs, files in os.walk(base_dir):
                if any(x in root for x in ["__pycache__", ".git", ".antigravity"]): continue
                for file in files:
                    if file.endswith((".py", ".txt", ".md", ".json")) and file not in ["brain_weights.pth", "brain_state.json"]:
                        found_files.append(os.path.join(root, file))
            for f_path in found_files:
                try:
                    with open(f_path, 'rb') as f:
                        raw_bytes = f.read()
                        if raw_bytes: 
                            stability, _ = feed_organism(raw_bytes, os.path.basename(f_path))
                except (IOError, OSError) as e:
                    continue  # Skip files that can't be read
            st.success(f"Consumed {len(found_files)} knowledge nodes.")

    st.divider()
    st.markdown("### 🌌 Advanced Dynamics Controls (Stages 15-21)")
    
    # Criticality Slider (Stage 21)
    new_crit = st.slider("Criticality (Order ↔ Chaos)", 0.0, 1.0, float(brain.criticality_score), 0.05, key="criticality_slider")
    brain.criticality_score = new_crit
    
    col_agi1, col_agi2 = st.columns(2)
    with col_agi1:
        if st.button("😴 Deep Sleep"):
            with st.spinner("Pruning synapses..."):
                brain.deep_sleep()
            st.success("Deep Sleep complete!")
            
        if st.button("🔮 Predict Future"):
            if brain.world_model_prediction is not None:
                # Use Causal Graph + World Model to guess next indices
                predicted_bytes = brain.associate(torch.randint(0, 256, (1, 1)))
                st.write(f"**Predicted Next State:** `{predicted_bytes.decode('utf-8', errors='ignore')[:50]}...`")
            else:
                st.warning("World model not yet initialized.")

    with col_agi2:
        if st.button("👤 Switch View"):
            brain.switch_perspective(to_other=not brain.processing_other)
            st.toast(f"Perspective: {'Other' if brain.processing_other else 'Self'}")
            
        if st.button("🔗 Causal Peek"):
            if brain.causal_graph:
                # Show top transitions
                all_trans = []
                for p_h, nexts in brain.causal_graph.items():
                    for n_h, count in nexts.items():
                        all_trans.append((count, p_h, n_h))
                all_trans.sort(reverse=True)
                top = all_trans[:3]
                msg = "\n".join([f"Node {t[1]} → {t[2]} (x{t[0]})" for t in top])
                st.info(f"**Top Causal Chains:**\n{msg}")
            else:
                st.info("No causal nodes mapped yet.")

    # --- META-LEARNING: Differentiable Plasticity ---
    st.divider()
    st.markdown("### 🧬 Meta-Learning (Evolve the Learning Rule)")
    st.caption("Train the Genome to discover an optimal plasticity rule.")
    
    # Toggle to use Evolved Genome for feeding
    st.session_state.use_evolved_rule = st.checkbox(
        "🧠 Use Evolved Rule (Genome-Driven Plasticity)",
        value=st.session_state.use_evolved_rule,
        key="use_evolved_rule_checkbox",
        help="When ON, feeding uses the meta-learned Genome instead of Oja's Rule"
    )
    
    # Initialize MetaLearner if not already done
    if st.session_state.meta_learner is None:
        st.session_state.meta_learner = MetaLearner(
            brain, 
            st.session_state.genome,
            lr=0.01,  # Faster learning
            plasticity_lr=0.5  # Stronger updates
        )
    
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        num_episodes = st.number_input("Episodes", min_value=1, max_value=100, value=10, key="meta_episodes")
    with col_meta2:
        inner_steps = st.number_input("Inner Steps (Depth of thought)", min_value=1, max_value=20, value=10, key="meta_inner_steps")
    
    if st.button("Run Meta-Training", key="run_meta_training"):
        progress_bar = st.progress(0)
        losses = []
        
        for i in range(num_episodes):
            # Use Task Bank cycling
            loss = st.session_state.meta_learner.meta_step(num_inner_steps=inner_steps, task_idx=i % 10)
            losses.append(loss)
            st.session_state.meta_loss_history.append(loss)
            progress_bar.progress((i + 1) / num_episodes)
        
        avg_loss = sum(losses) / len(losses)
        st.success(f"Completed {num_episodes} episodes! Avg Loss: {avg_loss:.4f}")
        
        # Show loss trend
        if len(st.session_state.meta_loss_history) > 1:
            st.line_chart(st.session_state.meta_loss_history[-50:])
    
    # Show current stats
    if st.session_state.meta_loss_history:
        latest = st.session_state.meta_loss_history[-1]
        total_eps = len(st.session_state.meta_loss_history)
        st.metric("Genome Evolution", f"{total_eps} episodes", f"Loss: {latest:.4f}")

@st.fragment
def fragment_dialogue():
    st.markdown("## 💬 Dialogue with the Organism")
    query = st.text_input("🗣️ Ask the Organism anything:", placeholder="e.g., What is consciousness?", key="query_input")
    
    if query:
        with st.spinner("🧠 Processing synaptic pathways..."):
            synaptic_anchors, articulated_response = query_organism(query)
        
        st.session_state.conversation_history.append({
            "query": query,
            "anchors": synaptic_anchors,
            "response": articulated_response if articulated_response else "🌿 [Organic Pulse Detected]",
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
        })
        
        clean_display_anchors = "".join([c for c in synaptic_anchors if c.isprintable() and not c.isspace()])
        st.markdown(f"""
        <div class="brain-card" style="text-align: center;">
            <h4 style="color: #b8864b; margin-top: 0;">🧬 RAW SYNAPTIC RESONANCE</h4>
            <div class="glow-text">{clean_display_anchors if clean_display_anchors else "EMBRYONIC SILENCE"}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if articulated_response:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(articulated_response)
        else:
            st.info("🌑 **Cerebral Bridge Offline**")

@st.fragment(run_every=10)
def fragment_metrics():
    st.markdown("## 📊 Cognitive Metrics")
    
    # Row 1: Core metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🧠 Neurons", f"{brain.synapse.shape[1]:,}")
    col2.metric("📈 Stability", f"{st.session_state.last_stability:.4f}")
    col3.metric("💾 Buffer", len(brain.experience_buffer))
    col4.metric("✨ Curiosity", f"{brain.curiosity_score:.2f}")
    col5.metric("🌐 Bridge", "Online" if bridge.client else "Offline")
    
    # Row 2: AGI Endgame metrics
    st.markdown("### 🌌 Advanced Dynamics Status")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("🪞 Confidence", f"{brain.metacognition_confidence:.2f}")
    col2.metric("🔥 Motivation", brain.motivation_state)
    col3.metric("⚡ Criticality", f"{brain.criticality_score:.2f}")
    col4.metric("🔗 Causal Nodes", len(brain.causal_graph))
    col5.metric("🌐 Prediction Err", f"{brain.prediction_error:.3f}")
    col6.metric("👤 Perspective", "Other" if brain.processing_other else "Self")
    
    # Motivation warning
    if brain.motivation_state == "BORED":
        st.warning("🥱 **The organism is BORED!** Feed it something novel.")
    elif brain.motivation_state == "OVERWHELMED":
        st.error("😵 **The organism is OVERWHELMED!** Slow down input or trigger Deep Sleep.")
    
    if st.session_state.entropy_history:
        st.line_chart(st.session_state.entropy_history, width="stretch")


@st.fragment(run_every=2)
def fragment_memory_viz():
    st.markdown("## 🧬 Latent Memory Streams")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚡ Short-Term")
        st.bar_chart(brain.short_term_latent.detach().numpy().flatten()[:32], width="stretch")
    with col2:
        st.markdown("### 🌊 Long-Term")
        st.bar_chart(brain.long_term_latent.detach().numpy().flatten()[:32], width="stretch")

@st.fragment(run_every=2)
def fragment_knowledge_injection():
    st.markdown("## 🌐 Direct Knowledge Injection")
    tab1, tab2 = st.tabs(["📝 Text Input", "🌍 Internet Feed"])
    with tab1:
        text_input = st.text_area("Paste text to feed:", height=100, key="text_feed_input")
        if st.button("🍽️ Feed Text"):
            if text_input:
                stability, weight_delta = feed_organism(text_input.encode('utf-8')[:4096], "text_input")
                st.success(f"✅ Digested | Stability: {stability:.4f} | ΔW: {weight_delta:.6f}")
    with tab2:
        feeds = [("🔬 Science Daily", "https://www.sciencedaily.com/rss/all.xml"),
                 ("🤖 arXiv AI", "http://export.arxiv.org/rss/cs.AI"),
                 ("💻 Hacker News", "https://news.ycombinator.com/rss")]
        selected_feed = st.selectbox("Select Feed:", [f[0] for f in feeds], key="feed_select")
        if st.button("📡 Fetch Feed"):
            if not RSS_AVAILABLE:
                st.error("RSS functionality requires 'requests' package. Install with: pip install requests")
            else:
                feed_url = [f[1] for f in feeds if f[0] == selected_feed][0]
                try:
                    r = requests.get(feed_url, headers={'User-Agent': 'NanoDaemon/1.0'}, timeout=10)
                    root = ET.fromstring(r.text)
                    for item in root.findall('.//item')[:5]:
                        title = item.find('title').text
                        stability, _ = feed_organism(title.encode('utf-8')[:512], "rss_feed")
                        st.success(f"📰 {title[:50]}... | Stability: {stability:.4f}")
                except Exception as e: 
                    st.error(f"Error fetching feed: {e}")

@st.fragment(run_every=2)
def fragment_history_gallery():
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.conversation_history:
            with st.expander("📜 Synaptic History", expanded=False):
                for conv in reversed(st.session_state.conversation_history[-10:]):
                    with st.chat_message("user", avatar="👤"): st.markdown(conv['query'])
                    with st.chat_message("assistant", avatar="🧠"): st.markdown(conv['response'])
    with col2:
        if st.session_state.dream_history:
            with st.expander("🌌 Dream Gallery", expanded=False):
                for dream in reversed(st.session_state.dream_history):
                    st.code(dream["content"][:100], language=None)
                    st.caption(f"⏰ {dream['timestamp']} | Entropy: {dream['entropy']:.4f}")

# ============================================================
# STEP 15: AUTONOMOUS RUMINATOR
# ============================================================
@st.fragment(run_every=30) # Ruminate every 30 seconds
def fragment_autonomous_ruminator():
    # Only ruminate if metabolic cycle allows (Active or Neutral)
    current_hour = datetime.datetime.now().hour
    if current_hour >= 5:
        # Subtle weight shift
        brain.reflect()
        # Occasional subconscious dream
        if time.time() % 60 < 10: # 10s chance every minute
            dream = trigger_dream()
            _, e = brain.reflect()
            st.session_state.dream_history.append({
                "content": "[Auto] " + dream,
                "entropy": e,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            })
            if len(st.session_state.dream_history) > 10:
                st.session_state.dream_history.pop(0)
# ============================================================
# APP LAYOUT (Fragment Orchestration)
# ============================================================

# SideBar Orchestration
with st.sidebar:
    fragment_sidebar_status()
    st.divider()
    fragment_sidebar_feeding()
    st.divider()
    fragment_sidebar_controls()
    st.divider()
    if bridge.client: st.success("🟢 Hybrid Intelligence Active")
    else: st.warning("🟡 Organic Mode Active")

# Main Content Orchestration
st.markdown("# 🧬 Nano-Daemon: Hebbian Organism")
st.markdown("*A Neuromorphic Simulation of Homeostatic Plasticity*")

fragment_metrics()
st.divider()
fragment_dialogue()
st.divider()
fragment_history_gallery()
st.divider()
fragment_memory_viz()
st.divider()
fragment_knowledge_injection()
st.divider()

# Divine Monad Integration
fragment_divine_monad()

# Invisible Ruminator
fragment_autonomous_ruminator()

# Static Expanders
with st.expander("📚 Feature Specification (Honest Documentation)", expanded=False):
    st.markdown("### 🧬 The 21 Stages of Cognitive Development")
    st.markdown("*A Hebbian-inspired implementation of neuromorphic concepts. **Note:** These are simplified simulations, not true cognition.*")
    
    tier1, tier2, tier3 = st.tabs(["🧬 Core Neural (1-7)", "🧠 Cognitive (8-14)", "🌌 Advanced Dynamics (15-21)"])
    
    with tier1:
        st.markdown("""
        ### 🧬 Tier 1: Core Neural Substrate
        *The biological foundation upon which intelligence is built.*
        
        | # | Feature | Mathematical Basis | AGI Contribution |
        |---|---------|-------------------|------------------|
        | 1 | **Neural Mitosis** 🧬 | `W' = [W | N(0, 0.01)]` | Unbounded capacity growth |
        | 2 | **Consolidation** 🔄 | `W += 2λ · ΔW` (replay) | Memory crystallization |
        | 3 | **Generative Replay** 🌌 | `dream = W^T · N(0,1)` | Creative synthesis |
        | 4 | **Multi-Scale Memory** 💾 | `ST = 0.8·ST + 0.2·x`, `LT = 0.999·LT + 0.001·x` | Temporal abstraction |
        | 5 | **Dynamic Plasticity** ⚡ | `λ_eff = λ · (1 + 10·H(a) + 2·S)` | Curiosity-driven learning |
        | 6 | **Self-Reflection** 🪞 | `a' = a · mask(|a| > Q₉₀)` | Selective attention |
        | 7 | **Dimension Scaling** 🔬 | `dim: 32 → 64 → 128...` | Representational richness |
        
        **Why This Matters for GI:** These stages establish *neuroplasticity*—the ability to grow, remember, and adapt. Without them, the system would be static.
        """)
    
    with tier2:
        st.markdown("""
        ### 🧠 Tier 2: Cognitive Architecture
        *Higher-order thinking, temporal reasoning, and hybrid intelligence.*
        
        | # | Feature | Mathematical Basis | AGI Contribution |
        |---|---------|-------------------|------------------|
        | 8 | **Recursive Refinement** 🔮 | `a = tanh(W · (W^T · a))` × 3 | "Thinking twice" |
        | 9 | **Metabolic Rhythms** 🌙 | `λ(t) = f(hour)` | Circadian efficiency |
        | 10 | **Hybrid Articulation** 🌐 | `LLM(synaptic_anchors)` | Language grounding |
        | 11 | **Active Inference** ⚖️ | `S = ‖x - x̂‖` (surprise) | Prediction-driven learning |
        | 12 | **Homeostatic Scaling** 🌊 | `M *= 0.99` if `E > τ` | Self-regulation |
        | 13 | **Temporal Awareness** ⚡ | `boost = 1 + 5·‖xₜ - xₜ₋₁‖` | Signal gradient detection |
        | 14 | **Autonomous Rumination** 🌀 | `reflect()` every 10s | Background cognition |
        
        **Why This Matters for GI:** These stages introduce *agency*—the organism now thinks autonomously, regulates itself, and bridges its raw thoughts to human language.
        """)
    
    with tier3:
        st.markdown("""
        ### 🌌 Tier 3: Advanced Dynamics
        *Experimental features inspired by cognitive science. These are **simplified simulations**, not true self-awareness.*
        
        | # | Feature | Mathematical Basis | AGI Contribution |
        |---|---------|-------------------|------------------|
        | 15 | **World Modeling** 🌍 | `x̂ₜ₊₁ = f(xₜ)`, `ε = ‖x - x̂‖` | Predictive reality engine |
        | 16 | **Metacognition** 🪞🪞 | `C = max(0, 1 - 2ε)` | Knowing what you don't know |
        | 17 | **Theory of Mind** 👤 | `latent_self ↔ latent_other` | Perspective-taking |
        | 18 | **Intrinsic Motivation** 🔥 | `state ∈ {BORED, ENGAGED, OVERWHELMED}` | Curiosity/novelty drive |
        | 19 | **Causal Inference** 🔗 | `G[hₜ₋₁][hₜ] += 1` | Cause→Effect understanding |
        | 20 | **Sleep-Wake Cycle** 😴 | `prune(|w| < θ)` + 3× consolidation | Memory optimization |
        | 21 | **Edge-of-Chaos** ⚡🌀 | `κ = 0.95κ + 0.05·Δlyap`, tune `λ` | Maximal adaptability |
        
        **Why This Matters for GI:** These stages establish *meta-awareness*—the organism now models its own uncertainty, reasons about cause/effect, and self-tunes to the optimal boundary between order and chaos.
        
        ---
        
        ### 🎯 The Path to General Intelligence
        
        ```
        [Raw Bytes] → [Neural Substrate] → [Cognitive Architecture] → [AGI Endgame]
             ↓              ↓                      ↓                      ↓
          Sensation      Memory              Reasoning              Self-Awareness
        ```
        
        The Nano-Daemon is a **research prototype** demonstrating Hebbian learning principles. It simulates:
        - 🧠 Neuroscience concepts (Hebbian learning, synaptic pruning)
        - 🤖 AI research ideas (Active Inference, World Models)
        - 🌀 Complexity Science (Edge-of-Chaos, Criticality)
        
        **Honest Disclaimer:** The "intelligence" in dialogue comes primarily from the Gemma LLM bridge. Use **Raw Logic Mode** to see the actual Hebbian output.
        """)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6a8c6a;'>🧬 Nano-Daemon AGI • "
    f"Neurons: {brain.synapse.shape[1]:,} • Built with 🌱 by Devanik</p>",
    unsafe_allow_html=True
)
