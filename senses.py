import requests
import xml.etree.ElementTree as ET
import threading
import time
import torch
import os
import json
import random
from watchdog.events import FileSystemEventHandler
from core import PlasticCortex

# Global lock for telemetry file access
telemetry_lock = threading.Lock()

def update_telemetry(count, last_item, stability, entropy):
    """Safely updates the brain state telemetry using atomic writes."""
    telemetry = {
        "files_eaten": count,
        "last_file": last_item,
        "brain_stability": stability,
        "neuron_activity": max(0, entropy) # Ensure positive entropy for gauge
    }
    with telemetry_lock:
        try:
            # Atomic write: Write to temp, then rename
            temp_path = "brain_state.json.tmp"
            with open(temp_path, "w", encoding='utf-8') as f:
                json.dump(telemetry, f)
            
            # Replace the existing file (this is atomic on most OSs)
            if os.path.exists("brain_state.json"):
                os.remove("brain_state.json")
            os.rename(temp_path, "brain_state.json")
        except Exception as e:
            pass

class InternetSense:
    """The Organism's connection to the global information stream."""
    def __init__(self, brain, feeder=None):
        self.brain = brain
        self.feeder = feeder
        self.feeds = [
            "https://news.ycombinator.com/rss",
            "https://www.sciencedaily.com/rss/all.xml",
            "http://export.arxiv.org/rss/cs.AI"
        ]
        self.is_active = False

    def start(self):
        self.is_active = True
        threading.Thread(target=self._poll_loop, daemon=True).start()
        print("🌐 INTERNET SENSE ONLINE: Listening to global entropy...")

    def _poll_loop(self):
        while self.is_active:
            print("🌐 Organism searching for global sustenance...")
            for url in self.feeds:
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) NanoDaemon/1.0'}
                    # Randomize feed order to avoid predictable patterns
                    response = requests.get(url, headers=headers, timeout=15)
                    if response.status_code == 200:
                        # Safety check for XML content
                        try:
                            root = ET.fromstring(response.text)
                            items = root.findall('.//item')
                            # Digest up to 5 items per feed for rapid initialization
                            for item in items[:5]:
                                title_elem = item.find('title')
                                title = title_elem.text if title_elem is not None else "Unknown"
                                self.digest_content(title)
                                time.sleep(1) # Slow down digestion to "savor" the data
                        except ET.ParseError:
                            print(f"⚠️ XML Parse Error on {url}")
                except Exception as e:
                    print(f"⚠️ Web Sense glitch ({url}): {e}")
            
            # Short rest before next hunt (5 minutes)
            print("💤 Organism resting after a big meal...")
            time.sleep(300) 

    def digest_content(self, text):
        raw_bytes = text.encode('utf-8')[:4096]
        data = torch.tensor(list(raw_bytes), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            activation, stability = self.brain(data)
        
        # Update metrics if feeder is available
        if self.feeder:
            self.feeder.processed_count += 1
            update_telemetry(
                self.feeder.processed_count,
                f"WEB: {text[:30]}...",
                stability,
                stability # Entropy
            )
        
        print(f"📡 Web Content Digested: {text[:50]}...")

class UniversalFeeder(FileSystemEventHandler):
    def __init__(self, brain):
        self.brain = brain
        self.processed_count = 0
        from intelligence_bridge import GemmaBridge
        self.bridge = GemmaBridge()
        print("👁️ SENSORY CORTEX V1.2 ONLINE: Detecting Creation & Modification")

    def hallucinate(self):
        """The Organism dreams by reversing its own logic."""
        print("\n🌌 DREAM STATE INITIATED...")
        
        # Noise must match the Brain's current Hidden Dimension (mitotic growth)
        hidden_dim = self.brain.synapse.shape[1]
        noise = torch.randn(1, hidden_dim) 
        
        # 2. Pass it through the brain weights backwards
        # [1, 1024] x [1024, 32] = [1, 32] (The original signal)
        with torch.no_grad():
            thought_vector = torch.matmul(noise, self.brain.synapse.t())
            
        # 3. Decode thoughts to bytes
        dream_bytes = []
        # The dream will be a sequence of 32 "Concept Bytes"
        for val in thought_vector[0]:
            byte_val = int((val.item() + 1) * 128) 
            byte_val = max(0, min(255, byte_val))
            dream_bytes.append(byte_val)
            
        # 4. Materialize the dream
        try:
            with open("dream_result.txt", "wb") as f:
                f.write(bytes(dream_bytes))
            print(f"✨ DREAM MATERIALIZED: dream_result.txt (Size: {len(dream_bytes)} bytes)")
        except Exception as e:
            print(f"❌ NIGHTMARE ERROR: {e}")

    def process_event(self, path):
        filename = os.path.basename(path)
        
        # 1. IGNORE SELF (Prevent Loops)
        ignore_list = [
            "brain_state.json", 
            "brain_state.json.tmp", 
            "brain_weights.pth", 
            "dream_result.txt", 
            "RESPONSE.txt",
            "index.html"
        ]
        if filename in ignore_list or filename.endswith('.tmp'):
            return

        # 2. COMMAND CHECK (PRIORITY) - Check this BEFORE reading file
        if "DREAM" in filename:
            self.hallucinate()
            return
            
        if filename == "QUERY.txt":
            try:
                print(f"👂 Organism listening to query...")
                with open(path, 'rb') as f:
                    query_bytes = f.read(1024)
                
                if not query_bytes: return
                
                query_text = query_bytes.decode('utf-8', errors='ignore').strip()
                
                # 1. RAW ASSOCIATION (The Hebbian Ground Truth)
                response_bytes = self.brain.associate(torch.tensor(list(query_bytes), dtype=torch.long).unsqueeze(0))
                synaptic_anchors = response_bytes.decode('utf-8', errors='ignore')
                
                # 2. HYBRID ARTICULATION (The 'Cherry on Top')
                final_thought = self.bridge.articulate(query_text, synaptic_anchors)
                
                with open("RESPONSE.txt", "w", encoding='utf-8') as f:
                    f.write(f"PROMPT: {query_text}\n")
                    f.write(f"SYNAPTIC ANCHORS: {synaptic_anchors}\n")
                    f.write(f"HYBRID ARTICULATION: {final_thought}")
                
                print(f"💡 HYBRID THOUGHT: {final_thought[:100]}...")
                
                # Update dashboard
                update_telemetry(self.processed_count, f"THOUGHT: {final_thought[:30]}...", 0.9, 0.9)
                return
            except Exception as e:
                print(f"⚠️ Query glitch: {e}")
                return

        # 3. DIGESTION
        try:
            with open(path, 'rb') as f:
                raw_bytes = f.read(4096)
            
            # If empty, ignore
            if not raw_bytes: return

            data = torch.tensor(list(raw_bytes), dtype=torch.long).unsqueeze(0)
            brain_state, stability = self.brain(data)
            
            self.processed_count += 1
            print(f"[{self.processed_count}] Digested: {filename} | Stability: {stability:.4f}")
            
            # Telemetry
            update_telemetry(
                self.processed_count,
                filename,
                stability,
                stability
            )

        except Exception as e:
            pass 

    def feed_all_existing(self, directory_path):
        """Initial feeding session for all existing files in the directory."""
        print(f"🍽️ INITIAL FEEDING SESSION: {os.path.abspath(directory_path)}")
        for root, dirs, files in os.walk(directory_path):
            # Ignore __pycache__ and other hidden folders
            if "__pycache__" in root or ".git" in root:
                continue
                
            for filename in files:
                file_path = os.path.join(root, filename)
                # Ensure we don't digest the tracking files
                if filename not in ["brain_state.json", "dream_result.txt", "index.html"]:
                     self.process_event(file_path)
        print("✅ INITIAL FEEDING COMPLETE. Organism is satiated.")

    # Listen for BOTH Creation and Modification
    def on_modified(self, event):
        if not event.is_directory:
            self.process_event(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self.process_event(event.src_path)