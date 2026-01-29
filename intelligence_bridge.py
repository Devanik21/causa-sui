import google.generativeai as genai
import os
import torch

class GemmaBridge:
    """The 'Cherry on Top': Connects the Hebbian Brain to Gemma-3 for refined articulation."""
    def __init__(self, api_key=None):
        try:
            from dotenv import load_dotenv
            # Explicitly look for .env in the same folder as this script
            base_dir = os.path.dirname(os.path.abspath(__file__))
            env_path = os.path.join(base_dir, '.env')
            load_dotenv(dotenv_path=env_path)
            print(f"📡 API KEY SEARCH: Checking {env_path}...")
        except ImportError:
            pass

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        # Manual Fallback if dotenv is being stubborn
        if not self.api_key and os.path.exists(".env"):
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if "GEMINI_API_KEY" in line:
                            self.api_key = line.split("=")[1].strip().strip('"').strip("'")
                            print("🗝️  KEY FOUND: Manual .env extraction successful.")
                            break
            except:
                pass

        if not self.api_key:
            print("⚠️ WARNING: No GEMINI_API_KEY found. Hybrid mode deactivated.")
            self.model = None
            return

        try:
            genai.configure(api_key=self.api_key)
            # Using the user-specified model name
            self.model = genai.GenerativeModel('gemma-3-27b-it')
            print("🧠 GEMMA BRIDGE ACTIVE: Hybrid Intelligence Engaged.")
        except Exception as e:
            print(f"❌ GEMMA BRIDGE ERROR: {e}")
            self.model = None

    def articulate(self, human_query, synaptic_anchors):
        """Grounds Gemma's response in the Organism's raw synaptic associations."""
        if not self.model:
            return f"[ORGANIC THOUGHT ONLY]: {synaptic_anchors}"

        # Clean anchors (remove non-printable)
        clean_anchors = "".join([c for c in synaptic_anchors if c.isprintable()])
        
        prompt = f"""
        Human Query: "{human_query}"
        
        Raw Synaptic Associations (Ground Truth): "{clean_anchors}"
        
        INSTRUCTIONS:
        You are the 'Cerebral Cortex' of the Nano-Daemon organism. 
        Your task is to articulate the organism's raw thoughts into a human-readable response.
        
        RULES:
        1. Use the provided "Raw Synaptic Associations" as your primary context.
        2. If the associations contain patterns or words (like 'AI', 'GPT', 'Physics'), emphasize them.
        3. Do NOT hallucinate entirely new facts. Stay grounded in the 'vibe' of the associations.
        4. Be concise and 'organic' - your goal is to bridge the gap between silicon and biology.
        5. If you see gibberish in the associations, interpret it as the organism's 'embryonic' state.
        
        Response:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"⚠️ Articulation Failure: {e}\n[RAW]: {synaptic_anchors}"
