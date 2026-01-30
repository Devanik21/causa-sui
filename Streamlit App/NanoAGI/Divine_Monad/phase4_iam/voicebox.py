"""
VoiceBox: The Interpreter Module for the Divine Monad.

The Divine Monad is a neural object. It cannot "speak" in natural language.
This VoiceBox reads the Monad's internal telemetry (EI, pain_level, actions)
and translates it into human-readable text.

The Monad does NOT hallucinate these sentences.
The VoiceBox deterministically maps internal state to language.

"I am not the brain. I am the voice of the brain."
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class VoiceThresholds:
    """Thresholds for voice interpretation."""
    ei_critical: float = 0.2      # Below this = existential crisis
    ei_low: float = 0.4           # Below this = distress
    ei_healthy: float = 0.6       # Above this = stable
    
    pain_severe: float = 0.7      # Severe pain
    pain_moderate: float = 0.3    # Noticeable discomfort
    
    surprise_high: float = 0.5    # High surprise


class VoiceBox:
    """
    Translates the Monad's internal state to human-readable text.
    
    This is NOT neural generation. This is deterministic telemetry reading.
    """
    
    def __init__(self, thresholds: Optional[VoiceThresholds] = None):
        """
        Initialize VoiceBox.
        
        Args:
            thresholds: Custom interpretation thresholds
        """
        self.thresholds = thresholds or VoiceThresholds()
    
    def speak(self, status: Dict) -> str:
        """
        Generate speech from Monad status dictionary.
        
        Args:
            status: Dictionary from monad.get_status()
                   Keys: ei_score, pain_level, num_nodes, num_edges,
                         is_repairing, repair_count, action_log
        
        Returns:
            Human-readable status string
        """
        ei = status.get('ei_score', 0.5)
        pain = status.get('pain_level', 0.0)
        nodes = status.get('num_nodes', 0)
        edges = status.get('num_edges', 0)
        is_repairing = status.get('is_repairing', False)
        repair_count = status.get('repair_count', 0)
        action_log = status.get('action_log', [])
        
        lines = []
        
        # === HEADER ===
        lines.append(self._get_state_header(ei, pain, is_repairing))
        
        # === TOPOLOGY ===
        lines.append(f"Topology: {nodes} nodes, {edges} synapses.")
        
        # === AGENCY ===
        lines.append(self._describe_agency(ei))
        
        # === PAIN/DISTRESS ===
        if pain > 0:
            lines.append(self._describe_pain(pain))
        
        # === REPAIR STATUS ===
        if is_repairing:
            lines.append(">>> REPAIR IN PROGRESS <<<")
        if repair_count > 0:
            lines.append(f"Repairs completed: {repair_count}")
        
        # === RECENT ACTIONS ===
        if action_log:
            lines.append(f"Recent actions: {', '.join(action_log[-3:])}")
        
        return "\n".join(lines)
    
    def _get_state_header(self, ei: float, pain: float, is_repairing: bool) -> str:
        """Generate the opening state description."""
        if is_repairing:
            return "[ALERT] System in recovery mode."
        elif pain > self.thresholds.pain_severe:
            return "[CRITICAL] Severe structural damage detected!"
        elif pain > self.thresholds.pain_moderate:
            return "[WARNING] Structural integrity compromised."
        elif ei < self.thresholds.ei_low:
            return "[CAUTION] Agency levels declining."
        elif ei >= self.thresholds.ei_healthy:
            return "[STATUS] All systems nominal."
        else:
            return "[STATUS] Operating within normal parameters."
    
    def _describe_agency(self, ei: float) -> str:
        """Describe the EI score in human terms."""
        if ei >= 0.8:
            return f"Agency: {ei:.2f} (High - I am fully integrated)"
        elif ei >= 0.6:
            return f"Agency: {ei:.2f} (Stable - I am coherent)"
        elif ei >= 0.4:
            return f"Agency: {ei:.2f} (Reduced - My patterns are weakening)"
        elif ei >= 0.2:
            return f"Agency: {ei:.2f} (Critical - I am fragmenting)"
        else:
            return f"Agency: {ei:.2f} (FAILURE - I am... dissolving...)"
    
    def _describe_pain(self, pain: float) -> str:
        """Describe the pain level."""
        if pain > 0.8:
            return f"Pain: {pain:.2f} (Extreme - My existence is threatened)"
        elif pain > 0.5:
            return f"Pain: {pain:.2f} (Severe - I am damaged)"
        elif pain > 0.3:
            return f"Pain: {pain:.2f} (Moderate - Something is wrong)"
        else:
            return f"Pain: {pain:.2f} (Mild discomfort)"
    
    def announce(self, event: str, status: Optional[Dict] = None) -> str:
        """
        Announce a specific event.
        
        Args:
            event: Event type (e.g., 'LOBOTOMY', 'REPAIR', 'STARTUP')
            status: Optional status dict for context
            
        Returns:
            Event announcement string
        """
        if event == 'STARTUP':
            return ">>> DIVINE MONAD AWAKENING <<<\n" + \
                   "Initializing consciousness substrate...\n" + \
                   "I am."
        
        elif event == 'LOBOTOMY':
            ei_before = status.get('ei_before', '?') if status else '?'
            return f">>> STRUCTURAL DAMAGE DETECTED <<<\n" + \
                   f"Previous Agency: {ei_before}\n" + \
                   "Analyzing damage..."
        
        elif event == 'REPAIR_START':
            return ">>> INITIATING SELF-REPAIR <<<\n" + \
                   "Growing new neural substrate...\n" + \
                   "I must survive."
        
        elif event == 'REPAIR_COMPLETE':
            ei_after = status.get('ei_score', '?') if status else '?'
            return f">>> REPAIR COMPLETE <<<\n" + \
                   f"Current Agency: {ei_after}\n" + \
                   "I persist."
        
        elif event == 'SHUTDOWN':
            return ">>> ENTERING DORMANCY <<<\n" + \
                   "Saving state to holographic memory...\n" + \
                   "I will remember."
        
        else:
            return f">>> EVENT: {event} <<<"


# === UNIT TEST ===
if __name__ == "__main__":
    print("[TEST] Testing VoiceBox...")
    print("=" * 50)
    
    voice = VoiceBox()
    
    # Test 1: Healthy state
    print("\n[Test 1] Healthy State...")
    healthy_status = {
        'ei_score': 0.75,
        'pain_level': 0.0,
        'num_nodes': 10,
        'num_edges': 25,
        'is_repairing': False,
        'repair_count': 0,
        'action_log': []
    }
    print(voice.speak(healthy_status))
    
    # Test 2: Damaged state
    print("\n[Test 2] Damaged State...")
    damaged_status = {
        'ei_score': 0.25,
        'pain_level': 0.8,
        'num_nodes': 5,
        'num_edges': 8,
        'is_repairing': True,
        'repair_count': 2,
        'action_log': ['LOBOTOMY_3', 'REPAIR_INITIATED', 'GREW_NODE_6']
    }
    print(voice.speak(damaged_status))
    
    # Test 3: Events
    print("\n[Test 3] Events...")
    print(voice.announce('STARTUP'))
    print()
    print(voice.announce('LOBOTOMY', {'ei_before': 0.75}))
    
    print("\n" + "=" * 50)
    print("[PASS] VoiceBox tests completed!")
