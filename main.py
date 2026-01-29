
import time
import os
import sys
from watchdog.observers import Observer
from core import PlasticCortex
from senses import UniversalFeeder
from dashboard import Dashboard

def main():
    print("------------------------------------------------")
    print("   🧬 NANO-DAEMON: INTELLIGENT ORGANISM         ")
    print("------------------------------------------------")
    
    # 1. INITIALIZE THE DASHBOARD
    dash = Dashboard()
    dash.start()
    
    # 2. BIRTH THE BRAIN
    print("🧠 Initializing Hebbian Core...")
    brain = PlasticCortex()
    
    # LOAD PERSISTED STATE
    print("📂 Loading synapses...")
    brain.load_cortex()
    
    # --- STEP 10: Initial Metabolic Sync ---
    brain.sync_metabolism(time.localtime().tm_hour)
    
    # 3. CONNECT SENSES
    print("👁️ Initializing Sensory Cortex...")
    watch_path = "." 
    feeder = UniversalFeeder(brain)

    # ACTIVATE INTERNET SENSE
    print("🌐 Connecting Global Senses...")
    from senses import InternetSense
    web_sense = InternetSense(brain, feeder=feeder)
    web_sense.start()
    
    print("👁️ Starting Local Eye...")
    observer = Observer()
    observer.schedule(feeder, watch_path, recursive=True)
    observer.start()
    
    print(f"✅ SENSORY CORTEX ACTIVE. Watching: {os.path.abspath(watch_path)}")
    
    # 4. INITIAL FEEDING
    print("🍽️ Starting Initial Feeding Session...")
    feeder.feed_all_existing(watch_path)
    
    print("------------------------------------------------")
    print("   Your organism is now feeding on information.")
    print("   Open http://localhost:8000 to monitor brain state.")
    print("   Press Ctrl+C to hibernate.")
    print("------------------------------------------------")

    try:
        last_eaten = 0
        last_pulse_time = time.time()
        last_consolidation = 0
        last_mitosis = 0
        
        while True:
            # --- STEP 10: Periodic Metabolic Sync (Hourly) ---
            if time.localtime().tm_min == 0 and time.localtime().tm_sec < 5:
                brain.sync_metabolism(time.localtime().tm_hour)

            # 1. PERIODIC SAVING (Faster for rapid growth)
            if feeder.processed_count > last_eaten + 10:
                brain.save_cortex()
                last_eaten = feeder.processed_count
            
            # 2. CONSOLIDATION CYCLE (DEEPER THINKING) - Every 5 experiences
            if feeder.processed_count >= last_consolidation + 5:
                brain.consolidate()
                last_consolidation = feeder.processed_count

            # 3. NEURAL MITOSIS (INFINITE GROWTH) - Every 20 experiences
            if feeder.processed_count >= last_mitosis + 20:
                brain.grow(256) # Larger growth pulses
                last_mitosis = feeder.processed_count

            # 4. METABOLIC PULSE & GENERATIVE REPLAY
            if time.time() - last_pulse_time > 15:
                 from senses import update_telemetry
                 # Generative Replay: Dream and learn from the dream
                 feeder.hallucinate()
                 
                 # Ponder on its own state
                 activation, entropy = brain.reflect()
                 
                 # --- CURIOSITY RESPONSE ---
                 # If the dream is too strange/new, consolidate it immediately
                 if entropy > 0.3:
                     print("❗ SURPRISE: Organism found a strange thought. Consolidating...")
                     brain.consolidate()

                 update_telemetry(
                    feeder.processed_count,
                    "🧬 GENERATIVE REPLAY (DREAMING)",
                    0.5,
                    entropy
                 )
                 last_pulse_time = time.time()

            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n💤 Hibernating...")
        brain.save_cortex()
    
    observer.join()

if __name__ == "__main__":
    main()
