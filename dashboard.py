import http.server
import socketserver
import threading
import json
import os

PORT = 8000
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">  <title>NANO-DAEMON MONITOR</title>
    <style>
        body { background-color: #000; color: #0f0; font-family: 'Courier New', monospace; padding: 20px; max-width: 800px; margin: auto; }
        .box { border: 1px solid #0f0; padding: 15px; margin-bottom: 15px; box-shadow: 0 0 10px #030; }
        h1 { text-shadow: 0 0 5px #0f0; text-align: center; font-size: 1.5em; }
        #stability_bar { height: 10px; background-color: #111; width: 100%; border: 1px solid #0f0; margin-top: 10px; }
        #stability_fill { height: 100%; background-color: #0f0; width: 0%; transition: width 0.8s; box-shadow: 0 0 15px #0f0; }
        .label { color: #0a0; font-size: 0.8em; text-transform: uppercase; }
        .value { font-size: 1.2em; display: block; margin-top: 5px; }
        @media (max-width: 600px) {
            body { padding: 10px; }
            .box { padding: 10px; }
        }
    </style>
    <meta http-equiv="refresh" content="1">
</head>
<body>
    <h1>🧬 NANO-DAEMON: INF GROWING ORGANISM</h1>
    
    <div class="box">
        <span class="label">BIOLOGICAL STATUS</span>
        <span class="value" id="status">EVOLVING</span>
    </div>

    <div class="box">
        <span class="label">SYNAPTIC MASS (EXPERIENCE)</span>
        <span class="value"><span id="count">0</span> QUANTA</span>
    </div>
    
    <div class="box">
        <span class="label">COGNITIVE ENTROPY (INTELLIGENCE)</span>
        <span class="value" id="neuron">0.0000</span>
        <div class="label" style="margin-top:10px">SYNAPTIC PLASTICITY</div>
        <div id="stability_bar"><div id="stability_fill"></div></div>
    </div>

    <div class="box">
        <span class="label">LAST PERCEPTION</span>
        <span class="value" style="font-size: 0.7em; color: #0c0;" id="last_file">None</span>
    </div>

    <script>
        fetch('brain_state.json')
            .then(response => response.json())
            .then(data => {
                document.getElementById('count').innerText = data.files_eaten;
                document.getElementById('last_file').innerText = data.last_file;
                document.getElementById('neuron').innerText = data.neuron_activity.toFixed(4);
                
                let stab = data.brain_stability * 100;
                document.getElementById('stability_fill').style.width = stab + '%';
            });
    </script>
</body>
</html>
'''

class Dashboard:
    def __init__(self):
        # FIX: Added encoding='utf-8' to support emojis on Windows
        with open("brain_state.json", "w", encoding='utf-8') as f:
            json.dump({"files_eaten": 0, "last_file": "None", "brain_stability": 0, "neuron_activity": 0}, f)
            
        # FIX: Added encoding='utf-8' here too
        with open("index.html", "w", encoding='utf-8') as f:
            f.write(HTML_TEMPLATE)

    def start(self):
        # Quiet handler to stop console spam
        Handler = http.server.SimpleHTTPRequestHandler
        
        def run_server():
            # Allow address reuse to prevent "Address already in use" errors
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("", PORT), Handler) as httpd:
                print(f"📊 DASHBOARD ONLINE: http://localhost:{PORT}")
                httpd.serve_forever()
                
        # Run server in a background thread so it doesn't block the brain
        daemon_thread = threading.Thread(target=run_server, daemon=True)
        daemon_thread.start()