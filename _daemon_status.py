"""Read daemon state and log tail."""
import json, os, time
state_path = "/content/sub1quant/_download_state.json"
log_path = "/content/sub1quant/_download.log"

if os.path.exists(state_path):
    with open(state_path) as f:
        try:
            print("STATE:", json.dumps(json.load(f), indent=2))
        except Exception as e:
            print(f"STATE parse err: {e}")
else:
    print(f"no state yet at {state_path}")

if os.path.exists(log_path):
    with open(log_path) as f:
        lines = f.readlines()
    print(f"--- LOG ({len(lines)} lines) ---")
    for ln in lines[-15:]:
        print(ln.rstrip())
else:
    print(f"no log yet at {log_path}")
