import urllib.request, urllib.error
import os

TOKEN = os.getenv("BRIDGE_TOKEN", "pNWps4V97_opsqamUqTwvr8HK4JI_bDd")
BASE = os.getenv("BRIDGE_URL", "https://prime-apparently-types-explanation.trycloudflare.com")

try:
    req = urllib.request.Request(BASE + "/health", headers={"X-Bridge-Token": TOKEN})
    print("HEALTH OK:", urllib.request.urlopen(req, timeout=10).read().decode()[:300])
except urllib.error.HTTPError as e:
    print(f'HEALTH HTTP {e.code}: {e.read()[:200].decode()}')
except Exception as e:
    print(f'HEALTH ERR: {e}')

try:
    req = urllib.request.Request(BASE + "/exec", method="POST",
                                 headers={"X-Bridge-Token": TOKEN, "Content-Type": "application/json"},
                                 data=b'{"code":"print(2+2)"}')
    print("EXEC OK:", urllib.request.urlopen(req, timeout=10).read().decode()[:200])
except urllib.error.HTTPError as e:
    print(f'EXEC HTTP {e.code}: {e.read()[:200].decode()}')
except Exception as e:
    print(f'EXEC ERR: {e}')
