import urllib.request, urllib.error
try:
    req = urllib.request.Request('https://comparative-representative-off-keyboards.trycloudflare.com/health')
    print("HEALTH OK:", urllib.request.urlopen(req, timeout=10).read().decode()[:200])
except urllib.error.HTTPError as e:
    print(f'HEALTH HTTP {e.code}: {e.read()[:200].decode()}')
except Exception as e:
    print(f'HEALTH ERR: {e}')

try:
    req = urllib.request.Request('https://comparative-representative-off-keyboards.trycloudflare.com/exec', method='POST',
                                 headers={'X-Bridge-Token': 'y4x9SYvUGW2NuNc9wXVeoHJC3gsKuzP3', 'Content-Type': 'application/json'},
                                 data=b'{"code":"print(2)"}')
    print("EXEC OK:", urllib.request.urlopen(req, timeout=10).read().decode()[:200])
except urllib.error.HTTPError as e:
    print(f'EXEC HTTP {e.code}: {e.read()[:200].decode()}')
except Exception as e:
    print(f'EXEC ERR: {e}')
