"""Probe HuggingFace for a gemma-4-E2B model. Runs on kernel."""
import urllib.request, urllib.error, json, sys, time

# Try several candidate paths
candidates = [
    "google/gemma-4-E2B",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E2B-pt",
    "google/gemma4-E2B",
    "google/gemma4-E2B-it",
    "google/gemma-4b",
    "google/gemma-3n-E4B",
    "google/gemma-3-E4B-it",
    "google/gemma-3-E2B-it",
    "google/gemma-3-E2B",
    # try without "google/" if user has private
]

for repo in candidates:
    url = f"https://huggingface.co/api/models/{repo}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sub1quant-probe"})
        with urllib.request.urlopen(req, timeout=10) as r:
            j = json.loads(r.read())
            sibs = j.get("siblings", [])
            print(f"HIT  {repo}  - {len(sibs)} files")
            for s in sibs[:10]:
                print(f"   {s.get('rfilename')}")
            if len(sibs) > 10:
                print(f"   ...{len(sibs)-10} more")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"MISS {repo}")
        elif e.code in (401, 403):
            print(f"PRIV {repo}  - code {e.code}")
        else:
            print(f"ERR  {repo}  - code {e.code}")
    except Exception as e:
        print(f"ERR  {repo}  - {type(e).__name__}: {e}")
print("---")

# Also try searching
search = "https://huggingface.co/api/models?search=gemma+4+E2B&limit=20"
try:
    req = urllib.request.Request(search, headers={"User-Agent": "sub1quant-probe"})
    with urllib.request.urlopen(req, timeout=10) as r:
        j = json.loads(r.read())
        print(f"SEARCH results ({len(j)}):")
        for m in j[:15]:
            print(f"  {m.get('id')}  downloads={m.get('downloads', '?')}")
except Exception as e:
    print(f"search error: {e}")
