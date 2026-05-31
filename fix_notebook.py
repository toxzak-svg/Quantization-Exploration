import json

with open('notebook/streaming_inference_demo.ipynb') as f:
    nb = json.load(f)

# Fix cell 2 (0-indexed)
cell = nb['cells'][2]
cell['source'] = [
    'import sys\n',
    'import os\n',
    '\n',
    '# Add parent directory to path\n',
    "sys.path.insert(0, os.path.dirname(os.path.abspath('.')))\n",
    '\n',
    'from stream_inference_gguf import GGUFReader, ternary_gemv, unpack_ternary\n',
    'import torch\n',
    'import time\n',
    'from pathlib import Path'
]

with open('notebook/streaming_inference_demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Fixed notebook')