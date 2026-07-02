import json, pathlib
nb = json.loads(pathlib.Path(r'C:\Users\Zwmar\projects\sub1quant\notebook\cross_layer_mi_colab.ipynb').read_text())
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source']).strip()
    first = src.split('\n', 1)[0][:70]
    print(f'  {i:2d} [{c["cell_type"]:8s}] {first!r}')