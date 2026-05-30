import torch
from collections import Counter

data = torch.load('quantized/gemma-4-E2B-sub1bit.pt', map_location='cpu', weights_only=True)
q = data.get('quantized', {})

print(f'Total entries: {len(q)}')

# Analyze shapes
shapes = []
for idx, entry in q.items():
    u_shape = entry['U_shape']
    vt_shape = entry['Vt_shape']
    shapes.append((u_shape, vt_shape))

# Show unique shapes
unique_shapes = set((str(s[0]), str(s[1])) for s in shapes)
print('Unique U,Vt shape pairs:')
for us, vs in unique_shapes:
    count = sum(1 for s in shapes if str(s[0]) == us and str(s[1]) == vs)
    print(f'  U={us}, Vt={vs} -> {count} layers')

# Gemma4 config: hidden_size=1536, intermediate_size=6144, num_layers=35
# 35 layers * (Q,K,V,O,gate,up,down) = 35*7 = 245 weights per section
# But we're seeing 316, so maybe it's counting something else
print(f'\nExpected: 35 layers * 7 weights = 245, but got {len(q)}')

# Check the ranks
ranks = [entry['U_shape'][1] for entry in q.values()]
rank_counts = Counter(ranks)
print(f'\nRank distribution: {dict(rank_counts)}')