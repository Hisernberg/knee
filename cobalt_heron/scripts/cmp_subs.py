"""PQ of submission A against submission B treated as ground truth + area stats."""
import sys, numpy as np, pandas as pd
from pycocotools import mask as mu
sys.path.insert(0, '.')
from ch.metric import PQAccumulator
def load(f):
    d = pd.read_csv(f); d['stem'] = d.filament_id.str.rsplit('_', n=1).str[0]
    return {s: [{'size': [2048, 2048], 'counts': c.encode()} for c in g.segmentation_rle] for s, g in d.groupby('stem')}
A, B = load(sys.argv[1]), load(sys.argv[2]); acc = PQAccumulator()
for s in set(A) | set(B): acc.add(B.get(s, []), A.get(s, []))
aa = [mu.area(r) for v in A.values() for r in v]; bb = [mu.area(r) for v in B.values() for r in v]
print(acc); print('A areas', np.percentile(aa, [10, 25, 50, 75, 90]).astype(int), len(aa)); print('B areas', np.percentile(bb, [10, 25, 50, 75, 90]).astype(int), len(bb))
