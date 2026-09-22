"""Support-aware fusion: keep base instances whose filter score is high, or moderately high AND confirmed
by an independent submission (IoU > iou_thr).  Base masks are kept as-is (disjoint).
usage: fuse_support.py base_scored.csv other.csv out.csv hi lo iou_thr
base_scored.csv must contain a `score` column (from ch.submit --keep-scores)."""
import sys, numpy as np, pandas as pd
from pycocotools import mask as mu
base, other, out, hi, lo, it = sys.argv[1], sys.argv[2], sys.argv[3], *map(float, sys.argv[4:7])
B = pd.read_csv(base); O = pd.read_csv(other)
for d in (B, O): d['stem'] = d.filament_id.str.rsplit('_', n=1).str[0]
og = {s: [{'size': [2048, 2048], 'counts': c.encode()} for c in g.segmentation_rle] for s, g in O.groupby('stem')}
rows = []
for s, g in B.groupby('stem', sort=True):
    br = [{'size': [2048, 2048], 'counts': c.encode()} for c in g.segmentation_rle]
    sup = np.zeros(len(br))
    if og.get(s): sup = np.asarray(mu.iou(br, og[s], [0] * len(og[s]))).max(1)
    k = 0
    for (_, r), su in zip(g.iterrows(), sup):
        if r.score > hi or (r.score > lo and su > it):
            k += 1; rows.append({'filament_id': f'{s}_{k}', 'segmentation_rle': r.segmentation_rle})
pd.DataFrame(rows).to_csv(out, index=False); print(out, len(rows))
