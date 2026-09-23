"""Exact 2048 PQ of the submit path on OOF prob maps."""
import argparse, json
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
from ch.data import load_coco, record_rles
from ch.metric import PQAccumulator
from ch.post import instances
from ch.submit import to2048

ap = argparse.ArgumentParser(); ap.add_argument('--probs', nargs='+'); ap.add_argument('--cfg'); a = ap.parse_args()
cfg = json.loads(a.cfg); recs = load_coco(); acc = PQAccumulator()
for f in sorted(Path(a.probs[0]).glob('*.npz')):
    p = np.mean([np.load(Path(d) / f.name)['p'].astype(np.float32) for d in a.probs], 0)
    lab, _ = instances(p[0], p[1], **cfg); l2 = to2048(lab, p[0], cfg['t'])
    pr = [mu.encode(np.asfortranarray((l2 == k).astype(np.uint8))) for k in range(1, l2.max() + 1)]
    for _, anns in recs[f.stem]:
        acc.add(record_rles(anns), pr)
print(acc)
