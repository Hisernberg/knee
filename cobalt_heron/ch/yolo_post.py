"""YOLO-seg raw predictions (json: stem -> [{conf, rle}]) -> disjoint instances; OOF tuning; fusion with U-Net subs."""
import argparse, json
import numpy as np, pandas as pd
from pycocotools import mask as mu
from ch.data import load_coco, record_rles
from ch.metric import PQAccumulator

H = 2048


def disjoint(items, conf, min_area):
    """Greedy by confidence: each pixel goes to the highest-conf instance; drop small leftovers."""
    items = sorted([x for x in items if x['conf'] >= conf], key=lambda x: -x['conf'])
    occ = np.zeros((H, H), bool); out = []
    for x in items:
        m = mu.decode({'size': [H, H], 'counts': x['rle'].encode()}).astype(bool) & ~occ
        if m.sum() < min_area: continue
        occ |= m; out.append((x['conf'], mu.encode(np.asfortranarray(m.astype(np.uint8)))))
    return out


def tune(preds_path):
    P = json.load(open(preds_path)); recs = load_coco()
    grid = [(c, a) for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] for a in [0, 200, 500]]
    accs = {g: PQAccumulator() for g in grid}
    for stem, items in P.items():
        gts = [record_rles(a) for _, a in recs[stem]]
        for c, a in grid:
            pr = [r for _, r in disjoint(items, c, a)]
            for g in gts: accs[(c, a)].add(g, pr)
    for g, acc in sorted(accs.items(), key=lambda kv: -kv[1].pq)[:8]:
        print(g, acc)


def write(preds_path, conf, min_area, out):
    P = json.load(open(preds_path)); rows = []
    for stem, items in sorted(P.items()):
        for k, (_, r) in enumerate(disjoint(items, conf, min_area), 1):
            rows.append({'filament_id': f'{stem}_{k}', 'segmentation_rle': r['counts'].decode()})
    pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle']).to_csv(out, index=False); print(out, len(rows))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--preds'); ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--min-area', type=int, default=200); ap.add_argument('--out')
    a = ap.parse_args()
    tune(a.preds) if a.cmd == 'tune' else write(a.preds, a.conf, a.min_area, a.out)
