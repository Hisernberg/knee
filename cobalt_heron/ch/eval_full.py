"""Exact 2048 PQ of the full submit path (instances -> filter -> 2048 upsample -> optional morph) on OOF maps."""
import argparse, json, pickle, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from pycocotools import mask as mu
from ch.data import load_coco, record_rles
from ch.metric import PQAccumulator
from ch.post import instances
from ch.submit import to2048
from ch.inst_feats import inst_features

A = None
def work(f):
    F = pickle.load(open(A.filt, 'rb')); cfg = F['cfg']; recs = load_coco()
    p = np.load(f)['p'].astype(np.float32); img = np.load(Path(A.cache) / Path(f).name)['img']
    lab, _ = instances(p[0], p[1], **cfg)
    res = {}
    if lab.max():
        sc = F['model'].predict(inst_features(lab, p[0], p[1], img))
    gts = [record_rles(a) for _, a in recs[Path(f).stem]]
    for thr in A.thrs:
        l = lab
        if lab.max():
            keep = sc > thr; lut = np.zeros(lab.max() + 1, np.int32); lut[1:][keep] = np.arange(1, keep.sum() + 1); l = lut[lab]
        for up in A.ups:
            l2 = to2048(l, p[0], cfg['t'], up)
            for mo in A.morphs:
                pr = []
                for k in range(1, l2.max() + 1):
                    m = (l2 == k).astype(np.uint8)
                    if mo > 0: m = cv2.dilate(m, np.ones((2 * mo + 1,) * 2, np.uint8))
                    elif mo < 0: m = cv2.erode(m, np.ones((-2 * mo + 1,) * 2, np.uint8))
                    if m.any(): pr.append(mu.encode(np.asfortranarray(m)))
                acc = PQAccumulator()
                for g in gts: acc.add(g, pr)
                res[(thr, up, mo)] = acc
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--probs'); ap.add_argument('--filt'); ap.add_argument('--cache', default='/home/user/work/c1024')
    ap.add_argument('--thrs', type=float, nargs='+', default=[0.35]); ap.add_argument('--ups', nargs='+', default=['prob'])
    ap.add_argument('--morphs', type=int, nargs='+', default=[0]); ap.add_argument('--every', type=int, default=1)
    A = ap.parse_args()
    files = sorted(Path(A.probs).glob('*.npz'))[::A.every]
    tot = {}
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(work, files):
            for k, v in r.items(): tot.setdefault(k, PQAccumulator()).merge(v)
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1].pq): print(k, v)
