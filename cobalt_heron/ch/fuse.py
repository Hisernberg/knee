"""Fuse U-Net instances (with filter scores) and YOLO instances (with conf).
keep U-Net inst if score > t1, or score > t2 and a YOLO inst (conf>=cy) matches it (IoU > 0.3);
add YOLO inst with conf >= c_add whose overlap with every kept U-Net inst is < 0.2 of its own area."""
import argparse, itertools, json, pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from pycocotools import mask as mu
from ch.data import load_coco, record_rles
from ch.metric import PQAccumulator
from ch.post import instances
from ch.submit import to2048
from ch.inst_feats import inst_features
from ch.yolo_post import disjoint

A = None


def unet_insts(prob_file, img, F, cfg):
    p = np.load(prob_file)['p'].astype(np.float32)
    lab, _ = instances(p[0], p[1], **cfg)
    if lab.max() == 0: return [], np.zeros(0)
    sc = F['model'].predict(inst_features(lab, p[0], p[1], img))
    l2 = to2048(lab, p[0], cfg['t'], 'prob1.2')
    rles, scs = [], []
    for k in range(1, lab.max() + 1):
        m = l2 == k
        if m.any(): rles.append(mu.encode(np.asfortranarray(m.astype(np.uint8)))); scs.append(sc[k - 1])
    return rles, np.array(scs)


def prep(stem, prob_file, img, yitems):
    F = pickle.load(open(A.filt, 'rb'))
    U, us = unet_insts(prob_file, img, F, F['cfg'])
    Y = disjoint(yitems, 0.1, 200); yc = np.array([c for c, _ in Y]); Y = [r for _, r in Y]
    iou = np.asarray(mu.iou(U, Y, [0] * len(Y))) if U and Y else np.zeros((len(U), len(Y)))
    ua = np.array([mu.area(r) for r in U], float); ya = np.array([mu.area(r) for r in Y], float)
    inter = iou * (ua[:, None] + ya[None, :]) / (1 + iou) if len(U) and len(Y) else iou
    return dict(stem=stem, U=U, us=us, Y=Y, yc=yc, iou=iou, yfrac=inter / np.maximum(ya[None, :], 1) if len(Y) else inter)


def select(d, t1, t2, cy, c_add):
    iou, yc = d['iou'], d['yc']
    match = (iou > 0.3) & (yc[None, :] >= cy) if iou.size else np.zeros((len(d['U']), 0), bool)
    ku = (d['us'] > t1) | ((d['us'] > t2) & match.any(1) if match.shape[1] else d['us'] > t1)
    out = [r for r, k in zip(d['U'], ku) if k]
    if len(d['Y']):
        cov = d['yfrac'][ku].max(0) if ku.any() else np.zeros(len(d['Y']))
        out += [r for r, c, v in zip(d['Y'], yc, cov) if c >= c_add and v < 0.2]
    return out


def _prep_one(args):
    return prep(*args)


def load_all(prob_dir, cache, preds, stems=None, npy=False):
    P = json.load(open(preds)); jobs = []
    for f in sorted(Path(prob_dir).glob('*.npz')):
        if stems is not None and f.stem not in stems: continue
        img = np.load(Path(cache) / (f.stem + ('.npy' if npy else '.npz')))
        img = img if npy else img['img']
        jobs.append((f.stem, str(f), img, P.get(f.stem, [])))
    with ProcessPoolExecutor(4) as ex:
        return list(ex.map(_prep_one, jobs))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--probs'); ap.add_argument('--preds'); ap.add_argument('--filt')
    ap.add_argument('--cache', default='/home/user/work/c1024'); ap.add_argument('--params', default=None); ap.add_argument('--out')
    A = ap.parse_args()
    if A.cmd == 'tune':
        D = load_all(A.probs, A.cache, A.preds); recs = load_coco()
        grid = list(itertools.product([0.35, 0.4, 0.5], [0.15, 0.2, 0.25, 0.3], [0.2, 0.3], [0.5, 0.6, 0.7, 1.1]))
        acc = {g: PQAccumulator() for g in grid}
        for d in D:
            gts = [record_rles(a) for _, a in recs[d['stem']]]
            for g in grid:
                pr = select(d, *g)
                for gt in gts: acc[g].add(gt, pr)
        for g, a in sorted(acc.items(), key=lambda kv: -kv[1].pq)[:10]: print(g, a)
        base = [g for g in grid if g[1] == 0.35 and g[3] == 1.1]
    else:
        t1, t2, cy, ca = map(float, A.params.split(','))
        D = load_all(A.probs, A.cache, A.preds, npy=True); rows = []
        for d in D:
            occ = np.zeros((2048, 2048), bool); k = 0
            for r in select(d, t1, t2, cy, ca):
                m = mu.decode(r).astype(bool) & ~occ
                if m.sum() < 100: continue
                occ |= m; k += 1
                rows.append({'filament_id': f"{d['stem']}_{k}", 'segmentation_rle': mu.encode(np.asfortranarray(m.astype(np.uint8)))['counts'].decode()})
        pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle']).to_csv(A.out, index=False); print(A.out, len(rows))
