"""Learn an instance keep/drop filter on OOF and measure pooled PQ with grouped CV."""
import argparse, json, pickle
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
from sklearn.ensemble import HistGradientBoostingRegressor
from ch.post import instances
from ch.tune import gt1024, lab_to_rles
from ch.inst_feats import inst_features, match_targets, FEATS
from ch.metric import PQAccumulator


def collect(prob_dirs, cfg, cache_dir):
    gt = gt1024(); rows = []
    for f in sorted(Path(prob_dirs[0]).glob('*.npz')):
        p = np.mean([np.load(Path(d) / f.name)['p'].astype(np.float32) for d in prob_dirs], 0)
        img = np.load(Path(cache_dir) / f.name)['img']
        lab, _ = instances(p[0], p[1], **cfg)
        X = inst_features(lab, p[0], p[1], img); pr = lab_to_rles(lab)
        hit, iou = match_targets(pr, gt[f.stem])
        rows.append(dict(stem=f.stem, X=X, hit=hit, iou=iou, pr=pr, gt=gt[f.stem]))
    return rows


def pq_with(rows, keep_fn):
    acc = PQAccumulator()
    for r in rows:
        k = keep_fn(r); pr = [x for x, kk in zip(r['pr'], k) if kk]
        for g in r['gt']: acc.add(g, pr)
    return acc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--probs', nargs='+'); ap.add_argument('--cfg')
    ap.add_argument('--cache', default='/home/user/work/c1024'); ap.add_argument('--save', default=None)
    a = ap.parse_args(); cfg = json.loads(a.cfg)
    rows = collect(a.probs, cfg, a.cache)
    print('instances', sum(len(r['pr']) for r in rows), 'baseline', pq_with(rows, lambda r: [True] * len(r['pr'])))
    # grouped 4-fold CV over images
    idx = np.arange(len(rows)); rng = np.random.RandomState(0); rng.shuffle(idx); grp = np.array_split(idx, 4)
    for r in rows: r['pred'] = np.zeros(len(r['pr']))
    for gi in grp:
        tr = [rows[i] for i in idx if i not in set(gi)]
        X = np.concatenate([r['X'] for r in tr]); y = np.concatenate([r['hit'] for r in tr])
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30).fit(X, y)
        for i in gi:
            if len(rows[i]['pr']): rows[i]['pred'] = m.predict(rows[i]['X'])
    for th in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        print('thr', th, pq_with(rows, lambda r: r['pred'] > th))
    if a.save:
        X = np.concatenate([r['X'] for r in rows]); y = np.concatenate([r['hit'] for r in rows])
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30).fit(X, y)
        pickle.dump({'model': m, 'cfg': cfg, 'feats': FEATS}, open(a.save, 'wb')); print('saved', a.save)


if __name__ == '__main__':
    main()
