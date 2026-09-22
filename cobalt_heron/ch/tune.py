"""Grid-search post-processing on OOF prob maps with the exact pooled PQ (at 1024 or 2048)."""
import argparse, itertools, json, pickle
from pathlib import Path
import numpy as np, cv2
from pycocotools import mask as mu
from concurrent.futures import ProcessPoolExecutor
from ch.data import load_coco, record_rles, WORK
from ch.metric import PQAccumulator
from ch.post import instances

GT_CACHE = WORK / 'gt1024.pkl'


def gt1024():
    if GT_CACHE.exists():
        return pickle.load(open(GT_CACHE, 'rb'))
    out = {}
    for stem, rs in load_coco().items():
        recs = []
        for _, anns in rs:
            rl = []
            for r in record_rles(anns):
                m2 = cv2.resize(mu.decode(r).astype(np.float32), (1024, 1024), interpolation=cv2.INTER_AREA) >= 0.5
                rl.append(mu.encode(np.asfortranarray(m2.astype(np.uint8))))
            recs.append(rl)
        out[stem] = recs
    pickle.dump(out, open(GT_CACHE, 'wb')); return out


def lab_to_rles(lab):
    n = lab.max()
    return [mu.encode(np.asfortranarray((lab == i).astype(np.uint8))) for i in range(1, n + 1)]


_G = {}
def _eval(args):
    cfg, files = args
    acc = PQAccumulator()
    for f in files:
        p = np.load(f)['p'].astype(np.float32)
        lab, _ = instances(p[0], p[1], **cfg)
        pr = lab_to_rles(lab)
        for g in _G['gt'][Path(f).stem]:
            acc.add(g, pr)
    return cfg, acc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--probs', required=True); ap.add_argument('--top', type=int, default=15); ap.add_argument('--grid', default='{"t":[0.3,0.4,0.5,0.6],"te":[1.1],"merge":[0,6,12],"min_area":[30,80]}'); ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    _G['gt'] = gt1024()
    files = sorted(Path(a.probs).glob('*.npz'))
    grid = []
    G = json.loads(a.grid)
    for t, te, mg, ma, ms in itertools.product(G['t'], G['te'], G['merge'], G['min_area'], G.get('min_score', [0.0])):
        grid.append(dict(t=t, te=te, merge=mg, min_area=ma, min_score=ms))
    with ProcessPoolExecutor(a.procs) as ex:
        res = list(ex.map(_eval, [(c, files) for c in grid]))
    res.sort(key=lambda r: -r[1].pq)
    for c, acc in res[:a.top]:
        print(json.dumps(c), acc)


if __name__ == '__main__':
    main()
