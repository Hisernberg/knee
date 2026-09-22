"""Test prob maps (1024) -> 2048 disjoint instances -> submission.csv."""
import argparse, json, pickle
from pathlib import Path
import numpy as np, cv2, pandas as pd
from scipy import ndimage as ndi
from pycocotools import mask as mu
from ch.post import instances
from ch.inst_feats import inst_features


def to2048(lab, prob, t, mode=None):
    """Upsample 1024 label map to 2048 using the smooth upsampled prob as the fg support."""
    import os
    mode = mode or os.environ.get('CH_UP', 'prob')
    if mode == 'nearest':
        return cv2.resize(lab.astype(np.float32), (2048, 2048), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    if mode.startswith('prob'):
        t = t * float(mode[4:]) if len(mode) > 4 else t
    p2 = cv2.resize(prob, (2048, 2048), interpolation=cv2.INTER_LINEAR)
    l2 = cv2.resize(lab.astype(np.int32).astype(np.float32), (2048, 2048), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    fg = (p2 > t) & (cv2.dilate((l2 > 0).astype(np.uint8), np.ones((5, 5))) > 0)
    _, (iy, ix) = ndi.distance_transform_edt(l2 == 0, return_indices=True)
    return np.where(fg, l2[iy, ix], 0)


def rows_for(stem, lab2, scores=None):
    rows = []
    for k in range(1, lab2.max() + 1):
        m = lab2 == k
        if m.sum() == 0: continue
        e = mu.encode(np.asfortranarray(m.astype(np.uint8)))
        r = {'filament_id': f'{stem}_{len(rows) + 1}', 'segmentation_rle': e['counts'].decode('ascii')}
        if scores is not None: r['score'] = float(scores[k - 1])
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--probs', nargs='+', required=True)
    ap.add_argument('--cfg', default=None); ap.add_argument('--filt', default=None); ap.add_argument('--thr', type=float, default=0.35)
    ap.add_argument('--imgs', default='/home/user/work/test1024'); ap.add_argument('--keep-scores', action='store_true'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    F = pickle.load(open(a.filt, 'rb')) if a.filt else None
    cfg = F['cfg'] if F else json.loads(a.cfg)
    files = sorted(Path(a.probs[0]).glob('*.npz')); rows = []
    for f in files:
        p = np.mean([np.load(Path(d) / f.name)['p'].astype(np.float32) for d in a.probs], 0)
        lab, _ = instances(p[0], p[1], **cfg)
        if F is not None and lab.max() > 0:
            img = np.load(Path(a.imgs) / f'{f.stem}.npy')
            sc = F['model'].predict(inst_features(lab, p[0], p[1], img)); keep = sc > a.thr
            lut = np.zeros(lab.max() + 1, np.int32); lut[1:][keep] = np.arange(1, keep.sum() + 1); lab = lut[lab]
            sc = sc[keep]
        else:
            sc = np.ones(lab.max())
        rows += rows_for(f.stem, to2048(lab, p[0], cfg['t']), sc if a.keep_scores else None)
    df = pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle'] + (['score'] if a.keep_scores else [])); df.to_csv(a.out, index=False)
    print(a.out, len(files), 'images', len(df), 'instances')


if __name__ == '__main__':
    main()
