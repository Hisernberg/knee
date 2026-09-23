"""Per-instance features + OOF match targets for a learned keep/drop filter."""
import numpy as np, cv2
from scipy import ndimage as ndi
from pycocotools import mask as mu

FEATS = ['area', 'pmean', 'pmax', 'p90', 'emax', 'elong', 'r', 'contrast', 'bbox_fill', 'n_nb',
         'p50', 'f07', 'major', 'minor', 'dark_rel', 'nn_dist', 'n_inst', 'rim_p']


def inst_features(lab, prob, edge, img):
    n = lab.max(); out = []
    if n == 0: return np.zeros((0, len(FEATS)), np.float32)
    objs = ndi.find_objects(lab)
    cy, cx = np.array(lab.shape) / 2
    bg = cv2.medianBlur(img, 31).astype(np.float32)
    cents = []
    for k, sl in enumerate(objs, 1):
        m = lab[sl] == k; pv = prob[sl][m]; ev = edge[sl][m]
        ys, xs = np.nonzero(m)
        area = m.sum()
        if area >= 5:
            cov = np.cov(np.stack([ys, xs])); w = np.sort(np.linalg.eigvalsh(cov))
            elong = np.sqrt(max(w[1], 1e-3) / max(w[0], 1e-3))
        else:
            elong = 1.0
        yy = ys.mean() + sl[0].start; xx = xs.mean() + sl[1].start; cents.append((yy, xx))
        r = np.hypot(yy - cy, xx - cx) / (lab.shape[0] / 2)
        contrast = float((bg[sl][m] - img[sl][m].astype(np.float32)).mean())
        major = 4 * np.sqrt(max(w[1], 1e-3)) if area >= 5 else 1.0
        minor = 4 * np.sqrt(max(w[0], 1e-3)) if area >= 5 else 1.0
        disk_med = float(np.median(img[img > 30])) if (img > 30).any() else 128.0
        dark_rel = float(img[sl][m].astype(np.float32).mean() / max(disk_med, 1))
        y0, y1, x0, x1 = max(sl[0].start - 6, 0), sl[0].stop + 6, max(sl[1].start - 6, 0), sl[1].stop + 6
        ring = (lab[y0:y1, x0:x1] == 0)
        rim_p = float(prob[y0:y1, x0:x1][ring].mean()) if ring.any() else 0.0
        out.append([area, pv.mean(), pv.max(), np.percentile(pv, 90), ev.max(), elong, r, contrast,
                    area / m.size, 0, np.median(pv), (pv > 0.7).mean(), major, minor, dark_rel, 0, n, rim_p])
    c = np.array(cents)
    if len(c) > 1:
        d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
        for i in range(len(out)):
            out[i][9] = int(((d[i] < 60) & (d[i] > 0)).sum())
            out[i][15] = float(np.sort(d[i])[1])
    else:
        for i in range(len(out)): out[i][15] = 2000.0
    return np.array(out, np.float32)


def match_targets(pred_rles, gt_records):
    """Per prediction: fraction of records where it hits (IoU>0.5), and mean IoU when hit."""
    n = len(pred_rles); hits = np.zeros(n); ious = np.zeros(n)
    for g in gt_records:
        if not g or not n: continue
        iou = np.asarray(mu.iou(pred_rles, g, [0] * len(g)))  # (n, ng)
        best = iou.max(1); h = best > 0.5
        hits += h; ious += np.where(h, best, 0)
    R = max(len(gt_records), 1)
    return hits / R, np.where(hits > 0, ious / np.maximum(hits, 1), 0)
