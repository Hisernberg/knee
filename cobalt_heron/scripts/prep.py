"""Cache 1024px grayscale images + soft consensus masks + folds."""
import sys; sys.path.insert(0, '.')
import numpy as np, cv2, json
from concurrent.futures import ProcessPoolExecutor
from pycocotools import mask as mu
from ch.data import *

S = 1024
recs = load_coco()
stems = sorted(recs)

def one(stem):
    img = read_gray(train_path(stem))
    small = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
    soft = np.zeros((S, S), np.float32); edge = np.zeros((S, S), np.float32)
    for _, anns in recs[stem]:
        rl = record_rles(anns)
        m = np.zeros((H, W), np.uint8); lab = np.zeros((H, W), np.int32)
        for k, r in enumerate(rl, 1):
            d = mu.decode(r).astype(bool); m |= d; lab[d] = k
        soft += cv2.resize(m.astype(np.float32), (S, S), interpolation=cv2.INTER_AREA)
        # touching-boundary map between different instances
        dil = cv2.dilate(lab.astype(np.float32), np.ones((5, 5)), iterations=1)
        ero = -cv2.dilate(-lab.astype(np.float32), np.ones((5, 5)), iterations=1)
        b = ((dil != ero) & (lab > 0)).astype(np.float32)
        edge += cv2.resize(b, (S, S), interpolation=cv2.INTER_AREA)
    n = len(recs[stem])
    np.savez_compressed(WORK / 'c1024' / f'{stem}.npz', img=small, soft=(soft / n * 255).astype(np.uint8),
                        edge=(np.clip(edge / n, 0, 1) * 255).astype(np.uint8))
    return stem

if __name__ == '__main__':
    (WORK / 'c1024').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(4) as ex:
        for i, s in enumerate(ex.map(one, stems)):
            if i % 100 == 0: print(i, flush=True)
    # folds grouped by month (YYYYMM)
    months = sorted({s[:6] for s in stems}); rng = np.random.RandomState(0); rng.shuffle(months)
    fm = {m: i % 5 for i, m in enumerate(months)}
    json.dump({s: fm[s[:6]] for s in stems}, open(WORK / 'folds.json', 'w'))
    tc = WORK / 'test1024'; tc.mkdir(exist_ok=True)
    for p in test_paths():
        np.save(tc / f'{p.stem}.npy', cv2.resize(read_gray(p), (S, S), interpolation=cv2.INTER_AREA))
    print('done')
