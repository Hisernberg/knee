# cobalt-heron: YOLO-seg trainer (fold-0 holdout model + full-data model), outputs raw instance preds.
import subprocess, sys, os, json, glob, random, time
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics', 'pycocotools'], check=False)
from pathlib import Path
import numpy as np, cv2, torch
from pycocotools import mask as mu

MODEL = os.environ.get('CH_YOLO', 'yolo11m-seg.pt'); IMGSZ = int(os.environ.get('CH_IMGSZ', 1536))
EPOCHS = int(os.environ.get('CH_EPOCHS', 60)); RUNS = os.environ.get('CH_RUNS', 'f0,full').split(',')
ann = glob.glob('/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json', recursive=True)[0]
ROOT = Path(ann).parent.parent; TR = ROOT / 'train' / 'train_images'; TE = ROOT / 'test' / 'test_images'
W = Path('/kaggle/working'); OUT = W / 'out'; OUT.mkdir(exist_ok=True)
d = json.load(open(ann)); H = 2048
anns = {}
for a in d['annotations']: anns.setdefault(a['image_id'], []).append(a)
recs = {}
for im in d['images']: recs.setdefault(Path(im['file_name']).stem, []).append(im['id'])
stems = sorted(recs)
months = sorted({s[:6] for s in stems}); rng = np.random.RandomState(0); rng.shuffle(months)
fm = {m: i % 5 for i, m in enumerate(months)}; folds = {s: fm[s[:6]] for s in stems}

def rle(a): return mu.merge(mu.frPyObjects(a['segmentation'], H, H))
def medoid(stem):
    ids = recs[stem]
    if len(ids) == 1: return ids[0]
    U = [mu.merge([rle(a) for a in anns.get(i, [])]) if anns.get(i) else mu.encode(np.zeros((H, H), np.uint8, order='F')) for i in ids]
    M = np.asarray(mu.iou(U, U, [0] * len(U))); return ids[int(np.argmax(M.sum(1)))]

def poly_lines(stem):
    lines = []
    for a in anns.get(medoid(stem), []):
        m = mu.decode(rle(a))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if cv2.contourArea(c) < 4 or len(c) < 3: continue
            c = c.reshape(-1, 2) / H
            lines.append('0 ' + ' '.join(f'{v:.6f}' for v in c.ravel()))
    return lines

LAB = {s: poly_lines(s) for s in stems}
print('labels ready', flush=True)

def make_ds(name, train_stems, val_stems):
    base = W / 'ds' / name
    for split, ss in (('train', train_stems), ('val', val_stems)):
        (base / split / 'images').mkdir(parents=True, exist_ok=True); (base / split / 'labels').mkdir(parents=True, exist_ok=True)
        for s in ss:
            dst = base / split / 'images' / f'{s}.jpg'
            if not dst.exists(): os.symlink(TR / f'{s}.jpeg', dst)
            (base / split / 'labels' / f'{s}.txt').write_text('\n'.join(LAB[s]))
    y = base / 'data.yaml'; y.write_text(f'path: {base}\ntrain: train/images\nval: val/images\nnames:\n  0: filament\n'); return y

def predict(model, paths, fn):
    res = {}
    for p in paths:
        r = model.predict(str(p), imgsz=IMGSZ, conf=0.03, iou=0.7, retina_masks=True, verbose=False, max_det=150)[0]
        items = []
        if r.masks is not None:
            ms = r.masks.data.cpu().numpy() > 0.5; cf = r.boxes.conf.cpu().numpy()
            for m, c in zip(ms, cf):
                if m.shape != (H, H): m = cv2.resize(m.astype(np.uint8), (H, H), interpolation=cv2.INTER_NEAREST) > 0
                e = mu.encode(np.asfortranarray(m.astype(np.uint8)))
                items.append({'conf': float(c), 'rle': e['counts'].decode()})
        res[Path(p).stem] = items
    json.dump(res, open(OUT / fn, 'w'))

from ultralytics import YOLO
ngpu = torch.cuda.device_count(); dev = ','.join(str(i) for i in range(ngpu)) if ngpu else 'cpu'
print('device', dev, flush=True)
for run in RUNS:
    t0 = time.time()
    if run == 'f0':
        y = make_ds(run, [s for s in stems if folds[s] != 0], [s for s in stems if folds[s] == 0])
    else:
        y = make_ds(run, stems, [s for s in stems if folds[s] == 0][:40])
    m = YOLO(MODEL)
    m.train(data=str(y), imgsz=IMGSZ, epochs=EPOCHS, batch=4 if IMGSZ > 1024 else 8, device=dev, workers=4,
            project=str(W / 'runs'), name=run, degrees=10, flipud=0.5, fliplr=0.5, mosaic=0.5, close_mosaic=10,
            scale=0.3, hsv_h=0, hsv_s=0, hsv_v=0.3, overlap_mask=False, mask_ratio=2, patience=100, plots=False, amp=True, cos_lr=True)
    best = W / 'runs' / run / 'weights' / 'last.pt'
    os.system(f'cp {best} {OUT}/{run}_last.pt')
    m = YOLO(str(best))
    if run == 'f0':
        predict(m, [TR / f'{s}.jpeg' for s in stems if folds[s] == 0], 'f0_val_preds.json')
    predict(m, sorted(TE.glob('*.jpeg')), f'{run}_test_preds.json')
    print(run, 'done', time.time() - t0, flush=True)
