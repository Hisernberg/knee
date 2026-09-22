"""Predict 1024px prob/edge maps (flip TTA) for a list of images, save float16 npz."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from ch.feats import make_input
from ch.train import build


def load(path):
    ck = torch.load(path, map_location='cpu')
    m = build(ck['enc']); m.load_state_dict(ck['sd']); m.eval()
    return m.to(memory_format=torch.channels_last)


@torch.no_grad()
def predict(models, img, tta=True):
    x = torch.from_numpy(make_input(img))[None].contiguous(memory_format=torch.channels_last)
    dev = next(models[0].parameters()).device; x = x.to(dev)
    acc = 0; n = 0
    flips = [(), (3,), (2,), (2, 3)] if tta else [()]
    for m in models:
        for f in flips:
            xi = torch.flip(x, f) if f else x
            o = torch.sigmoid(m(xi))
            acc = acc + (torch.flip(o, f) if f else o); n += 1
    return (acc / n)[0].float().cpu().numpy()  # (2, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--src', required=True, help='dir with <stem>.npz (key img) or <stem>.npy')
    ap.add_argument('--stems', default=None, help='json folds file + fold, e.g. folds.json:0')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-tta', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("CH_THREADS", 4)))
    models = [load(p) for p in a.models]
    if torch.cuda.is_available():
        models = [m.cuda() for m in models]
    src = Path(a.src); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(list(src.glob('*.npz')) + list(src.glob('*.npy')))
    if a.stems:
        fp, fo = a.stems.rsplit(':', 1); folds = json.load(open(fp))
        files = [f for f in files if folds.get(f.stem) == int(fo)]
    for i, f in enumerate(files):
        img = np.load(f)['img'] if f.suffix == '.npz' else np.load(f)
        p = predict(models, img, not a.no_tta)
        np.savez_compressed(out / f'{f.stem}.npz', p=p.astype(np.float16))
        if i % 20 == 0: print(i, len(files), flush=True)


if __name__ == '__main__':
    main()
