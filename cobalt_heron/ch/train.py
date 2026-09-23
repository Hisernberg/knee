"""CPU-friendly U-Net training on 1024px cache with random crops."""
import argparse, json, math, random, time, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
from ch.feats import make_input
from ch.data import WORK

def build(enc, dec=(128, 64, 48, 32, 16)):
    return smp.Unet(enc, encoder_weights='imagenet', in_channels=3, classes=2, decoder_channels=dec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)  # -1 = all data
    ap.add_argument('--enc', default='resnet18')
    ap.add_argument('--steps', type=int, default=1100)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--crop', type=int, default=384)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache', default=str(WORK / 'c1024'))
    ap.add_argument('--folds', default=str(WORK / 'folds.json'))
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    folds = json.load(open(a.folds))
    stems = [s for s, f in folds.items() if f != a.fold]
    X, Y, E, P = [], [], [], []
    for s in stems:
        d = np.load(Path(a.cache) / f'{s}.npz')
        X.append(make_input(d['img'])); Y.append(d['soft']); E.append(d['edge'])
        ys, xs = np.nonzero(d['soft'] > 0); P.append((ys, xs))
    print('loaded', len(X), flush=True)
    model = build(a.enc).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    C = a.crop; S = 1024; t0 = time.time(); run = 0
    for step in range(a.steps):
        lr = a.lr * (min(1, (step + 1) / 50)) * (0.5 * (1 + math.cos(math.pi * step / a.steps)) * 0.97 + 0.03)
        for g in opt.param_groups: g['lr'] = lr
        xb, yb = [], []
        for _ in range(a.bs):
            i = random.randrange(len(X)); ys, xs = P[i]
            if len(ys) and random.random() < 0.7:
                k = random.randrange(len(ys)); cy, cx = ys[k] + random.randint(-C // 3, C // 3), xs[k] + random.randint(-C // 3, C // 3)
            else:
                ang = random.random() * 2 * math.pi; r = random.random() ** 0.5 * 440
                cy, cx = int(512 + r * math.sin(ang)), int(512 + r * math.cos(ang))
            y0 = min(max(cy - C // 2, 0), S - C); x0 = min(max(cx - C // 2, 0), S - C)
            x = X[i][:, y0:y0 + C, x0:x0 + C]
            y = np.stack([Y[i][y0:y0 + C, x0:x0 + C], E[i][y0:y0 + C, x0:x0 + C]]).astype(np.float32) / 255
            k = random.randrange(4); x = np.rot90(x, k, (1, 2)); y = np.rot90(y, k, (1, 2))
            if random.random() < 0.5: x = x[:, :, ::-1]; y = y[:, :, ::-1]
            x = x.copy()
            if random.random() < 0.8:
                g = math.exp(random.uniform(-0.3, 0.3)); x = np.clip(x, 0, 1) ** g
                x = x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)
            xb.append(x); yb.append(y.copy())
        xb = torch.from_numpy(np.stack(xb)).float().contiguous(memory_format=torch.channels_last)
        yb = torch.from_numpy(np.stack(yb))
        out = model(xb)
        p = torch.sigmoid(out[:, 0])
        bce = F.binary_cross_entropy_with_logits(out[:, 0], yb[:, 0], pos_weight=torch.tensor(2.0))
        dice = 1 - (2 * (p * yb[:, 0]).sum() + 1) / (p.sum() + yb[:, 0].sum() + 1)
        ebce = F.binary_cross_entropy_with_logits(out[:, 1], yb[:, 1], pos_weight=torch.tensor(3.0))
        loss = bce + dice + 0.5 * ebce
        opt.zero_grad(); loss.backward(); opt.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 50 == 0 or step == a.steps - 1:
            print(f'step {step} loss {run:.4f} lr {lr:.2e} {time.time() - t0:.0f}s', flush=True)
        if (step + 1) % 500 == 0 or step == a.steps - 1:
            torch.save({'enc': a.enc, 'sd': model.state_dict()}, a.out)
    print('saved', a.out)

if __name__ == '__main__':
    main()
