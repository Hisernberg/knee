"""Hypothesis simulator: score every estimator against synthetic truths built from the
real operator, real counts and real priors, under competing hypotheses about how the
organizer generated f*.

Run:  python -m trafficflow.t4.simulate  [--splits validation private] [--seeds 2]
"""
from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from .data import FAMILY, PANELS, SPLITS
from .estimators import ESTIMATORS, Ctx, context
from .metric import score
from .solvers import kl_fit, l2_fit


def common_gravity(x: Ctx) -> np.ndarray:
    """Geometric mean of the three released priors (their shared O x D x d^-1 structure)."""
    return np.exp(np.mean([np.log(np.maximum(x.P.prior[s], 1e-9)) for s in SPLITS], axis=0))


def truths(x: Ctx, rng: np.random.Generator) -> dict:
    """name -> (truth vector, counts the truth is consistent with)."""
    n = len(x.b)
    out = {
        # H_A: truth is a ridge/NNLS fit of the split prior to hidden mainline targets
        #      => it equals the Euclidean projection of b onto {S f = c, f >= 0}.
        "H_A_l2proj(b)": l2_fit(x.S, x.c, x.b),
        # H_B: same, but the generator's prior differs from the released one by LN(eps)
        "H_B_l2proj(b*LN.03)": l2_fit(x.S, x.c, x.b * np.exp(0.03 * rng.standard_normal(n))),
        "H_B_l2proj(b*LN.10)": l2_fit(x.S, x.c, x.b * np.exp(0.10 * rng.standard_normal(n))),
        # H_C / H_D: other projection geometries
        "H_C_klproj(b)": kl_fit(x.S, x.c, x.b),
        "H_D_chi2proj(b)": l2_fit(x.S, x.c, x.b, w=x.b),
        # H_F: truth only shares the gravity structure with b (independent month noise)
        "H_F_l2proj(grav*LN.6)": l2_fit(x.S, x.c, common_gravity(x) * np.exp(0.6 * rng.standard_normal(n))),
    }
    # H_G: released counts carry 2% segment noise around A f*
    ctrue = x.c * (1 + 0.02 * rng.standard_normal(len(x.c)))
    out["H_G_l2proj(b; c+2%noise)"] = l2_fit(x.S, ctrue, x.b)
    return out


def run(splits=("validation", "private"), seeds=1, estimators=None) -> pd.DataFrame:
    estimators = estimators or ESTIMATORS
    rows = []
    for panel in PANELS:
        for split in splits:
            x = context(panel, split)
            Am, cm = x.P.Am(split), x.P.cm(split)
            nz = len(x.P.zones)
            ests = {k: f(x) for k, f in estimators.items()}
            for seed in range(seeds):
                rng = np.random.default_rng(1000 * seed + PANELS.index(panel))
                for hname, fstar in truths(x, rng).items():
                    for ename, f in ests.items():
                        r = score(f, fstar, x.b, Am, cm, x.P.dest, nz)
                        r.update(panel=panel, family=FAMILY[panel], split=split, seed=seed,
                                 hypothesis=hname, estimator=ename)
                        rows.append(r)
    return pd.DataFrame(rows)


def summarise(D: pd.DataFrame, metric: str = "S_ODME") -> pd.DataFrame:
    fam = D.groupby(["hypothesis", "estimator", "split", "seed", "family"])[metric].mean()
    tot = fam.groupby(["hypothesis", "estimator", "split", "seed"]).mean()
    return tot.groupby(["hypothesis", "estimator", "split"]).mean().unstack("split")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["validation", "private"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="/home/user/work/t4/sim_results.csv")
    a = ap.parse_args()
    D = run(a.splits, a.seeds)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    D.to_csv(a.out, index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    for m in ("S_ODME", "S_od", "S_dev", "S_attr", "S_link"):
        print(f"\n===== {m} (family-averaged) =====")
        print(summarise(D, m).round(4))


if __name__ == "__main__":
    main()
