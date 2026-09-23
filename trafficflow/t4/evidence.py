"""Reverse-engineering diagnostics for the Task 4 generator (prints a report).

    python -m trafficflow.t4.evidence  > /home/user/work/t4/evidence.txt

E1 corridor/screenline structure      E4 (q - c) explained by projection duals
E2 prior structure and noise          E5 bound on a hidden prior perturbation
E3 counts vs simulated PM flows       E6 published baseline numbers reproduced
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from .data import FAMILY, PANELS, SPLITS, load
from .estimators import context
from .metric import score
from .obs import mainline_pm
from .solvers import kl_fit, l2_fit, ridge_nn
from .structure import link_segment, segment_counts, segment_matrix


def e1_structure():
    print("\n## E1 structure")
    for p in PANELS:
        P = load(p)
        nz = len(P.zones)
        seg = link_segment(P)
        meas = P.measured("validation")
        screen = int((seg[meas] >= 0).sum())
        same = all(np.ptp(v) == 0 for s in SPLITS for v in segment_counts(P, s).values())
        print(f"{p:11s} zones={nz:3d} paths={len(P.path_ids):5d} (= nz(nz-1)/2: {len(P.path_ids) == nz*(nz-1)//2}) "
              f"measured={int(meas.sum()):3d} screenlines={screen} segments_covered={len(set(seg[meas]))}/{nz-1} "
              f"rank(A_meas)={np.linalg.matrix_rank(P.Am('validation'))} counts identical within segment={same}")


def e2_prior():
    print("\n## E2 prior: log b = alpha_o + beta_d + g(d-o) + noise;  A b / c profile")
    for p in PANELS:
        P = load(p)
        nz, n = len(P.zones), len(P.path_ids)
        L = np.log(np.array([P.prior[s] for s in SPLITS]))
        X = np.zeros((n, 2 * nz + 1))
        X[np.arange(n), P.orig] = 1
        X[np.arange(n), nz + P.dest] = 1
        X[:, -1] = np.log(P.dest - P.orig)
        beta, *_ = np.linalg.lstsq(X, L.mean(0), rcond=None)
        r_mean = (L.mean(0) - X @ beta).std()
        r_split = np.mean([(L[i] - X @ np.linalg.lstsq(X, L[i], rcond=None)[0]).std() for i in range(3)])
        sig = np.sqrt(max(r_split ** 2 - r_mean ** 2, 0) * 1.5)
        cc = np.corrcoef(L)[np.triu_indices(3, 1)].mean()
        S = segment_matrix(P)
        sc = segment_counts(P, "validation")
        c = np.array([sc[k][0] for k in range(nz - 1)])
        prof = S @ P.prior["validation"] / c
        print(f"{p:11s} ln(d) coef={beta[-1]:.2f} resid(meanprior)={r_mean:.3f} resid(split)={r_split:.3f} "
              f"=> per-split noise sigma~{sig:.2f}; xsplit logcorr={cc:.3f}; Ab/c ends={prof[0]:.2f},{prof[-1]:.2f} "
              f"max={prof.max():.2f}")


def _seg_q(P, split):
    seg = link_segment(P)
    ns = len(P.zones) - 1
    m = mainline_pm(P.panel, split, split == "train")
    qa = pd.Series(m["mean"]).reindex(P.link_ids).to_numpy()
    use = np.isfinite(qa) & (seg >= 0)
    q = np.array([qa[use & (seg == k)].mean() for k in range(ns)])
    nm = np.array([(P.measured(split) & (seg == k)).sum() for k in range(ns)], float)
    sc = segment_counts(P, split)
    c = np.array([sc[k][0] for k in range(ns)])
    return q, c, nm


def e3_e4_e5():
    print("\n## E3/E4/E5 counts vs simulated PM (15-19h) mainline means q;  q - c = lam * mu/n_k ?")
    rng = np.random.default_rng(7)
    for p in PANELS:
        P = load(p)
        S = segment_matrix(P)
        for split in SPLITS:
            q, c, nm = _seg_q(P, split)
            y = q - c
            r = np.log(c / q)
            out = [f"{p:11s} {split[:3]} median c/q={np.median(c / q):.3f} std log(c/q)={r.std():.3f} "
                   f"ends={r[0]:+.2f},{r[-1]:+.2f}"]
            res = {}
            for name, b in [("L2", P.prior[split]),
                            ("L2-wrongprior", P.prior["private" if split != "private" else "validation"])]:
                f, mu = l2_fit(S, c, b, return_dual=True)
                x = mu / nm
                lam = (x @ y) / (x @ x)
                res[name] = (1 - np.sum((y - lam * x) ** 2) / np.sum(y ** 2), lam, y - lam * x, mu)
            f, mu = kl_fit(S, c, P.prior[split], return_dual=True)
            x = mu * c / nm
            lam_kl = (x @ y) / (x @ x)
            r2_kl = 1 - np.sum((y - lam_kl * x) ** 2) / np.sum(y ** 2)
            f, mu = l2_fit(S, c, P.prior[split], w=P.prior[split], return_dual=True)
            x = mu / nm
            r2_chi = 1 - np.sum((y - (x @ y) / (x @ x) * x) ** 2) / np.sum(y ** 2)
            r2, lam, e, mu0 = res["L2"]
            out.append(f"R2: L2={r2:.3f} (lam={lam:.1f}) L2(wrong prior)={res['L2-wrongprior'][0]:.3f} "
                       f"KL={r2_kl:.3f} CHI2={r2_chi:.3f}")
            # E5: roughness a hidden LN(eps) prior perturbation would add
            rough = []
            for eps in (0.02, 0.05):
                _, mu2 = l2_fit(S, c, P.prior[split] * np.exp(eps * rng.standard_normal(len(P.path_ids))),
                                return_dual=True)
                rough.append(np.diff(lam * (mu0 - mu2) / nm).std())
            out.append(f"rough(diff e)={np.diff(e).std():.0f} vs eps.02={rough[0]:.0f} eps.05={rough[1]:.0f}")
            print(" | ".join(out))


def e6_published():
    print("\n## E6 published numbers under H_A (truth = Euclidean projection of the split prior)")
    rows = []
    for p in PANELS:
        for split in ("validation", "private"):
            x = context(p, split)
            fstar = l2_fit(x.S, x.c, x.b)
            Am, cm, nz = x.P.Am(split), x.P.cm(split), len(x.P.zones)
            other = "private" if split == "validation" else "validation"
            cand = {
                "official baseline (train c, split b, lam .05)": (ridge_nn(x.S, x.nk, x.c_train, x.b, 0.05), x.b),
                "old recipe (split c, OTHER month b)": (ridge_nn(x.S, x.nk, x.c, x.P.prior[other], 0.05),
                                                        x.P.prior[other]),
            }
            for k, (f, bb) in cand.items():
                r = score(f, fstar, bb, Am, cm, x.P.dest, nz)
                r.update(panel=p, family=FAMILY[p], split=split, est=k)
                rows.append(r)
    D = pd.DataFrame(rows)
    fam = D.groupby(["est", "split", "family"])[["S_od", "S_link", "S_dev", "S_attr", "S_ODME"]].mean()
    print(fam.groupby(["est", "split"]).mean().round(4).to_string())
    d = D[(D.panel == "D12_I5_N") & (D.split == "validation") & D.est.str.startswith("old")]
    print(f"old recipe D12_I5_N validation S_od = {d.S_od.iloc[0]:.4f}  (commit 7c7269b reports 0.308665)")
    print("published: baseline S_ODME 0.8359 (validation); 'the two leaderboards agree to 0.008'")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    e1_structure()
    e2_prior()
    e3_e4_e5()
    e6_published()
