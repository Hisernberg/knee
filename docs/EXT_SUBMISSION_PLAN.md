# Extended-Raptor submission plan (2026-09-22)

Prepared for the Kaggle account `kragglenote2forwork`, competition `rsna-knee-abnormality-detection`
(4,185 teams, deadline 2026-10-22). Everything below was measured through the Kaggle API and the public
notebooks/datasets themselves; every number is attributed.

## 1. Where the account stands

| Item | Fact |
|---|---|
| Public LB | **0.942, rank 304 of 4,185** (5 submissions used, all on 2026-09-21) |
| Best scored submission | notebook `rsna-knee-dinosaur-v5` **version 1**, a fork of Roman Tamrazov's *RSNA Knee \| DINOsaur V5?* (= Mattia Angeli's public 0.943 *Speedy Raptors* stack + a home-trained ConvNeXt arm) |
| Versions 2-5 | "unhandled error while rerunning your code" on the hidden set. They are an experimental *MAST v4* rewrite that turned the stack's normal fall-backs (a study failing to decode, a fallback member, a neutral fill) into fatal `RuntimeError`s; the 3-study public run passes, the 1,300-study hidden run does not |
| Baseline to keep | v1 = 0.942. The ConvNeXt arm is weak (below the ~0.90 solo level at which an extra member starts to pay), so v1 sits 0.001 **below** its own anchor |

Score to rank (public LB, 2026-09-22):

| Public score | Rank | Teams at that score |
|---|---|---|
| 0.958 | 1 | 1 |
| 0.956 | 2 | 1 |
| 0.955 | 3 | 8 |
| 0.950 | 39 | 11 |
| 0.946 | 80 | 11 |
| 0.945 | 91 | 29 |
| 0.944 | 120 | 28 |
| 0.943 | 148 | 100 |
| 0.942 | 248 | 269 |
| 0.941 | 517 | 420 |

Rank 1-3 needs **0.956+** on the current board (ties at 0.955 hold ranks 3-10).

## 2. What the public assets can and cannot do

* Every top public notebook (Mattia Angeli, Evgeniy Dvorkin, maverickss26, Jiwei Liu, prvsiyan, renta.k ...) is the
  same graph: 20 DINOv2-small members + 5 DINOv3 folds + RadImageNet heads ("transformer stack", OOF macro-AUC
  0.84-0.86) blended with a CoAtNet "Raptor" family (public solo 0.917-0.932). Forks plateau at **0.939-0.943**.
* Measured on the public data: DINO stage OOF 0.840 on gold-58 / 0.850 vs weak labels; RadImageNet heads OOF 0.842 /
  0.815; their blend peaks at 0.861 gold-58. The Raptor/CoAt family carries the score.
* Adding a member below ~0.90 solo **lowers** the blend (starkhushi: home 2.5D members at 15 % weight -> 0.936);
  swapping backbones on the same labels/geometry is worth ~+0.001; a differently-sampled corpus is worth 0.006-0.011
  for a single model (dreaddevelopment's own ablation).
* The strongest public single model, `dreaddevelopment/raptor-knee-finespacing` (v9: 80 slices, 2-98 % span, 140 mm,
  **0.932 public solo**), and `raptor-knee-widedense` (v4: 64 slices, 6-94 %, **0.927 at 62 windows**) are used by
  **no** 0.943 fork. Their author explicitly recommends ensembling them with the max-span (v5) checkpoint.
* The "0.957" in the title of `kminsher/rsna-fast-parent-0-957` is a local diagnostic number, not a leaderboard
  score; the notebook is the 0.939/0.941 recipe.
* Label tables: `stevenleehans` v4 blend is the best clean LLM teacher on gold-58 (0.893); `yunusgmsoy` v5 and both
  `barun2104` tables copy the 58 gold labels verbatim (AUC 1.000) and must never be used for validation.
* Teams at 0.955-0.958 (Scott Willis 151 submissions, takoi & charmq 149, dalab 122 ...) train their own model
  families over weeks with full out-of-fold validation. No combination of public checkpoints has reached 0.947.

**Honest expectation:** with public assets the ceiling is about 0.944-0.946 (rank ~80-120). Rank 1-3 is not reachable
this way, and this document does not claim otherwise.

## 3. The submission notebook that is ready

`kaggle/ext/notebook/rsna-knee-raptor-ext-v1.ipynb` (+ `kernel-metadata.json`), built by
`kaggle/ext/build_ext_notebook.py` from Mattia Angeli's public notebook (`kaggle/ext/source/`, Apache-2.0):

1. the complete 0.943 graph runs unchanged and its output is kept as `submission_anchor.csv`;
2. the Raptor worker also scores v9 and v4 at their training geometry, with weight 0.0 in the anchor blend (the anchor
   stays byte-identical); a failure of an extra view is logged and isolated, never fatal;
3. a CPU cell (`kaggle/ext/variants_cell.py`) rebuilds the anchor from its saved parts and refuses to write variants
   unless the reproduction is exact, then writes five files. Verified locally on the real 3-study outputs of the
   account's own public run: parity 0.0, six-view path and failed-view fallback both exercised.

One GPU run therefore yields five submissions (code competition: a submission is a notebook version + an output file):

| File | What it is | Expected public LB |
|---|---|---|
| `submission_anchor.csv` (C1) | exact Speedy Raptors 0.943 recipe | **0.943** (the source notebook's own score; rank ~148, +0.001 vs baseline) |
| `submission_main.csv` (C2, = `submission.csv`) | public-Raptor stage widened from 4 to 6 views: v5 .40, v9 .20, v8 .14, v4 .10, v10 .08, v5-reverse .08; rest unchanged | **0.943-0.945** (median 0.944; rank ~90-150) |
| `submission_c3.csv` (C3) | balanced six views + CoAt family weighted by gold-58 quality (Global96 .45 / D4 .30 / residual .25), family share .45 | 0.942-0.946 (higher variance) |
| `submission_c4.csv` (C4) | C2 with a flat 0.65 outer weight (no per-target LB probing) | 0.941-0.944 on public, the safer private-LB pick |
| `submission_c5.csv` (C5) | C2 with CoAt family share 0.50 | 0.942-0.945 |

Public-LB resolution is about +/-0.001 (roughly 400 hidden studies), so C2-C5 mainly tell us the *direction* for the
next run. Final selection rule: C1 or C2 as the safe pick, C4 as the second pick for the private 70 %.

Estimated runtime on 2xT4: 3-5 h (the 4-view stack runs in about 90 min plus the D4/Global96 children; two extra
views add roughly 35-45 min). Budget guard: 8 h inside the notebook.

## 4. Blocker: weekly GPU quota

`kaggle kernels push` was refused with **"Maximum weekly GPU quota of 30.00 hours reached"**. Kaggle's quota resets
every **Saturday at 00:00 UTC** (next: 2026-09-26 00:00 UTC). Nothing GPU-based can run before that; the notebook's
graph needs exactly two GPUs, so a CPU dry run is not possible either. Submitting an already-finished notebook version
needs no GPU, but the account has no unsubmitted finished version worth submitting.

## 5. Runbook (one command once quota is back)

```bash
pip install kaggle pandas numpy            # credentials: ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
python kaggle/ext/run_and_submit.py --build-dir kaggle/ext/notebook          # push -> poll -> submit 5 files -> scores
python kaggle/ext/run_and_submit.py --build-dir kaggle/ext/notebook --submit-only --version 1 --variants c3,c4   # later
```
`kaggle/ext/run_log.jsonl` records every push, status, submission and score. Rebuild from a newer upstream notebook with
`kaggle kernels pull mattiaangeli/bend-the-knee-to-speedy-raptors-the-original -p src -m` then
`python kaggle/ext/build_ext_notebook.py --source src/<file>.ipynb --out kaggle/ext/notebook --owner <user> --title "RSNA Knee Raptor Ext V1"`.

GPU budget for the week after the reset (30 h): the first run costs ~4-5 h and yields five submissions. Two further
runs (re-weighted variants chosen from the first scores) fit comfortably. Training a new arm on the pre-decoded
`knee-raptor-corpus` (44-slice, 15-85 % geometry, the weakest of the family) would cost 8-10 h on T4x2 for a member
expected at ~0.90-0.92 solo, i.e. at best +0.001; it is not recommended while quota is the constraint.

## 6. What top-3 would actually take

Own CoAtNet-class family trained on a dense, wide-span 80-96 slice corpus at 336-384 px with clean LLM labels and
5-fold OOF on all 4,407 studies (to pick blend weights honestly), several seeds/geometries, plus the public stack as a
minority member. That is roughly 100-200 GPU-hours on modern hardware and cannot be done inside a 30 h/week T4 quota.
