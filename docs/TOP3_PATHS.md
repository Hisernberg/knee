# Two realistic paths to a top-3 finish (RSNA Knee Abnormality Detection), and how each is verified

Written 2026-09-22. Board: rank 1 = 0.958, rank 2 = 0.956, ranks 3-10 = 0.955; best public fork = 0.943; this
account = 0.942. Deadline 2026-10-22 (team-merger deadline 2026-10-15). No public-asset combination has exceeded
about 0.946, so both paths below train new models. Neither is a certainty; each has an explicit go/no-go gate.

## The one thing every path needs: a dense corpus and honest validation

* **Corpus96** (`kaggle/corpus/build_corpus96.py`, running now as three CPU-only Kaggle kernels
  `rsna-knee-corpus96-part0..2`, no GPU quota used): every training study as 96 slices x 336 px, 140 mm crop,
  2-98 % span, slot widths 26/22/18/12/18. This is the geometry of the strongest public single models
  (finespacing 0.932 solo, max-span 0.928 solo). Output ~48 GB in three parts, attachable to any Kaggle kernel.
* **Labels**: `kaggle/corpus/labels_consensus.csv`, the probability mean of the two clean public LLM label tables
  (0.894 macro-AUC on gold-58; the two tables that copy gold verbatim are excluded). Confidence weights: 1.0 when
  both teachers commit, 0.7 when one is unknown, 0.3 when both are unknown.
* **Validation**: the 58 expert-labelled studies are never trained on (per-epoch gate, SE about 0.02), and every
  model writes out-of-fold predictions on the 4,349 report-only studies (fold = stable hash of the UID). Blend
  weights are chosen on OOF (n = 4,349, resolution about 0.001), never on the public leaderboard.
* **Trainer**: `kaggle/corpus/train_raptor96.py` (CoAtNet-2 384 px 2.5D windows, per-finding attention MIL,
  weighted soft BCE, one-cycle AdamW, AMP, DataParallel, top-3 SWA). Checkpoints use the public Raptor key layout
  (`model/arch/res/lab/epoch/gold_auc`), so a trained fold drops into the extended submission notebook as one
  more Raptor view with no new inference code.

## Path 1: own CoAtNet-2 family on the dense corpus, rented GPU (fast, ~$30-80)

What: 5 folds x 12 epochs of `train_raptor96.py`, two seeds if budget allows; blend by OOF; ship weights as a
private Kaggle dataset (allowed by the rules; must be made public if it wins).

| Hardware | GPU-hours (5 folds) | Cost | Wall-clock |
|---|---|---|---|
| Kaggle T4x2 only | ~145 | $0 | ~5 weeks of quota: does not fit the deadline |
| Kaggle T4x2 + Colab Pro+ link (+30 h/week) | ~145 | $50/month | ~2.5 weeks if the promotion still applies |
| RunPod / Vast.ai RTX 4090 x5 in parallel | ~53 | $18-40 | one evening |
| RunPod / Vast.ai A100 80 GB x5 | ~43 | $30-70 | one evening |
| Colab Pro+ A100 (background execution) | ~43 | ~500 compute units | 2-3 sessions |

Expected effect (estimate, labelled as such): the public finespacing checkpoint alone is 0.932 on the LB with a
single epoch-selected model and 4,349-study labels; a 5-fold, SWA, dense-corpus family with consensus labels and
flip augmentation should land at 0.935-0.945 solo. Blended with the public stack (which stays a minority member),
the honest expectation is **0.947-0.952** (rank ~25-70), with top-3 possible only if the new family is markedly
stronger than the public one. Go/no-go gate after fold 0: gold-58 >= 0.92 and OOF vs consensus labels >= 0.90;
otherwise stop spending.

Runbook (any Linux GPU box):
```bash
pip install torch torchvision timm pandas scikit-learn
# download the three corpus parts (kaggle kernels output kragglenote2forwork/rsna-knee-corpus96-partK -p corpus/partK)
for f in 0 1 2 3 4; do python kaggle/corpus/train_raptor96.py --corpus-dirs corpus/part0,corpus/part1,corpus/part2 \
   --train-csv data/train.csv --labels kaggle/corpus/labels_consensus.csv --fold $f --epochs 12 --bs 4 --accum 2 --out runs/f$f; done
python scripts/blend_oof.py --runs runs --train-csv data/train.csv      # OOF blend report
kaggle datasets create -p runs_upload                                    # private weights dataset
```
Then add the five `raptor96_swa.pt` files as views in `kaggle/ext/build_ext_notebook.py` (same ARMS mechanism as
v9/v4) and re-blend on OOF.

## Path 2: self-distillation from the public ensemble plus a stronger teacher (free, slower)

What: use Kaggle's separate TPU quota (20 h/week, v5e-8, PyTorch/XLA) or the weekly T4 budget for a reduced
schedule (3 folds x 8 epochs x 48 slices, ~25 T4x2 hours), and improve the *targets* rather than only the model:

1. Run the public 0.943 ensemble on the 4,407 training studies (one inference run, ~5 T4x2 hours) to get image
   based pseudo-labels; blend them with the report consensus (report labels under-report incidental findings such
   as effusion, synovitis and contusion, exactly where image models beat reports on gold-58).
2. Train the corpus96 model on the blended targets (self-distillation). In past RSNA competitions this style of
   pseudo-labelling was worth +0.003 to +0.008 for weakly labelled targets (see the technique survey below).
3. Blend on OOF; the gold-58 gate stays the veto.

Expected effect (estimate): 0.945-0.950. It is the cheaper path, and it composes with Path 1 (same corpus, same
trainer, different label file).

## What is not a path

* More public checkpoints: everything above 0.92 solo is already in the extended notebook; the rest are 0.88-0.917
  and lower the blend.
* Per-target weight probing on the public leaderboard: +/-0.001 and it overfits the 30 % public split.
* Re-labelling reports with a hosted LLM API: the competition rules require locally run, freely accessible models.

## Schedule that fits the deadline

| When | Action | GPU source |
|---|---|---|
| now | corpus96 parts (running), trainer, labels: done | Kaggle CPU |
| Sat 2026-09-26 | scheduled: extended-Raptor inference run, five submissions (anchor/main/c3/c4/c5) | Kaggle T4x2 (~5 h) |
| Sat-Sun | Path 1 fold 0 on a rented 4090 (gate) or Kaggle TPU/T4 reduced run | external / TPU |
| week of 09-28 | remaining folds, OOF blend, weights dataset, extended notebook v2 with the new family | external |
| 10-05 to 10-15 | Path 2 pseudo-label round if fold-0 gate passed; final two selections chosen on OOF | Kaggle T4x2 |

Compute facts and prices above come from the cited Kaggle docs and vendor pages in the research notes
(`docs/research/compute_feasibility.md`).

## Update after the technique survey (2026-09-22, see docs/research/techniques.md)

The public evidence changes the priority order:

1. **Labels are the lever.** Every 0.947-0.958 single-model result in the "best single model" thread credits label
   work, not encoder size (DINOv2-S -> B was +0.001, within noise). Measured here on gold-58: treating "not
   addressed" as absent for ACL, MCL, Baker's and fracture and imputing silent synovitis from the effusion field
   lifts the consensus table from 0.893 to **0.900** (fracture 0.815 -> 0.897). `kaggle/corpus/labels_v2_silence_policy.csv`
   is that table with confidence weights; it is the default label file for training from now on.
2. **Small models at 224-288 px are enough.** The #1 team uses the smallest ResNet (5-fold 0.947); a single-fold
   CoAtNet at 224 reached 0.950. At 224 px with a random bag of 32 windows per study, one fold costs about 4 T4x2
   hours, so **5 folds (~20 h) fit inside one week of Kaggle GPU quota** after Saturday's inference run. Path 2 is
   therefore the primary free path; Path 1 (rented GPU) only buys speed and a second seed.
3. **Then self-distillation.** Blend OOF predictions 0.5/0.5 with the report labels and retrain (lumbar-2024 2nd
   place; "OOF pseudo-labels well correlated with LB" in this competition). Expected +0.003 to +0.008.
4. Cheap extras with evidence: flip TTA (+0.001 to +0.003), 25 -> 50 epochs (+0.004 reported), a 90 mm meniscus
   crop as a second view for the weak lateral columns (+0.002 to +0.005, speculative).

Revised expectation if steps 1-3 land: a family at 0.947-0.950 solo, 0.950-0.954 blended with the public stack
(rank ~10-40). Top-3 (0.956+) needs the distillation round and the anatomy crop to both pay; it is possible, not
assured, and the OOF/gold gates decide it, not the public leaderboard.

## Corpus96 build result (2026-09-22 21:50 UTC)

All three CPU kernels completed: 1,469 + 1,469 + 1,469 = **4,407 studies, 0 incomplete, 0 slice decode failures**,
mean 4.72 of 5 slots filled per study, 14-22 minutes per part. Outputs (`vols_partK.npy` (n,96,336,336) uint8,
`masks_partK.npy`, `ids_partK.npy`, `meta_partK.json`) live in the kernel outputs
`kragglenote2forwork/rsna-knee-corpus96-part0..2` and are mounted by `kaggle/corpus/build_train_kernels.py`.
