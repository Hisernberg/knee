# Kaggle workflow (code competition)

`rsna-knee-abnormality-detection` is a **code competition**: a submission is a completed notebook version
that wrote `/kaggle/working/submission.csv` with internet off in ≤ 9 hours (T4 ×2). Uploading a CSV from
your laptop is rejected. The API path that works is:

```
kaggle kernels push -p <folder with notebook + kernel-metadata.json>      # runs the notebook on Kaggle
kaggle kernels status <owner>/<slug>                                     # wait for "complete"
kaggle competitions submit -c rsna-knee-abnormality-detection \
    -k <owner>/<slug> -v <version> -f submission.csv -m "<message>"     # submit that version
kaggle competitions submissions -c rsna-knee-abnormality-detection -v   # public score
```

## Submission file contract (verified)

```
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```
One row per `StudyInstanceUID` in `test.csv` (3 rows in the public stub, ≈ 1,300 in the hidden rerun),
probabilities in [0, 1], no NaN. `kneemri.schema.validate_submission` enforces exactly this before the
file is written, and the notebook's last cell re-validates the written file.

## Steps

1. **Train offline** (GPU box, data under `data/`): `scripts/run_all.sh` or the individual commands in the
   root README. Checkpoints land in `work/runs/<arm>/fold*/best_filament_unet.pth`.
2. **Upload weights** as a private Kaggle dataset:
   `python kaggle/upload_weights.py --owner <you> --slug rsna-knee-kneemri-weights --runs work/runs`
   (re-run with `--update` for new versions).
3. **Edit `kaggle/variants.yaml`**: set `owner`, the dataset slug, and the checkpoint globs.
4. **Dry-run the build**: `python kaggle/build_notebook.py --variant v1_arm_a_5fold --out build/v1`
   creates a self-contained notebook (package source embedded, no pip, no internet) and its
   `kernel-metadata.json` (`enable_internet: false`, `machine_shape: NvidiaTeslaT4`, competition + dataset
   sources). `tests/test_notebook_build.py` executes the same cells locally on synthetic data.
5. **Run the loop**: `export KAGGLE_API_TOKEN=...` then
   `python kaggle/submit_loop.py --owner <you> --all --max-concurrent 2`
   Pushes each variant, polls until complete, submits each finished version, waits for the public score,
   and appends everything to `kaggle/submission_log.jsonl`. `--dry-run` prints the commands only;
   `--no-submit` pushes and waits without submitting. The daily limit (5) is respected with
   `--max-per-day`.

## Notes

* The token is read from the environment and never written to disk or logs by these scripts.
* Kaggle silently falls back to P100 for an invalid `machine_shape`; the current torch image cannot run on
  P100 (`cudaErrorNoKernelImageForDevice`). Keep `NvidiaTeslaT4`.
* Notebook cells are validated with `compile()` at build time (a `from __future__` line inside an indented
  block is not caught by `ast.parse`).
