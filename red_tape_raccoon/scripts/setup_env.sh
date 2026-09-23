#!/usr/bin/env bash
# Reproducible local environment for red_tape_raccoon.
#
#   bash red_tape_raccoon/scripts/setup_env.sh            # clone + deps + public tasks (+ handbook if approved)
#   BUILD_IMAGE=1 bash red_tape_raccoon/scripts/setup_env.sh
#
# Credentials are read from the standard CLI locations only:
#   ~/.cache/huggingface/token   (HF token; handbook needs manual approval on the HF page)
#   ~/.kaggle/kaggle.json        (Kaggle legacy API key)
#   red_tape_raccoon/vendor/chi-bench/.env  (ANTHROPIC_API_KEY for the judge + agent provider keys)
# Nothing under vendor/ is ever committed (see .gitignore).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="${VENDOR:-$HERE/vendor}"
CHI_REPO="${CHI_REPO:-https://github.com/actava-ai/chi-bench}"
CHI_COMMIT="${CHI_COMMIT:-2395e9e2ae7d42cdf2750dc3e67daeb0e672b910}"
DATA_REV="${DATA_REV:-chi-bench-v1.0.0}"

mkdir -p "$VENDOR"
if [ ! -d "$VENDOR/chi-bench/.git" ]; then
  git clone "$CHI_REPO" "$VENDOR/chi-bench"
fi
git -C "$VENDOR/chi-bench" fetch --depth 200 origin main || true
git -C "$VENDOR/chi-bench" checkout -q "$CHI_COMMIT"

cd "$VENDOR/chi-bench"
uv sync --extra dev
# Our harness is installed into the benchmark's venv so Harbor can import it.
uv pip install -e "$HERE"

uv run hf download actava/chi-bench --repo-type dataset --revision "$DATA_REV" --local-dir data/
echo "$DATA_REV" > data/.chi-bench-version

if uv run hf download actava/managed-care-operations-handbook --repo-type dataset \
     --local-dir data/skills/ >/dev/null 2>&1; then
  echo "handbook: downloaded ($(find data/skills -name '*.md' | wc -l) markdown files)"
else
  echo "handbook: NOT AUTHORIZED yet -- request access at" \
       "https://huggingface.co/datasets/actava/managed-care-operations-handbook"
  echo "          (competition rules forbid running trials before approval)"
fi
rm -rf data/skills/.cache

[ -f .env ] || cp .env.example .env

if [ "${BUILD_IMAGE:-0}" = "1" ]; then
  if [ -f /root/.ccr/ca-bundle.crt ]; then
    # Sandboxed hosts re-terminate TLS at an egress proxy: bake its CA into a local variant.
    cp /root/.ccr/ca-bundle.crt docker/ccr-ca-bundle.crt
    python3 "$HERE/scripts/local_dockerfile.py" docker/Dockerfile docker/Dockerfile.local
    docker build --network host \
      --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" --build-arg https_proxy="${HTTPS_PROXY:-}" \
      --build-arg NO_PROXY="${NO_PROXY:-}" --build-arg no_proxy="${NO_PROXY:-}" \
      -f docker/Dockerfile.local -t chi-bench:latest .
  else
    uv run cb docker build
  fi
  uv run cb data verify
fi
