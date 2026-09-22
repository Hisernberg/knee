# Compute feasibility notes (research agent, 2026-09-22)

Estimates marked [est]; all other numbers are from the linked sources.

## Kaggle limits
| Item | Value | Source |
|---|---|---|
| GPU weekly quota | 30 h, floating (sometimes higher), resets Saturday 00:00 UTC | https://www.kaggle.com/docs/efficient-gpu-usage , https://www.kaggle.com/discussions/product-feedback/173129 |
| GPU hardware | 1x P100 16 GB or 2x T4 16 GB; 29 GB RAM | https://www.kaggle.com/docs/notebooks |
| TPU quota | 20 h/week, 9 h per session, separate from the GPU quota; v5e-8 (8 x 16 GB) | https://www.kaggle.com/docs/tpu , https://www.kaggle.com/discussions/product-announcements/607202 |
| Session length | 12 h CPU/GPU, 9 h TPU | https://www.kaggle.com/docs/notebooks |
| Disk / output | 20 GB /kaggle/working, 20 GB saved output per notebook | https://www.kaggle.com/docs/notebooks |
| Concurrency | 1 interactive + 2 commit GPU sessions; batch CPU sessions up to 5 | https://www.kaggle.com/discussions/general/105509 , https://www.kaggle.com/product-feedback/483684 |
| Private storage | 200 GB datasets + 200 GB models | https://www.kaggle.com/discussions/product-announcements/512322 |
| Extra hours via Colab | Colab Pro / Pro+ linked accounts: +15 / +30 Kaggle GPU h/week (promotion) | https://www.kaggle.com/docs/notebooks |
| Competition notebook | <= 9 h, internet off; freely available external data and pretrained models allowed | competition overview page |

Quota is charged per GPU session-hour: two concurrent commit sessions halve wall-clock, not quota.

## PyTorch/XLA on Kaggle TPU
Officially supported (PJRT); timm CoAtNet was developed on TPUs and runs well there (MaxViT less so). Pitfalls: TPU
queueing/instability reports, torch_xla version mismatches, 16 GB per v5e core (per-core batch ~8-16 [est]), XLA
recompiles on shape changes (fixed batch sizes, fixed window counts), input pipeline must feed 8 cores, bf16 +
MpDeviceLoader, 9 h cap (checkpoint every epoch), TPU notebooks cannot be the submission notebook (train on TPU,
infer on GPU). Sources: https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb ,
https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k , https://www.kaggle.com/discussions/product-feedback/298618

## Prices (Sept 2026)
Colab Pro $9.99/mo (100 CU), Pro+ $49.99/mo (500 CU), PAYG $9.99/100 CU; burn rates T4 1.19, L4 1.71, A100-40 5.40,
A100-80 7.52 CU/h (http://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/). On-demand USD/h: RunPod A100
80 GB $1.59 secure / $1.19 community, H100 PCIe $2.89 / $1.99, RTX 4090 $0.74 / $0.34; Lambda A100 40 GB $1.99,
H100 PCIe $3.29; Vast.ai A100 80 GB $0.67-1.09, 4090 $0.14-0.35 (https://www.runpod.io/pricing , https://lambda.ai/pricing ,
https://vast.ai/pricing/gpu/RTX-4090).

## Throughput, coatnet_rmlp_2_rw_384 at 384 px AMP
Measured: RTX 3090 inference 273.9 img/s (timm benchmark CSV), 43 GMACs, 73.9 M params. Estimates [est, +/-30 %]
train / infer img/s: P100 18 / 60; T4 22 / 80 (x2 DDP 42 / 160); 3090 75 / 274; 4090 115-125 / 430; A100 80 GB
130-150 / 500; H100 230-260 / 850; TPU v5e-8 250-400 / 1000; L4 50-60 / 200.

## Rules
External training and uploading weights as a private dataset is standard and allowed; winners must publish
training code, inference code and weights (public Kaggle dataset), CC-BY-NC 4.0; external data/models must be
freely and equally accessible at minimal cost. Rules: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules

## Cost/time for 5 folds x 12 epochs x 4,400 studies x 96 slices [est]
20.3 M training samples + 5.1 M validation inferences.
| Hardware | GPU-hours | Cost | Reality |
|---|---|---|---|
| Kaggle P100 | ~340 | $0 | 11+ weeks |
| Kaggle T4x2 | ~145 | $0 | ~5 weeks of quota; only a cut-down run (~25 h: 32 slices, 3 folds, 8 epochs) fits |
| Kaggle T4x2 + Colab Pro+ link | ~145 | $50/mo | ~2.5 weeks |
| Kaggle TPU v5e-8 | ~15-25 | $0 | 1-2 weeks of TPU quota plus XLA risk |
| Colab Pro+ A100 | ~43 | ~325 CU (~$33) | 2-3 background sessions |
| RunPod 4090 | ~53 | $18-39 | ~11 h wall-clock with 5 pods |
| Vast.ai 4090 | ~53 | $8-19 | cheapest, variable reliability |
| RunPod A100 80 GB | ~43 | $51-68 | ~9 h with 5 pods |
| Vast.ai A100 80 GB | ~43 | $29-47 | |
| RunPod / Lambda H100 | ~23-27 | $54-99 | |
