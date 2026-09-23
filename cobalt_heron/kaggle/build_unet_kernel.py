"""Build a Kaggle CPU/GPU script kernel that embeds the ch package and runs prep -> train -> test inference.
usage: python kaggle/build_unet_kernel.py <slug> "<train args>" [--gpu]
"""
import sys, json
from pathlib import Path
slug, targs = sys.argv[1], sys.argv[2]; gpu = '--gpu' in sys.argv
import re
fold = int(re.search(r'--fold (-?\d+)', targs).group(1))
here = Path(__file__).resolve().parent.parent
src = {p.name: p.read_text() for p in (here / 'ch').glob('*.py')}
prep = (here / 'scripts' / 'prep.py').read_text()
body = f'''# cobalt-heron U-Net kernel (auto-generated)
import subprocess, sys, os, glob, json
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch', 'pycocotools'], check=False)
SRC = {json.dumps(src)}
PREP = {json.dumps(prep)}
W = Path('/kaggle/working'); (W / 'ch').mkdir(exist_ok=True)
for k, v in SRC.items(): (W / 'ch' / k).write_text(v)
(W / 'prep.py').write_text(PREP)
ann = glob.glob('/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json', recursive=True)[0]
env = dict(os.environ, CH_DATA=str(Path(ann).parent.parent), CH_WORK='/kaggle/temp/work', PYTHONPATH=str(W))
Path('/kaggle/temp/work').mkdir(parents=True, exist_ok=True)
def run(cmd):
    print('>>', cmd, flush=True); subprocess.run(cmd, shell=True, check=True, env=env, cwd=str(W))
run('python prep.py')
run('python -m ch.train --out /kaggle/working/model.pt {targs}')
run('python -m ch.infer --models /kaggle/working/model.pt --src /kaggle/temp/work/test1024 --out /kaggle/temp/testprob')
run('cd /kaggle/temp && tar cf /kaggle/working/testprob.tar testprob')
if {fold} >= 0:
    run('python -m ch.infer --models /kaggle/working/model.pt --src /kaggle/temp/work/c1024 --stems /kaggle/temp/work/folds.json:{fold} --out /kaggle/temp/oofprob')
    run('cd /kaggle/temp && tar cf /kaggle/working/oofprob.tar oofprob')
'''
d = here / 'kaggle' / 'unet' / slug; d.mkdir(parents=True, exist_ok=True)
(d / 'run.py').write_text(body)
json.dump({"id": f"kragglenote2forwork/{slug}", "title": slug, "code_file": "run.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
           "competition_sources": ["filament-segmentation-2026"], "dataset_sources": [], "kernel_sources": [],
           "model_sources": []}, open(d / 'kernel-metadata.json', 'w'), indent=1)
print(d)
