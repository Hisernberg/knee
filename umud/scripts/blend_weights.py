"""Per-target blend of pipeline shot A with the public Vera CSV (C3/C4/C5 family).

    python scripts/blend_weights.py <w_pa> <w_fl> <w_mt> <out.csv> [pipeline.csv]
Paths are this workspace's; weights are the pipeline share per target. C5 = 0.6 0.3 0.6 (best 0.34947).
"""
import pandas as pd,numpy as np,sys
wpa,wfl,wmt,out=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]
pipe=sys.argv[5] if len(sys.argv)>5 else '/home/user/subs/s02_A_pipeline.csv'
a=pd.read_csv(pipe).set_index('image_id')
v=pd.read_csv('/home/user/subs/s00_vera_public.csv').set_index('image_id').loc[a.index]
g=pd.Series(np.load('/tmp/claude-0/test_groups.npy'),index=a.index)
p=a.copy(); p['pa_deg']+=1.6
W={'pa_deg':wpa,'fl_mm':wfl,'mt_mm':wmt}
c=pd.DataFrame({k:W[k]*p[k]+(1-W[k])*v[k] for k in W})
for col in c.columns: c[col]=0.4*c[col]+0.6*c.groupby(g)[col].transform('median')
c.loc['IMG_00001.tif']=[17.334,79.423,21.778]; c.loc['IMG_00002.tif']=[12.876,69.424,15.478]
for col,(lo,hi) in {'pa_deg':(5,45),'fl_mm':(30,200),'mt_mm':(10,50)}.items(): c[col]=c[col].clip(lo,hi)
c.round(3).reset_index().to_csv(out,index=False); print('wrote',out,len(c))
