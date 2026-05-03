import pandas as pd
import numpy as np

df = pd.read_csv(r'e:\iot-classification\evaluation\unseen\predict\vendor_analysis.csv')
ms = df[df['device_type'] == 'MEDIA_SERVER'].copy()

print('=== MEDIA_SERVER vendor distribution ===')
print(f'Total vendors (unique): {len(ms)}')
print(f'Vendors with support=1 : {(ms["support"]==1).sum()}')
print(f'Vendors with support<=5: {(ms["support"]<=5).sum()}')
print(f'Vendors with f1=0.0    : {(ms["f1"]==0.0).sum()}')
print(f'Vendors with f1=0 AND support=1: {((ms["f1"]==0.0)&(ms["support"]==1)).sum()}')
print(f'Vendors with f1=0 AND support>1: {((ms["f1"]==0.0)&(ms["support"]>1)).sum()}')

print()
print('=== F1 buckets ===')
buckets = [(0,0,'F1=0'), (0.001,0.5,'0<F1<0.5'), (0.5,0.8,'0.5<=F1<0.8'),
           (0.8,0.999,'0.8<=F1<1.0'), (1.0,1.0,'F1=1.0')]
for lo, hi, label in buckets:
    if lo == hi:
        mask = ms['f1'] == lo
    else:
        mask = (ms['f1'] >= lo) & (ms['f1'] <= hi)
    print(f'  {label:<18}: {mask.sum():>4} vendors, total support={ms.loc[mask,"support"].sum()}')

print()
print('=== Top-10 highest-support MEDIA_SERVER vendors ===')
print(ms.nlargest(10,'support')[['vendor','support','TP','FP','FN','precision','recall','f1']].to_string(index=False))

print()
print('=== Vendors with support>5 and f1=0 ===')
bad = ms[(ms['f1']==0)&(ms['support']>5)].sort_values('support',ascending=False)
print(bad[['vendor','support','TP','FP','FN','precision','recall','f1']].to_string(index=False))

print()
print('=== Macro F1 contribution simulation ===')
n_vendors = len(ms)
n_zero = (ms['f1']==0.0).sum()
n_one  = (ms['f1']==1.0).sum()
print(f'  {n_vendors} vendors equally weighted in macro F1')
print(f'  {n_zero} vendors contribute F1=0.0  ({n_zero/n_vendors:.1%})')
print(f'  {n_one}  vendors contribute F1=1.0  ({n_one/n_vendors:.1%})')
print(f'  Mean F1 among non-zero vendors : {ms[ms["f1"]>0]["f1"].mean():.4f}')
print(f'  Support-weighted avg F1 (micro): {(ms["f1"]*ms["support"]).sum()/ms["support"].sum():.4f}')

print()
print('=== Compare: VPN ===')
vp = df[df['device_type'] == 'VPN'].copy()
n_v = len(vp)
n_v_zero = (vp['f1']==0.0).sum()
print(f'  VPN vendors total: {n_v}')
print(f'  VPN F1=0 vendors : {n_v_zero} ({n_v_zero/n_v:.1%})')
print(f'  VPN support=1    : {(vp["support"]==1).sum()}')
print(f'  VPN support-weighted F1: {(vp["f1"]*vp["support"]).sum()/vp["support"].sum():.4f}')
