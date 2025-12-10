# generate_synthetic.py
import os, csv, numpy as np
OUT = 'collected'
os.makedirs(OUT, exist_ok=True)

def save_sample(writer, sample_id, x,y,t, pressure=None):
    fname = os.path.join(OUT, f'{writer}__{sample_id}.csv')
    if pressure is None:
        pressure = [0.5]*len(x)
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['writer_id','sample_id','stroke_id','point_id','x','y','t','pressure','penDown'])
        for i in range(len(x)):
            w.writerow([writer, sample_id, 0, i, float(x[i]), float(y[i]), float(t[i]), float(pressure[i]), True])

# create N normal and M irregular samples
N = 6
M = 6
for i in range(N):
    t = np.linspace(0,1,80)
    x = np.cumsum(0.5*np.cos(2*np.pi*1.2*t) + 0.1*np.random.randn(len(t)))
    y = np.cumsum(0.5*np.sin(2*np.pi*1.2*t) + 0.1*np.random.randn(len(t)))
    save_sample('synth','normal'+str(i), x, y, t)

for i in range(M):
    t = np.linspace(0,1,80)
    # add tremor: small high-frequency sinusoid -> irregular
    tremor = 0.05*np.sin(2*np.pi*30*t)
    x = np.cumsum(0.5*np.cos(2*np.pi*1.2*t) + tremor + 0.15*np.random.randn(len(t)))
    y = np.cumsum(0.5*np.sin(2*np.pi*1.2*t) + tremor + 0.15*np.random.randn(len(t)))
    save_sample('synth','irr'+str(i), x, y, t)

# generate labels.csv
with open('labels.csv','w') as f:
    for i in range(N):
        f.write(f"synth__normal{i},0\n")
    for i in range(M):
        f.write(f"synth__irr{i},1\n")
print("Generated synthetic data and labels.csv")
