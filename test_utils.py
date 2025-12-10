# test_utils.py
import numpy as np
from utils import compute_kinematic_features

# synthetic simple stroke
t = np.linspace(0, 1, 10)
x = np.sin(2*np.pi*1.5*t)  # oscillatory x
y = np.cos(2*np.pi*1.5*t)  # oscillatory y

feats = compute_kinematic_features(x, y, t)
print("Features:", feats)
