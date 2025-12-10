# utils.py
import numpy as np
import pandas as pd
from typing import Dict

def load_sample_csv(path: str) -> pd.DataFrame:
    """
    Load a sample CSV saved by the collector and sort by stroke & point.
    Expected columns: ['writer_id','sample_id','stroke_id','point_id','x','y','t','pressure','penDown']
    """
    df = pd.read_csv(path)
    df = df.sort_values(['stroke_id', 'point_id']).reset_index(drop=True)
    return df

def compute_kinematic_features(xs, ys, ts) -> Dict[str, float]:
    """
    Compute simple kinematic features from 1D sequences xs, ys, ts (timestamps).
    Returns a dict with keys:
      v_mean, v_std, v_max, a_mean, a_std, curv_mean, curv_std, n_points, duration, jerk_mean
    """
    # convert inputs to 1D numpy arrays
    xs = np.asarray(xs).astype(np.float64).flatten()
    ys = np.asarray(ys).astype(np.float64).flatten()
    ts = np.asarray(ts).astype(np.float64).flatten()

    feats = {}
    # If too few points, return zeros
    if xs.size < 2 or ys.size < 2 or ts.size < 2:
        for k in ['v_mean','v_std','v_max','a_mean','a_std','curv_mean','curv_std','n_points','duration','jerk_mean']:
            feats[k] = 0.0
        return feats

    # differences
    dt = np.diff(ts)
    dx = np.diff(xs)
    dy = np.diff(ys)

    # avoid zero dt
    dt = dt.astype(np.float64)
    dt[dt == 0] = 1e-6

    # velocities
    vx = dx / dt
    vy = dy / dt
    v = np.sqrt(vx**2 + vy**2)

    # acceleration (difference of velocity over time)
    if v.size >= 2:
        dv = np.diff(v)
        dt2 = dt[1:] if dt.size > 1 else np.array([1e-6], dtype=np.float64)
        dt2 = dt2.astype(np.float64)
        dt2[dt2 == 0] = 1e-6
        a = dv / dt2
    else:
        a = np.array([])

    # jerk = derivative of acceleration magnitude (approx)
    jerk = np.abs(np.diff(a)) if a.size >= 2 else np.array([])

    # curvature (discrete approximation using gradients)
    # Use gradient which handles endpoints gracefully
    x1 = np.gradient(xs)
    x2 = np.gradient(x1)
    y1 = np.gradient(ys)
    y2 = np.gradient(y1)
    denom = (x1**2 + y1**2)**1.5
    denom[denom == 0] = 1e-6
    curv = np.abs(x1 * y2 - y1 * x2) / denom

    # build features (use nan-safe aggregations)
    feats['v_mean'] = float(np.nanmean(v)) if v.size>0 else 0.0
    feats['v_std']  = float(np.nanstd(v))  if v.size>0 else 0.0
    feats['v_max']  = float(np.nanmax(v))  if v.size>0 else 0.0

    feats['a_mean'] = float(np.nanmean(a)) if a.size>0 else 0.0
    feats['a_std']  = float(np.nanstd(a))  if a.size>0 else 0.0

    feats['curv_mean'] = float(np.nanmean(curv)) if curv.size>0 else 0.0
    feats['curv_std']  = float(np.nanstd(curv))  if curv.size>0 else 0.0

    feats['n_points'] = int(xs.size)
    feats['duration'] = float(ts.max() - ts.min())
    feats['jerk_mean'] = float(np.nanmean(jerk)) if jerk.size>0 else 0.0

    return feats

def df_sample_to_feature_vector(df: pd.DataFrame) -> Dict[str, float]:
    """
    Convert a sample DataFrame (sorted by stroke/point) to a feature dictionary.
    It flattens strokes into one long sequence (stroke boundaries not explicitly used here).
    """
    xs = df['x'].values
    ys = df['y'].values
    ts = df['t'].values
    return compute_kinematic_features(xs, ys, ts)
