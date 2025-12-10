# train_baseline.py
import glob
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

from utils import load_sample_csv, df_sample_to_feature_vector

COL_DIR = 'collected'
LABEL_CSV = 'labels.csv'  # Create: sample_filename,label e.g. user1__sample1,1
MODEL_OUT = 'baseline_model.joblib'


def build_dataset(col_dir=COL_DIR):
    """
    Load all CSVs in collected/ and return a DataFrame where each row is
    a sample and columns are features (computed via df_sample_to_feature_vector).
    Index is filename without .csv (e.g. user__sample).
    """
    feats_list = []
    keys = []
    pattern = os.path.join(col_dir, '*.csv')
    files = sorted(glob.glob(pattern))
    for f in files:
        try:
            df = load_sample_csv(f)
            feats = df_sample_to_feature_vector(df)
            key = os.path.basename(f).replace('.csv', '')
            feats_list.append(feats)
            keys.append(key)
        except Exception as e:
            print(f"Warning: failed to process {f}: {e}")

    if len(feats_list) == 0:
        # return empty dataframe with no columns
        return pd.DataFrame()

    # DataFrame from list of dicts
    Xdf = pd.DataFrame(feats_list, index=keys).fillna(0)
    return Xdf


if __name__ == '__main__':
    X = build_dataset()
    if X.shape[0] == 0:
        print("No samples found in 'collected/'. Use the collector to add samples.")
        exit(1)

    if not os.path.exists(LABEL_CSV):
        print("Create a labels.csv with rows: sample_filename,label (e.g. user1__sample1,1)")
        print("Found samples (first 50):")
        print('\n'.join(list(X.index)[:50]))
        exit(1)

    # read labels.csv (no header)
    labels = pd.read_csv(LABEL_CSV, header=None, names=['sample', 'label'])
    label_map = labels.set_index('sample')['label'].to_dict()

    X_rows = []
    y_rows = []
    keys_rows = []
    for idx, row in X.iterrows():
        if idx in label_map:
            X_rows.append(row.values)
            y_rows.append(int(label_map[idx]))
            keys_rows.append(idx)
        else:
            # sample with no label - skip
            pass

    if len(y_rows) == 0:
        print("No labeled samples matched. Check labels.csv sample keys (should match filenames in collected/).")
        print("Found samples (first 50):")
        print('\n'.join(list(X.index)[:50]))
        exit(1)

    X_arr = np.vstack(X_rows)
    y_arr = np.array(y_rows)

    # standardize features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_arr)

    # classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)

    # cross-validation (stratified)
    n_splits = 5 if len(y_arr) >= 5 else max(2, len(y_arr))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    try:
        scores = cross_val_score(clf, Xs, y_arr, cv=cv, scoring='f1')
        print('F1 CV scores:', np.round(scores, 4), 'mean:', round(float(scores.mean()), 4))
    except Exception as e:
        print("Warning: cross_val_score failed:", e)

    # fit final model
    clf.fit(Xs, y_arr)

    # save bundle with model, scaler and feature names
    model_bundle = {
        'model': clf,
        'scaler': scaler,
        'features': list(X.columns)
    }
    joblib.dump(model_bundle, MODEL_OUT)
    print('Saved model to', MODEL_OUT)
