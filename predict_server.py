# predict_server.py
from flask import Flask, request, jsonify
import joblib, os, pandas as pd
from utils import df_sample_to_feature_vector

MODEL_FILE = 'baseline_model.joblib'
app = Flask(__name__)

model_bundle = None
if os.path.exists(MODEL_FILE):
    try:
        model_bundle = joblib.load(MODEL_FILE)
        print("Loaded model bundle from", MODEL_FILE)
    except Exception as e:
        print("Failed to load model:", e)
        model_bundle = None
else:
    print("Model not found. Train model with: python train_model.py")

@app.route('/predict_file', methods=['POST'])
def predict_file():
    if model_bundle is None:
        return jsonify({'error':'model_missing'}), 400
    if 'sample' not in request.files:
        return jsonify({'error':'no_file'}), 400
    f = request.files['sample']
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({'error':'bad_csv','message':str(e)}), 400
    try:
        feats = df_sample_to_feature_vector(df)
        feature_names = model_bundle.get('features', list(feats.keys()))
        X = [feats.get(k,0.0) for k in feature_names]
        Xs = model_bundle['scaler'].transform([X])
        pred = float(model_bundle['model'].predict_proba(Xs)[0,1])
        return jsonify({'irregularity_score': pred})
    except Exception as e:
        return jsonify({'error':'inference_failed','message':str(e)}), 500

@app.route('/predict_json', methods=['POST'])
def predict_json():
    if model_bundle is None:
        return jsonify({'error':'model_missing'}), 400
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({'error':'bad_json','message':str(e)}), 400
    strokes = data.get('strokes', None)
    if strokes is None:
        return jsonify({'error':'no_strokes'}), 400
    rows = []
    for si, stroke in enumerate(strokes):
        for pi, p in enumerate(stroke):
            rows.append({
                'writer_id': data.get('writer_id','anon'),
                'sample_id': data.get('sample_id','s'),
                'stroke_id': si,
                'point_id': pi,
                'x': p.get('x'),
                'y': p.get('y'),
                't': p.get('t'),
                'pressure': p.get('pressure'),
                'penDown': p.get('penDown')
            })
    df = pd.DataFrame(rows)
    try:
        feats = df_sample_to_feature_vector(df)
        feature_names = model_bundle.get('features', list(feats.keys()))
        X = [feats.get(k,0.0) for k in feature_names]
        Xs = model_bundle['scaler'].transform([X])
        pred = float(model_bundle['model'].predict_proba(Xs)[0,1])
        return jsonify({'irregularity_score': pred})
    except Exception as e:
        return jsonify({'error':'inference_failed','message':str(e)}), 500

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

