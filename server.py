from flask import Flask, request, jsonify, send_from_directory
import os, csv, datetime
import pandas as pd

# YOU MUST HAVE THIS BEFORE ANY @app.route
app = Flask(__name__, static_folder='.')

OUT_DIR = 'collected'
os.makedirs(OUT_DIR, exist_ok=True)


@app.route('/')
def index():
    return send_from_directory('.', 'collector.html')

@app.route('/', methods=['GET'])
def index():
    if model_bundle is None:
        return "Inference server running — model NOT found (run training).", 200
    return "Inference server running — model loaded.", 200


@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json(force=True)

    writer = data.get('writer_id', 'anon')
    sample = data.get('sample_id', f's_{int(datetime.datetime.now().timestamp())}')
    created = data.get('created', datetime.datetime.now().isoformat())
    strokes = data.get('strokes', [])

    fname = os.path.join(OUT_DIR, f'{writer}__{sample}.csv')
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['writer_id', 'sample_id', 'stroke_id', 'point_id', 'x', 'y', 't', 'pressure', 'penDown'])

        for si, stroke in enumerate(strokes):
            for pi, p in enumerate(stroke):
                w.writerow([
                    writer, sample, si, pi,
                    p.get('x'), p.get('y'), p.get('t'),
                    p.get('pressure'), p.get('penDown')
                ])

    return jsonify({'status': 'ok', 'file': fname})
# server.py (add near other routes)
from flask import request

LABEL_FILE = 'labels.csv'

@app.route('/label', methods=['POST'])
def label_sample():
    """
    Expects JSON: { 'sample_key': 'user__sample', 'label': 0 or 1 }
    Appends to labels.csv if not already present.
    """
    data = request.get_json(force=True)
    sample = data.get('sample_key')
    label = data.get('label')
    if sample is None or label is None:
        return jsonify({'error':'missing_fields'}), 400

    # create file if it doesn't exist
    exists = os.path.exists(LABEL_FILE)
    # load to check duplicates
    if exists:
        with open(LABEL_FILE,'r') as f:
            lines = [l.strip().split(',')[0] for l in f if l.strip()]
            if sample in lines:
                return jsonify({'status':'already_labeled'}), 200

    with open(LABEL_FILE, 'a') as f:
        f.write(f"{sample},{int(label)}\n")
    return jsonify({'status':'ok','sample':sample,'label':int(label)})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
