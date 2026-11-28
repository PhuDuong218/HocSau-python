"""Reconstruct aggregate metrics from the existing annotated confusion matrix.

This is analysis of saved evidence, NOT a new checkpoint evaluation.
No per-image predictions, filter weights, or face images can be recovered here.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Transcribed from the 49 visible annotations in this repository's PNG.
MATRIX = np.array([
    [142, 0, 34, 273, 216, 261, 32],
    [27, 0, 2, 25, 17, 37, 3],
    [72, 0, 79, 211, 259, 264, 139],
    [14, 0, 17, 1534, 103, 78, 28],
    [21, 0, 24, 234, 752, 185, 17],
    [51, 0, 45, 284, 396, 455, 16],
    [19, 0, 90, 95, 93, 26, 508],
], dtype=int)
SOURCE_SHA256 = '4ddbb97de3b1cc07b5eb5958cf9268b8bfcc9f0bf9f6ec4ec7eb1502efdcff6b'


def analyze(output_dir=None):
    source = ROOT / 'results/plots/baseline_confusion_matrix.png'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Source PNG changed; review all 49 annotations before updating the matrix/hash.')
    if MATRIX.sum(axis=1).tolist() != [958, 111, 1024, 1774, 1233, 1247, 831]:
        raise ValueError('Row totals disagree with notebook support counts.')
    # Counts fully determine aggregate classification metrics. Expanded indices
    # are mathematical representations of counts, not recovered image records.
    actual, predicted = [], []
    for i in range(7):
        for j in range(7):
            actual.extend([i] * int(MATRIX[i, j]))
            predicted.extend([j] * int(MATRIX[i, j]))
    report = classification_report(actual, predicted, labels=list(range(7)),
                                   target_names=LABELS, output_dict=True, zero_division=0)
    # Cross-check against the original report printed in the saved notebook.
    notebook = json.loads((ROOT/'notebooks/02_baseline.ipynb').read_text(encoding='utf-8'))
    streams = '\n'.join(''.join(o.get('text', [])) for c in notebook['cells']
                         for o in c.get('outputs', []) if o.get('output_type') == 'stream')
    for label in LABELS:
        row = next(line.split() for line in streams.splitlines() if line.split() and line.split()[0] == label)
        expected = [f"{report[label][key]:.2f}" for key in ['precision', 'recall', 'f1-score']]
        if row[1:4] != expected or int(row[4]) != int(report[label]['support']):
            raise ValueError(f'Saved notebook report disagrees with image for {label}')
    output_dir = Path(output_dir or ROOT/'results/evaluation/saved_baseline')
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        'source_type': 'saved_confusion_matrix_annotations', 'new_inference': False,
        'source_file': 'results/plots/baseline_confusion_matrix.png', 'source_sha256': digest,
        'cross_check': '49 annotations manually reviewed; all row totals and rounded per-class notebook metrics match',
        'samples': int(MATRIX.sum()), 'correct': int(MATRIX.trace()),
        'wrong': int(MATRIX.sum() - MATRIX.trace()), 'accuracy': report['accuracy'],
        'macro_precision': report['macro avg']['precision'], 'macro_recall': report['macro avg']['recall'],
        'macro_f1': report['macro avg']['f1-score'], 'weighted_f1': report['weighted avg']['f1-score'],
        'labels': LABELS, 'confusion_matrix': MATRIX.tolist(), 'classification_report': report,
        'limitation': 'The baseline notebook selected its checkpoint on this same test split. This is not independent test performance. Individual image identities and checkpoint weights are unavailable in these aggregate results.'
    }
    (output_dir/'metrics_from_saved_matrix.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (output_dir/'classification_report.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'precision', 'recall', 'f1-score', 'support'])
        for label in LABELS + ['macro avg', 'weighted avg']:
            writer.writerow([label]+[report[label][key] for key in ['precision','recall','f1-score','support']])
    with (output_dir/'confusion_matrix.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['true / predicted']+LABELS)
        for label,row in zip(LABELS,MATRIX):
            writer.writerow([label]+row.tolist())
    (output_dir/'saved_notebook_output.txt').write_text(streams, encoding='utf-8')
    print(f"Saved evidence analysis: {metrics['correct']}/{metrics['samples']}, accuracy={metrics['accuracy']:.6f}, macro-F1={metrics['macro_f1']:.6f}")
    return metrics


if __name__ == '__main__':
    analyze()
