"""
Analyze the CSV output from run_on_images.py and print accuracy metrics.

Expects CSV with columns: filename, freshness_label, good_conf, spoiled_conf

Ground truth is inferred from filename prefixes:
  FRESH-*   → actual good
  otherwise → actual spoiled (HALF-FRESH, SPOILED)
"""

import csv, statistics

with open('output/test_decisions_log.csv') as f:
    rows = list(csv.DictReader(f))

total = len(rows)
pred_good = [r for r in rows if r['freshness_label'] == 'good']
pred_spoiled = [r for r in rows if r['freshness_label'] == 'spoiled']
good_confs = [float(r['good_conf']) for r in rows]

# The val folder has good/ and spoiled/ subdirs - check filenames
# FRESH prefix = actual good, HALF-FRESH/SPOILED = actual spoiled
actual_good_rows = [r for r in rows if r['filename'].upper().startswith('FRESH')]
actual_spoiled_rows = [r for r in rows if not r['filename'].upper().startswith('FRESH')]

tp = sum(1 for r in actual_good_rows if r['freshness_label'] == 'good')
fn = sum(1 for r in actual_good_rows if r['freshness_label'] == 'spoiled')
fp = sum(1 for r in actual_spoiled_rows if r['freshness_label'] == 'good')
tn = sum(1 for r in actual_spoiled_rows if r['freshness_label'] == 'spoiled')

accuracy = (tp + tn) / total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print('=== MODEL ACCURACY ===')
print(f'Total val images:  {total}')
print(f'  Actual Good:     {len(actual_good_rows)}')
print(f'  Actual Spoiled:  {len(actual_spoiled_rows)}')
print()
print(f'  True Positives (good predicted as good):       {tp}')
print(f'  False Negatives (good predicted as spoiled):   {fn}')
print(f'  False Positives (spoiled predicted as good):   {fp}')
print(f'  True Negatives  (spoiled predicted as spoiled):{tn}')
print()
print(f'  Accuracy:  {accuracy*100:.1f}%')
print(f'  Precision: {precision*100:.1f}%')
print(f'  Recall:    {recall*100:.1f}%')
print(f'  F1 Score:  {f1:.3f}')
print()
print('=== CONFIDENCE STATS ===')
print(f'  Mean  good_conf: {statistics.mean(good_confs):.4f}')
print(f'  Median good_conf:{statistics.median(good_confs):.4f}')
