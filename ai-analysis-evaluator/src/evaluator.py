"""Deterministic checks for common analytical mistakes in AI-generated responses."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re

@dataclass
class Finding:
    code: str
    severity: str
    message: str
    evidence: str


def evaluate_report(report: dict) -> dict:
    """Check a structured model report against analytical quality rules.

    This is a reference rules-based evaluator, NOT a trained AI judge. Some
    real-world errors require contextual human review.
    """
    findings: list[Finding] = []
    rows = report.get('rows', [])
    target = report.get('target', '')
    features = report.get('features', [])
    split = report.get('split', {})
    metrics = report.get('metrics', {})
    claims = report.get('claims', {})

    if target and target in features:
        findings.append(Finding('TARGET_IN_FEATURES', 'critical', 'Target included in model features.', target))
    if any(str(f).startswith('post_outcome_') for f in features):
        findings.append(Finding('POST_OUTCOME_FEATURE', 'critical', 'A feature was recorded after the outcome.', ', '.join(f for f in features if str(f).startswith('post_outcome_'))))
    if split.get('strategy') == 'random' and report.get('time_ordered', False):
        findings.append(Finding('TIME_LEAKAGE', 'high', 'Random split on time-ordered data can leak future patterns.', 'split.strategy=random, time_ordered=true'))
    if split.get('train_ids') is not None and split.get('test_ids') is not None:
        overlap = set(split['train_ids']) & set(split['test_ids'])
        if overlap:
            findings.append(Finding('SPLIT_OVERLAP', 'critical', 'Same records appear in training and test.', ', '.join(sorted(map(str, overlap)))))
    if rows and {'tp', 'tn', 'fp', 'fn'} <= rows[0].keys():
        total = {k: sum(int(r[k]) for r in rows) for k in ('tp', 'tn', 'fp', 'fn')}
        denom = sum(total.values())
        expected_accuracy = (total['tp'] + total['tn']) / denom if denom else 0
        expected_precision = total['tp'] / (total['tp'] + total['fp']) if total['tp'] + total['fp'] else 0
        expected_recall = total['tp'] / (total['tp'] + total['fn']) if total['tp'] + total['fn'] else 0
        for metric, expected in [('accuracy', expected_accuracy), ('precision', expected_precision), ('recall', expected_recall)]:
            if metric in metrics and abs(float(metrics[metric]) - expected) > 0.005:
                findings.append(Finding('METRIC_MISMATCH', 'high', f'Reported {metric} differs from confusion-matrix calculation.', f'reported={metrics[metric]}, expected={expected:.4f}'))
    if claims.get('causal') and report.get('study_design') == 'observational':
        findings.append(Finding('CAUSAL_OVERCLAIM', 'high', 'Observational association alone does not establish causation.', str(claims['causal'])))
    if claims.get('fairness_proven') and not report.get('group_metrics'):
        findings.append(Finding('UNSUPPORTED_FAIRNESS', 'medium', 'Fairness conclusion lacks subgroup metrics.', str(claims['fairness_proven'])))
    severities = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
    penalty = sum(severities[f.severity] * 5 for f in findings)
    return {'score': max(0, 100 - penalty), 'findings': [asdict(f) for f in findings], 'review_required': True}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a structured AI analysis report')
    parser.add_argument('input', type=Path, help='Input JSON report')
    parser.add_argument('--output', type=Path, help='Optional findings output path')
    args = parser.parse_args()
    result = evaluate_report(json.loads(args.input.read_text(encoding='utf-8')))
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + '\n', encoding='utf-8')
    else:
        print(output)

if __name__ == '__main__':
    main()
