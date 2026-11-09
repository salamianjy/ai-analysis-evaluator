# AI Analysis Evaluator

A **rules-based benchmark** for spotting common errors in structured AI-generated data science reports. Built as a portfolio demonstration of task design, quantitative checking, traceable evaluation findings, and transparent rubrics.

> This tool does **not** call an LLM or assess unstructured free-form answers. Its heuristics are examples, not comprehensive safeguards, and human review remains necessary.

## Why this project?

AI assistants can produce polished reports that contain data leakage, inconsistent performance metrics, or conclusions unsupported by the study design. This project provides a small reproducible evaluation harness that finds **specific, auditable** failure modes.

## What it evaluates

| Check | Example problem | Severity |
|---|---|---|
| Target contamination | Target field used as a feature | Critical |
| Outcome leakage | Field prefixed `post_outcome_` included | Critical |
| Time-order leakage | Random split on time-ordered data | High |
| Train-test overlap | Same IDs in both partitions | Critical |
| Metric verification | Reported accuracy, precision, or recall differs from confusion counts | High |
| Causal overreach | Causal claim from observational design | High |
| Unsupported fairness claim | Fairness claimed with no subgroup metrics | Medium |

These are deliberately limited detection rules. For example, the prefix heuristic for outcome leakage depends on column naming; it cannot prove that any feature truly leaks information. The metric check uses aggregate counts, so it assumes binary classification and micro-aggregated confusion counts.

## Try it

Requires Python 3.10+; no third-party dependencies.

```bash
python -m src.evaluator examples/valid_report.json
python -m src.evaluator examples/flawed_report.json --output findings.json
python -m unittest discover -s tests -v
```

The first input should score 100/100. The second produces named findings with evidence and a lower rubric score. `review_required` remains true regardless of score.

## Data and reproducibility

Both example datasets are **synthetic JSON fixtures** written for this project. No personal, client, or proprietary information is included. Assessment outputs are deterministic for the supplied inputs.

## Extend the benchmark

- Add representative borderline cases and measure false-positive/false-negative rates.
- Score model responses against curated human-annotated reference judgments.
- Add schema validation, calibration plots, and regression-vs-classification checks.
- Require dataset provenance, baseline comparisons, and reproducibility artifacts.
- Expand handling of time-series forecasting and group-level evaluations.

## Responsible use

This project is newly created as a demonstration of skills. It is **not** a production-grade AI safety evaluator or evidence of previous employment. Please review and understand the implementation before representing it in applications.
