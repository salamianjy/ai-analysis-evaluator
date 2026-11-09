import unittest
from src.evaluator import evaluate_report

class EvaluatorTests(unittest.TestCase):
    def basic(self):
        return {'target': 'churn', 'features': ['age'], 'rows': [{'tp': 40, 'tn': 40, 'fp': 10, 'fn': 10}], 'metrics': {'accuracy': .8, 'precision': .8, 'recall': .8}, 'split': {'strategy': 'stratified', 'train_ids': [1,2], 'test_ids': [3,4]}}

    def codes(self, report):
        return {f['code'] for f in evaluate_report(report)['findings']}

    def test_clean_report(self):
        self.assertEqual(self.codes(self.basic()), set())
        self.assertEqual(evaluate_report(self.basic())['score'], 100)

    def test_target_and_post_outcome_leakage(self):
        r = self.basic(); r['features'] = ['age', 'churn', 'post_outcome_refund']
        self.assertTrue({'TARGET_IN_FEATURES', 'POST_OUTCOME_FEATURE'} <= self.codes(r))

    def test_time_leakage(self):
        r = self.basic(); r['time_ordered'] = True; r['split']['strategy'] = 'random'
        self.assertIn('TIME_LEAKAGE', self.codes(r))

    def test_overlap(self):
        r = self.basic(); r['split']['test_ids'] = [2,3]
        self.assertIn('SPLIT_OVERLAP', self.codes(r))

    def test_metrics(self):
        r = self.basic(); r['metrics']['accuracy'] = .95
        self.assertIn('METRIC_MISMATCH', self.codes(r))

    def test_causal_claim(self):
        r = self.basic(); r['study_design'] = 'observational'; r['claims'] = {'causal': 'Campaign caused retention'}
        self.assertIn('CAUSAL_OVERCLAIM', self.codes(r))

    def test_fairness_claim(self):
        r = self.basic(); r['claims'] = {'fairness_proven': True}
        self.assertIn('UNSUPPORTED_FAIRNESS', self.codes(r))

    def test_zero_denominator(self):
        r = self.basic(); r['rows'] = [{'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}]; r['metrics'] = {'accuracy': 0, 'precision': 0, 'recall': 0}
        self.assertNotIn('METRIC_MISMATCH', self.codes(r))

if __name__ == '__main__':
    unittest.main()
