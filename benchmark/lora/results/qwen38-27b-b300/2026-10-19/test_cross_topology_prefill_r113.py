import copy
import unittest

from audit_cross_topology_prefill_r113 import prefill_scores
from audit_topology_original_r99 import numerical_stats


class PrefillDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.case = {'input_ids': [10, 11, 12]}
        self.response = {'meta_info': {
            'prompt_tokens': 3,
            'input_token_logprobs': [[None, 10], [-1.0, 11], [-2.0, 12]],
            'input_top_logprobs': [None, [[-0.5, 21]], [[-0.25, 22]]],
        }}

    def test_prompt_token_scores_not_top_token_probabilities(self):
        self.assertEqual(prefill_scores(self.response, self.case), [(-1.0, 21), (-2.0, 22)])

    def test_rejects_different_prompt_token_identity(self):
        self.response['meta_info']['input_token_logprobs'][1][1] = 99
        with self.assertRaises(AssertionError):
            prefill_scores(self.response, self.case)

    def test_rejects_truncated_prefill(self):
        self.response['meta_info']['input_top_logprobs'].pop()
        with self.assertRaises(AssertionError):
            prefill_scores(self.response, self.case)

    def test_rejects_nonfinite_scores(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            response = copy.deepcopy(self.response)
            response['meta_info']['input_token_logprobs'][1][0] = value
            with self.assertRaises(AssertionError):
                prefill_scores(response, self.case)

    def test_numerical_error_and_argmax_agreement_are_separate(self):
        left = prefill_scores(self.response, self.case)
        right = [(-1.0, 99), (-2.0, 22)]
        stats = numerical_stats(list(zip(left, right)))
        self.assertEqual(stats['mean_abs_nat'], 0)
        self.assertEqual(stats['argmax_match_fraction'], .5)
        self.assertTrue(stats['passes_logprob_gate'])

    def test_large_errors_fail_existing_numerical_gate(self):
        left = prefill_scores(self.response, self.case)
        right = [(-1.2, 21), (-2.2, 22)]
        self.assertFalse(numerical_stats(list(zip(left, right)))['passes_logprob_gate'])


if __name__ == '__main__':
    unittest.main()
