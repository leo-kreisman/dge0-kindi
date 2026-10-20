"""Synthetic timing checks only; these do not measure model performance."""
import unittest

from verify_mac import timing_metrics


class TimingTests(unittest.TestCase):
    def test_prefill_and_terminal_cleanup_excluded_from_decode(self):
        result = timing_metrics(10, [20, 20.5, 21], 22)
        self.assertEqual(result['decode_tokens_per_second'], 2)
        self.assertEqual(result['end_to_end_tokens_per_second'], 0.25)
        self.assertEqual(result['first_token_seconds'], 10)
        self.assertEqual(result['wall_seconds'], 12)
        self.assertEqual(result['token_elapsed_seconds'], [10, 10.5, 11])
        self.assertTrue(result['short_sample'])

    def test_no_decode_estimate_from_one_or_zero_tokens(self):
        for times in [[], [20]]:
            self.assertIsNone(timing_metrics(10, times, 22)['decode_tokens_per_second'])
        self.assertIsNone(timing_metrics(10, [], 22)['first_token_seconds'])

    def test_zero_clock_interval_does_not_invent_speed(self):
        self.assertIsNone(timing_metrics(10, [10, 10], 10)['decode_tokens_per_second'])


if __name__ == '__main__':
    unittest.main()
