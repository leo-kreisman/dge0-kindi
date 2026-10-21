"""Synthetic timing checks only; these do not measure model performance."""
import unittest

from verify_mac import timing_metrics, check_modes, GIB
import copy


class TimingTests(unittest.TestCase):
    def run_fixture(self):
        rows=[]
        for case,cached in zip(['initial','repeat','edited_tail','changed_prefix'],[0,2000,2000,0]):
            rows.append({'case':case,'tokens':[1,2], 'stats':{
                'prefix_cache':{'last_cached_tokens':cached},
                'memory':{'resident_peak_bytes':GIB,'footprint_bytes':GIB},
                'joint_budget':{'managed_peak_bytes':100,'managed_limit_bytes':200},
                'expert_io':{'packed_read_calls':1}}})
        return {mode:{'rows':copy.deepcopy(rows)} for mode in ('baseline','io','cached','packed')}

    def test_comparison_rejects_missing_rows(self):
        run=self.run_fixture()
        self.assertTrue(all(check_modes(run).values()))
        run['io']['rows'].pop()
        self.assertFalse(all(check_modes(run).values()))

    def test_comparison_requires_parity_memory_and_packed_use(self):
        for mutate in [
            lambda r:r['io']['rows'][0].update(tokens=[9]),
            lambda r:r['cached']['rows'][0]['stats']['memory'].update(footprint_bytes=0),
            lambda r:r['packed']['rows'][0]['stats']['memory'].update(footprint_bytes=9*GIB),
            lambda r:[x['stats']['expert_io'].update(packed_read_calls=0) for x in r['packed']['rows']],
        ]:
            run=self.run_fixture(); mutate(run)
            self.assertFalse(all(check_modes(run).values()))

    def test_adaptive_and_restart_require_telemetry(self):
        run=self.run_fixture()
        for mode,loaded in [('adaptive',0),('profiled',12)]:
            run[mode]=copy.deepcopy(run['cached'])
            for row in run[mode]['rows']:
                row['stats']['expert_residency']={'enabled':True,'observations':20,'loaded_entries':loaded}
        self.assertTrue(all(check_modes(run).values()))
        for mode in ('adaptive','profiled'):
            bad=copy.deepcopy(run)
            del bad[mode]['rows'][0]['stats']['expert_residency']
            self.assertFalse(all(check_modes(bad).values()))
        bad=copy.deepcopy(run)
        for row in bad['profiled']['rows']:
            row['stats']['expert_residency']['loaded_entries']=0
        self.assertFalse(all(check_modes(bad).values()))

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
