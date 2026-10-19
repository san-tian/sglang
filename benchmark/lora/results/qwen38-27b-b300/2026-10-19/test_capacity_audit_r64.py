"""Negative controls for false capacity gains; CPU-only, no GPU acceptance."""
import unittest
from audit_capacity_residency_r64 import IncompleteSnapshot,normalize_snapshot,parse_prometheus,peak_same_scrape

def snapshot(values, metric='kv_used_tokens'):
    return [{'metric':metric,'labels':{'tp_rank':str(i),'pp_rank':'0'},'value':v} for i,v in enumerate(values)]

class CapacityEvidenceTests(unittest.TestCase):
    def test_tp_duplicates_are_not_summed(self):
        rows=[]
        for name in ('kv_used_tokens','num_running_reqs','decode_sum_seq_lens'):
            rows+=snapshot([123]*8,name)
        x=normalize_snapshot(rows,1)
        self.assertEqual(x['totals']['kv_used_tokens'],123)
        self.assertEqual(x['leaders'],[0])

    def test_dp_independent_workers_are_summed(self):
        rows=[]
        for name in ('kv_used_tokens','num_running_reqs','decode_sum_seq_lens'):
            rows+=snapshot(range(1,9),name)
        self.assertEqual(normalize_snapshot(rows,8)['totals']['kv_used_tokens'],36)

    def test_distinct_worker_time_peaks_never_add(self):
        samples=[]
        for stamp,values in [(1,[80,0,0,0,0,0,0,0]),(2,[0,70,0,0,0,0,0,0])]:
            rows=[]
            for name in ('kv_used_tokens','num_running_reqs','decode_sum_seq_lens'):
                rows+=snapshot(values,name)
            samples.append({'monotonic':stamp,'normalized':normalize_snapshot(rows,8)})
        # One worker may have80 at t1, another70 at t2; observed peak is80, not150.
        self.assertEqual(peak_same_scrape(samples)['normalized']['totals']['kv_used_tokens'],80)

    def test_missing_dp_workers_cannot_pass(self):
        rows=[]
        for name in ('kv_used_tokens','num_running_reqs','decode_sum_seq_lens'):
            rows+=snapshot([1]*7,name)
        with self.assertRaises(IncompleteSnapshot):normalize_snapshot(rows,8)

    def test_priorities_excluded_and_escaped_labels_preserved(self):
        raw='sglang:kv_used_tokens{tp_rank="0",priority="",model_name="a,\\\"b"} 4\nsglang:kv_used_tokens{tp_rank="0",priority="1"} 1000\n'
        rows=parse_prometheus(raw)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['value'],4)
        self.assertEqual(rows[0]['labels']['model_name'],'a,"b')

    def test_nonfinite_metrics_cannot_pass(self):
        for value in ('NaN','+Inf','-Inf'):
            with self.assertRaises(AssertionError):parse_prometheus('sglang:kv_used_tokens{tp_rank="0"} '+value)

    def test_duplicate_or_ambiguous_leaders_cannot_pass(self):
        raw='sglang:kv_used_tokens{tp_rank="0"} 4\n'*2
        with self.assertRaises(AssertionError):parse_prometheus(raw)
        rows=[]
        for name in ('kv_used_tokens','num_running_reqs','decode_sum_seq_lens'):
            rows+=snapshot([1]*8,name)
        extra=snapshot([2])[0];extra['labels']['model_name']='different-model';rows.append(extra)
        with self.assertRaises(AssertionError):normalize_snapshot(rows,1)

if __name__=='__main__':unittest.main()
