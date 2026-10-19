"""Negative controls for independent pre-forcing raw score reconstruction."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from audit_active_raw_r65 import native_scores, numerical_stats, check_statistics


class RawAuditControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'capture.rank-0'
        self.case={'input_ids':[11,12]};self.cont=[13,14]
        self.response={'output_ids':[13,14],'meta_info':{'prompt_tokens':2,'completion_tokens':2,
            'finish_reason':{'type':'length','length':2},
            'input_token_logprobs':[[None,11],[-1.25,12]],
            'input_top_logprobs':[None,[[-.5,12]]]}}
        self.rows=[{'step':0,'token':13,'logprob':-1.,'argmax':15},
                   {'step':1,'token':14,'logprob':-2.,'argmax':16}]

    def tearDown(self):self.tmp.cleanup()

    def run_capture(self):
        self.path.write_text(''.join(json.dumps(v)+'\n' for v in self.rows))
        return native_scores(self.response,self.path,self.case,self.cont)

    def test_original_logits_source(self):
        self.assertEqual(self.run_capture(),{'prefill':[(-1.25,12)],
                         'prefill_last':[(-1.,15)],'decode':[(-2.,16)]})

    def test_nonfinite_capture_rejected(self):
        self.rows[1]['logprob']=float('nan')
        with self.assertRaises(AssertionError):self.run_capture()

    def test_nonfinite_prompt_rejected(self):
        self.response['meta_info']['input_token_logprobs'][1][0]=float('inf')
        with self.assertRaises(AssertionError):self.run_capture()

    def test_repeated_step_rejected(self):
        self.rows[1]['step']=0
        with self.assertRaises(AssertionError):self.run_capture()

    def test_wrong_forced_path_rejected(self):
        self.response['output_ids'][1]=17
        with self.assertRaises(AssertionError):self.run_capture()

    def test_wrong_capture_token_rejected(self):
        self.rows[1]['token']=17
        with self.assertRaises(AssertionError):self.run_capture()

    def test_truncated_toplogprobs_rejected(self):
        self.response['meta_info']['input_top_logprobs'].pop()
        with self.assertRaises(AssertionError):self.run_capture()

    def test_green_summary_cannot_hide_changed_scores(self):
        zero=numerical_stats([((-1.,13),(-1.,13))])
        changed=numerical_stats([((-1.25,13),(-1.,13))])
        with self.assertRaises(AssertionError):check_statistics(changed,zero)


if __name__=='__main__':unittest.main(verbosity=2)
