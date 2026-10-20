"""Aggregate sealed measurements for the requested progress report; no GPU rerun."""
import argparse,csv,hashlib,json,math
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(snapshot,evidence):
    manifest=json.loads((snapshot/'manifest.json').read_text());proof={}
    def read(name):
        p=snapshot/name;assert digest(p)==manifest['files_sha256'][name],name
        proof[name]=digest(p);return json.loads(p.read_text())
    live=json.loads((evidence/'live-progress-r153.json').read_text())
    rows=[];natural_identity=[]
    def add(config,control,point,source):
        row={'configuration':config,'control':control,'input_tokens':point['input_tokens'],
             'output_tokens':128 if point['input_tokens']==128 else 256,
             'concurrency':point['concurrency'],'cache_label':point['cache'],
             'requests':point['requests'],'output_tokens_s_mean':point['throughput_tokens_s_mean'],
             'e2e_p95_s':point['e2e_s']['p95'],'ttft_p95_s':point['ttft_s']['p95'],
             'tpot_p95_ms':1000*point['tpot_s']['p95'],'source_report':source}
        assert all(math.isfinite(row[k]) and row[k]>0 for k in ('output_tokens_s_mean','e2e_p95_s','ttft_p95_s','tpot_p95_ms'))
        rows.append(row)
    source='cache-threeway-comparison-r49.json';x=read(source)
    assert x['requests_per_side']==636 and set(x['natural_outputs_matching_independent_M'].values())=={636}
    for point in x['points']:
        for side,control in [('cached','D-LRU'),('uncached','D-nocache'),('merged','M')]:add('TP1 R34',control,point[side],source)
    for config,source in [('TP1 R55 active','active-single-threeway-audit-r71.json'),('TP2','tp2-threeway-audit-r58.json'),('attention DP2','dpa2-threeway-audit-r58.json'),('TP4','tp4-threeway-audit-r58.json'),('attention DP4','dpa4-threeway-audit-r58.json'),('TP8','tp8-native-kernel-raw-audit-r100.json')]:
        x=read(source);assert x['passed'] and x['natural_D_M_outputs_equal'] and x['raw_native_payloads_audited']==1908
        natural_identity.append({'configuration':config,'D_M_equal_sequences':636,'D_B_equal_sequences':x['base_output_ids_equal_M_count'],'requests_per_control':636,'source_report':source,'interpretation':'Natural sequence agreement only; weights differ between adapted model and original base. No task quality or shared-prefix D/B logprob comparison.'})
        for point in x['speed_points']:
            for side,control in [('d','D'),('m','M'),('base','B')]:add(config,control,point['controls'][side],source)
    source='dpa8-native-stage-raw-audit-r151.json';x=read(source)
    assert x['passed'] and x['natural_requests']==1272 and x['natural_output_ids_match']
    for side,control in [('d','D'),('m','M')]:
        for point in x['controls'][side]['speed_points']:add('attention DP8',control,point,source)
    source='dcp8x2-merge-baseline-observation-r114.json';x=read(source);assert x['baseline_original_evidence_checked'] and x['native_requests']==636
    for point in x['points']:add('TP8+DCP2', 'M',point,source)
    assert len(rows)==384
    assert len({tuple(x[k] for k in ('configuration','control','input_tokens','concurrency','cache_label')) for x in rows})==384
    for config in {r['configuration'] for r in rows}:
        for control in {r['control'] for r in rows if r['configuration']==config}:
            selected=[r for r in rows if r['configuration']==config and r['control']==control]
            assert len(selected)==16
            assert {(r['input_tokens'],r['concurrency'],r['cache_label']) for r in selected}=={(t,c,m) for t in (128,2048) for c in (1,4,16,32) for m in ('cold','hot')}
    precision=[]
    for label,name in [('TP1 R55 active','active-cache-raw-accuracy-audit-r65.json'),('TP4','tp4-independent-audit-r40.json'),('attention DP4','dpa4-independent-audit-r40.json'),('TP8','tp8-original-precision-audit-r99.json'),('attention DP8','dpa8-precision-stage-raw-audit-r145.json'),('TP8+DCP2','dcp8x2-original-precision-stage-r115.json')]:
        z=read(name);a=z['accuracy'];scores={}
        for phase in ('prefill','prefill_last','decode'):
            parts=[v['scores'][phase] for v in a.values()]
            assert all(v['mean_abs_nat']==v['p99_abs_nat']==v['max_abs_nat']==0 and v['argmax_match_fraction']==1 for v in parts)
            scores[phase]=sum(v['count'] for v in parts)
        assert sum(scores.values())==114692
        precision.append({'configuration':label,'comparison':'D versus independent M, same topology','case_pairs':248,'scores':scores,'mean_P99_max_nat':[0,0,0],'argmax_match_fraction':1,'source_report':name})
    # Earlier TP2/DP2 published summaries are re-read, not relabeled as new rank/raw audits.
    for label,name in [('TP2','tp2-full-d-r29/summary.json'),('attention DP2','dpa2-full-d-r29/summary.json')]:
        p=evidence/name;z=json.loads(p.read_text());assert (z['cases'],z['rounds'],z['decode_tokens'])==(30,4,256)
        assert all(v['mean_abs_nat']==v['p99_abs_nat']==v['max_abs_nat']==0 and v['argmax_match_fraction']==1 for v in z['scores'].values())
        assert sum(v['count'] for v in z['scores'].values())==111184
        precision.append({'configuration':label,'comparison':'D versus independent M, same topology','case_pairs':240,'scores':{k:v['count'] for k,v in z['scores'].items()},'mean_P99_max_nat':[0,0,0],'argmax_match_fraction':1,'source_summary':name,'source_summary_sha256':digest(p),'scope':'Existing summary re-read only; not a new complete raw audit.'})
    mixed=read('mixed-six-comparison-r104.json');assert mixed['passed'] and mixed['native_requests']==1152 and mixed['natural_outputs_identical']
    million=read('million-measured-full-audit-r60.json');assert million['passed'] and million['formal_native_requests_audited']==18
    failures={k:read(v) for k,v in [('DCP2_mixed','dcp8x2-mixed-failure-audit-r116.json'),('DCP2_cross_topology','tp8-vs-dcp8x2-merged-prefill-r113.json'),('DP8_cross_topology','tp8-vs-attention-dp8-prefill-r139.json')]}
    prefix=read('active-prefix-raw-audit-r70.json');assert prefix['passed'] and prefix['scores']['max_abs_nat']==0
    capacity=read('capacity-tp8-independent-audit-r64.json');assert capacity['passed'] and capacity['native_requests_audited']==13
    report={'as_of_utc':live['as_of_utc'],'full_goal_complete':False,'new_GPU_measurements_from_this_aggregation':False,
            'model':live['model'],'model_directory_label':'Qwen3.8-27B','MoE_tested':False,
            'definitions':{'D':'original base plus dynamically mounted LoRA','M':'same base/adapter independently merged offline, serving without LoRA','B':'original base, no adapter'},
            'native_speed_points':rows,'precision':precision,'natural_D_B_sequence_agreement':natural_identity,
            'zero_base_control':{'scores':32,'cache_modes':['cold','hot'],'max_abs_nat':0,'hot_cached_tokens':1600,'source_report':'active-prefix-raw-audit-r70.json','scope':'Zero adapter compared to original base at last-prefill+15decode scores per cold/hot case; no real-adapter D/B shared-prefix or task-quality metric.'},
            'mixed_policy':{k:mixed[k] for k in ('native_requests','identical_request_payloads','natural_outputs_identical','pooled_points','throughput_ratios','scope')},
            'million_context':{k:million[k] for k in ('formal_native_requests_audited','total_tokens_per_request','points','logprob_alignment_measured','long_context_quality_accepted','scope')},
            'pending':{'DP8_base_native_complete_at_snapshot':live['base_native_complete'],'DP8_completed_base_groups':live['base_completed_groups'],'DP8_required_base_groups':48,'DP8_equal_byte_KV_report_exists':live['capacity_report_exists'],'DP8_resource_released':live['resource_released'],'full_4P4D_accepted':False,'R55_TP4_DP4_and_isolated_speed_accepted':False},
            'TP8_equal_byte_capacity':{'bytes_per_rank':capacity['full_attention_KV_bytes_per_rank'],'requests':13,'raw_gauge_snapshots':capacity['memory_gauge_snapshots_verified'],'c8_observed_peak_tokens':capacity['cases'][2]['peak_observed_logical_KV_tokens'],'running_requests_at_same_peak':capacity['cases'][2]['running_requests_at_that_same_peak']},
            'failure_statistics':{'DCP2_mixed_decode':failures['DCP2_mixed']['scores']['decode'],'DCP2_cross_topology':failures['DCP2_cross_topology']['prefill'],'DP8_cross_topology':failures['DP8_cross_topology']['prefill']},
            'failure_reports':{k:manifest['files_sha256'][v] for k,v in [('DCP2_mixed','dcp8x2-mixed-failure-audit-r116.json'),('DCP2_cross_topology','tp8-vs-dcp8x2-merged-prefill-r113.json'),('DP8_cross_topology','tp8-vs-attention-dp8-prefill-r139.json')]},
            'source_reports_sha256':proof,'live_observation_sha256':digest(evidence/'live-progress-r153.json'),
            'limitations':['R28/R34/R55 are distinct runtimes; only compute paired within-cohort ratios.','Cold/hot refers to prefix-KV flush; projection caches are prewarmed, first fill and registration are separate.','Separate-time controls and background work prevent isolated universal speed claims.','TP1 fixedKV65536 can queue at c32*2304=73728 tokens.','Natural output agreement and logprob alignment do not establish task accuracy or long-context quality.','Original base has different weights; real-adapter D/B shared-prefix logprob difference and held-out quality are not measured.']}
    (evidence/'progress-overview-r153.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (evidence/'throughput-all-configurations-r153.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'native_speed_points':len(rows),'source_reports_verified':len(proof),'precision_configurations':len(precision),'full_goal_complete':False}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();main(a.snapshot,a.evidence)
