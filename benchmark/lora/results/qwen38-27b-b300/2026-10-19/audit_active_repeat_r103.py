"""Rehash and compare the two complete active-policy natural mixed controls."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
IMAGE='sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records={};configs={};points={};proofs={}
for i in [2,3]:
    auditfile=R/f'mixed-cohort-audit-{i}-active-r93.json';audit=json.loads(auditfile.read_text());assert audit['passed'] and audit['natural_requests']==192 and audit['actual_runtime_image']==IMAGE
    directory=R/f'perf-active-mix-{i}-active-r84';assert sha(directory/'summary.json')==audit['summary_sha256']
    cfg=json.loads((directory/'server-info.json').read_text());cfg=cfg.get('server_args',cfg);configs[i]=cfg
    assert cfg['tp_size']==cfg['attn_dp_size']==1 and cfg['mem_fraction_static']==.52 and cfg['max_total_num_tokens']==65536 and cfg['max_mamba_cache_size']==256
    assert cfg['lora_merged_bf16_cache_policy']=='active' and cfg['lora_merged_bf16_cache_size_mb']==36864
    records[i]={}
    for name,h in audit['original_request_files_sha256'].items():
        p=directory/name;assert sha(p)==h;raw=json.loads(p.read_text());records[i][name]={'request':raw['request'],'output_ids':raw['events'][-1]['data']['output_ids']}
    assert len(records[i])==192
    points[i]=audit['points'];proofs[str(i)]={'cohort_audit_sha256':sha(auditfile),'raw_requests_rehashed':192}
assert records[2]==records[3],'Repeated active-policy request payload or natural output differs'
keys=['tp_size','attn_dp_size','context_length','dtype','max_total_num_tokens','max_mamba_cache_size','mem_fraction_static','attention_backend','page_size','chunked_prefill_size','max_running_requests']
assert all(configs[2][k]==configs[3][k] for k in keys)
comparison=[]
for cache in ['cold','hot']:
    controls={str(i):next(v for v in points[i] if v['cache']==cache) for i in [2,3]}
    comparison.append({'cache_label':cache,'controls':controls,'second_over_first_throughput':controls['3']['throughput_tokens_s_mean']/controls['2']['throughput_tokens_s_mean']})
result={'passed':True,'original_requests_rehashed':384,'identical_request_payloads':True,'natural_outputs_identical':True,'comparisons':comparison,'proofs':proofs,'source_sha256':sha(Path(__file__)),'scope':'Two adjacent complete active-policy R84 controls, GPU2/R55/mem.52/KV65536/SSM256, actual KV cached_tokens reported. Reproducibility observation only: reverse LRU/no-cache and full1152 matrix remain required. No isolated policy speedup, larger topology, true prefix-hit, capacity or overall acceptance.'}
out=R/'active-repeat-comparison-r103.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
