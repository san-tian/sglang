"""Observe only the completed DCP2 merged baseline, before D/base controls exist.

Reuse the already versioned R100 per-side original-evidence audit verbatim.
This report cannot accept DCP2 precision, MoL speed, capacity or another topology.
"""
import ast,hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
source=R/'audit_topology_native_r100.py'
# Bind the exact original auditor via the independently sealed R102 source list.
manifest=json.loads((R/'source-topology-audits-r102.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_files'][source.name]
original=source.read_text();tree=ast.parse(original)
side_loop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='side')
# R85 produces its normal kernel-summary after all three sides. Store the same
# analyzer's baseline summary at a separate path, leaving that future path free.
d=R/'perf-dcp8x2-m-r30'
assert json.loads((d/'complete.json').read_text())=={'requests':636,'profile':True}
summary_path=d/'kernel-summary-baseline-r114.json'
assert not summary_path.exists()
subprocess.run(['python3',str(R/'analyze_profiles_r30.py'),str(d/'profile'),'--output',str(summary_path)],check=True,capture_output=True)
# Execute exact original prefix+one loop body, changing only the chosen side,
# parser argv and location of the already existing kernel-summary format.
selected=original[:original.index("assert rows_by_side['d']==rows_by_side['m']")]
assert selected.count("for side in ['d','m','base']:")==1
selected=selected.replace("for side in ['d','m','base']:","for side in ['m']:",1)
selected=selected.replace("directory/'kernel-summary.json'","directory/'kernel-summary-baseline-r114.json'")
import sys
saved=sys.argv;sys.argv=[str(source),'dcp8x2'];ns={'__file__':str(source),'__name__':'baseline_fragment'}
try:exec(compile(selected,str(source)+'[merge-only]','exec'),ns)
finally:sys.argv=saved
assert ns['latency_statistics']==192
rows=ns['rows_by_side']['m'];traces=ns['traces_by_side']['m']
assert len(rows)==636 and len(traces)==8
cfg=ns['configurations']['m'];assert cfg['model_path'].rstrip('/')=='/work/qwen38-mol-development-20261019/merged/real_r16'
original_rows=[json.loads(s) for s in (d/'requests.jsonl').read_text().splitlines()]
hits={cache:sorted({v['cached_tokens'] for v in original_rows if v['cache']==cache}) for cache in ['cold','hot']}
report={'scope':'Completed DCP2 independent-merged baseline only; no dynamic/base speed ratio, D/M precision, capacity, cross-topology numerical parity, attentionDP or4P4D acceptance.',
 'baseline_original_evidence_checked':True,'whole_topology_accepted':False,'native_requests':636,'groups':48,'speed_points':16,'latency_statistics_recomputed':192,
 'raw_payloads_sha256':ns['proofs']['m'],'summary_sha256':hashlib.sha256((d/'summary.json').read_bytes()).hexdigest(),
 'complete_sha256':hashlib.sha256((d/'complete.json').read_bytes()).hexdigest(),'auditor_reused_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'observer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'selected_audit_fragment_sha256':hashlib.sha256(selected.encode()).hexdigest(),
 'source_manifest_sha256':hashlib.sha256((R/'source-topology-audits-r102.json').read_bytes()).hexdigest(),
 'actual_placement':ns['placement_proofs']['m'],'launch':ns['launches']['m'],
 'configuration':{k:cfg[k] for k in ['model_path','enable_lora','tp_size','attn_dp_size','dcp_size','dcp_comm_backend','dtype','context_length','mem_fraction_static','max_mamba_cache_size','max_running_requests','max_total_num_tokens','attention_backend','linear_attn_prefill_backend','bf16_gemm_backend','enable_deterministic_inference','disable_cuda_graph']},
 'actual_cached_tokens_values':hits,'points':ns['summaries']['m']['points'],'rank_kernel_traces':traces,
 'limitations':'TTFT and TPOT independently reconstructed from original SSE; E2E percentile inputs retain original client HTTP elapsed measurements. No forcing during timing. Kernel event counts/names are not critical-path or E2E time attribution. Cold/hot labels alone do not prove KV reuse.'}
out=R/'dcp8x2-merge-baseline-observation-r114.json'
with out.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({k:report[k] for k in ['scope','baseline_original_evidence_checked','whole_topology_accepted','native_requests','groups','speed_points','latency_statistics_recomputed','actual_cached_tokens_values']}))
