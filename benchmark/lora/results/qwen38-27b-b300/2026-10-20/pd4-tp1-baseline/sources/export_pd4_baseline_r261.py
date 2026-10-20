"""Export baseline evidence, never weights, compiled caches, or old-container environment."""
import csv,hashlib,json,shutil,tarfile,time
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'pd4-baseline-r253';D=R/'publish-pd4-r261';D.mkdir(exist_ok=False)
audit=json.loads((E/'audit-r256.json').read_text());assert audit['all_five_phases_complete']
assert json.loads((E/'final-state-r260/complete.json').read_text())['all_208_overlay_hashes_match']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n in ['topology.json','switch-before.json','cancelled-legacy-queues.json','suite-progress.json','suite-finished.json','baseline-complete-r264.json','audit-r256.json','residency-audit-r265.json','host-client-cpu-r258.jsonl']:
 if (E/n).exists():shutil.copyfile(E/n,D/n)
for p in E.glob('launch-*.json'):shutil.copyfile(p,D/p.name)
(D/'sources').mkdir()
for n in ['launch_pd4_baseline_r253.py','measure_pd4_baseline_r254.py','run_pd4_baseline_r255.py','audit_pd4_baseline_r256.py','monitor_pd4_cpu_r258.py','finalize_pd4_baseline_r260.py','export_pd4_baseline_r261.py','residency_pd4_r262.py','capacity_pd4_r263.py','audit_residency_pd4_r265.py','precision_slots_pd4_r266.py']:shutil.copyfile(R/n,D/'sources'/n)
for phase in ['smoke','speed','natural','capacity','capacity-r263','slots','residency-paced','slots-precision-r266','final-state-r260']:
 with tarfile.open(D/(phase+'-raw.tgz'),'w:gz',compresslevel=6) as t:t.add(E/phase,arcname=phase)
shutil.copyfile(E/'final-state-r260/runtime-overlay-audit.json',D/'runtime-overlay-audit.json')
with (D/'throughput.csv').open('w') as f:
 fields=['phase','input_tokens','max_output_tokens','concurrency','round','output_tokens_s','input_tokens_s','elapsed_s','ttft_mean_s','ttft_p95_s','tpot_mean_s','tpot_p95_s']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for phase in ['speed','natural']:
  for g in audit['phases'][phase]['groups']:
   if g['label'].startswith('warm-'):continue
   w.writerow({'phase':phase,'input_tokens':g['input_tokens'],'max_output_tokens':g['max_output_tokens'],'concurrency':g['concurrency'],'round':g['label'].rsplit('-r',1)[1],'output_tokens_s':g['output_tokens_s'],'input_tokens_s':g['input_tokens_s'],'elapsed_s':g['elapsed_s'],'ttft_mean_s':g['ttft_s']['mean'],'ttft_p95_s':g['ttft_s']['p95'],'tpot_mean_s':g['tpot_s']['mean'] if g['tpot_s'] else None,'tpot_p95_s':g['tpot_s']['p95'] if g['tpot_s'] else None})
shutil.copyfile(E/'slots-precision-r266/comparison.json',D/'mixed-lora-logprob-comparison.json')
(D/'raw-files-sha256.json').write_text(json.dumps({str(p.relative_to(E)):sha(p) for phase in ['smoke','speed','natural','capacity','capacity-r263','slots','residency-paced','slots-precision-r266','final-state-r260'] for p in (E/phase).rglob('*') if p.is_file()},indent=2))
(D/'runtime.json').write_text(json.dumps({'model':'Qwen/Qwen3.8-27B-FP8','model_revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a','image':audit['topology']['image'],'base_git_HEAD':'3c560089e1818a5217a73f2168273485a5350341','fork_overlay_revision':audit['topology']['runtime_overlay'],'verified_files':208,'note':'Inherited .git/OCI label is upstream; overlay hashes define actual engine changes.'},indent=2))
(D/'manifest.json').write_text(json.dumps({'unix_s':time.time(),'files_sha256':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()}},indent=2))
for p in D.rglob('*'):
 if p.is_file():assert p.stat().st_size<95*1024**2,(str(p),p.stat().st_size)
print(json.dumps({'files':len(list(D.rglob('*'))),'MiB':sum(p.stat().st_size for p in D.rglob('*') if p.is_file())/1024**2}))
