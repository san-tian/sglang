"""Bind before/after raw mixed audits to actual8GPU candidate placement."""
import hashlib,json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text())
def sha(n):return hashlib.sha256((R/n).read_bytes()).hexdigest()
old=read('dcp8x2-mixed-failure-audit-r116.json');new=read('dcp8x2-early-mixed-raw-audit-r120.json')
assert not old['passed'] and new['passed']
for report in [old,new]:
 assert report['raw_rank_capture_files_reparsed']==512 and report['raw_evidence_files_rehashed']==550
 assert sum(x['count'] for x in report['scores'].values())==768
for x in new['scores'].values():assert x['mean_abs_nat']==x['p99_abs_nat']==x['max_abs_nat']==0 and x['argmax_matches']==x['count']
assert sha('dcp8x2-early-mixed-r118/summary.json')==new['summary_sha256']
assert sha('dcp8x2-mixed-r31/summary.json')==old['summary_sha256']
name='qwen38-dcp8x2-d-early-r118'
obs=[json.loads(s) for s in (R/'actual-placement-observations-r118.jsonl').read_text().splitlines()]
matches=[x for x in obs if x['name']==name];assert len(matches)==1;placement=matches[0]
image=read('source-host-r118.json')['image']
assert placement['placement_matched'] and placement['image']==image
assert placement['observed_compute_gpus']==placement['expected_compute_gpus']==list(range(8))
assert len({x['pid'] for x in placement['scheduler_gpu_processes']})==8
actual=json.loads(subprocess.check_output(['docker','inspect',placement['container_id']],text=True))[0]
assert actual['Image']==image and not actual['State']['Running'] and not actual['State']['OOMKilled']
cmd=read(f'launch-{name}.json')
assert cmd[cmd.index('--gpus')+1]=='"device=0,1,2,3,4,5,6,7"'
assert cmd[cmd.index('--tp-size')+1]=='8' and cmd[cmd.index('--dcp-size')+1]=='2'
assert actual['Config']['Cmd']==cmd[cmd.index(image)+1:]
result={'passed':True,'at_unix':time.time(),'full_goal_complete':False,'old_failed_audit_sha256':sha('dcp8x2-mixed-failure-audit-r116.json'),'new_passed_audit_sha256':sha('dcp8x2-early-mixed-raw-audit-r120.json'),'old_scores':old['scores'],'new_scores':new['scores'],'actual_8GPU_placement':placement,'immutable_image':image,'launch_sha256':sha(f'launch-{name}.json'),'probe_source_sha256':sha('probe_dcp_chunk_invariance_r117.py'),'runtime_override_manifest_sha256':sha('source-runtime-candidate-r116.json'),'scope':'Exact original48-request mixed regression passes on new8GPU candidate. Not full new-image D/M, TP8-vs-DCP merged parity, natural speed, equal-byte KV, attentionDP8 or full4P4D acceptance.'}
with (R/'dcp-mixed-fix-placement-r120.json').open('x') as f:json.dump(result,f,indent=2)
print('Actual8GPU mixed-fix placement and before/after score evidence bound')
