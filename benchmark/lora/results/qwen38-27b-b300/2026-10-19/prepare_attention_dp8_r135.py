"""Continue the unaffected attentionDP8 protocol on original immutable R28."""
import ast
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parent
OLD='sha256:1092e415c3dfa77fa0cb4da3b6ba3dea469bd5d32687e7a241e0bcc028ebc878'
IMAGE='sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'
COMMIT='a589b336780051bbf26755cf1ea0aba56fbd029c'
def save(name,text):
    p=R/name
    assert not p.exists(),name
    ast.parse(text)
    p.write_text(text)
def sha(name):return hashlib.sha256((R/name).read_bytes()).hexdigest()

helper=(R/'host_common_r118.py').read_text().replace('r118','r135').replace(OLD,IMAGE)
save('host_common_r135.py',helper)
old=(R/'run_eight_gpu_recovery_r118.py').read_text()
mix=old[old.index('def mixed('):old.index('# Fail fast')]
loop=old[old.index("for top,dpa,extra in ["):old.rindex("(R/'eight-gpu-recovery-complete-r118.json')")]
loop=loop.replace("[('dcp8x2',False,['--dcp-size','2','--dcp-comm-backend','a2a']),('dpa8',True,[])]","[('dpa8',True,[])]")
assert 'for top,dpa,extra in ' in loop and loop.count('perf(name,port,top,')==3
assert loop.count('physical_rows=8191 if dpa else 32767')==1
body=(mix+loop).replace('r118','r135').replace('mamba=32 if dpa else 256','mamba=256')
header='''"""Original-image attentionDP8 continuation. DCP numerical failure remains rejected."""
from host_common_r135 import *
manifest=json.loads((R/'source-host-r135.json').read_text())
for n,h in manifest['source_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
assert not Path('/proc/1264524').exists()
assert not (R/'eight-gpu-recovery-complete-r118.json').exists()
assert json.loads((R/'tp8-vs-dcp8x2-candidate-prefill-r125.json').read_text())['passed'] is False
waitleases(['real-qkv-oracle-state-r129.json'],list(range(8)),timeout=172800)
oracle=json.loads((R/'real-qkv-oracle-state-r129.json').read_text())
assert oracle['returncode']==0 and not oracle['state']['Running']
assert json.loads((R/'real-qkv-r129/oracle-finished.json').read_text())['diagnosis_completed']
assert shutil.disk_usage('/').free>4*1024**3
assert not (R/'attention-dp8-resource-released-r135.json').exists()
'''
ending="""item=json.loads((R/'dpa8-precision-perf-capacity-complete-r135.json').read_text())
assert item['raw_protocol_complete'] and (item['precision_pairs'],item['gdn_pairs'],item['mixed_requests'])==(240,8,48)
assert item['performance']['requests_per_side']==636 and item['performance']['outputs_equal']
for side in ['m','d','base']:
 complete=json.loads((R/f'perf-dpa8-{side}-r135/complete.json').read_text())
 assert complete['requests']==636 and complete['profile']
for label in ['full','gdn']:
 for stage in ['reference','dynamic']:
  directory=R/f'dpa8-{label}-{stage}-r135'
  seal=json.loads((directory/'seal.json').read_text())
  assert all(hashlib.sha256((directory/name).read_bytes()).hexdigest()==h for name,h in seal['files'].items())
rows=[json.loads(line) for line in (R/'actual-placement-observations-r135.jsonl').read_text().splitlines()]
assert {x['name'] for x in rows}=={f'qwen38-dpa8-{role}-r135' for role in ['m-real_r16','m-gdn_fused','d','base','capacity']}
assert all(x['placement_matched'] for x in rows)
(R/'attention-dp8-cohort-complete-r135.json').write_text(json.dumps({'topologies':['dpa8'],'image':manifest['image'],'raw_protocol_complete':True,'independent_raw_audits_pending':True,'full_goal_complete':False}))
"""
driver=header+'try:\n'+''.join(' '+line+'\n' for line in (body+ending).splitlines())+'''finally:
 # This releases only an experiment resource lease, including on failure.
 # It never asserts model correctness or substitutes for R31/R118 success.
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
 if not apps.strip():
  state={'at_unix':time.time(),'resource_released':True,'observed_gpu_processes':apps,
         'cohort_completed':(R/'attention-dp8-cohort-complete-r135.json').exists(),
         'model_accuracy_accepted':False,'full_goal_complete':False,
         'scope':'GPU lease release only; no correctness/speed/capacity acceptance.'}
  with (R/'attention-dp8-resource-released-r135.json').open('x') as f:json.dump(state,f,indent=2)
'''
save('run_attention_dp8_r135.py',driver)
adaptations={}
for label in ['precision','native','mixed','capacity']:
    src=f'audit_recovery_{label}_r121.py'
    dst=f'audit_attention_dp8_{label}_r135.py'
    text=(R/src).read_text().replace('r118','r135').replace('r121','r135').replace(OLD,IMAGE)
    if label=='precision':
        text=text.replace("assert provenance['base_runtime_commit']==COMMIT", "assert provenance['runtime_commit']==provenance['commit']==COMMIT")
        text=text.replace("        assert provenance['runtime_source_override_manifest']=='source-runtime-candidate-r116.json'\n        assert digest(root/'source-runtime-candidate-r116.json')==provenance['source_files']['source-runtime-candidate-r116.json']\n",'')
        text=text.replace("'base_runtime_commit':COMMIT,'runtime_override_manifest_sha256':digest(args.root/'source-runtime-candidate-r116.json')", "'runtime_commit':COMMIT,'runtime_source_manifest_sha256':digest(args.root/'source-host-r135.json')")
        originals={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((R/src).read_text()).body if isinstance(n,ast.FunctionDef)}
        actual={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        for name in ['numerical_stats','check_statistics','native_scores','capture_scores']:
            assert originals[name]==actual[name],name
    text=text.replace('(32 if dp==8 else 256)','256').replace('(32 if dpa and tp==8 else 256)','256')
    save(dst,text)
    adaptations[src]={'source_sha256':sha(src),'adapted':dst,'adapted_sha256':sha(dst)}

watch=(R/'watch_recovery_audits_r122.py').read_text()
watch=watch.replace('source-recovery-audits-r122.json','source-attention-dp8-audits-r135.json')
watch=watch.replace('dcp-recovery-process-r118.json','attention-dp8-process-r135.json')
watch=watch.replace('run_eight_gpu_recovery_r118.py','run_attention_dp8_r135.py')
watch=watch.replace("['dcp8x2','dpa8']","['dpa8']")
for label in ['precision','native','mixed','capacity']:
    watch=watch.replace(f'audit_recovery_{label}_r121.py',f'audit_attention_dp8_{label}_r135.py')
watch=watch.replace('r118','r135').replace('r121','r135').replace('r122','r135')
watch=watch.replace('topology-recovery-audits-finished-r135.json','attention-dp8-raw-audits-finished-r135.json')
watch=watch.replace('Full recovered DCP2 and attentionDP8 raw audits','Original-image attentionDP8 full raw audits')
save('watch_attention_dp8_r135.py',watch)

files=['perf_endpoint_r85.py','capacity_residency_r61.py','analyze_profiles_r30.py',
       'client-r30/compare_frozen.py','client-r30/compare_merged.py','client-r30/compare_mixed.py',
       'launch-qwen38-tp2-d-r28.json','watch_actual_placement_r92.py',
       'host_common_r135.py','run_attention_dp8_r135.py']
record={'image':IMAGE,'runtime_image':IMAGE,'commit':COMMIT,'runtime_commit':COMMIT,
        'source_files':{n:sha(n) for n in files},'mamba_cache_cli_global':256,'expected_mamba_cache_per_DP_worker':32,'mamba_slots_per_request':5,'scope':'Independent original-image attentionDP8 continuation. DCP2 rejected candidate remains unresolved. Original clients, counts and numerical gates retained; no original completion marker is synthesized.'}
with (R/'source-host-r135.json').open('x') as f:json.dump(record,f,indent=2)
files=[f'audit_attention_dp8_{x}_r135.py' for x in ['precision','native','mixed','capacity']]+[
    'watch_attention_dp8_r135.py','source-host-r135.json','audit_active_raw_r65.py','audit_topology_original_r99.py']
record={'runtime_image':IMAGE,'source_files':{n:sha(n) for n in files},'adaptations':adaptations,
        'scope':'Namespace/image/provenance adaptations only; finite, numerical, raw-score, inventory, native-latency, trace, actual-placement and same-scrape residency assertions retained.'}
with (R/'source-attention-dp8-audits-r135.json').open('x') as f:json.dump(record,f,indent=2)
print('Prepared original-image attentionDP8 full protocol and independent raw auditors')
