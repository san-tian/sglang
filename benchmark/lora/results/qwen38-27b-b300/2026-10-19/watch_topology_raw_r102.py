"""Read-only per-topology original precision/native/mixed/placement/capacity audits."""
import hashlib,json,subprocess,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
source=json.loads((R/'source-topology-audits-r102.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    for n,h in source['source_files'].items():assert sha(R/n)==h,n
    assert sha(R/'source-host-r85.json')==source['parent_source_manifest_sha256']
verify();deadline=time.monotonic()+172800
result={'passed':False,'full_goal_complete':False,'topologies':{},'scope':'Full TP8/DCP2/attentionDP8 original precision, native SSE statistics, mixed routing and actual placement with separate equal-byte KV reports. No overall goal acceptance from this observer alone.'}
def wait(path,pid,keyword):
    while not path.exists():
        proc=Path(f'/proc/{pid}')
        assert proc.exists() and (proc/'stat').read_text().split()[2]!='Z',('Actual coordinator terminal without required evidence',path.name,pid)
        assert keyword.encode() in (proc/'cmdline').read_bytes()
        if time.monotonic()>deadline:raise TimeoutError('Audit observation window expired; GPU workers must not restart merely for this timeout')
        time.sleep(5)
def run(script,args,output,log):
    assert not (R/output).exists();verify()
    with (R/log).open('x') as f:subprocess.run(['python3',str(R/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    d=json.loads((R/output).read_text());assert d['passed'];return d
try:
    for top in ['tp8','dcp8x2','dpa8']:
        wait(R/f'{top}-precision-perf-capacity-complete-r31.json',954883,'run_eight_gpu_suite_r85.py');verify()
        precision_name=f'{top}-original-precision-audit-r99.json'
        precision=run('audit_topology_original_r99.py',['--root',str(R),'--topology',top,'--output',str(R/precision_name)],precision_name,f'{top}-original-precision-audit-r99.log')
        assert precision['case_pairs']==248 and precision['score_count']==114692
        native_name=f'{top}-native-kernel-raw-audit-r100.json'
        native=run('audit_topology_native_r100.py',[top],native_name,f'{top}-native-kernel-raw-audit-r100.log')
        assert native['raw_native_payloads_audited']==1908 and native['latency_statistics_recomputed']==576
        mixed_name=f'{top}-mixed-raw-audit-r101.json'
        mixed=run('audit_topology_mixed_r101.py',['--root',str(R),'--topology',top,'--output',str(R/mixed_name)],mixed_name,f'{top}-mixed-raw-audit-r101.log')
        assert mixed['mixed_requests']==48 and sum(s['count'] for s in mixed['scores'].values())==768
        expected=[f'qwen38-{top}-m-real_r16-'+('r85' if top=='tp8' else 'r31'),f'qwen38-{top}-m-gdn_fused-r31',f'qwen38-{top}-d-r31',f'qwen38-{top}-base-r41',f'qwen38-{top}-capacity-r31']
        observations=[json.loads(s) for s in (R/'actual-placement-observations-r92.jsonl').read_text().splitlines()]
        placement={}
        for name in expected:
            rows=[r for r in observations if r['name']==name];assert len(rows)==1
            r=rows[0];assert r['placement_matched'] and r['image']==source['runtime_image'] and r['observed_compute_gpus']==r['expected_compute_gpus']==list(range(8))
            assert len({v['pid'] for v in r['scheduler_gpu_processes']})==8
            placement[name]={'container_id':r['container_id'],'observed_compute_gpus':r['observed_compute_gpus'],'unix_s':r['unix_s']}
        cap_name=f'capacity-{top}-independent-audit-r64.json'
        wait(R/cap_name,893959,'watch_capacity_audit_r64.py');cap=json.loads((R/cap_name).read_text())
        assert cap['passed'] and cap['native_requests_audited']==13 and cap['full_attention_KV_bytes_per_rank']==536870912
        detail={'passed':True,'precision_audit_sha256':sha(R/precision_name),'native_audit_sha256':sha(R/native_name),'mixed_audit_sha256':sha(R/mixed_name),'capacity_audit_sha256':sha(R/cap_name),'actual_placement':placement,'full_goal_complete':False}
        with (R/f'{top}-original-audits-finished-r102.json').open('x') as f:json.dump(detail,f,indent=2)
        result['topologies'][top]=detail;print('audited',top,flush=True)
    result['passed']=True
except BaseException:result['error']=traceback.format_exc();raise
finally:
    result['unix_s']=time.time()
    with (R/'topology-original-audits-finished-r102.json').open('x') as f:json.dump(result,f,indent=2)
