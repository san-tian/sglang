"""Adapt only source identity, new namespace and provenance, retaining raw audits."""
import ast,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
OLDIMAGE='sha256:a2b256e2b1370dc3968977d0c4af04a7f908ed0d4dfa54f3631c13730a117288'
IMAGE='sha256:1092e415c3dfa77fa0cb4da3b6ba3dea469bd5d32687e7a241e0bcc028ebc878'
parents={}
def save(old,new,s):
 ast.parse(s);(R/new).write_text(s)
 parents[new]={'parent':old,'parent_sha256':hashlib.sha256((R/old).read_bytes()).hexdigest()}
old='audit_topology_original_r99.py';s=(R/old).read_text()
s=s.replace(OLDIMAGE,IMAGE).replace('-r30','-r118').replace('source-host-r85.json','source-host-r118.json')
s=s.replace("        recorded_commit=provenance.get('runtime_commit',provenance.get('commit'))\n        assert recorded_commit==(COMMIT[:8] if legacy else COMMIT)","        assert not legacy\n        assert provenance['base_runtime_commit']==COMMIT\n        assert provenance['runtime_source_override_manifest']=='source-runtime-candidate-r116.json'\n        assert digest(root/'source-runtime-candidate-r116.json')==provenance['source_files']['source-runtime-candidate-r116.json']")
s=s.replace("'runtime_commit':COMMIT", "'base_runtime_commit':COMMIT,'runtime_override_manifest_sha256':digest(args.root/'source-runtime-candidate-r116.json')")
# Numerical helper implementations must be exactly the old implementations.
fns=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)}
a,b=fns((R/old).read_text()),fns(s)
for n in ['numerical_stats','check_statistics','native_scores','capture_scores']:assert a[n]==b[n],n
save(old,'audit_recovery_precision_r121.py',s)
old='audit_topology_native_r100.py';s=(R/old).read_text().replace(OLDIMAGE,IMAGE)
for a,b in [('-r30','-r118'),('-r31','-r118'),('-r41','-r118'),("'r30'","'r118'"),("'r31'","'r118'"),("'r41'","'r118'"),("'r85'","'r118'"),('actual-placement-observations-r92.jsonl','actual-placement-observations-r118.jsonl'),('-r100.json','-r121.json')]:s=s.replace(a,b)
s=s.replace("'scope':'Three natural636-request", "'scope':'Candidate R116 image, new R118 namespace; original R100 raw numerical reconstruction retained. Three natural636-request")
save(old,'audit_recovery_native_r121.py',s)
old='audit_topology_mixed_r101.py';s=(R/old).read_text().replace("suffix='r30' if legacy else 'r31'","suffix='r118'")
s=s.replace("'Original per-rank mixed routing", "'Candidate R116 image, new R118 namespace; original per-rank mixed routing")
save(old,'audit_recovery_mixed_r121.py',s)
old='audit_capacity_residency_r64.py';s=(R/old).read_text().replace(OLDIMAGE,IMAGE).replace('-r63','-r118').replace('-r31','-r118').replace('-r64.json','-r121.json')
s=s.replace("'scope':'Observed lower bounds", "'scope':'Candidate R116 image, new R118 namespace; original R64 same-scrape audit retained. Observed lower bounds")
save(old,'audit_recovery_capacity_r121.py',s)
(R/'recovery-audit-adaptations-r121.json').write_text(json.dumps({'parents':parents,'precision_numerical_helpers_ast_unchanged':['numerical_stats','check_statistics','native_scores','capture_scores'],'changes':'Image ID, paths/namespace, parent-runtime-plus-two-overrides provenance, output names and scope labels only. All raw-score, count, finite, statistic, capture agreement, natural response, trace, placement, same-scrape KV and equal-byte assertions retained.','gpu_tests_executed':False},indent=2))
print('Prepared four versioned raw auditors; original numerical helpers AST unchanged')
