"""Export finished FP8 trials with raw data and source seals; omit tensor/cache files."""
import hashlib,json,shutil,subprocess,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'fp8-mtp-r228';D=R/'publish-fp8-r248';D.mkdir(exist_ok=False)
assert (E/'final-state-r246/complete.json').exists() and (E/'natural-suite-complete-r237.json').exists()
audit=json.loads((E/'audit-r247.json').read_text());assert all(p['complete'] and p['failure'] is None for v in audit['variants'] for p in v['phases'].values())
assert not json.loads((E/'runtime-overlay-audit-r248.json').read_text())['mismatches']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for p in E.glob('*.json'):
 if not p.name.startswith('audit-') or p.name=='audit-r247.json':shutil.copyfile(p,D/p.name)
modeldir=D/'model-provenance';modeldir.mkdir()
for n in ['model-revision.json','config.json','model.safetensors.index.json','download-command.json','startup-failed.json']:
 p=R/'fp8-mtp-r226'/n
 if p.exists():shutil.copyfile(p,modeldir/n)
sources=['launch_fp8_mtp_r226.py','launch_fp8_variants_r228.py','launch_fp8_off_r230.py','measure_fp8_r229.py','measure_fp8_r232.py','measure_fp8_r239.py','measure_fp8_r244.py','suite_fp8_r233.py','retry_slots_fp8_r240.py','retry_slots_fp8_r244.py','residency_fp8_r241.py','natural_suite_fp8_r242.py','audit_fp8_r247.py','audit_fp8_r235.py','finalize_fp8_r246.py','export_fp8_r248.py']
(D/'sources').mkdir()
for n in sources:shutil.copyfile(R/n,D/'sources'/n)
(D/'source-seal.json').write_text(json.dumps({'files_sha256':{n:sha(D/'sources'/n) for n in sources}},indent=2))
raw={}
for k in ['colocated','pd','off']:
 for f in sorted((E/k).iterdir()):
  if not f.is_dir():continue
  for p in f.rglob('*'):
   assert not p.is_symlink(),p
   if p.is_file():raw[str(p.relative_to(E))]=sha(p)
  with tarfile.open(D/(k+'-'+f.name+'-raw.tgz'),'w:gz',compresslevel=6) as t:t.add(f,arcname=k+'/'+f.name)
with tarfile.open(D/'final-state-and-logs.tgz','w:gz',compresslevel=6) as t:t.add(E/'final-state-r246',arcname='final-state-r246')
with tarfile.open(D/'frozen-BF16-independent-merge-reference.tgz','w:gz',compresslevel=6) as t:t.add(R/'precision-1p1d-reference-r168',arcname='precision-1p1d-reference-r168')
shutil.copyfile(E/'final-state-r246/parser-checks.json',D/'parser-checks.json');shutil.copyfile(E/'final-state-r246/complete.json',D/'final-state-summary.json')
(D/'raw-files-sha256.json').write_text(json.dumps(raw,indent=2))
(D/'runtime.json').write_text(json.dumps({'image':'sha256:5be81e13803be5cd8d8c80cbcfa6505a9bd847305b1868a8d95e3240939ab934','base_git_HEAD':'3c560089e1818a5217a73f2168273485a5350341','fork_overlay_revision':'0116d3a12bc893ae43d8dbfd594280eccdc8467c','fork_overlay_files_verified':26,'note':'Container .git and inherited OCI labels remain upstream; 26 Python overlay files were SHA256 checked against the fork revision.'},indent=2))
# Store only configs and tensor provenance for synthetic adapters; never export weights.
(D/'scaled-adapter-configs').mkdir()
for p in sorted((E/'adapters-r239').glob('*/adapter_config.json')):shutil.copyfile(p,D/'scaled-adapter-configs'/(p.parent.name+'.json'))
files={str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()};(D/'manifest.json').write_text(json.dumps({'unix_s':time.time(),'files_sha256':files},indent=2))
for p in D.rglob('*'):
 if p.is_file():assert p.stat().st_size<95*1024**2,(p,p.stat().st_size)
print(json.dumps({'files':len(files),'raw_files':len(raw),'MiB':sum(p.stat().st_size for p in D.rglob('*') if p.is_file())/1024**2}),flush=True)
