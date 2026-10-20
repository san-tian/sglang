"""Preserve first slot-client failure; independent hardlinks resolve inside containers."""
import json,os,subprocess,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'fp8-mtp-r228';src=Path('/opt/dlami/nvme/qwen38-mol-experiment/adapters/real_r16')
failed=E/'pd/slots';assert (failed/'failed.json').exists();failed.rename(E/'pd/slots-attempt-r232')
dest=E/'adapters-r239';dest.mkdir(exist_ok=False)
config=json.loads((src/'adapter_config.json').read_text())
for i in range(32):
 d=dest/f'scaled-{i:02d}';d.mkdir();cfg=dict(config);cfg['lora_alpha']=float(config['lora_alpha'])*(i+1)/32
 (d/'adapter_config.json').write_text(json.dumps(cfg,indent=2));os.link(src/'adapter_model.safetensors',d/'adapter_model.safetensors')
(E/'slot-retry-provenance-r240.json').write_text(json.dumps({'unix_s':time.time(),'reason':'old alpha-scaled adapter symlinks use host absolute paths not visible inside new containers','remedy':'own directory, per-alpha config; hardlinked read-only-in-practice tensor source, no edits to old adapters','source_tensor_sha256':hashlib.sha256((src/'adapter_model.safetensors').read_bytes()).hexdigest(),'source_alpha':config['lora_alpha'],'synthetic':True},indent=2))
for kind in ['pd','colocated']:subprocess.run(['python3','-u',str(R/'measure_fp8_r239.py'),kind,'slots'],check=True)
(E/'slots-suite-complete-r240.json').write_text(json.dumps({'unix_s':time.time(),'speed_and_capacity_completed_before_slot_retry':True,'failed_first_slot_client_preserved':True}))
