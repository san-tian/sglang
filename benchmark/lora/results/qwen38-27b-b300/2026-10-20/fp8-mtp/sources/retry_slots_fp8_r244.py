"""Respect observed LoRA CPU LRU semantics: 33rd load evicts, rather than rejects."""
import json,subprocess,time,urllib.request
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'fp8-mtp-r228';failed=E/'pd/slots';assert (failed/'failed.json').exists();failed.rename(E/'pd/slots-attempt-r239')
for port in [36800,37800]:
 url=f'http://127.0.0.1:{port}'
 with urllib.request.urlopen(url+'/v1/models') as f:models=json.load(f)
 if any(x['id']=='overflow-33' for x in models['data']):
  q=urllib.request.Request(url+'/unload_lora_adapter',json.dumps({'lora_name':'overflow-33'}).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(q) as f:assert json.load(f)['success']
for k in ['pd','colocated']:subprocess.run(['python3','-u',str(R/'measure_fp8_r244.py'),k,'slots'],check=True)
(E/'slots-suite-complete-r240.json').write_text(json.dumps({'unix_s':time.time(),'completed_by':'retry_slots_fp8_r244.py','failed_clients_preserved':['slots-attempt-r232','slots-attempt-r239'],'correction':'max_loaded_loras is CPU LRU residency limit; 33rd load accepted and evicts one of 32.'}))
