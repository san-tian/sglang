"""Live parser probes and final primary/FP8 joint health; no serving stack edits."""
import concurrent.futures as cf,json,subprocess,time,urllib.request,uuid,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'fp8-mtp-r228'
for i in range(900):
 if (E/'natural-suite-complete-r237.json').exists():break
 time.sleep(1)
else:raise TimeoutError('natural suite')
O=E/'final-state-r246';O.mkdir(exist_ok=False)
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def call(port,path,data=None):
 q=urllib.request.Request(f'http://127.0.0.1:{port}'+path,None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(q,timeout=300) as f:raw=f.read().decode()
 try:return json.loads(raw)
 except ValueError:return raw
infos={}
for port in [31800,31801,35800,36800,37800,38800]:
 info=call(port,'/get_server_info');call(port,'/health');infos[str(port)]=info
save('server-info.json',infos)
basecase=json.loads((R/'precision-1p1d-reference-r168/contract.json').read_text())['corpus'][0]
for name,p,d,boot in [('original-bf16',31800,31801,31810),('fp8-pd',36800,37800,36810)]:
 q={'input_ids':basecase['input_ids'],'sampling_params':{'temperature':0,'max_new_tokens':16,'ignore_eos':True},'lora_path':'main','bootstrap_host':'127.0.0.1','bootstrap_port':boot,'bootstrap_room':uuid.uuid4().int&((1<<63)-1)}
 with cf.ThreadPoolExecutor(2) as pool:
  fp=pool.submit(call,p,'/generate',q);fd=pool.submit(call,d,'/generate',q);pr,dr=fp.result(),fd.result()
 assert len(dr['output_ids'])==16 and pr['output_ids']==dr['output_ids'][:1];save(name+'-joint-generate.json',{'request':q,'prefill_response':pr,'decode_response':dr})
checks={}
requests={
 'reasoning':{'model':'main','messages':[{'role':'user','content':'计算17+25，给出简短答案。'}],'temperature':0,'max_tokens':256,'chat_template_kwargs':{'enable_thinking':True}},
 'tool-call':{'model':'main','messages':[{'role':'user','content':'Call get_weather for Beijing. Do not explain.'}],'tools':[{'type':'function','function':{'name':'get_weather','description':'Get weather by city','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}],'tool_choice':'auto','temperature':0,'max_tokens':128,'chat_template_kwargs':{'enable_thinking':False}}}
for name,q in requests.items():
 try:
  response=call(35800,'/v1/chat/completions',q);msg=response['choices'][0]['message'];save(name+'-chat.json',{'request':q,'response':response})
  if name=='reasoning':checks[name]={'reasoning_parsed':bool(msg.get('reasoning_content')),'final_answer_present':bool(msg.get('content')),'answer_contains_42':'42' in (msg.get('content') or '')}
  else:
   calls=msg.get('tool_calls') or [];checks[name]={'tool_calls_parsed':bool(calls),'weather_function':any(x['function']['name']=='get_weather' for x in calls),'arguments':[json.loads(x['function']['arguments']) for x in calls]}
 except Exception:checks[name]={'error':traceback.format_exc()}
save('parser-checks.json',checks)
# All pacing controls must have ended with explicit null.
pace=[json.loads(l) for l in (E/'pd/residency-paced/pacing-controls.jsonl').read_text().splitlines()];assert pace[-1]['forward_sleep_time'] is None
save('placement.json',{'inventory':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader'],text=True),'unix_s':time.time()})
for n in ['qwen38-fp8-mtp-prefill-r226','qwen38-fp8-mtp-decode-r226','qwen38-fp8-mtp-colocated-r228','qwen38-fp8-mtp-prefill-r228','qwen38-fp8-mtp-decode-r228','qwen38-fp8-off-colocated-r230']:
 (O/(n+'.log')).write_text(subprocess.check_output(['docker','logs',n],stderr=subprocess.STDOUT,text=True))
save('complete.json',{'passed':True,'unix_s':time.time(),'original_BF16_primary_preserved_and_joint_smoke_passed':True,'fp8_PD_joint_smoke_passed':True,'pacing_reset':True,'parser_checks':checks,'FP8_precision_accepted':False})
print(json.dumps(checks),flush=True)
