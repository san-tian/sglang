"""Read-only host/client CPU samples during the baseline; no GPU API calls."""
import json,os,time
from pathlib import Path
R=Path('/opt/dlami/nvme/qwen38-mol-multigpu-20261019');E=R/'pd4-baseline-r253'
hz=os.sysconf('SC_CLK_TCK');previous={};previous_t=None;previous_host=None
with (E/'host-client-cpu-r258.jsonl').open('x') as f:
 while not (E/'suite-finished.json').exists():
  now=time.monotonic();processes=[]
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    cmd=(p/'cmdline').read_bytes().split(b'\0')
    if str(R/'measure_pd4_baseline_r254.py').encode() not in cmd:continue
    stat=(p/'stat').read_text().rsplit(')',1)[1].split();cpu=(int(stat[11])+int(stat[12]))/hz
    rss=int(stat[21])*os.sysconf('SC_PAGE_SIZE');old=previous.get(p.name)
    processes.append({'pid':int(p.name),'phase':cmd[-2].decode(),'rss_bytes':rss,'threads':int(stat[17]),'cpu_core_equivalent':(cpu-old)/(now-previous_t) if old is not None else None})
    previous[p.name]=cpu
   except (FileNotFoundError,ProcessLookupError,PermissionError):continue
  host=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]];host=host[:8]
  total=sum(host);idle=host[3]+host[4];pct=None
  if previous_host:
   dt=total-previous_host[0];pct=100*(1-(idle-previous_host[1])/dt) if dt else None
  f.write(json.dumps({'monotonic':now,'clients':processes,'host_cpu_busy_pct':pct,'cpu_count':os.cpu_count()})+'\n');f.flush()
  previous_host=(total,idle);previous_t=now;time.sleep(1)
