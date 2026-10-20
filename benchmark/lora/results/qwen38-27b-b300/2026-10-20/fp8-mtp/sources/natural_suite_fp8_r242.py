"""Short natural drift probes, isolated from timing scores."""
import json,subprocess,time,urllib.request,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'fp8-mtp-r228'
for i in range(900):
 if (E/'slots-suite-complete-r240.json').exists():break
 time.sleep(1)
else:raise TimeoutError('suite')
for k in ['colocated','off']:subprocess.run(['python3','-u',str(R/'measure_fp8_r239.py'),k,'natural'],check=True)
# PD load remains exclusive to its paced residency trial until it finishes.
for i in range(900):
 if (E/'pd'/'residency-paced'/'complete.json').exists():break
 if (E/'pd'/'residency-paced'/'failed.json').exists():raise RuntimeError('residency failed')
 time.sleep(1)
else:raise TimeoutError('residency')
subprocess.run(['python3','-u',str(R/'measure_fp8_r239.py'),'pd','natural'],check=True)
(E/'natural-suite-complete-r237.json').write_text(json.dumps({'unix_s':time.time(),'note':'8 short prompts x 2 rounds x cold/hot, 32 output tokens; not quantized LoRA-vs-merge acceptance.'}))
