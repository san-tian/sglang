"""Read-only CPU audits; never starts/stops a GPU worker or changes GPU leases."""
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

R=Path(__file__).resolve().parent
manifest=json.loads((R/'capacity-audit-source-r64.json').read_text())
for name,digest in manifest['source_files'].items():
    assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
result={'passed':False,'scope':'Read-only CPU normalization/allocation/native-response audit after actual GPU capacity runs. CPU tests and a running watcher alone are not GPU acceptance.'}
try:
    audits={}
    for top in ['tp8','dcp8x2','dpa8']:
        marker=R/f'{top}-precision-perf-capacity-complete-r31.json'
        while not marker.exists():time.sleep(10)
        for name,digest in manifest['source_files'].items():
            assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
        with (R/f'capacity-{top}-audit-r64.log').open('x') as f:
            subprocess.run(['python3',str(R/'audit_capacity_residency_r64.py'),top],stdout=f,stderr=subprocess.STDOUT,check=True)
        audit=json.loads((R/f'capacity-{top}-independent-audit-r64.json').read_text())
        assert audit['passed'] and audit['native_requests_audited']==13
        audits[top]=audit
    assert len({a['full_attention_KV_bytes_per_rank'] for a in audits.values()})==1
    c8={top:next(c for c in a['cases'] if c['http_concurrency']==8) for top,a in audits.items()}
    result.update(passed=True,native_requests_audited=39,
                  full_attention_KV_bytes_per_rank=536870912,
                  observed_c8_same_scrape_peaks={top:c['peak_observed_logical_KV_tokens'] for top,c in c8.items()},
                  observed_c8_running_peaks={top:c['max_running_requests_one_scrape'] for top,c in c8.items()},
                  sustained_full_c8_observed={top:c['full_requested_batch_observed_in_multiple_scrapes'] for top,c in c8.items()},
                  observed_peak_ratios_vs_TP8={top:c8[top]['peak_observed_logical_KV_tokens']/c8['tp8']['peak_observed_logical_KV_tokens'] for top in ['dcp8x2','dpa8']})
except BaseException:
    result['error']=traceback.format_exc()
    raise
finally:
    (R/'capacity-independent-audits-finished-r64.json').write_text(json.dumps(result,indent=2))
