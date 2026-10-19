"""Recompute all threeway speed points; distinguish SSE and client-end timing."""
import hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def quantile(values,q):return sorted(values)[math.ceil(len(values)*q)-1]


def main():
    output=ROOT/'threeway-percentiles-audit-r67.json';assert not output.exists()
    proofs={};total=0;points_checked=0;latency_statistics=0;selected=[]
    for top in ('tp2','dpa2','tp4','dpa4'):
        audit_file=ROOT/f'{top}-threeway-audit-r58.json';audit=read(audit_file)
        assert audit['passed'] and audit['raw_native_payloads_audited']==1908
        proofs[top]={'threeway_audit_sha256':sha(audit_file),'controls':{}}
        for side in ('d','m','base'):
            suffix='r41' if side=='base' else ('r29' if top.endswith('2') else 'r30')
            directory=ROOT/f'perf-{top}-{side}-{suffix}'
            rows=lines(directory/'requests.jsonl');groups=lines(directory/'groups.jsonl');summary=read(directory/'summary.json')
            assert len(rows)==636 and len(groups)==48 and len(summary['points'])==16
            for row in rows:
                path=directory/(row['label']+'.json')
                assert sha(path)==audit['raw_native_sha256'][side][row['label']]
                raw=read(path);events=raw['events'];first,last=events[0],events[-1]
                assert len(first['data']['output_ids'])==1
                assert last['data']['output_ids']==row['output_ids'] and len(row['output_ids'])==row['output_tokens']
                equal(row['ttft_s'],first['elapsed_s'])
                equal(row['tpot_s'],(last['elapsed_s']-first['elapsed_s'])/(row['output_tokens']-1))
                assert math.isfinite(row['e2e_s']) and row['e2e_s']>=last['elapsed_s']>first['elapsed_s']
            expected={(n,c,cache) for n in (128,2048) for c in (1,4,16,32) for cache in ('cold','hot')}
            assert {(p['input_tokens'],p['concurrency'],p['cache']) for p in summary['points']}==expected
            for point in summary['points']:
                key=(point['input_tokens'],point['concurrency'],point['cache'])
                rr=[r for r in rows if (r['input_tokens'],r['concurrency'],r['cache'])==key]
                gg=[g for g in groups if (g['input_tokens'],g['concurrency'],g['cache'])==key]
                assert len(rr)==point['requests']==3*key[1] and len(gg)==3
                assert {g['round'] for g in gg}=={0,1,2}
                rates=[]
                for g in gg:
                    batch=[r for r in rr if r['round']==g['round']];assert len(batch)==key[1]
                    rate=sum(r['output_tokens'] for r in batch)/g['elapsed_s']
                    equal(rate,g['output_throughput_tokens_s']);rates.append(rate)
                equal(math.fsum(rates)/3,point['throughput_tokens_s_mean'])
                for metric in ('e2e_s','ttft_s','tpot_s'):
                    values=[r[metric] for r in rr]
                    expected_stats={'mean':math.fsum(values)/len(values),
                                    'p50':quantile(values,.5),'p95':quantile(values,.95),'p99':quantile(values,.99)}
                    for name,value in expected_stats.items():equal(value,point[metric][name]);latency_statistics+=1
                points_checked+=1
                if key==(2048,32,'cold'):selected.append({'topology':top,'control':side,**point})
            proofs[top]['controls'][side]={'requests_sha256':sha(directory/'requests.jsonl'),
                                          'groups_sha256':sha(directory/'groups.jsonl'),'summary_sha256':sha(directory/'summary.json')}
            total+=len(rows)
    result={'passed':True,'native_requests_reparsed':total,'speed_points_recomputed':points_checked,
            'latency_statistics_recomputed':latency_statistics,'c32_cold_2048_256':selected,
            'input_proofs':proofs,'audit_source_sha256':sha(Path(__file__)),
            'scope':'Recomputed all D/M/base TP2/DP2/TP4/DP4 points from 7632 raw SSE payloads and client timing records. TTFT/TPOT independently reconstructed from SSE event timestamps; E2E uses retained client HTTP completion elapsed, checked against last-SSE lower bound. Percentiles use nearest rank and throughput is the mean of three completed batch rates. Concurrent background and unequal automatic KV pools remain; this is not isolated speedup or capacity evidence.'}
    output.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['passed','native_requests_reparsed','speed_points_recomputed','latency_statistics_recomputed']}))


if __name__=='__main__':main()
