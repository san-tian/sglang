#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
# 终端3：冻结前生成结果表；按实际证据填写，文件存在时不覆盖。
python3 - <<'PYTHON'
import csv,json
from pathlib import Path
r=Path('/work/mol/evidence')
p=r/'acceptance.csv'
a=json.loads(Path('/work/mol/adapter-A/adapter_config.json').read_text())
b=json.loads(Path('/work/mol/adapter-B/adapter_config.json').read_text())
rows=[('挂载闭环','A/B/zero','smoke-api'),
      ('生命周期','A/B/zero','lifecycle'),('D/M主测','A','main'),
      ('D/M主测','B','main-B'),('混流','base/A/B/zero','mixed'),
      ('真实缓存恢复','A','cache-restore')]
rows += [('长上下文','A；B另列独立结果',d.name)
         for d in sorted(r.glob('long-*')) if d.is_dir()]
fields=['test','adapter','rank_A_B','targets_A_B','tp_dp_dcp',
        'total_tokens','result','mean_abs_nat','p99_abs_nat',
        'max_abs_nat','argmax_match','evidence','notes']
with p.open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for test,adapter,folder in rows:
        w.writerow(dict(test=test,adapter=adapter,
            rank_A_B=f"{a['r']}/{b['r']}",
            targets_A_B=json.dumps([a['target_modules'],b['target_modules']]),
            tp_dp_dcp='1/1/1',total_tokens='待填写实际服务总长',
            result='待核对' if (r/folder).exists() else '未测',
            evidence=folder,notes='按阶段/逐例记录；模板不自动判通过'))
print('结果表：',p,'；逐项填写通过/失败/未测，并引用原始证据。')
PYTHON
