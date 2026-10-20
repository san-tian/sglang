#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
set -e
curl --fail --silent --show-error --retry 120 --retry-connrefused --retry-delay 2 \
  http://127.0.0.1:30002/health -o /dev/null
python3 - <<'PYTHON'
import json
from pathlib import Path
from urllib.request import urlopen
root = Path('/work/mol/evidence')
for side, port, expected in [('D', 30000, '/work/mol/base'),
                             ('M-A-restored', 30002, '/work/mol/merged-A')]:
    with urlopen(f'http://127.0.0.1:{port}/model_info', timeout=30) as f:
        info = json.load(f)
    (root/f'{side}-before-cache-model-info.json').write_text(json.dumps(info, indent=2))
    assert info['model_path'] == expected, (side, info)
    print(side, info['model_path'])
print('D为原始基座，M为A独立merge；继续执行下方缓存检查')
PYTHON
