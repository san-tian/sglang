#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
# 终端3；所有客户端与D/M日志均停止写入后执行。
# 新一轮实验使用新evidence目录。
test ! -e /work/mol/evidence/SHA256SUMS.json
test ! -e /work/mol/evidence/executed-runbook
git rev-parse HEAD > /work/mol/evidence/final-source-commit.txt
git diff --binary HEAD > /work/mol/evidence/final-source.patch
cp -r /work/mol/runbook /work/mol/evidence/executed-runbook
python3 - <<'PYTHON'
import hashlib, json
from pathlib import Path
root = Path('/work/mol/evidence')
manifest = root/'SHA256SUMS.json'
assert not manifest.exists(), '证据目录已经冻结'
rows = []
for path in sorted(root.rglob('*')):
    if not path.is_file():
        continue
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    rows.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()})
manifest.write_text(json.dumps(rows, indent=2))
print('frozen_files=', len(rows))
PYTHON
