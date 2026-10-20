#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
nvidia-smi > /work/mol/evidence/gpu-environment.txt
python3 - <<'PYTHON'
import hashlib, json
from pathlib import Path
root = Path('/work/mol')
rows = []
for name in ('base', 'adapter-A', 'adapter-B'):
    folder = root/name
    assert folder.is_dir(), folder
    for path in sorted(folder.rglob('*')):
        if not path.is_file():
            continue
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b''):
                h.update(chunk)
        rows.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()})
(root/'evidence/input-files.json').write_text(json.dumps(rows, indent=2))
print('recorded_files=', len(rows))
PYTHON
