#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
# 适用于本参考merge工具支持的单文件plain LoRA；输出目录必须是新的。
python3 - <<'PYTHON'
import shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import load_file, save_file
src, dst = Path('/work/mol/adapter-A'), Path('/work/mol/adapter-zero')
weights = src/'adapter_model.safetensors'
assert weights.is_file() and not dst.exists()
state = load_file(str(weights))
keys = [k for k in state if '.lora_B.' in k]
assert keys, '没有找到lora_B，先检查导出格式'
with safe_open(str(weights), framework='pt', device='cpu') as f:
    metadata = f.metadata()
shutil.copytree(src, dst)
for key in keys:
    state[key].zero_()
save_file(state, str(dst/'adapter_model.safetensors'), metadata=metadata)
print('zero_B_tensors=', len(keys))
PYTHON

python3 - <<'PYTHON'
import json
from pathlib import Path
from transformers import AutoTokenizer
root = Path('/work/mol/evidence')
texts = [s.strip() for s in (root/'prompts.txt').read_text().splitlines() if s.strip()]
assert len(texts) == 30, 'prompts.txt需恰好30条非空文本'
tok = AutoTokenizer.from_pretrained('/work/mol/base')
rows = [{'id': f'case-{i:02d}', 'input_ids': tok.encode(s, add_special_tokens=False)} for i,s in enumerate(texts)]
assert all(2 <= len(r['input_ids']) <= 3840 for r in rows)
assert max(len(r['input_ids']) for r in rows) >= 128, '至少准备一条128-token以上输入以测试前缀缓存'
(root/'corpus.json').write_text(json.dumps(rows))
print('cases=', len(rows), 'input_lengths=', [len(r['input_ids']) for r in rows])
PYTHON
