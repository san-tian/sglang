#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
# 终端3；第1步输入检查通过后执行一次。已有文件时不会覆盖。
python3 - <<'PYTHON'
import json
from pathlib import Path
r=Path('/work/mol')
p=r/'evidence/versions.json'
c=json.loads((r/'base/config.json').read_text())
record={
  'model_revision':'待填写：模型仓库commit或内部权重版本',
  'tokenizer_revision':'待填写：tokenizer仓库commit或内部版本',
  'environment_image_or_build':'待填写：镜像digest；裸机填写环境构建版本',
  'architecture':c.get('architectures'),
  'config_max_position_embeddings':c.get('max_position_embeddings'),
  'official_context_source':'待填写：官方模型卡或配置的链接及revision',
  'official_native_total_tokens':None,
  'official_extended_total_tokens':None,
  'official_extension_rope_scaling':None,
  'initial_test':{'context_length':4096,'tp_size':1,'dtype':'bfloat16'}
}
with p.open('x') as f:
    json.dump(record,f,ensure_ascii=False,indent=2)
print('请编辑',p,'：替换待填写项，填写官方原生/扩展总长；无扩展写null。')
print('官方总长包含输入+输出；config字段仅供核对，不能自动当作官方承诺。')
PYTHON
