#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
# 已安装CUDA匹配的SGLang依赖；在三个终端/容器中检查相同路径。
# 将现有模型与两份adapter放到以下目录，或分别建立指向它们的软链接。
# /work/mol/base/       config.json、tokenizer文件、model.safetensors.index.json、权重分片
# /work/mol/adapter-A/  adapter_config.json、adapter_model.safetensors
# /work/mol/adapter-B/  adapter_config.json、adapter_model.safetensors
python3 - <<'PYTHON'
import json, platform
from pathlib import Path
from importlib.metadata import version
p=Path('/work/mol')
for f in ('base/config.json','base/model.safetensors.index.json',
          'adapter-A/adapter_config.json','adapter-A/adapter_model.safetensors',
          'adapter-B/adapter_config.json','adapter-B/adapter_model.safetensors'):
    assert (p/f).is_file(), f'缺少输入文件：{p/f}'
for name in ('adapter-A','adapter-B'):
    cfg=json.loads((p/name/'adapter_config.json').read_text())
    assert 1 <= cfg['r'] <= 32, '示例max-lora-rank=32；超出时修改启动参数后重新检查'
    print(name, 'rank=',cfg['r'], 'alpha=',cfg['lora_alpha'], 'targets=',cfg['target_modules'])
for name in ('adapter-zero','merged-A'):
    assert not (p/name).exists(), f'请使用新的实验目录：{p/name}已存在'
(p/'evidence').mkdir(parents=True,exist_ok=True)
versions={'python':platform.python_version(),
          'packages':{k:version(k) for k in ('torch','sglang','transformers','safetensors')}}
(p/'evidence/installed-versions.json').write_text(json.dumps(versions,indent=2))
print('输入目录与rank检查通过；模型/tokenizer revision和镜像ID仍需写入versions.json')
PYTHON
python3 -m pip freeze > /work/mol/evidence/pip-freeze.txt
