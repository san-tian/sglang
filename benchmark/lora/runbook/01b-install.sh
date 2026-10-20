#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
set -e
cd /work/mol
git clone https://github.com/san-tian/sglang.git sglang-mol
cd sglang-mol
git checkout 19955ce786d7c573f0a48a77231d5f251866fb1c
# 服务仍固定使用上面的提交；以下只补两份标准库检查客户端。
git show 63a980b42f89c327fcd76968af30cd3f809eecc9:benchmark/lora/check_lifecycle.py > benchmark/lora/check_lifecycle.py
git show 63a980b42f89c327fcd76968af30cd3f809eecc9:benchmark/lora/check_cache_restore.py > benchmark/lora/check_cache_restore.py
python3 -m pip install --no-deps -e ./python
mkdir -p /work/mol/evidence
git rev-parse HEAD > /work/mol/evidence/source-commit.txt
printf '%s\n' 63a980b42f89c327fcd76968af30cd3f809eecc9 > /work/mol/evidence/clients-commit.txt
python3 - <<'PYTHON'
import torch, sglang
from importlib.metadata import version
assert torch.cuda.is_available(), '当前环境无法访问GPU，先检查CUDA和容器GPU挂载'
assert torch.cuda.device_count() >= 2, '这份双服务示例需要至少2张可见GPU'
print('torch=', torch.__version__, 'CUDA=', torch.version.cuda)
print('sglang=', version('sglang'), 'source=', sglang.__file__)
print('GPUs=', [torch.cuda.get_device_name(i) for i in range(2)])
PYTHON
python3 -m sglang.launch_server --help > /work/mol/evidence/launch-help.txt
python3 benchmark/lora/merge_reference.py --help > /work/mol/evidence/merge-help.txt
python3 benchmark/lora/compare_merged.py --help > /work/mol/evidence/compare-help.txt
