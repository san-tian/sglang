#!/usr/bin/env bash
# 终端2；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
set -e
set -o pipefail
test -f /work/mol/merged-A/MERGE_COMPLETE
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path /work/mol/merged-A --host 127.0.0.1 --port 30002 \
  --tp-size 1 --dtype bfloat16 --context-length "${MOL_CONTEXT_LENGTH:-4096}" \
  --disable-cuda-graph --enable-deterministic-inference \
  --attention-backend triton --linear-attn-prefill-backend triton \
  --bf16-gemm-backend torch \
  --enable-custom-logit-processor 2>&1 | tee -a /work/mol/evidence/M-A-restored-server.log
