#!/usr/bin/env bash
# 终端2；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
# 先完成A的主矩阵；终端2按Ctrl-C停止M-A，等待进程退出，再执行：
# 只停止该终端的M；终端1中的D继续运行。
set -e
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python3 benchmark/lora/merge_reference.py \
  --base /work/mol/base --adapter /work/mol/adapter-B \
  --output /work/mol/merged-B --device cuda 2>&1 | tee /work/mol/evidence/merge-B.log
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path /work/mol/merged-B --host 127.0.0.1 --port 30002 \
  --tp-size 1 --dtype bfloat16 --context-length "${MOL_CONTEXT_LENGTH:-4096}" \
  --disable-cuda-graph --enable-deterministic-inference \
  --attention-backend triton --linear-attn-prefill-backend triton \
  --bf16-gemm-backend torch \
  --enable-custom-logit-processor 2>&1 | tee -a /work/mol/evidence/M-B-server.log
