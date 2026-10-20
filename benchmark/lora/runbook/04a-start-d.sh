#!/usr/bin/env bash
# 终端1；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
cd /work/mol/sglang-mol
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path /work/mol/base --host 127.0.0.1 --port 30000 \
  --tp-size 1 --dtype bfloat16 --context-length "${MOL_CONTEXT_LENGTH:-4096}" \
  --enable-lora --lora-backend merged_bf16 --max-lora-rank 32 \
  --lora-target-modules qkv_proj o_proj gate_up_proj down_proj \
  --max-loaded-loras 10 --max-loras-per-batch 4 \
  --disable-cuda-graph --enable-deterministic-inference \
  --attention-backend triton --linear-attn-prefill-backend triton \
  --bf16-gemm-backend torch \
  --enable-custom-logit-processor 2>&1 | tee -a /work/mol/evidence/D-server.log
