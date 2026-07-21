#!/bin/bash
# =============================================================================
# Mooncake HIP Transport 禁用 Patch
# =============================================================================
# 问题: mooncake 编译时 -DUSE_HIP=ON -DENABLE_MULTI_PROTOCOL=ON, GPU 显存同时
#       注册到 "rdma" 和 "hip" 两个 transport。multi_transport.cpp 优先级
#       hip(4) > rdma(2), 跨节点传输错误选择 HIP IPC (只能同节点), 导致:
#       - hipIpcOpenMemHandle failed (Error code: 17)
#       - 进程被 SIGKILL (exit 137)
#
# 修复: 1. 禁用 HIP transport 安装 (#ifdef USE_HIP → USE_HIP_DISABLED)
#       2. 降低 HIP 优先级 (return 4 → return 0)
#       让 RDMA 成为唯一 GPU 显存传输路径
#
# 验证: patch 后编译的 engine.so md5 与镜像中安装的完全一致
#       (0e1b8fdd2d39e007b58c3e1711ec398b)
#
# 用法: 在容器内执行
#   bash /scripts/patch_mooncake.sh           # apply + build
#   bash /scripts/patch_mooncake.sh --check    # 只检查是否已 patch
# =============================================================================

set -e

MOONCAKE_SRC="${MOONCAKE_SRC:-/sgl-workspace/Mooncake/mooncake-transfer-engine}"
CHECK_ONLY=false
[ "${1:-}" = "--check" ] && CHECK_ONLY=true

# ---- 幂等检查 ----
already_patched=false
if grep -q "USE_HIP_DISABLED" "${MOONCAKE_SRC}/src/transfer_engine_impl.cpp" 2>/dev/null; then
  already_patched=true
fi

if $already_patched; then
  echo "✅ Mooncake already patched (USE_HIP_DISABLED found)"
  if $CHECK_ONLY; then exit 0; fi
  echo "   Skipping patch, rebuilding to verify..."
else
  if $CHECK_ONLY; then
    echo "❌ Mooncake not patched"
    exit 1
  fi

  echo "[1/4] Patching transfer_engine_impl.cpp: 禁用 HIP transport 安装"
  sed -i 's/^#ifdef USE_HIP$/#ifdef USE_HIP_DISABLED/' "${MOONCAKE_SRC}/src/transfer_engine_impl.cpp"
  echo "  Done."

  echo "[2/4] Patching multi_transport.cpp: 降低 HIP 优先级 (4 → 0)"
  sed -i 's/if (p == "hip") return 4;/if (p == "hip") return 0; \/\/ patched: rdma priority for cross-node/' "${MOONCAKE_SRC}/src/multi_transport.cpp"
  echo "  Done. RDMA now has higher priority than HIP."
fi

echo "[3/4] Verifying patches..."
grep -n "USE_HIP_DISABLED" "${MOONCAKE_SRC}/src/transfer_engine_impl.cpp" && echo "  ✅ transfer_engine_impl.cpp patched" || { echo "  ❌ patch failed"; exit 1; }
grep -n "return 0; .*patched" "${MOONCAKE_SRC}/src/multi_transport.cpp" && echo "  ✅ multi_transport.cpp patched" || { echo "  ❌ patch failed"; exit 1; }

echo "[4/4] Building mooncake..."
cd /sgl-workspace/Mooncake/build
make -j$(nproc) 2>&1 | tail -5
make install 2>&1 | tail -3
ldconfig

echo ""
echo "✅ Mooncake patched and built successfully."
echo "   Patched .so: /opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so"
