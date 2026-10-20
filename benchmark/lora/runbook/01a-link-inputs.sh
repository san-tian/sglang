#!/usr/bin/env bash
# 终端3；执行顺序与人工检查点见同目录README.md。
set -eo pipefail
# 终端3：只修改下面三条路径，指向已有的本地权重；不会下载权重。
set -e
BASE_SRC="${BASE_SRC:?请传入本地权重绝对路径}"
ADAPTER_A_SRC="${ADAPTER_A_SRC:?请传入本地权重绝对路径}"
ADAPTER_B_SRC="${ADAPTER_B_SRC:?请传入本地权重绝对路径}"
for src in "$BASE_SRC" "$ADAPTER_A_SRC" "$ADAPTER_B_SRC"; do
  test -d "$src" || { echo "输入目录不存在：$src"; exit 1; }
done
mkdir -p /work/mol
for dst in base adapter-A adapter-B; do
  if test -e "/work/mol/$dst" || test -L "/work/mol/$dst"; then
    echo "保留已有目录：/work/mol/$dst；核对后跳过本块，不覆盖"; exit 1
  fi
done
ln -s "$BASE_SRC" /work/mol/base
ln -s "$ADAPTER_A_SRC" /work/mol/adapter-A
ln -s "$ADAPTER_B_SRC" /work/mol/adapter-B
ls -ld /work/mol/base /work/mol/adapter-A /work/mol/adapter-B
