#!/usr/bin/env python3
"""Patch the AITER package inside a running SGLang container.

Applies 3 fixes from the v0.5.14-cp-layersplit-v17-patched image:
  1. rope.py      – positions.clone() in 8 RoPE functions (EAGLE draft decode crash)
  2. jit/core.py   – raise RuntimeError → logger.warning (duplicate GEMM config shapes)
  3. pa_mqa_logits – _amd_iglp_sched_barrier(0x0) before early return (scheduler hint)

Usage (inside the container):
    python3 /scripts/patch_aiter.py          # apply
    python3 /scripts/patch_aiter.py --check   # dry-run, show what would change
    python3 /scripts/patch_aiter.py --revert   # revert (restore from .orig)

Usage (from host, via docker exec):
    docker exec sglang-prefill python3 /scripts/patch_aiter.py
"""

import os
import re
import sys
import shutil
from pathlib import Path

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

# Try common AITER install locations inside the container
AITER_SEARCH_PATHS = [
    "/sgl-workspace/aiter/aiter",
    "/opt/conda/lib/python3.12/site-packages/aiter",
    "/opt/conda/lib/python3.11/site-packages/aiter",
    os.path.expanduser("~/.local/lib/python3.12/site-packages/aiter"),
]

# ----------------------------------------------------------------------------
# Patch definitions
# ----------------------------------------------------------------------------

# Patch 1: rope.py — clone positions to guarantee stride(-1)==1 for C++ kernel
ROPE_PATCH = {
    "file": "ops/rope.py",
    "marker": "# PATCHED-rope-clone",
    "insertions": [
        # (function_name, anchor_line, patch_lines)
        (
            "rope_cached_positions_fwd",
            "    )\n    rope_cached_positions_fwd_impl(",
            "    )\n"
            "    # PATCHED-rope-clone: force positions stride(1)==1 for C++ kernel\n"
            "    if positions.numel() == 1 or positions.stride(-1) != 1:\n"
            "        positions = positions.clone()\n"
            "    rope_cached_positions_fwd_impl(",
        ),
        (
            "rope_cached_positions_2c_fwd",
            "    )\n    rope_cached_positions_2c_fwd_impl(",
            "    )\n"
            "    # PATCHED-rope-clone: force positions stride(1)==1 for C++ kernel assertion\n"
            "    if positions.numel() == 1 or positions.stride(-1) != 1:\n"
            "        positions = positions.clone()\n"
            "    rope_cached_positions_2c_fwd_impl(",
        ),
        (
            "rope_cached_positions_fwd_inplace",
            "    nope_first: bool,\n) -> Tensor:\n    rope_cached_positions_fwd_impl(",
            "    nope_first: bool,\n) -> Tensor:\n"
            "    # PATCHED-rope-clone: force positions stride(1)==1 for C++ kernel\n"
            "    if positions.numel() == 1 or positions.stride(-1) != 1:\n"
            "        positions = positions.clone()\n"
            "    rope_cached_positions_fwd_impl(",
        ),
        (
            "rope_cached_positions_2c_fwd_inplace",
            "    nope_first: bool,\n) -> Tensor:\n    rope_cached_positions_2c_fwd_impl(",
            "    nope_first: bool,\n) -> Tensor:\n"
            "    # PATCHED-rope-clone: .contiguous() is not enough for [1,1] tensors\n"
            "    # with stride=(8,8) because is_contiguous() returns True for\n"
            "    # single-element tensors; force stride(1)==1 via clone().\n"
            "    if positions.numel() == 1 or positions.stride(-1) != 1:\n"
            "        positions = positions.clone()\n"
            "    rope_cached_positions_2c_fwd_impl(",
        ),
        (
            "rope_cached_positions_offsets_fwd",
            "    )\n    rope_cached_positions_offsets_fwd_impl(",
            "    )\n"
            "    # PATCHED-rope-clone: force positions stride(1)==1 for C++ kernel\n"
            "    if positions.numel() == 1 or positions.stride(-1) != 1:\n"
            "        positions = positions.clone()\n"
            "    rope_cached_positions_offsets_fwd_impl(",
        ),
        (
            "rope_cached_positions_offsets_2c_fwd",
            "    )\n    rope_cached_positions_offsets_2c_fwd_impl(",
            "    )\n"
            "    # PATCHED-rope-clone: force positions stride(1)==1\n"
            "    if positions.numel() == 1 or positions.stride(-1) != 1:\n"
            "        positions = positions.clone()\n"
            "    rope_cached_positions_offsets_2c_fwd_impl(",
        ),
        (
            "rope_cached_positions_offsets_fwd_inplace",
            "    nope_first: bool,\n) -> Tensor:\n    rope_cached_positions_offsets_fwd_impl(",
            "    nope_first: bool,\n) -> Tensor:\n"
            "    # PATCHED-rope-clone: force positions stride(1)==1 for C++ kernel\n"
            "    if positions.numel() == 1 or positions.stride(-1) != 1:\n"
            "        positions = positions.clone()\n"
            "    rope_cached_positions_offsets_fwd_impl(",
        ),
        (
            "rope_cached_positions_offsets_2c_fwd_inplace",
            "    nope_first: bool,\n) -> Tensor:\n    rope_cached_positions_offsets_2c_fwd_impl(",
            "    nope_first: bool,\n) -> Tensor:\n"
            "    # PATCHED-rope-clone: force positions stride(1)==1 for C++ kernel\n"
            "    if positions.numel() == 1 or positions.stride(-1) != 1:\n"
            "        positions = positions.clone()\n"
            "    rope_cached_positions_offsets_2c_fwd_impl(",
        ),
    ],
}

# Patch 2: jit/core.py — raise RuntimeError → logger.warning
JIT_CORE_PATCH = {
    "file": "jit/core.py",
    "marker": "# PATCHED-jit-core-warning",
    "old": """                raise RuntimeError(
                    f"Found {dup_count} duplicate shape entries during merge of '{merge_name}'. "
                    f"Auto-resolved by keeping best performing (lowest 'us') for each shape "
                    f"and saved back to source config files. Please re-run.\\n"
                    f"Duplicate rows:\\n{dup_rows.to_string(index=False)}\\n"
                    f"Updated files:\\n{saved_info}"
                )""",
    "new": """                # PATCHED-jit-core-warning: don't crash on duplicate GEMM shapes
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    f"Found {dup_count} duplicate shape entries during merge of '{merge_name}'. "
                    f"Auto-resolved by keeping best performing (lowest 'us') for each shape "
                    f"and saved back to source config files. Continuing.\\n"
                    f"Updated files:\\n{saved_info}"
                )""",
}

# Patch 3: pa_mqa_logits.py — _amd_iglp_sched_barrier before early return
PA_MQA_PATCH = {
    "file": "ops/triton/gluon/pa_mqa_logits.py",
    "marker": "# PATCHED-sched-barrier",
    "replacements": [
        # (old, new)
        (
            "    if context_length == 0:\n        return",
            "    if context_length == 0:\n        _amd_iglp_sched_barrier(0x0)  # PATCHED-sched-barrier\n        return",
        ),
        (
            "    if split_context_length <= 0:\n        return",
            "    if split_context_length <= 0:\n        _amd_iglp_sched_barrier(0x0)  # PATCHED-sched-barrier\n        return",
        ),
    ],
}

ALL_PATCHES = [ROPE_PATCH, JIT_CORE_PATCH, PA_MQA_PATCH]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def find_aiter_root() -> Path:
    """Locate the aiter package directory."""
    for p in AITER_SEARCH_PATHS:
        if Path(p, "__init__.py").exists():
            return Path(p)
    # Fallback: try importing
    try:
        import aiter
        return Path(aiter.__file__).parent
    except ImportError:
        pass
    print("ERROR: Cannot locate the aiter package. Set AITER_ROOT env var.")
    sys.exit(1)


def apply_text_patch(filepath: Path, old: str, new: str, marker: str, dry_run: bool, revert: bool) -> bool:
    """Apply a single text replacement. Returns True if changed."""
    content = filepath.read_text()

    if marker in content:
        if revert:
            # Restore from .orig
            orig = filepath.with_suffix(filepath.suffix + ".orig")
            if orig.exists():
                shutil.copy2(orig, filepath)
                print(f"  ✅ reverted {filepath.name}")
                return True
            print(f"  ⚠️  no .orig for {filepath.name}, skipping revert")
            return False
        print(f"  ⏭️  already patched {filepath.name}")
        return False

    if old not in content:
        print(f"  ❌ anchor not found in {filepath.name}")
        return False

    if dry_run:
        print(f"  [DRY-RUN] would patch {filepath.name}")
        return False

    # Backup
    shutil.copy2(filepath, filepath.with_suffix(filepath.suffix + ".orig"))
    filepath.write_text(content.replace(old, new, 1))
    print(f"  ✅ patched {filepath.name}")
    return True


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    dry_run = "--check" in sys.argv
    revert = "--revert" in sys.argv
    mode = "revert" if revert else ("check" if dry_run else "apply")

    print(f"AITER patch script — mode: {mode}")
    print(f"  image: v0.5.14-cp-layersplit-v17-patched")
    print()

    aiter_root = find_aiter_root()
    print(f"AITER root: {aiter_root}")
    print()

    changed = 0

    # --- Patch 1: rope.py ---
    print("[1/3] rope.py — positions.clone() fix")
    rope_path = aiter_root / ROPE_PATCH["file"]
    if not rope_path.exists():
        print(f"  ❌ {rope_path} not found")
    else:
        content = rope_path.read_text()
        if ROPE_PATCH["marker"] in content:
            if revert:
                orig = rope_path.with_suffix(".orig")
                if orig.exists():
                    shutil.copy2(orig, rope_path)
                    print(f"  ✅ reverted rope.py")
                    changed += 1
                else:
                    print(f"  ⚠️  no .orig, skipping")
            else:
                print(f"  ⏭️  already patched")
        else:
            new_content = content
            for func_name, old, new in ROPE_PATCH["insertions"]:
                if old in new_content:
                    new_content = new_content.replace(old, new, 1)
                    print(f"  ✅ {func_name}")
                else:
                    print(f"  ⚠️  {func_name} anchor not found, skipping")

            if new_content != content and not dry_run:
                shutil.copy2(rope_path, rope_path.with_suffix(".orig"))
                rope_path.write_text(new_content)
                changed += 1
            elif dry_run and new_content != content:
                changed += 1
    print()

    # --- Patch 2: jit/core.py ---
    print("[2/3] jit/core.py — RuntimeError → warning")
    jit_path = aiter_root / JIT_CORE_PATCH["file"]
    if apply_text_patch(jit_path, JIT_CORE_PATCH["old"], JIT_CORE_PATCH["new"], JIT_CORE_PATCH["marker"], dry_run, revert):
        changed += 1
    print()

    # --- Patch 3: pa_mqa_logits.py ---
    print("[3/3] pa_mqa_logits.py — sched_barrier before early return")
    mqa_path = aiter_root / PA_MQA_PATCH["file"]
    if not mqa_path.exists():
        print(f"  ❌ {mqa_path} not found")
    else:
        content = mqa_path.read_text()
        if PA_MQA_PATCH["marker"] in content:
            if revert:
                orig = mqa_path.with_suffix(".orig")
                if orig.exists():
                    shutil.copy2(orig, mqa_path)
                    print(f"  ✅ reverted pa_mqa_logits.py")
                    changed += 1
            else:
                print(f"  ⏭️  already patched")
        else:
            new_content = content
            for old, new in PA_MQA_PATCH["replacements"]:
                if old in new_content:
                    new_content = new_content.replace(old, new, 1)
                    print(f"  ✅ barrier inserted")
                else:
                    print(f"  ⚠️  anchor not found, skipping")

            if new_content != content and not dry_run:
                shutil.copy2(mqa_path, mqa_path.with_suffix(".orig"))
                mqa_path.write_text(new_content)
                changed += 1
            elif dry_run and new_content != content:
                changed += 1
    print()

    # Summary
    if changed > 0:
        print(f"✅ {changed} file(s) {'reverted' if revert else 'patched'}")
        if not dry_run and not revert:
            print("\nRestart SGLang service for changes to take effect.")
    else:
        print("No changes made.")


if __name__ == "__main__":
    main()
