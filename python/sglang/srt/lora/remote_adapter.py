"""Resolve HTTP LoRA archives to a process-shared, atomic local directory."""

import fcntl
import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


def resolve_lora_path(path: str | None) -> str | None:
    if path is None or urllib.parse.urlsplit(path).scheme not in {"http", "https"}:
        return path
    root = Path(
        os.environ.get(
            "SGLANG_LORA_DOWNLOAD_CACHE", Path.home() / ".cache/sglang/lora_downloads"
        )
    )
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = hashlib.sha256(path.encode()).hexdigest()
    destination = root / key
    with (root / (key + ".lock")).open("a") as lock:
        # All TP processes share this cache. Publish only after complete validation.
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = destination / ".adapter-root"
        if marker.is_file():
            adapter = destination / marker.read_text()
            if adapter.resolve().is_relative_to(destination.resolve()) and _is_adapter(
                adapter
            ):
                return str(adapter)
            raise ValueError("Invalid cached LoRA archive")
        with tempfile.TemporaryDirectory(prefix=key + "-", dir=root) as temporary:
            temporary = Path(temporary)
            archive = temporary / "archive"
            with (
                urllib.request.urlopen(path, timeout=1200) as response,
                archive.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            with archive.open("rb") as source:
                magic = source.read(4)
            # Inspect content, since staged artifacts need not keep their suffix.
            if magic == b"\x28\xb5\x2f\xfd":
                import zstandard

                expanded = temporary / "archive.tar"
                with archive.open("rb") as source, expanded.open("wb") as output:
                    zstandard.ZstdDecompressor().copy_stream(source, output)
                archive = expanded
            unpacked = temporary / "unpacked"
            unpacked.mkdir()
            with tarfile.open(archive, "r:*") as tar:
                for member in tar:
                    name = PurePosixPath(member.name)
                    if (
                        name.is_absolute()
                        or ".." in name.parts
                        or not (member.isfile() or member.isdir())
                    ):
                        raise ValueError("Unsafe member in LoRA archive")
                    target = unpacked.joinpath(*name.parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with (
                            tar.extractfile(member) as source,
                            target.open("xb") as output,
                        ):
                            shutil.copyfileobj(source, output)
            candidates = [
                p.parent
                for p in unpacked.rglob("adapter_config.json")
                if _is_adapter(p.parent)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "LoRA archive must contain exactly one adapter_config.json with adapter weights"
                )
            relative = candidates[0].relative_to(unpacked)
            (unpacked / ".adapter-root").write_text(str(relative))
            os.replace(unpacked, destination)
        return str(destination / relative)


def _is_adapter(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(
        (path / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
