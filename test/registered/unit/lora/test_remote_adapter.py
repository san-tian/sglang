"""CPU HTTP/archive tests; no model or GPU imports."""

import concurrent.futures
import http.server
import importlib.util
import io
import os
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

source = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/lora/remote_adapter.py"
)
spec = importlib.util.spec_from_file_location("remote_adapter", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestRemoteAdapter(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / "cache"
        env = patch.dict(os.environ, SGLANG_LORA_DOWNLOAD_CACHE=str(self.cache))
        env.start()
        self.addCleanup(env.stop)
        self.body = self.archive()
        self.hits = 0
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/opaque")
                    self.end_headers()
                    return
                owner.hits += 1
                self.send_response(200)
                self.end_headers()
                self.wfile.write(owner.body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = "http://127.0.0.1:" + str(self.server.server_address[1]) + "/opaque"

    @staticmethod
    def archive(mode="w:gz", prefix="", extra=None):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode=mode) as tar:
            for name, data in {
                prefix + "adapter_config.json": b'{"r":16}',
                prefix + "adapter_model.safetensors": b"fixture weights",
                **(extra or {}),
            }.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return output.getvalue()

    def check_resolved(self):
        path = Path(module.resolve_lora_path(self.url))
        self.assertEqual(
            (path / "adapter_model.safetensors").read_bytes(), b"fixture weights"
        )
        return path

    def test_local_and_hf_unchanged(self):
        for path in [None, "/data/adapter", "org/adapter"]:
            self.assertEqual(module.resolve_lora_path(path), path)
        self.assertEqual(self.hits, 0)

    def test_gzip_without_suffix(self):
        self.check_resolved()
        self.assertEqual(self.hits, 1)

    def test_plain_tar(self):
        self.body = self.archive(mode="w")
        self.check_resolved()

    def test_zstd_without_suffix(self):
        try:
            import zstandard
        except ImportError:
            self.skipTest("zstandard not installed")
        self.body = zstandard.ZstdCompressor().compress(self.archive(mode="w"))
        self.check_resolved()

    def test_nested_adapter(self):
        self.body = self.archive(prefix="sampler_weights/")
        self.assertEqual(self.check_resolved().name, "sampler_weights")

    def test_redirect(self):
        self.url = self.url.replace("/opaque", "/redirect")
        self.check_resolved()

    def test_concurrent_download_once(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            paths = list(pool.map(module.resolve_lora_path, [self.url] * 8))
        self.assertEqual(len(set(paths)), 1)
        self.assertEqual(self.hits, 1)

    def test_reject_parent_traversal(self):
        self.body = self.archive(extra={"../escape": b"x"})
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.check_resolved()
        self.assertFalse((self.cache / "escape").exists())

    def test_reject_absolute(self):
        self.body = self.archive(extra={"/absolute": b"x"})
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.check_resolved()

    def test_reject_link(self):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as tar:
            info = tarfile.TarInfo("symlink")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../escape"
            tar.addfile(info)
        self.body = output.getvalue()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.check_resolved()

    def test_reject_two_adapters(self):
        self.body = self.archive(
            extra={"other/adapter_config.json": b"{}", "other/adapter_model.bin": b"x"}
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.check_resolved()

    def test_reject_training_state(self):
        self.body = self.archive(extra={})
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as tar:
            info = tarfile.TarInfo("optimizer.pt")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
        self.body = output.getvalue()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.check_resolved()

    def test_failure_is_not_cached(self):
        self.body = b"not an archive"
        with self.assertRaises(tarfile.ReadError):
            self.check_resolved()
        self.body = self.archive()
        self.check_resolved()
        self.assertEqual(self.hits, 2)

    def test_cached_metadata_revalidated(self):
        path = self.check_resolved()
        (path / "adapter_model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "Invalid cached"):
            self.check_resolved()


if __name__ == "__main__":
    unittest.main()
