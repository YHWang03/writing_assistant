"""按 PDF 内容和解析版本缓存结果；同进程并发请求按指纹串行去重。"""

from hashlib import sha256
import json
from pathlib import Path
from threading import Lock

from ..domain.library import atomic_write

CACHE_VERSION = "metadata-v1"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "pdf_metadata"
_locks_guard = Lock()
_locks = {}


class PDFCache:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else DEFAULT_CACHE_DIR
        self.results = {}

    def key(self, content: bytes):
        return sha256(CACHE_VERSION.encode() + b"\0" + content).hexdigest()

    def lock(self, key):
        with _locks_guard:
            return _locks.setdefault((str(self.directory.resolve()), key), Lock())

    def read(self, key):
        if key in self.results:
            return dict(self.results[key])
        try:
            result = json.loads((self.directory / f"{key}.json").read_text(encoding="utf-8"))
            if (isinstance(result, dict) and result.get("fingerprint") == key
                    and result.get("status") in ("success", "failed")):
                return result
        except (OSError, ValueError):
            pass
        return None

    def write(self, key, result):
        self.results[key] = dict(result)
        if result["status"] == "failed":
            # 失败只在本解析器生命周期内抑制；重启可在环境修复后重新尝试。
            return
        atomic_write(self.directory / f"{key}.json", json.dumps(result, ensure_ascii=False))
