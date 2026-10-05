"""按 PDF 内容和解析版本缓存结果；同进程并发请求按指纹串行去重。"""

from hashlib import sha256
import json
from pathlib import Path
from threading import Lock

from ..domain.library import atomic_write

CACHE_VERSION = "metadata-v2-ocr-filename-year"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "pdf_metadata"
_locks_guard = Lock()
_locks = {}


class PDFCache:
    def __init__(self, directory=None):
        '''初始化 PDF 缓存目录和进程内结果缓存。

        paras:
            directory: 磁盘缓存目录；None 使用项目默认缓存目录。
        '''
        self.directory = Path(directory) if directory is not None else DEFAULT_CACHE_DIR
        self.results = {}

    def key(self, content: bytes):
        '''结合缓存版本和 PDF 内容生成稳定指纹。

        paras:
            content: PDF 文件的原始字节。
        return: SHA-256 十六进制缓存键。
        '''
        return sha256(CACHE_VERSION.encode() + b"\0" + content).hexdigest()

    def lock(self, key):
        '''取得同一缓存目录和指纹共享的线程锁。

        paras:
            key: 缓存条目的内容指纹。
        return: 保护该缓存条目读写的 Lock。
        '''
        with _locks_guard:
            return _locks.setdefault((str(self.directory.resolve()), key), Lock())

    def read(self, key):
        '''优先读取进程内缓存，再读取并校验磁盘缓存。

        paras:
            key: 缓存条目的内容指纹。
        return: 缓存结果字典；未命中、损坏或不可读时为 None。
        '''
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
        '''更新进程内缓存；成功解析结果同时原子写入磁盘，失败结果仅驻留内存。

        paras:
            key: 缓存条目的内容指纹。
            result: 含解析状态和指纹的缓存结果字典。
        '''
        self.results[key] = dict(result)
        if result["status"] == "failed":
            # 失败只在本解析器生命周期内抑制；重启可在环境修复后重新尝试。
            return
        atomic_write(self.directory / f"{key}.json", json.dumps(result, ensure_ascii=False))
