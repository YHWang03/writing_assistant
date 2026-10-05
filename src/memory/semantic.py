'''本地多语言记忆向量检索，按内容缓存归一化向量，不依赖向量数据库。'''

from functools import lru_cache
from hashlib import sha256
import json
import logging
from pathlib import Path

from ..domain.library import atomic_write

MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
MODEL_DIR = Path(
    'C:/Users/Wang/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/'
    'snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
)
MIN_SIMILARITY = 0.3
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    '''按需复用本机 Hugging Face 缓存中的 Qwen 模型，不复制权重或联网下载。

    return: CPU 上的 SentenceTransformer；依赖或模型不可用时抛出异常供调用方降级。
    '''
    from sentence_transformers import SentenceTransformer
    if not (MODEL_DIR / 'config.json').is_file():
        raise FileNotFoundError(f'{MODEL_NAME} 本地缓存不存在，请检查 MODEL_DIR: {MODEL_DIR}')
    model = SentenceTransformer(str(MODEL_DIR), device='cpu', local_files_only=True)
    model.max_seq_length = 2048 # 模型能处理的最大输入长度
    return model


class SemanticIndex:
    '''每份记忆的可重建向量缓存，旧文本变更或模型变更后重新编码。'''

    def __init__(self, path: Path):
        '''读取缓存；缺失、损坏或模型不同则使用空缓存。

        paras:
            path: 与记忆 JSON 分开的向量缓存文件路径。
        '''
        self.path = path
        self.vectors = {}
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if data.get('model') == MODEL_NAME and isinstance(data.get('vectors'), dict):
                self.vectors = data['vectors']
        except (OSError, ValueError, AttributeError):
            pass

    def search(self, query: str, records: list[dict], limit: int) -> list[int]:
        '''先按余弦相似度筛选，再结合重要性重排；仅编码新增或变更文本。

        paras:
            query: 当前任务文本。
            records: 含 importance（1～5 分）的记忆记录列表。
            limit: 重排后返回的最大候选数。
        return: 当前 records 的索引列表；依赖、模型或缓存向量无效时抛出异常。
        '''
        import numpy as np
        model = _model()
        texts = [f"{r.get('name', '')}\n{r.get('description', '')}\n{r.get('body', '')}"
                 for r in records]
        keys = [sha256(text.encode('utf-8')).hexdigest() for text in texts] # 记忆记录的唯一标识，哈希值
        missing = {key: text for key, text in zip(keys, texts) if key not in self.vectors}
        if missing:
            vectors = model.encode(list(missing.values()), normalize_embeddings=True,
                                   show_progress_bar=False, batch_size=4)
            self.vectors.update(zip(missing, vectors.tolist()))
        current = {key: self.vectors[key] for key in keys}
        if missing or len(current) != len(self.vectors):
            self.vectors = current
            try:
                atomic_write(self.path, json.dumps({'model': MODEL_NAME, 'vectors': current}))
            except OSError as exc:
                logger.warning('记忆向量缓存保存失败，继续使用内存缓存: %s', exc)
        query_text = ('Instruct: Given the current task, retrieve relevant user preferences, '
                      'constraints and project facts from long-term memory.\nQuery: ' + query)
        query_vector = model.encode(query_text, normalize_embeddings=True,
                                    show_progress_bar=False)
        matrix = np.asarray([current[key] for key in keys])
        if matrix.ndim != 2 or matrix.shape[1] != len(query_vector) or not np.isfinite(matrix).all():
            self.vectors.clear()
            raise ValueError('记忆向量缓存无效，已清空；下次召回重新构建')
        similarities = matrix @ query_vector
        candidates = [i for i in np.argsort(-similarities).tolist()
                      if similarities[i] >= MIN_SIMILARITY][:limit * 2]
        candidates.sort(key=lambda i: -(0.8 * float(similarities[i])
                                        + 0.2 * (records[i].get('importance', 3) - 1) / 4))
        return candidates[:limit]
