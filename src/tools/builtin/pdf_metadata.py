"""Best-effort enrichment, never an existence gate for user PDFs."""
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from ._cite_key import title_key
from ...observability.tracing import span, emit
from .openalex import openalex_headers, reconstruct_abstract

FIELDS = ('title', 'authors', 'year', 'abstract')


def normalize(data):
    '''规范化 PDF 元数据的字符串、年份和关键词字段。

    paras:
        data: 从 PDF 提取或从检索结果转换的元数据字典。
    return: 规范化后的字典副本，缺失或无效字段使用空值。
    '''
    result = dict(data)
    for name in ('title', 'authors', 'abstract', 'doi', 'journal', 'volume', 'number', 'pages'):
        if not isinstance(result.get(name), str):
            result[name] = ''
        else:
            result[name] = result[name].strip()
    if not isinstance(result.get('year'), int) or isinstance(result['year'], bool) or result['year'] < 0:
        result['year'] = 0
    if not isinstance(result.get('keywords'), list):
        result['keywords'] = []
    result['keywords'] = [k for k in result['keywords'] if isinstance(k, str)]
    return result


def doi_key(value):
    '''去除 DOI 的常见前缀和首尾空白并统一大小写。

    paras:
        value: DOI 字符串，可包含 https://doi.org/ 或 doi: 前缀。
    return: 用于匹配的 DOI 字符串。
    '''
    return value.lower().strip().removeprefix('https://doi.org/').removeprefix('doi:')


class MetadataCompleter:
    """每篇依次尝试三个后端，各请求至多一次，首个匹配即采用。

    Eight-second socket timeout, no retries within each backend. This is not a
    strict wall-clock deadline (DNS and socket reads may take additional time).
    """
    def __init__(self):
        '''初始化按文件指纹或路径复用的补全缓存，成功与失败均缓存。
        '''
        self.cache = {}

    def complete(self, data):
        '''缺失关键字段时执行一轮多后端补全，缓存结果并保留字段来源和剩余缺项。

        paras:
            data: 包含文件路径或指纹的 PDF 元数据。
        return: 带补全状态、字段来源和缺失字段列表的元数据字典。
        '''
        result = normalize(data)
        origins = {k: 'pdf' for k in FIELDS if result.get(k)}
        missing = [k for k in FIELDS if not result.get(k)]
        status = 'not_needed'
        if missing:
            key = data.get('fingerprint') or data.get('file')
            emit('metadata_cache', cache_hit=key in self.cache)
            if key not in self.cache:
                self.cache[key] = self._lookup(result)
            candidate, status = self.cache[key]
            if candidate:
                for name in missing:
                    if candidate.get(name):
                        result[name] = candidate[name]
                        origins[name] = candidate['source']
        result.update(metadata_missing=[k for k in FIELDS if not result.get(k)],
                      field_sources=origins, completion_status=status)
        return result

    def _lookup(self, data):
        '''按顺序查询三个后端，唯一身份匹配且已知字段无冲突时采用候选。

        paras:
            data: 规范化后的 PDF 元数据。
        return: 候选及状态；无标题和 DOI 时不联网。
        '''
        doi, title = doi_key(data['doi']), title_key(data['title'])
        if not doi and not title:
            return None, 'insufficient_identity'
        for backend in ('openalex', 'semantic_scholar', 'arxiv'):
            try:
                items = self._fetch(backend, data)
                matches = [p for p in items if (
                    doi_key(p.get('doi', '')) == doi if doi else title_key(p['title']) == title)]
                if len(matches) != 1:
                    emit('metadata_match', backend=backend, status='no_unique_match')
                    continue
                candidate = matches[0]
                if ((title and title_key(candidate['title']) != title) or
                        (data['year'] and candidate['year'] and data['year'] != candidate['year'])):
                    emit('metadata_match', backend=backend, status='metadata_conflict')
                    continue
                candidate['source'] = backend
                emit('metadata_match', backend=backend, status='matched')
                return candidate, 'matched'
            except Exception as exc:
                emit('metadata_match', backend=backend, status='backend_failed',
                     error_type=type(exc).__name__)
        return None, 'all_backends_failed'

    def _fetch(self, backend, data):
        '''单次请求获取统一格式候选，不调用带重试的通用搜索链路。

        paras:
            backend: OpenAlex、Semantic Scholar 或 arXiv 后端名称。
            data: 含标题或 DOI 的元数据。
        return: 规范化候选列表；网络或格式错误向外抛出，由调用方切换后端。
        '''
        doi = doi_key(data['doi'])
        headers = {}
        if backend == 'openalex':
            query = {'filter': 'doi:https://doi.org/' + doi} if doi else {'search': data['title']}
            url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({**query, 'per-page': 5})
            headers = openalex_headers()
        elif backend == 'semantic_scholar':
            from .search import _s2_headers
            headers = _s2_headers()
            fields = 'title,authors,year,abstract,externalIds'
            if doi:
                url = 'https://api.semanticscholar.org/graph/v1/paper/DOI:' + urllib.parse.quote(doi, safe='') + '?fields=' + fields
            else:
                url = 'https://api.semanticscholar.org/graph/v1/paper/search?' + urllib.parse.urlencode(
                    {'query': data['title'], 'limit': 5, 'fields': fields})
        else:
            query = 'all:' + doi if doi else 'ti:"' + data['title'].replace('"', ' ') + '"'
            url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode(
                {'search_query': query, 'start': 0, 'max_results': 5})
        request = urllib.request.Request(url, headers=headers)
        with span('network', host=urllib.parse.urlparse(url).hostname, backend=backend,
                  url=url, attempt=1, timeout=8) as trace:
            with urllib.request.urlopen(request, timeout=8) as response:
                raw = response.read().decode('utf-8')
                trace['http_status'] = getattr(response, 'status', None)
        if backend == 'openalex':
            return [normalize(dict(title=p.get('display_name'), doi=p.get('doi'),
                authors=', '.join(a['author']['display_name'] for a in p.get('authorships', [])),
                year=p.get('publication_year'),
                abstract=reconstruct_abstract(p.get('abstract_inverted_index'))))
                for p in json.loads(raw).get('results', [])]
        if backend == 'semantic_scholar':
            payload = json.loads(raw)
            items = [payload] if doi else payload.get('data', [])
            return [normalize(dict(title=p.get('title'), doi=(p.get('externalIds') or {}).get('DOI'),
                authors=', '.join(a.get('name', '') for a in p.get('authors', [])),
                year=p.get('year'), abstract=p.get('abstract'))) for p in items]
        ns = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        return [normalize(dict(title=p.findtext('a:title', '', ns),
            doi=p.findtext('arxiv:doi', '', ns),
            authors=', '.join(a.findtext('a:name', '', ns) for a in p.findall('a:author', ns)),
            year=int(p.findtext('a:published', '0000', ns)[:4]),
            abstract=p.findtext('a:summary', '', ns)))
            for p in ET.fromstring(raw).findall('a:entry', ns)]
