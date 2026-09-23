"""由工具执行结果签发的来源记录，不接受模型声明的 source 作为证明。"""

import re
import unicodedata


def title_key(title):
    value = unicodedata.normalize('NFKD', title).casefold()
    return ' '.join(re.findall(r'[^\W_]+', ''.join(c for c in value if not unicodedata.combining(c))))


class ReferenceProvenance:
    def __init__(self):
        self.receipts = {}

    def record(self, title, kind):
        key = title_key(title)
        if key:
            self.receipts[key] = kind

    def lookup(self, title):
        return self.receipts.get(title_key(title))

    def admit(self, paper, verifier):
        kind = self.lookup(paper.title)
        if not kind:
            import json
            result = json.loads(verifier.execute(paper.title))
            if not result.get('verified'):
                raise ValueError('文献标题尚未通过搜索验证，禁止入库：' + result.get('reason', '验证失败'))
            kind = self.lookup(paper.title)
        if not kind:
            raise ValueError('缺少实际工具执行产生的来源凭据')
        from .paper import Source
        paper.source = Source.USER if kind == 'pdf' else Source.ONLINE if kind == 'search' else Source.LLM
        paper.provenance_kind = kind
        paper.provenance_title = title_key(paper.title)
        return paper
