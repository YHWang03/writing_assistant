"""引用证据与报告指纹。证据等级由程序约束，不接受模型自报原文已验证。"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

LABELS = {"abstract_support": "摘要支持",
          "insufficient": "证据不足", "contradiction": "存在矛盾", "failed": "检查失败"}
VERSION = 2

ACTIONS = {"abstract_support": "通过摘要级核查，无需修改。",
           "insufficient": "未确认：摘要未涉及或缺少摘要；保留待确认，必要时弱化论断或换用有摘要支持的引用，不自动读全文或判错。",
           "contradiction": "需修改：按矛盾原因修改论断或替换引用，交由 WritingAgent 修订后重新核查。",
           "failed": "技术检查未完成：有限重试仍失败，报告错误并等待恢复；不据此修改正文或判通过。"}


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def scan_claims(content):
    # 去掉未转义的行注释，保留行数。每次出现独立校验，不以最后一次引用覆盖前面的论断。
    content = re.sub(r"(?<!\\)%[^\n]*", "", content)
    pattern = r"\\(?:cite|citet|citep|citeauthor|citeyear)(?:\s*\[[^\]]*\])*\s*\{([^}]+)\}"
    claims = []
    for match in re.finditer(pattern, content):
        for key in match[1].split(','):
            key = key.strip()
            if key:
                claims.append({"cite_key": key, "line": content.count('\n', 0, match.start()) + 1,
                               "claim": content[max(0, match.start()-300):match.end()+300]})
    return claims


def sources_for(ref, claim):
    """仅使用共享文献库摘要；Citation 阶段不读取参考文献 PDF。"""
    sources = []
    if ref and ref.abstract:
        sources.append({"kind": "abstract", "location": "reference_library.abstract",
                        "text": ref.abstract[:1500]})
    return sources


def classify(result, sources, claim):
    base = {"claim": claim, "evidence": [], "status": "failed", "reason": "检查失败",
            "passed": False, "action": ACTIONS['failed'], "suggestion": "", "label": LABELS['failed']}
    if not isinstance(result, dict):
        return base
    base['reason'] = result.get('reason', '')
    base['suggestion'] = result.get('suggestion', '')
    index, quote = result.get('source_index'), result.get('evidence_quote')
    valid = (isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(sources)
             and isinstance(quote, str) and len(quote.strip()) >= 12
             and sources[index]['kind'] == 'abstract'
             and quote.strip() in sources[index]['text'])
    verdict = result.get('verdict')
    base['status'] = 'insufficient'
    if valid:
        source = sources[index]
        base['evidence'] = [{"quote": quote.strip(), "location": source['location'], "kind": source['kind']}]
        if verdict == '✅':
            base['status'] = 'abstract_support'
        elif verdict == '❌':
            base['status'] = 'contradiction'
    if not valid:
        base['reason'] = '没有可核对的证据摘录；' + base['reason']
    base['label'] = LABELS[base['status']]
    base['passed'] = base['status'] == 'abstract_support'
    base['action'] = ACTIONS[base['status']]
    if base['status'] == 'contradiction' and not base['suggestion']:
        base['suggestion'] = '依据摘要调整上述论断，或替换为能支持该论断的引用；不要编造替代文献。'
    return base


def snapshot(tex_path, refs):
    path = Path(tex_path)
    bib = path.with_name('references.bib')
    return {"tex": sha256(path.read_bytes()).hexdigest(),
            "bib": sha256(bib.read_bytes()).hexdigest() if bib.exists() else None,
            "library": digest([r.to_dict() for r in refs])}


def report_status(tex_path, refs, *, mark_stale=False):
    path = Path(tex_path).with_name('citation_evidence.json')
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('version') != VERSION or report['snapshot'] != snapshot(tex_path, refs):
            if mark_stale:
                from .library import atomic_write
                report['stale'] = True
                atomic_write(path, json.dumps(report, ensure_ascii=False, indent=2))
                text_path = path.with_name('citation_report.txt')
                if text_path.exists():
                    text = text_path.read_text(encoding='utf-8')
                    if not text.startswith('[已过期]'):
                        atomic_write(text_path, '[已过期] 正文、文献或来源已变化，以下旧结论不可沿用。\n\n' + text)
            return '引用核查报告已过期：正文、参考文献或来源已变化，需重新检查；不能沿用旧结论。'
        counts = Counter(item['status'] for item in report['items'])
        return '引用核查（当前版本）：' + '；'.join(f'{label} {counts[key]} 项' for key, label in LABELS.items()) + '。仅覆盖指定 TeX 文件，不展开子文件；编译成功不代表引用正确，语义判断仍需人工复核。'
    except (OSError, ValueError, KeyError, TypeError):
        return '没有可验证的当前版本引用核查报告，不能声明引用全部正确。'
