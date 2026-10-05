"""引用证据与报告指纹。证据等级由程序约束，不接受模型自报原文已验证。"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from .citation_syntax import STANDARD_CITATION, citation_keys
from .tex_sources import read_tex_sources

LABELS = {"abstract_support": "摘要支持",
          "insufficient": "证据不足", "contradiction": "存在矛盾", "failed": "检查失败"}
VERSION = 3

ACTIONS = {"abstract_support": "通过摘要级核查，无需修改。",
           "insufficient": "未确认：摘要未涉及或缺少摘要；保留待确认，必要时弱化论断或换用有摘要支持的引用，不自动读全文或判错。",
           "contradiction": "需修改：按矛盾原因修改论断或替换引用，交由 WritingAgent 修订后重新核查。",
           "failed": "技术检查未完成：有限重试仍失败，报告错误并等待恢复；不据此修改正文或判通过。"}


def digest(value):
    '''将可序列化对象按排序键编码为 JSON 并计算指纹。

    paras:
        value: 可 JSON 序列化的对象。
    return: SHA-256 十六进制摘要。
    '''
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def scan_claims(content):
    # 去掉未转义的行注释，保留行数。每次出现独立校验，不以最后一次引用覆盖前面的论断。
    '''提取 TeX 中每次标准引用出现的位置及其前后论断上下文。

    paras:
        content: 待扫描的 TeX 正文文本。
    return: 包含 cite_key、行号和上下文的记录列表。
    '''
    content = re.sub(r"(?<!\\)%[^\n]*", "", content)
    claims = []
    for match in STANDARD_CITATION.finditer(content):
        for key in citation_keys(match):
            if key:
                claims.append({"cite_key": key, "line": content.count('\n', 0, match.start()) + 1,
                               "claim": content[max(0, match.start()-300):match.end()+300]})
    return claims


def sources_for(ref, claim):
    '''从共享文献库构造摘要证据，不读取参考文献 PDF。

    paras:
        ref: 被引 Paper；None 或没有摘要时不提供证据。
        claim: 待核查的引用表述，当前保留此参数但不用于筛选摘要。
    return: 证据列表；摘要最多保留前 1500 字符，无摘要时返回空列表。
    '''
    sources = []
    if ref and ref.abstract:
        sources.append({"kind": "abstract", "location": "reference_library.abstract",
                        "text": ref.abstract[:1500]})
    return sources


def classify(result, sources, claim):
    '''校验证据摘录与摘要的对应关系，将模型判断映射为程序约束的证据等级。

    paras:
        result: 模型返回的引用判断字典或失败结果。
        sources: 可供核对的摘要证据列表。
        claim: 当前引用对应的论断上下文。
    return: 包含证据、状态、是否通过及处理建议的结果字典。
    '''
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
    '''计算正文、同目录参考文献文件和结构化文献库的版本指纹。

    paras:
        tex_path: 待核查的 TeX 正文路径。
        refs: 当前结构化文献对象序列。
    return: tex、bib 和 library 指纹字典；不存在的 bib 指纹为 None。
    '''
    path = Path(tex_path)
    bib = path.with_name('references.bib')
    return {"tex": digest(read_tex_sources(path)),
            "bib": sha256(bib.read_bytes()).hexdigest() if bib.exists() else None,
            "library": digest([r.to_dict() for r in refs])}


def report_status(tex_path, refs, *, mark_stale=False):
    '''比对引用报告版本和当前输入指纹，按需将旧报告标记为过期。

    paras:
        tex_path: 待核查的 TeX 正文路径。
        refs: 当前结构化文献对象序列。
        mark_stale: 是否在指纹不匹配时同步标记磁盘报告过期。
    return: 当前核查统计、过期提示或报告不可验证的说明。
    '''
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
        if report.get('error'):
            return '引用核查错误：' + report['error']
        counts = Counter(item['status'] for item in report['items'])
        return '引用核查（当前版本）：' + '；'.join(f'{label} {counts[key]} 项' for key, label in LABELS.items()) + '。覆盖主文件及静态引入章节；编译成功不代表引用正确，语义判断仍需人工复核。'
    except (OSError, ValueError, KeyError, TypeError):
        return '没有可验证的当前版本引用核查报告，不能声明引用全部正确。'
