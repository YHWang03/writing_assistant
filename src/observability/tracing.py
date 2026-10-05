'''本地可关联事件、单调计时和可选脱敏详情，不依赖外部追踪服务。'''

from contextlib import contextmanager
from functools import wraps
import json
import logging
import os
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

from .telemetry import _scope, call_scope

_details_dir = None
_run_id = uuid4().hex
_standard = set(logging.makeLogRecord({}).__dict__) | {'message', 'asctime'}
_secret = re.compile(r'api[-_]?key|authorization|password|secret|access[-_]?token|cookie', re.I)


def sanitize(value):
    '''递归遮蔽凭据及当前环境密钥，保留 API 返回的思考内容和配置。

    paras:
        value: 待记录的 JSON 兼容数据。
    return: 脱敏后的副本；不能保证识别自然语言中的所有个人信息。
    '''
    if isinstance(value, dict):
        return {str(k): '[REDACTED]' if _secret.search(str(k)) else sanitize(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        if value.lstrip().startswith(('{', '[')):
            try:
                return json.dumps(sanitize(json.loads(value)), ensure_ascii=False)
            except (ValueError, TypeError):
                pass
        for key, secret in os.environ.items():
            if _secret.search(key) and len(secret) >= 6:
                value = value.replace(secret, '[REDACTED]')
        value = re.sub(r'(?i)(Bearer\s+)[^\s\"\',;]+', r'\1[REDACTED]', value)
        value = re.sub(r'(https?://)[^/\s@]+@', r'\1[REDACTED]@', value)
        return re.sub(r'(?i)((?:api[-_]?key|token|password|secret)=)[^&\s]+', r'\1[REDACTED]', value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize(str(value))


def configure_details(directory):
    '''设置本次运行的详情目录并更新运行标识，不主动创建详情文件。

    paras:
        directory: 与 JSONL 同级的本次运行详情目录。
    '''
    global _details_dir, _run_id
    _details_dir = Path(directory)
    _run_id = uuid4().hex


def save_detail(value):
    '''默认将完整脱敏详情分文件保存；LOG_DETAILS=0 可关闭，写入失败返回错误标记。

    paras:
        value: 模型请求、响应或工具参数与结果。
    return: 相对 structured 目录的文件引用，或未记录原因。
    '''
    if os.getenv('LOG_DETAILS', '1').lower() not in {'1', 'true', 'yes'}:
        return {'recorded': False, 'reason': 'disabled'}
    if _details_dir is None:
        return {'recorded': False, 'reason': 'not_configured'}
    try:
        _details_dir.mkdir(parents=True, exist_ok=True)
        path = _details_dir / (uuid4().hex + '.json')
        payload = json.dumps(sanitize(value), ensure_ascii=False)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(payload)
        return {'recorded': True, 'path': _details_dir.name + '/' + path.name,
                'bytes': len(payload.encode('utf-8'))}
    except Exception as exc:
        return {'recorded': False, 'reason': type(exc).__name__}


def event_fields(record):
    '''合并上下文及所有 extra 字段，避免新字段被固定白名单丢弃。

    paras:
        record: 日志记录。
    return: 含版本与运行标识的脱敏字段。
    '''
    return sanitize({'schema_version': 2, 'run_id': _run_id, 'logger': record.name, **_scope.get(),
                     **{k: v for k, v in record.__dict__.items() if k not in _standard}})


def emit(event, **fields):
    '''记录继承当前调用关系的结构化事件。

    paras:
        event: 事件名。
        **fields: 事件属性。
    '''
    logging.getLogger(__name__).info(event, extra={**_scope.get(), **fields, 'event': event})


def result_status(result):
    '''识别工具常见的字符串及 JSON 错误结果，不改变原返回内容。

    paras:
        result: 工具执行结果。
    return: success 或 failed；语义是否完成仍由业务事件描述。
    '''
    if isinstance(result, str):
        if result.lstrip().startswith('Error:'):
            return 'failed'
        try:
            result = json.loads(result)
        except (ValueError, TypeError):
            return 'success'
    if isinstance(result, dict) and (result.get('error') or result.get('ok') is False or
                                    result.get('status') in {'error', 'failed'}):
        return 'failed'
    return 'success'


@contextmanager
def span(kind, **fields):
    '''记录成对的开始、结束事件；异常或中断仍保留耗时并向外抛出。

    paras:
        kind: 操作类别，生成 kind_start 与 kind_end。
        **fields: 操作归属与属性。
    return: 可补充状态和输出属性的字典；duration 单位为秒。
    '''
    parent = _scope.get().get('span_id')
    with call_scope(parent_span_id=parent, span_id=uuid4().hex, **fields):
        started = perf_counter()
        state = {'status': 'success'}
        emit(kind + '_start')
        try:
            yield state
        except BaseException as exc:
            state.update(status='cancelled' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                         error_type=type(exc).__name__, http_status=getattr(exc, 'code', None))
            raise
        finally:
            emit(kind + '_end', **state, duration=perf_counter() - started)


def traced(kind):
    '''给同步函数增加通用执行跨度，保留签名元信息及返回行为。

    paras:
        kind: 跨度类别。
    return: 函数装饰器。
    '''
    def decorate(function):
        '''包装目标函数并保留其元信息。

        paras:
            function: 需要记录执行跨度的同步函数。
        return: 包含计时及状态记录的包装函数。
        '''
        @wraps(function)
        def wrapped(*args, **kwargs):
            '''记录目标调用起止及返回状态；异常记录后继续抛出。

            paras:
                *args: 原样传给目标函数的位置参数。
                **kwargs: 原样传给目标函数的关键字参数。
            return: 目标函数的原始返回值。
            '''
            with span(kind, operation=function.__qualname__) as state:
                result = function(*args, **kwargs)
                state['status'] = result_status(result)
                return result
        return wrapped
    return decorate
