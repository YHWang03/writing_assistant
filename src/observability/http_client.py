'''保留 SDK 默认连接池与超时配置，记录其每次底层 HTTP 尝试。'''

from anthropic import DefaultHttpxClient
from .tracing import span


class TracedHttpClient(DefaultHttpxClient):
    '''观察同步 HTTP 请求，包括 SDK 内部重试，不修改其策略。'''

    def send(self, request, **kwargs):
        '''测量一次 HTTP 发送，非流式调用含完整响应体读取，不记录请求头。

        paras:
            request: SDK 构建的 HTTP 请求。
            **kwargs: 原样传给 HTTP 客户端的选项。
        return: 原响应；连接异常原样抛出。
        '''
        with span('http', host=request.url.host, url=str(request.url),
                  method=request.method, timing_scope='headers' if kwargs.get('stream') else 'full_response') as trace:
            response = super().send(request, **kwargs)
            trace.update(http_status=response.status_code,
                         status='failed' if response.status_code >= 400 else 'success',
                         request_id=response.headers.get('request-id') or response.headers.get('x-request-id'),
                         retry_after=response.headers.get('retry-after'))
            return response
