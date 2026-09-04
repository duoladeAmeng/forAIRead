# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import asyncio
import re

import httpx

from app.config import settings

_VERSION_SEG = re.compile(r"/v\d+$")


def _is_dashscope() -> bool:
    return "dashscope.aliyuncs.com" in settings.rerank_base_url


def _rerank_url() -> str:
    """把配置里的上游地址拼成 /rerank 端点。每次现算,改配置不用重启。

    地址带不带版本段都收:硅基流动和 Jina 的 rerank 在 /v1 下,Cohere 在 /v2 下,而
    「上游地址」按字面理解很容易只填到域名。少了版本段会打成 https://host/rerank,
    上游回 404,而且只在查询时以 hybrid_rerank 失败的形式冒出来,排障要绕一圈。
    DashScope 端点已含完整路径,直接用。"""
    base = settings.rerank_base_url.rstrip("/")
    if _is_dashscope():
        return base
    if not _VERSION_SEG.search(base):
        base += "/v1"
    return base + "/rerank"


_RETRY_STATUS = {429, 500, 502, 503, 504}
_RETRIES = 3          # 限流退避重试次数
_BACKOFF = 1.5        # 秒,每次翻倍


async def _post(url, json, headers, timeout):
    """带退避重试:子句拆分之后一个问题要打好几次 /rerank,上游按秒限流,
    突发流量会撞 429。这是瞬时限流不是坏请求,退一步再试就过去了;
    连试几次还不行才抛出去(评估脚本会把这一轮记成失败,不会静默算低分)。"""
    last: httpx.Response | None = None
    last_exc: Exception | None = None
    for i in range(_RETRIES + 1):
        try:
            async with httpx.AsyncClient() as c:
                resp = await c.post(url, json=json, headers=headers, timeout=timeout)
        except httpx.TransportError as e:
            # 连接层的抖动(断连、读超时、连不上)跟 429/5xx 一样是瞬时故障,一样该退避重试。
            # 早先只判状态码,c.post 抛出来就直接冒到顶:上游回 503 会重试,上游把连接
            # 掐了反而一次都不试。eval-rag 跑到第 200 多题被断一次,十几分钟整轮白跑。
            last_exc = e
            if i < _RETRIES:
                await asyncio.sleep(_BACKOFF * (2 ** i))
                continue
            raise
        if resp.status_code not in _RETRY_STATUS:
            return resp
        last = resp
        if i < _RETRIES:
            await asyncio.sleep(_BACKOFF * (2 ** i))
    if last is None and last_exc is not None:
        raise last_exc
    return last


async def rerank(query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """直连上游 /rerank 调重排模型。返回 [(原始索引, 相关分)] 按分降序,截断 top_n。

    支持两种上游格式:
    - Jina / Cohere / 硅基流动: {"query": ..., "documents": ..., "top_n": ...}
    - DashScope: {"model": ..., "input": {"query": ..., "documents": ...}, "parameters": {"top_n": ...}}
    """
    if not docs:
        return []
    n = top_n or len(docs)
    if _is_dashscope():
        payload = {"model": settings.rerank_model,
                   "input": {"query": query, "documents": docs},
                   "parameters": {"top_n": n}}
    else:
        payload = {"model": settings.rerank_model, "query": query, "documents": docs,
                   "top_n": n}
    resp = await _post(_rerank_url(), payload,
                       {"Authorization": f"Bearer {settings.rerank_api_key}"}, 60)
    resp.raise_for_status()
    data = resp.json()
    # DashScope 响应在 output.results,其他在 results
    results = data.get("output", data).get("results", data.get("results", []))
    ranked = sorted(((r["index"], float(r["relevance_score"])) for r in results),
                    key=lambda x: x[1], reverse=True)
    return ranked[:top_n] if top_n else ranked
