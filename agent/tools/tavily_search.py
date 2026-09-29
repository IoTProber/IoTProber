"""
Tavily 网页搜索工具
==================
暴露接口一览
-----------
1. tavily_web_search(query: str) -> str
   - LangChain @tool 装饰器包装，可直接挂载到 LangGraph ToolNode / create_react_agent。
   - 参数: query — 搜索关键词字符串。
   - 返回: 格式化的搜索摘要字符串（供 LLM 直接阅读）。

2. search_web(query: str, max_results: int = 5) -> list[dict]
   - 原始调用，返回 Tavily API 的结构化结果列表。
   - 每个 dict 包含: title, url, content, score。
   - 适合需要自行处理原始数据的场景。

3. search_web_formatted(query: str, max_results: int = 5) -> str
   - 调用 search_web 并将结果格式化为易读的 Markdown 字符串。
   - 适合直接填充 SearchState.search_results 字段。

4. TAVILY_TOOLS: list
   - 包含 [tavily_web_search] 的列表，方便一行注册到 agent。
   - 示例: agent = create_react_agent(llm, tools=TAVILY_TOOLS)

配置来源
--------
TAVILY_API_KEY 按以下顺序解析（与其余 API key 统一，不再需要第二份配置）：
  1. 环境变量 / .env 中的 TAVILY_API_KEY
  2. config/llm_config.local.json（优先）或 config/llm_config.json 的 "TAVILY": {"API_KEY": ...}
     —— 占位符 "YOUR_API_KEY" 视为未配置
"""

import json
import sys

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()

_client: TavilyClient | None = None


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)


def _resolve_api_key() -> str:
    """env/.env first, then the shared config/llm_config*.json ("TAVILY")."""
    key = os.getenv("TAVILY_API_KEY")
    if key:
        return key.strip()
    try:
        from path_config import LLM_CONFIG_FILE
        with open(LLM_CONFIG_FILE, encoding="utf-8") as fh:
            key = (json.load(fh).get("TAVILY") or {}).get("API_KEY", "")
    except Exception:
        key = ""
    key = (key or "").strip()
    return "" if key.startswith("YOUR_") else key


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = _resolve_api_key()
        if not api_key:
            raise EnvironmentError(
                "TAVILY_API_KEY not configured. Set it in the environment/.env, or put "
                '{"TAVILY": {"API_KEY": "tvly-..."}} into config/llm_config.local.json.'
            )
        _client = TavilyClient(api_key=api_key)
    return _client


def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    执行 Tavily 网页搜索，返回原始结构化结果。

    参数
    ----
    query       : 搜索关键词。
    max_results : 最多返回的结果条数，默认 5。

    返回
    ----
    list[dict]，每条结果包含:
        - title   (str)  : 页面标题
        - url     (str)  : 页面链接
        - content (str)  : 摘要内容
        - score   (float): 相关性评分
    """
    client = _get_client()
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=False,
    )
    results = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0.0),
            }
        )
    return results


def search_web_formatted(query: str, max_results: int = 5) -> str:
    """
    执行 Tavily 网页搜索，返回格式化的 Markdown 字符串。

    参数
    ----
    query       : 搜索关键词。
    max_results : 最多返回的结果条数，默认 5。

    返回
    ----
    str — 可直接赋值给 SearchState.search_results 的 Markdown 文本。
    """
    results = search_web(query=query, max_results=max_results)
    if not results:
        return f"No search results found for: {query}"

    lines = [f"## Search Results: {query}\n"]
    for idx, item in enumerate(results, start=1):
        lines.append(f"### {idx}. {item['title']}")
        lines.append(f"- **URL**: {item['url']}")
        lines.append(f"- **Score**: {item['score']:.2f}")
        lines.append(f"- **Summary**: {item['content']}\n")
    return "\n".join(lines)


@tool
def tavily_web_search(query: str) -> str:
    """
    Use the Tavily API to search the internet and retrieve the latest webpage content summaries related to the query.

    Parameters
    ----------
    query : The keyword or question to search for.

    Returns
    -------
    str — Formatted search results containing titles, links, and summaries.
    """
    try:
        return search_web_formatted(query=query, max_results=5)
    except EnvironmentError as exc:
        # The unseen ReAct loop routes here when the detector wants web
        # evidence; an unconfigured key must degrade to a textual "no result"
        # so the decision graph can continue (route back to the unseen node,
        # which proceeds without web evidence) instead of crashing the run.
        return (f"Web search unavailable: {exc} "
                "Continue with the evidence already available; do not call "
                "this tool again.")


TAVILY_TOOLS = [tavily_web_search]
