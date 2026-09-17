#!/usr/bin/env python3
"""
多源 + Semantic Scholar 混合架构论文搜索脚本
用于 paper-daily skill，搜索最近一个月的新论文和最近一年的高影响力论文。

支持的数据源（--sources，逗号分隔）：
  arxiv       arXiv API，按日期区间/关键词检索预印本
  openreview  OpenReview API，检索 ICLR/NeurIPS/ICML 等计算机顶会论文（无需 API key）

新增数据源只需在 SOURCE_REGISTRY 中注册一个 fetcher 函数，其余流程
（筛选、四维研究优先级预筛、去重、排序、输出）完全复用。
"""

import xml.etree.ElementTree as ET
import json
import re
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import urllib.request
import urllib.parse

# 同目录的共享配置解析模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_config import (  # noqa: E402
    clean_stem,
    find_config_path,
    resolve_workspace_subdir,
    sanitize_paper_title,
    scan_known_papers,
)

logger = logging.getLogger(__name__)


def _title_tokens(title: str) -> Set[str]:
    """标题分词，用于"新论文 ↔ 已有文献"的脉络关联。

    过滤掉过于通用的词，否则每篇论文都会和所有其他论文"相关"。
    """
    import re as _re

    stop = {
        'a', 'an', 'the', 'of', 'for', 'and', 'or', 'to', 'in', 'on', 'with', 'via',
        'is', 'are', 'be', 'by', 'from', 'at', 'as', 'that', 'this', 'it', 'its',
        'model', 'models', 'method', 'methods', 'approach', 'approaches', 'learning',
        'analysis', 'study', 'new', 'novel', 'toward', 'towards', 'using', 'based',
        'efficient', 'optimization',
        # 标题里的高频套话，做关联时会制造假匹配
        'understanding', 'rethinking', 'revisited', 'survey', 'guide', 'practical',
        'general', 'unified', 'simple', 'fast', 'better', 'improved', 'scaling',
    }
    words = _re.findall(r"[A-Za-z][A-Za-z0-9\-']+", title or '')
    return {w.lower() for w in words if len(w) > 2 and w.lower() not in stop}


def find_related_papers(paper: Dict, reference: Dict, limit: int = 3) -> List[Dict]:
    """找出这篇论文与知识库里已有论文的关联（按共同标题词排序）。

    Args:
        paper: 本次检索到的论文
        reference: scan_known_papers() 返回的 reference 映射
        limit: 最多返回几条

    Returns:
        [{'title', 'stem', 'source', 'shared'}, ...]
    """
    if not reference:
        return []

    tokens = _title_tokens(paper.get('title', ''))
    if not tokens:
        return []

    scored = []
    for stem, info in reference.items():
        other = _title_tokens(info.get('title') or '')
        if not other:
            continue
        shared = tokens & other
        if not shared:
            continue
        scored.append((len(shared), sorted(shared), stem, info))

    scored.sort(key=lambda x: (-x[0], x[2]))
    out = []
    for count, shared, stem, info in scored[:limit]:
        out.append({
            'title': info.get('title') or stem,
            'stem': stem,
            'source': info.get('source', ''),
            'path': info.get('path', ''),
            'shared_terms': shared,
            'shared_count': count,
        })
    return out


def _force_utf8_stdio():
    """Windows 控制台默认 GBK，直接 print 含中文/特殊字符的 JSON 会抛
    UnicodeEncodeError 并丢掉整份结果。统一改用 UTF-8 且不因编码失败中断。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


_force_utf8_stdio()


def title_to_note_filename(title: str) -> str:
    """将论文标题转换为论文稳定主干 `<论文标题>`（见 paperread 工作区 AGENT.md）。

    规则集中在 paper_config.sanitize_paper_title，保证与用户自己的 markdown
    解析流程、01-raw PDF 文件名、03-notes 文件夹名完全一致。
    """
    return sanitize_paper_title(title)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests library not found, using urllib for Semantic Scholar API")

# ---------------------------------------------------------------------------
# API 配置
# ---------------------------------------------------------------------------
ARXIV_NS = {
    'atom': 'http://www.w3.org/2005/Atom',
    'arxiv': 'http://arxiv.org/schemas/atom'
}

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = "title,abstract,publicationDate,citationCount,influentialCitationCount,url,authors,authors.affiliations,externalIds"

# 默认分类关键词映射（当配置中无用户自定义关键词时使用）
ARXIV_CATEGORY_KEYWORDS = {
    "cs.AI": "artificial intelligence",
    "cs.LG": "machine learning",
    "cs.CL": "computational linguistics natural language processing",
    "cs.CV": "computer vision",
    "cs.MM": "multimedia",
    "cs.MA": "multi-agent systems",
    "cs.RO": "robotics"
}

# ---------------------------------------------------------------------------
# 评分常量  —— 修改权重时只需编辑这里
# ---------------------------------------------------------------------------

# 各维度原始评分的满分值（归一化基准）
SCORE_MAX = 3.0

# 相关性评分：关键词在标题 / 摘要中匹配的加分
RELEVANCE_TITLE_KEYWORD_BOOST = 0.5
RELEVANCE_SUMMARY_KEYWORD_BOOST = 0.3
RELEVANCE_CATEGORY_MATCH_BOOST = 1.0

# 新近性阈值（天） -> 对应评分
RECENCY_THRESHOLDS = [
    (30, 3.0),
    (90, 2.0),
    (180, 1.0),
]
RECENCY_DEFAULT = 0.0

# 影响力信号：高影响力引用数归一化到 0-SCORE_MAX
# 含义：达到此引用数时视为满分
POPULARITY_INFLUENTIAL_CITATION_FULL_SCORE = 100

# 研究优先级权重。元数据排序只用于预筛，不代表全文质量。
WEIGHTS_NORMAL = {
    'relevance': 0.55,
    'recency': 0.15,
    'impact': 0.10,
    'evidence': 0.20,
}
# 高影响力候选仍以研究相关性为主，引用信号不得主导。
WEIGHTS_HOT = {
    'relevance': 0.50,
    'recency': 0.05,
    'impact': 0.25,
    'evidence': 0.20,
}

# Semantic Scholar 速率限制等待时间（秒）
S2_RATE_LIMIT_WAIT = 10
S2_CATEGORY_REQUEST_INTERVAL = 3

# Semantic Scholar API Key（可选，从配置文件读取）
S2_API_KEY = None

# ---------------------------------------------------------------------------
# OpenReview API 配置
# ---------------------------------------------------------------------------
OPENREVIEW_SEARCH_URL = "https://api2.openreview.net/notes/search"
OPENREVIEW_USER_AGENT = "paper-skills/0.1 (paper-daily)"
# OpenReview 单页最多返回 100 条
OPENREVIEW_PAGE_SIZE = 100
OPENREVIEW_REQUEST_INTERVAL = 1.0

# arXiv 请求参数：arXiv 高峰期单次响应可能超过 20s，超时给足；
# 关键词上限用于避免超长 OR 查询（0 表示不限制）
ARXIV_REQUEST_TIMEOUT = 120
ARXIV_MAX_KEYWORDS = 0

# 支持的论文数据源注册表。
# 新增数据源：写一个 fetcher(keywords, start_date, end_date, max_results) -> List[Dict]
# 并在 SOURCE_VALIDATORS 中登记，其余评分/去重/输出逻辑无需改动。
SOURCE_REGISTRY: Dict[str, Dict] = {}


def register_source(name: str, description: str, requires_key: bool = False):
    """把 fetcher 注册进 SOURCE_REGISTRY。"""
    def decorator(func):
        SOURCE_REGISTRY[name] = {
            'name': name,
            'description': description,
            'requires_key': requires_key,
            'fetch': func,
        }
        return func
    return decorator


def available_sources() -> List[str]:
    """返回所有已注册的数据源名称。"""
    return sorted(SOURCE_REGISTRY.keys())


def resolve_sources(raw: str) -> List[str]:
    """解析 --sources 参数，校验每个源是否已注册。"""
    requested = [s.strip().lower() for s in (raw or '').split(',') if s.strip()]
    if not requested:
        requested = ['arxiv']
    unknown = [s for s in requested if s not in SOURCE_REGISTRY]
    if unknown:
        raise ValueError(
            "未知数据源: %s（可用: %s）" % (', '.join(unknown), ', '.join(available_sources()))
        )
    # 去重但保持顺序
    seen = set()
    ordered = []
    for s in requested:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def parse_openreview_datetime(note: Dict) -> Optional[datetime]:
    """从 OpenReview note 中解析发布日期（毫秒时间戳），失败返回 None。"""
    # pdate 为正式发布日期，部分 venue 缺失；退回 cdate / tcdate（创建时间）
    for key in ('pdate', 'cdate', 'tcdate'):
        value = note.get(key)
        if not value:
            continue
        try:
            return datetime.fromtimestamp(float(value) / 1000.0)
        except (ValueError, TypeError, OSError):
            continue
    return None


def parse_openreview_note(note: Dict) -> Optional[Dict]:
    """把一条 OpenReview note 转换成与其他数据源一致的 paper 字典。"""
    content = note.get('content') or {}

    def field(name):
        raw = content.get(name)
        if isinstance(raw, dict):
            return raw.get('value')
        return raw

    title = field('title')
    if not title:
        return None

    paper = {
        'id': note.get('forum') or note.get('id'),
        'source': 'openreview',
        'title': str(title).strip(),
        'summary': str(field('abstract') or '').strip(),
        'authors': [str(a) for a in (field('authors') or [])],
        'affiliations': [],
        'categories': [],
        'venue': field('venue') or field('venueid') or '',
        'pdf_url': field('pdf') or '',
        'url': '',
    }

    # 指纹：优先取 arXiv 编号（OpenReview 的 paperhash 里常带 arxiv:xxxx），
    # 这样跨源去重才能命中同一篇论文
    paperhash = str(field('paperhash') or '')
    arxiv_match = re.search(r'arxiv[:\s]*(\d{4}\.\d{4,5})', paperhash, re.IGNORECASE)
    if arxiv_match:
        paper['arxiv_id'] = arxiv_match.group(1)

    forum = paper['id']
    if forum:
        paper['url'] = 'https://openreview.net/forum?id=%s' % forum
    if not paper['pdf_url'] and forum:
        paper['pdf_url'] = 'https://openreview.net/pdf?id=%s' % forum

    published = parse_openreview_datetime(note)
    if published:
        paper['published'] = published.isoformat()
        paper['published_date'] = published

    return paper


@register_source('openreview', 'OpenReview（ICLR/NeurIPS/ICML 等计算机顶会）')
def search_openreview_by_keywords(
    keywords: List[str],
    start_date: datetime,
    end_date: datetime,
    max_results: int = 100,
    max_retries: int = 2,
) -> List[Dict]:
    """使用关键词检索 OpenReview，并按日期区间过滤本地结果。

    OpenReview 的 search 接口不支持日期区间参数，因此这里拉取结果后
    在本地按 published_date 过滤；日期不可用的条目予以保留，避免漏掉
    尚未写入 pdate 的新论文（其后由新近性评分处理）。

    Args:
        keywords: 搜索关键词列表
        start_date: 开始日期
        end_date: 结束日期
        max_results: 最大结果数
        max_retries: 最大重试次数

    Returns:
        论文列表
    """
    terms = [k.strip() for k in keywords if k and k.strip()]
    if not terms:
        return []

    # OpenReview 的 term 是自由文本 AND 匹配，关键词过多会过窄，限制前 6 个
    term = ' '.join(terms[:6])
    logger.info("[OpenReview] Keyword search: %s", terms[:6])

    collected: List[Dict] = []
    seen_ids: Set[str] = set()
    offset = 0

    while len(collected) < max_results:
        params = {
            'term': term,
            'limit': min(OPENREVIEW_PAGE_SIZE, max_results - len(collected)),
            'offset': offset,
            'content': 'all',
            'source': 'forum',
        }
        url = '%s?%s' % (OPENREVIEW_SEARCH_URL, urllib.parse.urlencode(params))

        data = None
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': OPENREVIEW_USER_AGENT})
                with urllib.request.urlopen(req, timeout=45) as response:
                    data = json.loads(response.read().decode('utf-8'))
                break
            except Exception as e:
                logger.warning(
                    "[OpenReview] Error (attempt %d/%d): %s", attempt + 1, max_retries, e
                )
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 2)
        if data is None:
            logger.error("[OpenReview] Keyword search failed, returning partial results")
            break

        notes = data.get('notes') or []
        if not notes:
            break

        for note in notes:
            paper = parse_openreview_note(note)
            if not paper:
                continue
            key = paper.get('arxiv_id') or paper.get('id')
            if key and key in seen_ids:
                continue
            if key:
                seen_ids.add(key)

            published = paper.get('published_date')
            if published is not None:
                # 与 arXiv 侧保持一致：只保留目标区间内的论文
                if published.date() < start_date.date() or published.date() > end_date.date():
                    continue
            collected.append(paper)
            if len(collected) >= max_results:
                break

        offset += len(notes)
        if len(notes) < params['limit']:
            break
        time.sleep(OPENREVIEW_REQUEST_INTERVAL)

    logger.info("[OpenReview] Keyword search found %d papers", len(collected))
    return collected


def search_arxiv_adapter(
    keywords: List[str],
    start_date: datetime,
    end_date: datetime,
    max_results: int = 100,
    max_keywords: int = 0,
    timeout: int = 120,
) -> List[Dict]:
    """arXiv 源适配器：统一签名（关键词 + 日期区间）。"""
    return search_arxiv_by_keywords(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
        max_keywords=max_keywords,
        timeout=timeout,
    )


def fetch_for_source(source: str, **kwargs):
    """调用某个已注册数据源的 fetcher。"""
    entry = SOURCE_REGISTRY.get(source)
    if not entry:
        raise ValueError("未知数据源: %s" % source)
    return entry['fetch'](**kwargs)


def fetch_with_paper_config(source: str, **kwargs):
    """按配置调整检索规模后调用数据源 fetcher。

    - 保留全部 arXiv 关键词，由查询函数自动分批，避免超长 OR 查询
    - 用配置的 arxiv_request_timeout 覆盖默认超时

    通过包装函数实现，脚本内的自动测试与 paper_config 缺失场景也能正常工作。
    """
    if source in ('arxiv', 'arxiv_multi_source'):
        if not kwargs.get('max_keywords'):
            kwargs['max_keywords'] = ARXIV_MAX_KEYWORDS
        kwargs.setdefault('timeout', ARXIV_REQUEST_TIMEOUT)
    return fetch_for_source(source, **kwargs)


def search_papers_across_sources(
    sources: List[str],
    keywords: List[str],
    start_date: datetime,
    end_date: datetime,
    max_results_per_source: int = 100,
) -> Tuple[List[Dict], Dict[str, int]]:
    """按数据源逐一分发检索，单个源失败不影响其他源。

    Returns:
        (合并后的论文列表, {数据源: 命中数})
    """
    papers: List[Dict] = []
    counts: Dict[str, int] = {}

    for source in sources:
        try:
            fetched = fetch_with_paper_config(
                source,
                keywords=keywords,
                start_date=start_date,
                end_date=end_date,
                max_results=max_results_per_source,
            )
        except Exception as e:
            logger.warning("[%s] Search failed (non-fatal): %s", source, e)
            fetched = []
        counts[source] = len(fetched)
        papers.extend(fetched)

    logger.info("[Sources] Fetched %s", counts or '{}')
    return papers, counts



def load_research_config(config_path: str) -> Dict:
    """
    从 YAML 文件加载研究兴趣配置

    未显式给出路径时，使用 paper_config 的查找顺序（skill 自带 config.yaml →
    共享 config.yaml）。读取失败时打印明确错误并回退到内置默认配置。

    Args:
        config_path: 配置文件路径

    Returns:
        研究配置字典
    """
    import yaml

    resolved = config_path
    if not resolved:
        try:
            resolved = str(find_config_path(None))
        except FileNotFoundError as e:
            logger.error("%s", e)
            resolved = None

    if resolved:
        try:
            logger.info("Loading config from: %s", resolved)
            with open(resolved, 'r', encoding='utf-8-sig') as f:
                config = yaml.safe_load(f) or {}
            # 读取 Semantic Scholar API Key（如果配置了）
            global S2_API_KEY
            S2_API_KEY = config.get('semantic_scholar_api_key') or None
            # arXiv 请求参数
            global ARXIV_REQUEST_TIMEOUT, ARXIV_MAX_KEYWORDS
            ARXIV_REQUEST_TIMEOUT = int(config.get('arxiv_request_timeout', 120) or 120)
            ARXIV_MAX_KEYWORDS = int(config.get('arxiv_max_keywords', 0) or 0)
            return config
        except Exception as e:
            logger.error("Error loading config '%s': %s", resolved, e)
    else:
        logger.error("未找到任何配置文件。")

    # 主题无关的最小兜底：真实研究主题**只应来自 config.yaml 的 research_domains**。
    # 这里刻意不内置任何具体研究方向，避免把主题写死在脚本里；缺配置时会明确告警，
    # 检索结果为空即说明需要先创建配置（或用 paper-interests skill 生成）。
    logger.warning(
        "未加载到 research_domains，本次不会检索到任何论文。"
        "请在 config.yaml 配置研究领域，或调用 paper-interests skill。"
    )
    return {
        "research_domains": {},
        "excluded_keywords": ["workshop"],
    }


def calculate_date_windows(target_date: Optional[datetime] = None, days: int = 30) -> Tuple[datetime, datetime, datetime, datetime]:
    """
    计算两个时间窗口：最近N天和过去一年（除去最近N天）

    Args:
        target_date: 基准日期，如果为 None 则使用当前日期
        days: 最近搜索窗口的天数（默认30）

    Returns:
        (window_recent_start, window_recent_end, window_1y_start, window_1y_end)
    """
    if target_date is None:
        target_date = datetime.now()

    window_recent_start = target_date - timedelta(days=days)
    window_recent_end = target_date

    window_1y_start = target_date - timedelta(days=365)
    window_1y_end = target_date - timedelta(days=days + 1)

    return window_recent_start, window_recent_end, window_1y_start, window_1y_end


def search_arxiv_by_date_range(
    categories: List[str],
    start_date: datetime,
    end_date: datetime,
    max_results: int = 200,
    max_retries: int = 3
) -> List[Dict]:
    """
    使用 arXiv API 搜索指定日期范围内的论文
    
    Args:
        categories: arXiv 分类列表
        start_date: 开始日期
        end_date: 结束日期
        max_results: 最大结果数
        max_retries: 最大重试次数
        
    Returns:
        论文列表
    """
    # 构建分类查询
    category_query = "+OR+".join([f"cat:{cat}" for cat in categories])
    
    # 构建日期范围查询 (arXiv 使用 YYYYMMDD 格式)
    date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}0000+TO+{end_date.strftime('%Y%m%d')}2359]"
    
    # 组合查询
    full_query = f"({category_query})+AND+{date_query}"
    
    # 构建 URL
    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query={full_query}&"
        f"max_results={max_results}&"
        f"sortBy=submittedDate&"
        f"sortOrder=descending"
    )
    
    logger.info("[arXiv] Searching papers from %s to %s", start_date.date(), end_date.date())
    logger.debug("[arXiv] URL: %s...", url[:120])
    
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                xml_content = response.read().decode('utf-8')
                papers = parse_arxiv_xml(xml_content)
                logger.info("[arXiv] Found %d papers", len(papers))
                return papers
        except Exception as e:
            logger.warning("[arXiv] Error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                logger.info("[arXiv] Retrying in %d seconds...", wait_time)
                time.sleep(wait_time)
            else:
                logger.error("[arXiv] Failed after %d attempts", max_retries)
                return []
    
    return []


def _search_arxiv_by_keywords_single(
    keywords: List[str],
    start_date: datetime,
    end_date: datetime,
    max_results: int = 100,
    max_retries: int = 3,
    max_keywords: int = 30,
    timeout: int = 120,
) -> List[Dict]:
    """
    使用关键词直接搜索 arXiv 论文（不限分类）

    Args:
        keywords: 搜索关键词列表
        start_date: 开始日期
        end_date: 结束日期
        max_results: 最大结果数
        max_retries: 最大重试次数
        max_keywords: 参与查询的关键词上限（关键词过多会让 OR 查询过慢且过宽）
        timeout: 单次请求超时（秒）。arXiv 在高峰期可能单次响应 20s 以上

    Returns:
        论文列表
    """
    if max_keywords and len(keywords) > max_keywords:
        logger.info("[arXiv] Using first %d of %d keywords", max_keywords, len(keywords))
        keywords = keywords[:max_keywords]

    # 构建关键词查询 (在 title 和 abstract 中搜索)
    keyword_parts = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        # 如果关键词包含空格，用引号包裹进行精确匹配
        if ' ' in kw:
            keyword_parts.append(f'ti:"{kw}"+OR+abs:"{kw}"')
        else:
            keyword_parts.append(f"ti:{kw}+OR+abs:{kw}")

    if not keyword_parts:
        return []

    keyword_query = "+OR+".join([f"({p})" for p in keyword_parts])

    # 构建日期范围查询
    date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}0000+TO+{end_date.strftime('%Y%m%d')}2359]"

    full_query = f"({keyword_query})+AND+{date_query}"

    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query={urllib.parse.quote(full_query, safe='+:')}&"
        f"max_results={max_results}&"
        f"sortBy=relevance&"
        f"sortOrder=descending"
    )

    logger.info("[arXiv] Keyword search: %s", keywords)
    logger.debug("[arXiv] URL: %s...", url[:150])

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                xml_content = response.read().decode('utf-8')
                papers = parse_arxiv_xml(xml_content)
                logger.info("[arXiv] Keyword search found %d papers", len(papers))
                return papers
        except Exception as e:
            logger.warning("[arXiv] Keyword search error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                time.sleep(wait_time)
            else:
                logger.error("[arXiv] Keyword search failed after %d attempts", max_retries)
                return []

    return []


def search_arxiv_by_keywords(
    keywords: List[str],
    start_date: datetime,
    end_date: datetime,
    max_results: int = 100,
    max_retries: int = 3,
    max_keywords: int = 0,
    timeout: int = 120,
) -> List[Dict]:
    """分批查询 arXiv，避免研究主题过多导致单个 OR 请求过长。"""
    keywords = [kw.strip() for kw in keywords if str(kw).strip()]
    if max_keywords and len(keywords) > max_keywords:
        logger.info("[arXiv] Using first %d of %d keywords", max_keywords, len(keywords))
        keywords = keywords[:max_keywords]
    if not keywords:
        return []

    # 0 表示保留全部关键词；每批 20 个可避免 API 对超长查询返回 400。
    query_batch_size = 20
    keyword_batches = [
        keywords[index:index + query_batch_size]
        for index in range(0, len(keywords), query_batch_size)
    ]
    batch_max_results = max(1, (max_results + len(keyword_batches) - 1) // len(keyword_batches))
    all_papers: List[Dict] = []

    for batch_number, keyword_batch in enumerate(keyword_batches, 1):
        logger.info(
            "[arXiv] Keyword search batch %d/%d (%d keywords)",
            batch_number, len(keyword_batches), len(keyword_batch),
        )
        all_papers.extend(_search_arxiv_by_keywords_single(
            keywords=keyword_batch,
            start_date=start_date,
            end_date=end_date,
            max_results=batch_max_results,
            max_retries=max_retries,
            max_keywords=0,
            timeout=timeout,
        ))
        if batch_number < len(keyword_batches):
            time.sleep(1)

    logger.info("[arXiv] Keyword search found %d papers across all batches", len(all_papers))
    return all_papers


def search_semantic_scholar_hot_papers(
    query: str,
    start_date: datetime,
    end_date: datetime,
    top_k: int = 20,
    max_retries: int = 2
) -> List[Dict]:
    """
    使用 Semantic Scholar API 搜索指定时间范围内的高影响力论文

    Args:
        query: 搜索关键词
        start_date: 开始日期
        end_date: 结束日期
        top_k: 返回前 K 篇高影响力论文
        max_retries: 最大重试次数
        
    Returns:
        按高影响力引用数排序的论文列表
    """
    # 构建日期范围 (Semantic Scholar 使用 YYYY-MM-DD:YYYY-MM-DD 格式)
    date_range = f"{start_date.strftime('%Y-%m-%d')}:{end_date.strftime('%Y-%m-%d')}"
    
    # 构建请求参数
    params = {
        "query": query,
        "publicationDateOrYear": date_range,
        "limit": 100,  # 先拉取100篇相关度最高的
        "fields": SEMANTIC_SCHOLAR_FIELDS
    }
    
    headers = {
        "User-Agent": "StartMyDay-PaperFetcher/1.0"
    }
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
    
    logger.info("[S2] Searching hot papers from %s to %s", start_date.date(), end_date.date())
    logger.info("[S2] Query: '%s'", query)
    
    for attempt in range(max_retries):
        try:
            if HAS_REQUESTS:
                response = requests.get(
                    SEMANTIC_SCHOLAR_API_URL,
                    params=params,
                    headers=headers,
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()
            else:
                # 使用 urllib
                query_string = urllib.parse.urlencode(params)
                url = f"{SEMANTIC_SCHOLAR_API_URL}?{query_string}"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response:
                    data = json.loads(response.read().decode('utf-8'))
            
            papers = data.get("data", [])
            if not papers:
                logger.info("[S2] No papers found")
                return []
            
            # 本地二次过滤与排序
            valid_papers = []
            for p in papers:
                # 过滤掉没有标题或摘要的无效条目
                if not p.get("title") or not p.get("abstract"):
                    continue
                
                # 处理可能的 None 值
                inf_cit = p.get("influentialCitationCount") or 0
                cit = p.get("citationCount") or 0
                
                p["influentialCitationCount"] = inf_cit
                p["citationCount"] = cit
                
                # 标记来源
                p["source"] = "semantic_scholar"
                p["hot_score"] = inf_cit  # 使用高影响力引用数作为热度分数

                # 提取 affiliation 信息
                if p.get('authors') and not p.get('affiliations'):
                    affiliations = []
                    for a in p['authors']:
                        for affil in (a.get('affiliations') or []):
                            name = affil.get('name', '') if isinstance(affil, dict) else str(affil)
                            if name and name not in affiliations:
                                affiliations.append(name)
                    p['affiliations'] = affiliations
                
                valid_papers.append(p)
            
            # 按高影响力引用数倒序排列
            sorted_papers = sorted(
                valid_papers,
                key=lambda x: x["influentialCitationCount"],
                reverse=True
            )
            
            logger.info("[S2] Found %d valid papers, returning top %d", len(sorted_papers), top_k)
            return sorted_papers[:top_k]
            
        except Exception as e:
            error_msg = str(e)
            logger.warning("[S2] Error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            
            # 检查是否是 429 错误（Too Many Requests）
            is_rate_limit = False
            if HAS_REQUESTS and hasattr(e, 'response') and e.response is not None:
                is_rate_limit = e.response.status_code == 429
            else:
                is_rate_limit = "429" in error_msg or "Too Many Requests" in error_msg
            
            if attempt < max_retries - 1:
                # 对于 429 错误，使用更长的等待时间
                if is_rate_limit:
                    wait_time = S2_RATE_LIMIT_WAIT
                    logger.warning("[S2] Rate limit hit. Waiting %d seconds...", wait_time)
                else:
                    wait_time = (2 ** attempt) * 2
                    logger.info("[S2] Retrying in %d seconds...", wait_time)
                time.sleep(wait_time)
            else:
                logger.error("[S2] Failed after %d attempts", max_retries)
                return []
    
    return []


def search_hot_papers_from_categories(
    categories: List[str],
    start_date: datetime,
    end_date: datetime,
    top_k_per_category: int = 5,
    config: Optional[Dict] = None
) -> List[Dict]:
    """
    为多个 arXiv 分类搜索高影响力论文

    Args:
        categories: arXiv 分类列表
        start_date: 开始日期
        end_date: 结束日期
        top_k_per_category: 每个分类返回的论文数
        config: 研究配置（用于提取用户自定义关键词）

    Returns:
        合并后的高影响力论文列表
    """
    all_hot_papers = []
    seen_arxiv_ids = set()

    # 从配置中提取用户自定义的搜索关键词（更精准）
    user_queries = []
    if config:
        domains = config.get('research_domains', {})
        for domain_name, domain_config in domains.items():
            keywords = domain_config.get('keywords', [])
            # 取每个域的前3个关键词组合为查询
            if keywords:
                query = ' '.join(keywords[:3])
                user_queries.append(query)

    # 如果没有用户关键词，回退到分类关键词
    if not user_queries:
        user_queries = [ARXIV_CATEGORY_KEYWORDS.get(cat, cat) for cat in categories]

    # 去重查询
    seen_queries = set()
    unique_queries = []
    for q in user_queries:
        q_lower = q.lower()
        if q_lower not in seen_queries:
            seen_queries.add(q_lower)
            unique_queries.append(q)

    for query in unique_queries:

        try:
            papers = search_semantic_scholar_hot_papers(
                query=query,
                start_date=start_date,
                end_date=end_date,
                top_k=top_k_per_category
            )
        except Exception as e:
            logger.warning("[S2] Query '%s' failed: %s — skipping", query, e)
            papers = []

        # 去重（基于 arXiv ID）
        for p in papers:
            # 安全地从 externalIds 字典中提取 ArXiv 编号
            arxiv_id = p.get("externalIds", {}).get("ArXiv") if p.get("externalIds") else None
            
            # 统一写入 arxiv_id 字段，方便最后 Step 3 的全局去重
            p["arxiv_id"] = arxiv_id
            
            if arxiv_id and arxiv_id not in seen_arxiv_ids:
                seen_arxiv_ids.add(arxiv_id)
                all_hot_papers.append(p)
            elif not arxiv_id:
                # 没有 arXiv ID 的也保留（可能是其他来源的论文）
                all_hot_papers.append(p)
        
        time.sleep(S2_CATEGORY_REQUEST_INTERVAL)
    
    # 最终按影响力引用数排序
    all_hot_papers.sort(key=lambda x: x.get("influentialCitationCount", 0), reverse=True)
    
    return all_hot_papers


def parse_arxiv_xml(xml_content: str) -> List[Dict]:
    """
    解析 arXiv XML 结果
    
    Args:
        xml_content: XML 内容
        
    Returns:
        论文列表，每篇论文包含 ID、标题、作者、摘要等信息
    """
    papers = []
    
    try:
        root = ET.fromstring(xml_content)
        
        # 查找所有 entry 元素
        for entry in root.findall('atom:entry', ARXIV_NS):
            paper = {}
            
            # 提取 ID
            id_elem = entry.find('atom:id', ARXIV_NS)
            if id_elem is not None:
                paper['id'] = id_elem.text
                # 提取 arXiv ID（从 URL 中提取）
                match = re.search(r'arXiv:(\d+\.\d+)', paper['id'])
                if match:
                    paper['arxiv_id'] = match.group(1)
                else:
                    match = re.search(r'/(\d+\.\d+)$', paper['id'])
                    if match:
                        paper['arxiv_id'] = match.group(1)
            
            # 提取标题
            title_elem = entry.find('atom:title', ARXIV_NS)
            if title_elem is not None:
                paper['title'] = title_elem.text.strip()
            
            # 提取摘要
            summary_elem = entry.find('atom:summary', ARXIV_NS)
            if summary_elem is not None:
                paper['summary'] = summary_elem.text.strip()
            
            # 提取作者（及可选的 affiliation）
            authors = []
            affiliations = []
            for author in entry.findall('atom:author', ARXIV_NS):
                name_elem = author.find('atom:name', ARXIV_NS)
                if name_elem is not None:
                    authors.append(name_elem.text)
                affil_elem = author.find('arxiv:affiliation', ARXIV_NS)
                if affil_elem is not None and affil_elem.text:
                    affil = affil_elem.text.strip()
                    if affil and affil not in affiliations:
                        affiliations.append(affil)
            paper['authors'] = authors
            paper['affiliations'] = affiliations  # 可能为空列表
            
            # 提取发布日期
            published_elem = entry.find('atom:published', ARXIV_NS)
            if published_elem is not None:
                paper['published'] = published_elem.text
                # 解析日期
                try:
                    paper['published_date'] = datetime.fromisoformat(
                        paper['published'].replace('Z', '+00:00')
                    )
                except (ValueError, TypeError):
                    paper['published_date'] = None
            
            # 提取更新日期
            updated_elem = entry.find('atom:updated', ARXIV_NS)
            if updated_elem is not None:
                paper['updated'] = updated_elem.text
            
            # 提取分类
            categories = []
            for category in entry.findall('atom:category', ARXIV_NS):
                term = category.get('term')
                if term:
                    categories.append(term)
            paper['categories'] = categories
            
            # 提取 PDF 链接
            for link in entry.findall('atom:link', ARXIV_NS):
                if link.get('title') == 'pdf':
                    paper['pdf_url'] = link.get('href')
                    break
            
            # 提取主页面链接
            if 'id' in paper:
                paper['url'] = paper['id']
            
            # 标记来源
            paper['source'] = 'arxiv'
            
            papers.append(paper)
            
    except ET.ParseError as e:
        logger.error("Error parsing XML: %s", e)
        raise
    
    return papers


def calculate_relevance_score(
    paper: Dict,
    domains: Dict,
    excluded_keywords: List[str],
    focus_keywords: List[str] = None
) -> Tuple[float, Optional[str], List[str]]:
    """
    计算论文与研究兴趣的相关性评分

    当有 focus_keywords 时，以 focus 匹配为主导（高权重），
    已有兴趣域仅作为参考加分。

    Args:
        paper: 论文信息
        domains: 研究领域配置
        excluded_keywords: 排除关键词
        focus_keywords: 用户今日关注的关键词

    Returns:
        (相关性评分, 匹配的领域, 匹配的关键词列表)
    """
    focus_keywords = focus_keywords or []
    title = paper.get('title', '').lower()
    summary = paper.get('summary', '').lower() if 'summary' in paper else paper.get('abstract', '').lower()
    categories = set(paper.get('categories', []))

    # 检查排除关键词
    for keyword in excluded_keywords:
        if keyword.lower() in title or keyword.lower() in summary:
            return 0, None, []

    # ---- Focus 关键词独立评分（主导） ----
    focus_score = 0.0
    focus_matched = []
    if focus_keywords:
        for fk in focus_keywords:
            fk_lower = fk.lower().strip()
            if not fk_lower:
                continue
            if fk_lower in title:
                focus_score += 2.0  # 标题匹配：高分
                focus_matched.append(fk)
            elif fk_lower in summary:
                focus_score += 1.0  # 摘要匹配：中分
                focus_matched.append(fk)

    # ---- 已有兴趣域评分（参考） ----
    max_domain_score = 0
    best_domain = None
    domain_matched_keywords = []

    for domain_name, domain_config in domains.items():
        score = 0
        dm_keywords = []

        keywords = domain_config.get('keywords', [])
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in title:
                score += RELEVANCE_TITLE_KEYWORD_BOOST
                dm_keywords.append(keyword)
            elif keyword_lower in summary:
                score += RELEVANCE_SUMMARY_KEYWORD_BOOST
                dm_keywords.append(keyword)

        domain_categories = domain_config.get('arxiv_categories', [])
        for cat in domain_categories:
            if cat in categories:
                score += RELEVANCE_CATEGORY_MATCH_BOOST
                dm_keywords.append(cat)

        # Config documents priority on a 1-10 scale. Keep 5 backward-compatible
        # while allowing the configured priority to affect domain selection.
        priority = domain_config.get('priority', 5)
        try:
            priority = max(1.0, min(10.0, float(priority)))
        except (TypeError, ValueError):
            priority = 5.0
        weighted_score = min(score * (priority / 5.0), SCORE_MAX)

        if weighted_score > max_domain_score:
            max_domain_score = weighted_score
            best_domain = domain_name
            domain_matched_keywords = dm_keywords

    # ---- 合并评分 ----
    if focus_keywords:
        # Focus 模式：focus 为主，域匹配为辅（0.3 权重）
        total_score = focus_score + max_domain_score * 0.3
        all_matched = focus_matched + [k for k in domain_matched_keywords if k not in focus_matched]
        matched_domain = best_domain if best_domain else ("搜索结果" if focus_matched else None)
    else:
        # 普通模式：纯域评分
        total_score = max_domain_score
        all_matched = domain_matched_keywords
        matched_domain = best_domain

    return min(total_score, SCORE_MAX), matched_domain, all_matched


def calculate_recency_score(published_date: Optional[datetime]) -> float:
    """
    根据发布日期计算新近性评分
    
    Args:
        published_date: 发布日期
        
    Returns:
        新近性评分 (0-3)
    """
    if published_date is None:
        return 0
    
    now = datetime.now(published_date.tzinfo) if published_date.tzinfo else datetime.now()
    days_diff = (now - published_date).days
    
    for max_days, score in RECENCY_THRESHOLDS:
        if days_diff <= max_days:
            return score
    return RECENCY_DEFAULT


def calculate_abstract_evidence_score(summary: str) -> float:
    """
    评估摘要是否报告了可核验的研究信息，而不是推断论文质量。

    这是检索阶段的预筛信号。`novel`、`SOTA`、`first` 等作者自我评价
    不加分；只有问题、方法、证据、边界或可复核性信息才加分。

    Args:
        summary: 论文摘要

    Returns:
        摘要证据充分度 (0-3)
    """
    if not summary:
        return 0.0
    score = 0.0
    summary_lower = summary.lower()

    problem_indicators = [
        'we study', 'we investigate', 'we address', 'we consider',
        'research question', 'problem of', 'challenge of', 'objective'
    ]
    method_indicators = [
        'algorithm', 'estimator', 'optimizer', 'architecture', 'framework',
        'mechanism', 'dataset', 'benchmark', 'theorem', 'analysis'
    ]
    comparison_indicators = [
        'baseline', 'comparison', 'compared with', 'compared to',
        'ablation', 'versus', 'outperform', 'improve over'
    ]
    rigor_indicators = [
        'we prove', 'we show that', 'convergence rate', 'sample complexity',
        'upper bound', 'lower bound', 'regret bound', 'confidence interval',
        'statistically significant', 'multiple seeds', 'robustness'
    ]
    transparency_indicators = [
        'code is available', 'open-source', 'open source', 'we release',
        'limitations', 'failure case', 'computational cost'
    ]

    if any(ind in summary_lower for ind in problem_indicators):
        score += 0.5
    if any(ind in summary_lower for ind in method_indicators):
        score += 0.6
    if any(ind in summary_lower for ind in comparison_indicators):
        score += 0.6
    if re.search(r'(?<!\w)\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?x\b', summary_lower):
        score += 0.4
    if any(ind in summary_lower for ind in rigor_indicators):
        score += 0.8
    if any(ind in summary_lower for ind in transparency_indicators):
        score += 0.3

    return min(score, SCORE_MAX)


def calculate_quality_score(summary: str) -> float:
    """兼容旧调用；返回的是摘要证据充分度，不是全文质量。"""
    return calculate_abstract_evidence_score(summary)


def calculate_recommendation_score(
    relevance_score: float,
    recency_score: float,
    impact_score: float,
    evidence_score: float,
    is_hot_paper: bool = False
) -> float:
    """
    计算检索阶段的研究优先级分。

    权重定义在模块顶部常量 WEIGHTS_NORMAL / WEIGHTS_HOT 中。
    它只决定候选阅读顺序，不能解释为全文质量。对于高影响力候选，
    使用 WEIGHTS_HOT 适度提高影响力信号，但相关性仍占主导。

    Args:
        relevance_score: 相关性评分 (0-SCORE_MAX)
        recency_score: 新近性评分 (0-SCORE_MAX)
        impact_score: 引用影响力信号 (0-SCORE_MAX)
        evidence_score: 摘要证据充分度 (0-SCORE_MAX)
        is_hot_paper: 是否是高影响力论文

    Returns:
        研究优先级预筛分 (0-10)
    """
    scores = {
        'relevance': relevance_score,
        'recency': recency_score,
        'impact': impact_score,
        'evidence': evidence_score,
    }
    # 归一化到 0-10 分
    normalized = {k: (v / SCORE_MAX) * 10 for k, v in scores.items()}

    weights = WEIGHTS_HOT if is_hot_paper else WEIGHTS_NORMAL
    final_score = sum(normalized[k] * weights[k] for k in weights)

    return round(final_score, 2)


def filter_and_score_papers(
    papers: List[Dict],
    config: Dict,
    target_date: Optional[datetime] = None,
    is_hot_paper_batch: bool = False,
    focus_keywords: List[str] = None
) -> List[Dict]:
    """
    筛选和评分论文

    Args:
        papers: 论文列表
        config: 研究配置
        target_date: 目标日期（用于计算新近性）
        is_hot_paper_batch: 是否是高影响力论文批次

    Returns:
        筛选和评分后的论文列表
    """
    domains = config.get('research_domains', {})
    excluded_keywords = config.get('excluded_keywords', [])

    scored_papers = []

    for paper in papers:
        # 计算相关性
        relevance, matched_domain, matched_keywords = calculate_relevance_score(
            paper, domains, excluded_keywords, focus_keywords=focus_keywords or []
        )

        # 如果相关性为0，跳过
        if relevance == 0:
            continue

        # 计算新近性
        if 'published_date' in paper:
            recency = calculate_recency_score(paper.get('published_date'))
        else:
            # 对于 Semantic Scholar 的论文，使用 publicationDate
            pub_date_str = paper.get('publicationDate')
            if pub_date_str:
                pub_date = None
                for fmt in ('%Y-%m-%d', '%Y-%m', '%Y'):
                    try:
                        pub_date = datetime.strptime(pub_date_str, fmt)
                        break
                    except (ValueError, TypeError):
                        continue
                recency = calculate_recency_score(pub_date) if pub_date else 0
            else:
                recency = 0

        # 计算引用影响力信号。缺少引用数据时保持 0，不用新近性伪造热度，
        # 避免 recency 在综合分中被重复计算。
        if is_hot_paper_batch:
            # 高影响力论文：使用 influentialCitationCount
            inf_cit = paper.get('influentialCitationCount', 0)
            impact = min(
                inf_cit / (POPULARITY_INFLUENTIAL_CITATION_FULL_SCORE / SCORE_MAX),
                SCORE_MAX,
            )
        else:
            impact = 0.0

        # 摘要只能评估证据报告是否充分，不能评估全文质量。
        summary = paper.get('summary', '') if 'summary' in paper else paper.get('abstract', '')
        evidence = calculate_abstract_evidence_score(summary)

        # 计算研究优先级预筛分
        recommendation_score = calculate_recommendation_score(
            relevance, recency, impact, evidence, is_hot_paper_batch
        )

        # 添加评分信息
        paper['scores'] = {
            'relevance': round(relevance, 2),
            'recency': round(recency, 2),
            'impact': round(impact, 2),
            'evidence': round(evidence, 2),
            # 历史 JSON 兼容字段；新代码与报告应使用 impact/evidence。
            'popularity': round(impact, 2),
            'quality': round(evidence, 2),
            'recommendation': recommendation_score
        }
        paper['matched_domain'] = matched_domain
        paper['matched_keywords'] = matched_keywords
        paper['is_hot_paper'] = is_hot_paper_batch
        paper['score_type'] = 'research_priority_screening'
        paper['assessment_scope'] = 'metadata_and_abstract'

        scored_papers.append(paper)

    # 按研究优先级排序
    scored_papers.sort(key=lambda x: x['scores']['recommendation'], reverse=True)
    for rank, paper in enumerate(scored_papers, 1):
        paper['screening_rank'] = rank

    return scored_papers


# arXiv 适配器依赖 search_arxiv_by_keywords，故此处在所有函数定义完成后再注册
register_source('arxiv', 'arXiv 预印本（分类由 config.yaml 的 arxiv_categories 决定）')(
    search_arxiv_adapter
)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Search and filter papers across multiple sources with Semantic Scholar integration')
    parser.add_argument('--config', type=str,
                        default=None,
                        help='Path to research config file (default: skill config.yaml, then shared config.yaml)')
    parser.add_argument('--output', type=str, default='search_result.json',
                        help='Output JSON file path ("-" or "/dev/stdout" for stdout only)')
    parser.add_argument('--max-results', type=int, default=None,
                        help='Maximum number of results to fetch per source (default: config max_results or 200)')
    parser.add_argument('--top-n', type=int, default=None,
                        help='Number of top papers to return (default: config top_n or 10)')
    parser.add_argument('--target-date', type=str, default=None,
                        help='Target date (YYYY-MM-DD) for filtering')
    parser.add_argument('--categories', '--arxiv-categories', dest='categories', type=str,
                        default=None,
                        help='Comma-separated list of arXiv categories (default: config arxiv_categories)')
    parser.add_argument('--sources', type=str, default=None,
                        help='Comma-separated data sources (default: config sources). Available: %s'
                             % ','.join(available_sources()))
    parser.add_argument('--skip-hot-papers', action='store_true',
                        help='Skip searching hot papers from Semantic Scholar')
    parser.add_argument('--focus', type=str, default='',
                        help='User-specified focus keywords for today (comma-separated)')
    parser.add_argument('--focus-sources', type=str, default=None,
                        help='Data sources used in --focus mode (default: config sources)')
    parser.add_argument('--days', type=int, default=None,
                        help='Number of days to search back (default: config days or 30)')
    parser.add_argument('--exclude-known', action='store_true',
                        help='Exclude papers already in papers_dir / notes_dir / daily_dir history')
    parser.add_argument('--workspace', type=str, default=None,
                        help='Workspace root path (default: config workspace_path or PAPER_WORKSPACE_PATH)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr,
    )

    focus_keywords = [k.strip() for k in args.focus.split(',') if k.strip()] if args.focus else []
    if focus_keywords:
        logger.info("Focus keywords: %s", focus_keywords)

    config = load_research_config(args.config)

    # 未被命令行覆盖的参数回落到配置文件
    if args.max_results is None:
        args.max_results = int(config.get('max_results', 200) or 200)
    if args.top_n is None:
        args.top_n = int(config.get('top_n', 10) or 10)
    if args.days is None:
        args.days = int(config.get('days', 30) or 30)
    if not args.sources:
        args.sources = ','.join(config.get('sources') or ['arxiv'])
    if not args.categories:
        args.categories = ','.join(
            config.get('arxiv_categories') or ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV']
        )
    logger.info("max-results=%d top-n=%d days=%d", args.max_results, args.top_n, args.days)

    try:
        sources = resolve_sources(args.sources)
    except ValueError as e:
        logger.error("%s", e)
        return 1
    logger.info("Data sources: %s", sources)

    # 从配置收集检索关键词（各研究域的 keywords 合并去重）
    configured_keywords: List[str] = []
    seen_keywords = set()
    for _domain_name, domain_config in (config.get('research_domains') or {}).items():
        for kw in domain_config.get('keywords', []) or []:
            kw_clean = str(kw).strip()
            key = kw_clean.lower()
            if kw_clean and key not in seen_keywords:
                seen_keywords.add(key)
                configured_keywords.append(kw_clean)
    if not configured_keywords:
        logger.warning("No keywords found in config; falling back to arXiv category keywords")
        configured_keywords = [
            ARXIV_CATEGORY_KEYWORDS.get(cat, cat) for cat in args.categories.split(',')
        ]
    logger.info("Configured keywords: %d", len(configured_keywords))

    # 解析目标日期
    target_date = None
    if args.target_date:
        try:
            target_date = datetime.strptime(args.target_date, '%Y-%m-%d')
            logger.info("Target date: %s", args.target_date)
        except ValueError:
            logger.error("Invalid target date format: %s", args.target_date)
            return 1
    else:
        target_date = datetime.now()
        logger.info("Using current date: %s", target_date.strftime('%Y-%m-%d'))

    window_30d_start, window_30d_end, window_1y_start, window_1y_end = calculate_date_windows(target_date, days=args.days)
    logger.info("Date windows:")
    logger.info("  Recent %d days: %s to %s", args.days, window_30d_start.date(), window_30d_end.date())
    logger.info("  Past year (31-365 days): %s to %s", window_1y_start.date(), window_1y_end.date())

    # 解析分类
    categories = args.categories.split(',')

    all_scored_papers = []
    recent_papers = []
    hot_papers = []
    source_counts: Dict[str, int] = {}

    if focus_keywords:
        # ========== Focus 模式：关键词搜索为主 ==========
        logger.info("=" * 70)
        logger.info("Step 1 (Focus): Multi-source keyword search for: %s", focus_keywords)
        logger.info("=" * 70)

        # focus 模式下默认沿用配置的数据源，可用 --focus-sources 单独指定
        focus_source_names = resolve_sources(args.focus_sources or args.sources)
        logger.info("Focus data sources: %s", focus_source_names)

        focus_papers, source_counts = search_papers_across_sources(
            sources=focus_source_names,
            keywords=focus_keywords,
            start_date=window_30d_start,
            end_date=window_30d_end,
            max_results_per_source=args.max_results,
        )
        recent_papers = focus_papers

        if focus_papers:
            scored_focus = filter_and_score_papers(
                papers=focus_papers,
                config=config,
                target_date=target_date,
                is_hot_paper_batch=False,
                focus_keywords=focus_keywords
            )
            logger.info("Scored %d focus keyword papers", len(scored_focus))
            all_scored_papers.extend(scored_focus)
        else:
            logger.warning("No papers found for focus keywords")

        # Semantic Scholar 也按 focus 搜索（补充高引用论文）
        if not args.skip_hot_papers:
            logger.info("=" * 70)
            logger.info("Step 2 (Focus): Searching hot papers for focus keywords from Semantic Scholar")
            logger.info("=" * 70)

            focus_query = " ".join(focus_keywords)
            try:
                hot_papers = search_semantic_scholar_hot_papers(
                    query=focus_query,
                    start_date=window_1y_start,
                    end_date=window_1y_end,
                    top_k=20
                )
            except Exception as e:
                logger.warning("Semantic Scholar focus search failed (non-fatal): %s", e)
                hot_papers = []

            if hot_papers:
                scored_hot = filter_and_score_papers(
                    papers=hot_papers,
                    config=config,
                    target_date=target_date,
                    is_hot_paper_batch=True,
                    focus_keywords=focus_keywords
                )
                logger.info("Scored %d hot papers for focus", len(scored_hot))
                all_scored_papers.extend(scored_hot)

    else:
        # ========== 普通模式：按配置关键词多源检索 ==========
        logger.info("=" * 70)
        logger.info("Step 1: Multi-source search (last %d days) for configured interests", args.days)
        logger.info("=" * 70)

        recent_papers, source_counts = search_papers_across_sources(
            sources=sources,
            keywords=configured_keywords,
            start_date=window_30d_start,
            end_date=window_30d_end,
            max_results_per_source=args.max_results,
        )

        if recent_papers:
            scored_recent = filter_and_score_papers(
                papers=recent_papers,
                config=config,
                target_date=target_date,
                is_hot_paper_batch=False,
            )
            logger.info("Scored %d recent papers", len(scored_recent))
            all_scored_papers.extend(scored_recent)
        else:
            logger.warning("No recent papers found")

        # 搜索过去一年的高影响力论文（Semantic Scholar）
        if not args.skip_hot_papers:
            logger.info("=" * 70)
            logger.info("Step 2: Searching hot papers (past year) from Semantic Scholar")
            logger.info("=" * 70)

            try:
                hot_papers = search_hot_papers_from_categories(
                    categories=categories,
                    start_date=window_1y_start,
                    end_date=window_1y_end,
                    top_k_per_category=5,
                    config=config
                )
            except Exception as e:
                logger.warning("Semantic Scholar search failed (non-fatal): %s", e)
                hot_papers = []

            if hot_papers:
                scored_hot = filter_and_score_papers(
                    papers=hot_papers,
                    config=config,
                    target_date=target_date,
                    is_hot_paper_batch=True,
                )
                logger.info("Scored %d hot papers", len(scored_hot))
                all_scored_papers.extend(scored_hot)
            else:
                logger.warning("No hot papers found from Semantic Scholar")
        else:
            logger.info("Skipping hot paper search (disabled by user)")

    # ========== 第三步：合并结果并排序 ==========
    logger.info("=" * 70)
    logger.info("Step 3: Merging and ranking results")
    logger.info("=" * 70)
    
    # 按研究优先级排序
    all_scored_papers.sort(key=lambda x: x['scores']['recommendation'], reverse=True)
    
    # 去重（优先 arXiv ID，其次标题 normalize）。
    # 标题去重对所有来源生效：同一篇论文常同时出现在 arXiv 与 OpenReview 上，
    # 仅靠 ID 去重会重复推荐。
    seen_ids = set()
    seen_titles = set()
    unique_papers = []
    for p in all_scored_papers:
        arxiv_id = p.get('arxiv_id') or p.get('arxivId')
        title = p.get('title', '')
        title_normalized = re.sub(r'[^a-z0-9\s]', '', title.lower()).strip()

        if title_normalized and title_normalized in seen_titles:
            continue
        if arxiv_id:
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
        if title_normalized:
            seen_titles.add(title_normalized)
        unique_papers.append(p)
    
    logger.info("Total unique papers after deduplication: %d", len(unique_papers))

    if len(unique_papers) == 0:
        logger.warning("No papers matched the criteria!")
        return 1

    # ========== 第四步：标记知识库中已存在的论文 ==========
    # A 方案：检索列表保留全量并标记 already_known，推荐列表只取未命中历史的。
    known_stems = set()
    known_dirs = {}
    if args.exclude_known or config.get('exclude_known'):
        try:
            from paper_config import resolve_workspace_path

            workspace_root = resolve_workspace_path(config, args.workspace)
            known = scan_known_papers(workspace_root, config, log=logger.info)
            known_stems = known['stems']
            known_dirs = {k: str(v) for k, v in known['dirs'].items()}
        except Exception as e:
            logger.warning("扫描已存在论文失败，本次不去重：%s", e)

    known_count = 0
    related_total = 0
    for paper in unique_papers:
        # 论文稳定主干，与 01-raw / 03-notes 的命名规则一致
        stem = title_to_note_filename(paper.get('title', ''))
        paper['paper_stem'] = stem
        keys = {clean_stem(stem), clean_stem(paper.get('title', ''))}
        arxiv_id = paper.get('arxiv_id') or ''
        if arxiv_id:
            keys.add(clean_stem(arxiv_id))
        hits = keys & known_stems if known_stems else set()
        hit = bool(hits)
        paper['already_known'] = hit
        if hit:
            known_count += 1
            # 记录它已在知识库的哪一处（供日报标注"已在库"来源）
            hit_key = sorted(hits)[0]
            for slot in ('by_pdf', 'by_note', 'by_daily'):
                if hit_key in known.get(slot, {}):
                    paper['kb_source'] = slot.replace('by_', '')
                    paper['kb_path'] = known[slot][hit_key]
                    break

        # 与知识库已有论文的脉络关联：新论文该和哪几篇放在一起看
        related = find_related_papers(paper, known.get('reference') or {}, limit=3)
        if related:
            paper['related_papers'] = related
            related_total += 1
    if known_stems:
        logger.info("Known-library scan: %d papers already in library, %d papers have related prior work",
                    known_count, related_total)

    # 推荐候选：排除知识库里已有的（除非用户显式要求包含）
    include_known = bool(config.get('include_known_in_recommendations'))
    candidates = unique_papers if include_known else [p for p in unique_papers if not p['already_known']]

    if len(candidates) == 0:
        logger.warning("所有检索结果都已在知识库中，没有新的推荐论文。")
        # 不直接失败：仍然输出全量检索列表，便于日报记录

    # 取前 N 篇
    top_papers = candidates[:args.top_n]

    # 补充稳定主干与链接文件名（旧字段名保留，避免破坏既有用法）
    for paper in top_papers:
        paper['note_filename'] = paper.get('paper_stem') or title_to_note_filename(paper.get('title', ''))

    # 准备输出
    papers_by_source: Dict[str, int] = {}
    for p in unique_papers:
        src = p.get('source', 'unknown')
        papers_by_source[src] = papers_by_source.get(src, 0) + 1

    output = {
        'target_date': args.target_date or target_date.strftime('%Y-%m-%d'),
        'date_windows': {
            'recent_30d': {
                'start': window_30d_start.strftime('%Y-%m-%d'),
                'end': window_30d_end.strftime('%Y-%m-%d')
            },
            'past_year': {
                'start': window_1y_start.strftime('%Y-%m-%d'),
                'end': window_1y_end.strftime('%Y-%m-%d')
            }
        },
        'sources_searched': sources,
        'source_counts': source_counts,
        'papers_by_source': papers_by_source,
        'known_dirs': known_dirs,
        'total_recent': len(recent_papers),
        'total_hot': len(hot_papers),
        'total_unique': len(unique_papers),
        'total_known': known_count,
        'total_candidates': len(candidates),
        # A 方案：all_papers 是全量检索结果（含历史命中），日报的"本次检索列表"用它；
        # top_papers 是过滤历史后的推荐。
        'all_papers': [
            {
                'title': p.get('title', ''),
                'source': p.get('source', ''),
                'arxiv_id': p.get('arxiv_id', ''),
                'venue': p.get('venue', ''),
                'url': p.get('url', ''),
                'pdf_url': p.get('pdf_url', ''),
                'published': str(p.get('published', '')),
                'authors': p.get('authors', []),
                'paper_stem': p.get('paper_stem', ''),
                'matched_domain': p.get('matched_domain', ''),
                'matched_keywords': p.get('matched_keywords', []),
                'already_known': p.get('already_known', False),
                'kb_source': p.get('kb_source', ''),
                'related_papers': p.get('related_papers', []),
                'score': (p.get('scores') or {}).get('recommendation'),
            }
            for p in unique_papers
        ],
        'top_papers': top_papers
    }

    # 保存结果
    json_str = json.dumps(output, ensure_ascii=False, indent=2, default=str)
    if args.output == '-' or args.output == '/dev/stdout':
        sys.stdout.write(json_str)
        sys.stdout.write('\n')
    else:
        Path(args.output).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json_str)
        logger.info("Results saved to: %s", args.output)
        # 同时输出到 stdout
        print(json_str)

    logger.info("Top %d papers:", len(top_papers))
    for i, p in enumerate(top_papers, 1):
        hot_marker = " [HOT]" if p.get('is_hot_paper') else ""
        logger.info("  %d. %s... (Score: %s)%s", i, p.get('title', 'N/A')[:60], p['scores']['recommendation'], hot_marker)
    if known_count:
        logger.info("（另有 %d 篇已在知识库中，未计入推荐）", known_count)

    return 0


if __name__ == '__main__':
    sys.exit(main())
