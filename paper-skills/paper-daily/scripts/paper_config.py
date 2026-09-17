#!/usr/bin/env python3
"""
paper_config —— paper-daily / paper-conf / paper-interests 共享的配置解析模块

配置查找顺序（第一个存在的即为生效配置）：
  1. 显式传入的 config_path（命令行 --config）
  2. 环境变量 PAPER_SKILLS_CONFIG
  3. <skill 目录>/config.yaml          （skill 自带的默认配置）
  4. <paper-skills 根目录>/config.yaml （共享配置）

workspace 路径解析顺序：
  1. 显式传入的 workspace（命令行 --workspace）
  2. 环境变量 PAPER_WORKSPACE_PATH
  3. 配置中的 workspace_path
  4. 自动识别当前项目根目录
"""

import json
import os
from pathlib import Path
from typing import Optional

SKILL_ROOT = Path(__file__).resolve().parents[1]          # 如 .../paper-skills/paper-daily
PAPER_SKILLS_ROOT = SKILL_ROOT.parent                      # 如 .../paper-skills
WORKSPACE_MARKERS = ('01-raw', '02-markdown', '03-notes', '04-equation_problem')


def candidate_config_paths(config_path: Optional[str] = None) -> list:
    """按优先级返回候选配置文件路径。"""
    candidates = []
    if config_path:
        candidates.append(Path(config_path).expanduser())
    env_config = os.environ.get('PAPER_SKILLS_CONFIG')
    if env_config:
        candidates.append(Path(env_config).expanduser())
    candidates.append(SKILL_ROOT / 'config.yaml')
    candidates.append(PAPER_SKILLS_ROOT / 'config.yaml')
    return candidates


def find_config_path(config_path: Optional[str] = None) -> Path:
    """返回第一个存在的配置文件路径；都不存在时抛 FileNotFoundError。"""
    candidates = candidate_config_paths(config_path)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "未找到配置文件，已尝试：%s" % ', '.join(str(p) for p in candidates)
    )


def load_config(config_path: Optional[str] = None) -> dict:
    """加载并返回配置字典。"""
    import yaml

    path = find_config_path(config_path)
    with open(path, 'r', encoding='utf-8-sig') as f:
        return yaml.safe_load(f) or {}


def resolve_workspace_path(config: dict, workspace: Optional[str] = None) -> Path:
    """解析 workspace 路径：参数 > 环境变量 > 配置 > 当前项目根目录。"""
    raw = workspace or os.environ.get('PAPER_WORKSPACE_PATH') or config.get('workspace_path') or ''
    if raw and str(raw).strip():
        path = Path(str(raw)).expanduser()
        return (path if path.is_absolute() else Path.cwd() / path).resolve()

    candidates = [PAPER_SKILLS_ROOT.parent]
    here = Path.cwd().resolve()
    candidates.extend([here, *here.parents])
    for candidate in candidates:
        if sum((candidate / marker).is_dir() for marker in WORKSPACE_MARKERS) >= 2:
            return candidate
    return here


def resolve_path(config: dict, key: str, default=None, base: Optional[Path] = None, workspace: Optional[Path] = None) -> Path:
    """解析配置里的路径项。

    以 `@workspace/` 开头 -> 相对 workspace 根目录；否则相对 base（默认 skill 目录）；
    绝对路径直接使用。`@workspace/` 是 paperread 工作区用法的推荐写法。
    """
    raw = config.get(key)
    if raw is None or str(raw).strip() == '':
        raw = default
    if raw is None:
        raise ValueError("配置项 %s 未设置，且没有默认值" % key)

    text = str(raw).strip()
    if text.startswith('@workspace/'):
        if workspace is None:
            raise ValueError("配置项 %s 使用了 @workspace/ 前缀，但未解析出 workspace 路径" % key)
        return (workspace / text[len('@workspace/'):]).resolve()
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return ((base or SKILL_ROOT) / path).resolve()


def sanitize_paper_title(title: str, max_length: int = 120) -> str:
    """把论文标题转换为稳定主干 `<论文标题>`。

    规则对齐 paper-reading 工作区现有论文的命名风格（见 01-raw 里的 7 个 PDF）：
    路径非法字符与逗号分隔符换成 `_`，**空白一律换成下划线**（不用空格），
    连字符 `-` 保留（如 `lp-norm`），压缩连续下划线，去掉首尾的点与下划线。

    例：
        "Understanding MARS: When Scaling Momentum Correction Provably Helps"
        -> "Understanding_MARS_When_Scaling_Momentum_Correction_Provably_Helps"
        "Rethinking 3D Convolution in lp-norm Space"
        -> "Rethinking_3D_Convolution_in_lp_norm_Space"
    """
    import re

    text = str(title or '').strip()
    # 非法字符与逗号/顿号/分号 -> 下划线
    text = re.sub(r'[\\/:*?"<>|\t\n\r]+', '_', text)
    text = re.sub(r'[,\u3001;；]+', '_', text)
    # 空白与 ASCII 连字符 -> 下划线（与工作区现有命名风格一致：
    # "lp-norm" 存为 "lp_norm"、"Memory-Efficient" 存为 "Memory_Efficient"）
    text = re.sub(r'[\s\-]+', '_', text)
    # 压缩连续下划线，去掉首尾杂点
    text = re.sub(r'_{2,}', '_', text).strip(' ._')
    if len(text) > max_length:
        text = text[:max_length].rstrip(' ._')
    return text


def resolve_workspace_subdir(config: dict, key: str, default: str, workspace: Path) -> Path:
    """解析 workspace 内的子目录（如 01-raw / 03-notes / 08-daily）。

    `key` 与 `default` 分开是为了让调用方传入语义化的键名，
    但目录名本身始终来自 **配置**（默认值只是该目录的标准名）。
    """
    raw = (config or {}).get(key) or default
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    return workspace / path


def resolve_output_dir(config: dict, date: str, daily_dir: Optional[str] = None) -> Path:
    """解析本次输出目录：<daily_dir>/<YYYY-MM-DD>/

    相对路径按 skill 目录解析，绝对路径直接使用。
    """
    raw = daily_dir or config.get('daily_dir') or 'daily'
    base = Path(str(raw)).expanduser()
    if not base.is_absolute():
        base = SKILL_ROOT / base
    return base / date


def load_research_domains(config: dict) -> dict:
    """返回 research_domains 字典（缺失时返回空字典）。"""
    return config.get('research_domains') or {}


def clean_stem(path) -> str:
    """从文件名/文件夹名得到规范化主干，用于跨目录匹配同一篇论文。"""
    import re

    stem = str(path)
    stem = stem.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
    for ext in ('.markdown', '.md', '.pdf'):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    stem = re.sub(r'[^a-z0-9\u4e00-\u9fff]+', '', stem.lower())
    return stem


def scan_known_papers(workspace: Path, config: dict, log=None):
    """扫描知识库里已经存在的论文，供推荐去重与日报标记使用。

    来源（见 paperread 工作区 AGENT.md）：
      1. `papers_dir`（默认 01-raw）：已下载 PDF 的主干
      2. `notes_dir`（默认 03-notes）：已精读归档的论文文件夹名
      3. `daily_dir`（默认 08-daily）历史日期文件夹的检索结果：之前推荐过的论文

    Returns:
        {'stems': set[str], 'by_pdf': {...}, 'by_note': {...}, 'by_daily': {...},
         'reference': {stem: {'title': str, 'source': str, 'path': str}}, 'dirs': {...}}
    """
    result = {'stems': set(), 'by_pdf': {}, 'by_note': {}, 'by_daily': {},
              'reference': {}, 'dirs': {}}

    def remember(stem: str, display: str, source: str, path: str):
        """记录一篇已有论文的可读信息，供日报做"已有文献脉络"关联。"""
        item = result['reference'].setdefault(stem, {'title': '', 'source': '', 'path': ''})
        # 真正的标题比文件名主干更可读，优先保留
        if display and (not item['title'] or len(display) > len(item['title'])):
            item['title'] = display
        if source and not item['source']:
            item['source'] = source
        if path and not item['path']:
            item['path'] = path

    papers_dir = resolve_workspace_subdir(config, 'papers_dir', '01-raw', workspace)
    notes_dir = resolve_workspace_subdir(config, 'notes_dir', '03-notes', workspace)
    daily_dir = resolve_workspace_subdir(config, 'daily_dir', '08-daily', workspace)
    result['dirs'] = {'papers': papers_dir, 'notes': notes_dir, 'daily': daily_dir}

    for label, directory, pattern, source in (
        ('by_pdf', papers_dir, '*.pdf', 'pdf'),
        ('by_note', notes_dir, None, 'note'),
    ):
        if not directory.is_dir():
            if log:
                log("%s 不存在，跳过：%s", label, directory)
            continue
        if pattern:
            items = sorted(p for p in directory.glob(pattern) if p.is_file())
        else:
            items = sorted(p for p in directory.iterdir() if p.is_dir())
        for item in items:
            stem = clean_stem(item.name)
            if not stem:
                continue
            result[label][stem] = str(item)
            result['stems'].add(stem)
            # 文件夹/文件名去掉扩展名就是可读标题（下划线换空格更好读）
            display = item.stem if item.is_file() else item.name
            remember(stem, display.replace('_', ' ').strip(), source, str(item))

    # 3) 历史检索结果里的推荐论文
    if daily_dir.is_dir():
        result_name = config.get('daily_result_name') or 'search_result.json'
        for result_file in sorted(daily_dir.glob('*/%s' % result_name)):
            try:
                data = json.loads(result_file.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError):
                continue
            for paper in (data.get('top_papers') or []):
                title = paper.get('title', '')
                for candidate in (paper.get('paper_stem') or '', title):
                    stem = clean_stem(candidate)
                    if stem:
                        result['by_daily'].setdefault(stem, str(result_file))
                        result['stems'].add(stem)
                        remember(stem, title, 'daily', str(result_file))
        if log and result['by_daily']:
            log("daily history: %d previously recommended papers", len(result['by_daily']))

    return result


def collect_keywords(config: dict) -> list:
    """汇总所有研究域的关键词，去重且保持顺序。"""
    keywords = []
    seen = set()
    for domain in load_research_domains(config).values():
        for kw in (domain or {}).get('keywords', []) or []:
            kw_clean = str(kw).strip()
            key = kw_clean.lower()
            if kw_clean and key not in seen:
                seen.add(key)
                keywords.append(kw_clean)
    return keywords
