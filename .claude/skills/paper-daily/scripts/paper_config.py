#!/usr/bin/env python3
"""
paper_config —— paper-daily / paper-conf / paper-interests 共享的配置解析模块

配置查找顺序（第一个存在的即为生效配置）：
  1. 显式传入的 config_path（命令行 --config）
  2. 环境变量 PAPER_SKILLS_CONFIG
  3. <skill 目录>/config.yaml          （skill 自带的默认配置）
  4. <skills 根目录>/config.yaml （共享配置）

workspace 路径解析顺序：
  1. 显式传入的 workspace（命令行 --workspace）
  2. 环境变量 PAPER_WORKSPACE_PATH
  3. 配置中的 workspace_path
  4. 自动识别当前项目根目录
"""

import json
import os
import re
import subprocess
from datetime import date as _date, datetime as _datetime
from pathlib import Path
from typing import Optional

SKILL_ROOT = Path(__file__).resolve().parents[1]          # 如 .../.claude/skills/paper-daily
PAPER_SKILLS_ROOT = SKILL_ROOT.parent                      # 如 .../.claude/skills
WORKSPACE_MARKERS = ('01-raw', '02-markdown', '03-notes', '04-equation_problem')

# 月份目录（见工作区 .AGENT.md）：严格 YYYY-MM
MONTH_RE = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')


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


# ---------------------------------------------------------------------------
# 月份目录（YYYY-MM）与文件相对链接
# ---------------------------------------------------------------------------

def is_month_dir(name) -> bool:
    """判断是否为合法的月份目录名（严格 YYYY-MM）。"""
    return bool(MONTH_RE.match(str(name).strip()))


def month_of(value) -> str:
    """把日期/日期字符串归一为 `YYYY-MM`；无法识别时返回空串。

    支持 date/datetime、`2026-09-17`、`2026/9/1`、`20260917`、`2026-09`。
    """
    if value is None:
        return ''
    if isinstance(value, (_date, _datetime)):
        return value.strftime('%Y-%m')
    text = str(value).strip()
    match = re.match(r'^(\d{4})[-/.](\d{1,2})', text)
    if match:
        month = int(match.group(2))
        return '%04d-%02d' % (int(match.group(1)), month) if 1 <= month <= 12 else ''
    match = re.match(r'^(\d{4})(\d{2})', text)
    if match:
        month = int(match.group(2))
        return '%s-%s' % (match.group(1), match.group(2)) if 1 <= month <= 12 else ''
    return ''


def iter_month_dirs(root, descending: bool = False) -> list:
    """列出 root 下的月份目录（只认 YYYY-MM），默认按月份升序。"""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and is_month_dir(p.name)),
        key=lambda p: p.name,
        reverse=descending,
    )


def _normalize_suffix(suffix: str) -> str:
    text = str(suffix or '').strip()
    if not text:
        return ''
    return text if text.startswith('.') else '.' + text


def _fuzzy_candidates(root: Path, stem: str, suffix: str, want_dir: bool):
    """按规范化主干（忽略大小写、连字符、下划线、空格差异）在月份目录与根部查找。

    用于容错：论文标题里的 `-` 会被 `sanitize_paper_title()` 转成 `_`
    （例如 `Meta-Black-Box` vs `Meta_Black_Box`），精确匹配会漏。
    """
    key = clean_stem(stem)
    if not key:
        return None
    for month_dir in iter_month_dirs(root, descending=True):
        for child in sorted(month_dir.iterdir()):
            if want_dir and child.is_dir() and clean_stem(child.name) == key:
                return child
            if not want_dir and child.is_file() and child.name.lower().endswith(suffix.lower()) \
                    and clean_stem(child.name[: -len(suffix)] if suffix else child.name) == key:
                return child
    for child in sorted(root.iterdir()):
        if is_month_dir(child.name) or child.name.startswith('.'):
            continue
        if want_dir and child.is_dir() and clean_stem(child.name) == key:
            return child
        if not want_dir and child.is_file() and child.name.lower().endswith(suffix.lower()) \
                and clean_stem(child.name[: -len(suffix)] if suffix else child.name) == key:
            return child
    return None


def paper_path(root, stem: str, suffix: str = ''):
    """定位某论文在 root 下的真实文件：月份目录优先，根部平铺回落，最后按主干模糊匹配。"""
    stem = str(stem or '').strip()
    if not stem:
        return None
    suffix = _normalize_suffix(suffix)
    root = Path(root)
    for month_dir in iter_month_dirs(root, descending=True):
        candidate = month_dir / (stem + suffix)
        if candidate.is_file():
            return candidate
    flat = root / (stem + suffix)
    if flat.is_file():
        return flat
    return _fuzzy_candidates(root, stem, suffix, want_dir=False) if root.is_dir() else None


def paper_dir(root, stem: str):
    """定位某论文在 root 下的真实目录：月份目录优先，根部平铺回落，最后按主干模糊匹配。"""
    stem = str(stem or '').strip()
    if not stem:
        return None
    root = Path(root)
    for month_dir in iter_month_dirs(root, descending=True):
        candidate = month_dir / stem
        if candidate.is_dir():
            return candidate
    flat = root / stem
    if flat.is_dir():
        return flat
    return _fuzzy_candidates(root, stem, '', want_dir=True) if root.is_dir() else None


def month_paper_path(root, month: str, stem: str, suffix: str = '') -> Path:
    """构造 `<root>/<YYYY-MM>/<stem><suffix>`（不检查是否存在）。"""
    return Path(root) / str(month).strip() / (str(stem).strip() + _normalize_suffix(suffix))


def month_paper_dir(root, month: str, stem: str) -> Path:
    """构造 `<root>/<YYYY-MM>/<stem>/`（不检查是否存在）。"""
    return Path(root) / str(month).strip() / str(stem).strip()


def scan_paper_entries(root, kind: str = 'file', suffix: str = '') -> list:
    """递归列出 root 下的论文条目（月份目录 + 根部平铺）。

    kind='file'  匹配 `<root>/YYYY-MM/<stem><suffix>` 与 `<root>/<stem><suffix>`
    kind='dir'   匹配 `<root>/YYYY-MM/<stem>/`        与 `<root>/<stem>/`

    返回 `[{'month': 'YYYY-MM'|'', 'stem': str, 'path': Path}]`，按月份与名称排序。
    """
    root = Path(root)
    entries = []
    if not root.is_dir():
        return entries
    suffix = _normalize_suffix(suffix)
    groups = [(d.name, d) for d in iter_month_dirs(root)] + [('', root)]
    for month, directory in groups:
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            if child.name.startswith('.'):
                continue
            if kind == 'dir':
                if not child.is_dir() or is_month_dir(child.name):
                    continue
                stem = child.name
            else:
                if not child.is_file() or not child.name.lower().endswith(suffix.lower()):
                    continue
                stem = child.name[: -len(suffix)] if suffix else child.stem
            if stem.lower() in ('readme', 'index'):
                continue
            entries.append({'month': month, 'stem': stem, 'path': child})
    return entries


def rel_link(target, origin_dir) -> str:
    """生成从 origin_dir 指向 target 的标准 Markdown 相对路径（正斜杠）。"""
    target = Path(target)
    origin_dir = Path(origin_dir)
    try:
        rel = os.path.relpath(target, origin_dir)
    except ValueError:
        return target.as_posix()
    return rel.replace('\\', '/')


# ---------------------------------------------------------------------------
# 索引表格读写：统一论文登记表（01-raw/index.md）与月份 README
# ---------------------------------------------------------------------------

def parse_markdown_table(text: str, required=()) -> list:
    """解析文本中第一张包含全部 required 表头列的 Markdown 表格，返回 [dict]。"""
    lines = str(text or '').splitlines()
    required = [str(item) for item in required]
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith('|') or set(stripped) <= set('|-: '):
            continue
        headers = [cell.strip() for cell in stripped.strip('|').split('|')]
        if not all(any(req in header for header in headers) for req in required):
            continue
        rows = []
        for row_line in lines[index + 1:]:
            row_text = row_line.strip()
            if not row_text.startswith('|'):
                break
            if set(row_text) <= set('|-: '):
                continue
            cells = [cell.strip() for cell in row_text.strip('|').split('|')]
            if len(cells) < len(headers):
                cells += [''] * (len(headers) - len(cells))
            rows.append(dict(zip(headers, cells)))
        return rows
    return []


def table_cell(row: dict, *names) -> str:
    """从表格行里按表头关键字取单元格。"""
    for name in names:
        for key, value in (row or {}).items():
            if name in key:
                return (value or '').strip()
    return ''


def stem_from_cell(value: str) -> str:
    """从单元格取论文主干：支持 Markdown 链接、纯文件名与纯主干。"""
    text = str(value or '').strip()
    match = re.search(r'\[[^\]]*\]\(([^)]+)\)', text)
    if match:
        text = match.group(1)
    text = text.rsplit('/', 1)[-1].strip()
    for ext in ('.markdown', '.md', '.pdf'):
        if text.lower().endswith(ext):
            text = text[: -len(ext)]
            break
    return text.strip()


def git_first_commit_date(path, workspace) -> str:
    """该文件的首次提交日期（YYYY-MM-DD）；不是 git 仓库或查询失败时返回空串。"""
    try:
        proc = subprocess.run(
            ['git', '-C', str(workspace), '-c', 'core.quotePath=false', 'log',
             '--diff-filter=A', '--format=%ad', '--date=short', '--', str(path)],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ''
    if proc.returncode != 0:
        return ''
    dates = [line.strip() for line in proc.stdout.decode('utf-8', 'replace').splitlines() if line.strip()]
    return dates[-1] if dates else ''   # git log 新到旧 -> 最后一行是最早的新增日期


def read_archive_dates(workspace, config: dict) -> dict:
    """读取各论文的入库信息：`{stem: {'month', 'date', 'title'}}`。

    来源优先级：`<papers_dir>/<YYYY-MM>/README.md` 表格 → `<papers_dir>/index.md` 表格
    → git 首次提交日期。月份目录 README 是入库日期的权威记录。
    """
    workspace = Path(workspace)
    papers_dir = resolve_workspace_subdir(config, 'papers_dir', '01-raw', workspace)
    result = {}

    def remember(stem: str, month: str, when: str, title: str):
        stem = str(stem or '').strip()
        if not stem:
            return
        item = result.setdefault(stem, {'month': '', 'date': '', 'title': ''})
        if month and not item['month']:
            item['month'] = month
        if when and not item['date']:
            item['date'] = when
        if title and len(title) > len(item['title']):
            item['title'] = title

    month_readmes = [d / 'README.md' for d in iter_month_dirs(papers_dir, descending=True)]
    month_readmes.append(papers_dir / 'README.md')
    if (papers_dir / 'index.md').is_file():
        month_readmes.append(papers_dir / 'index.md')

    for readme in month_readmes:
        if not readme.is_file():
            continue
        try:
            text = readme.read_text(encoding='utf-8-sig')
        except OSError:
            continue
        rows = parse_markdown_table(text, required=['入库日期'])
        default_month = readme.parent.name if is_month_dir(readme.parent.name) else ''
        for row in rows:
            stem = table_cell(row, 'paper_stem', '主干') or stem_from_cell(table_cell(row, 'PDF'))
            if not stem:
                continue
            when = table_cell(row, '入库日期', '落盘日期')
            month = month_of(when) or default_month
            remember(stem, month, when, table_cell(row, '论文标题', '标题'))

    # 兼容旧版平铺索引：`- <论文标题> —— YYYY-MM-DD`
    # 可读标题里的 `ℓ_p`、`3D` 等字符归一化后可能与文件名主干不一致，
    # 因此先用归一化键精确匹配，再用相似度兜底（如 `ℓ_p-Norm` vs `lp_Norm`）。
    legacy = papers_dir / 'index.md'
    if legacy.is_file():
        try:
            legacy_text = legacy.read_text(encoding='utf-8-sig')
        except OSError:
            legacy_text = ''
        legacy_items = []
        for line in legacy_text.splitlines():
            match = re.match(r'^[-*]\s+(.*?)\s*[—–-]{2,}\s*(\d{4}-\d{2}-\d{2})', line.strip())
            if match:
                legacy_items.append((match.group(1).strip(), match.group(2)))
        if legacy_items:
            from difflib import SequenceMatcher

            candidates = [(clean_stem(e['stem']), e['stem'])
                          for e in scan_paper_entries(papers_dir, kind='file', suffix='.pdf')]
            used = set()
            for title, when in legacy_items:
                key = clean_stem(title)
                exact = next((stem for norm, stem in candidates if norm == key and stem not in used), None)
                best, score = None, 0.0
                if not exact:
                    for norm, stem in candidates:
                        if stem in used:
                            continue
                        ratio = SequenceMatcher(None, key, norm).ratio()
                        if ratio > score:
                            best, score = stem, ratio
                    exact = best if score >= 0.85 else None
                if exact:
                    used.add(exact)
                    remember(exact, month_of(when), when, title)

    # 兜底：用 git 首次提交日期补齐缺失的日期
    for entry in scan_paper_entries(papers_dir, kind='file', suffix='.pdf'):
        stem = entry['stem']
        item = result.setdefault(stem, {'month': entry['month'], 'date': '', 'title': ''})
        if not item['month']:
            item['month'] = entry['month']
        if not item['date']:
            item['date'] = git_first_commit_date(entry['path'], workspace)
            if not item['month']:
                item['month'] = month_of(item['date'])
    return result


def scan_known_papers(workspace: Path, config: dict, log=None):
    """扫描知识库里已经存在的论文，供推荐去重与日报标记使用。

    来源（见工作区 .AGENT.md）：
      1. `papers_dir`（默认 01-raw）：已下载 PDF 的主干（月份目录 `YYYY-MM/` 与根部平铺都支持）
      2. `notes_dir`（默认 03-notes）：已精读归档的论文文件夹名（月份目录下同样递归）
      3. `daily_dir`（默认 08-daily）历史日期文件夹的检索结果：之前推荐过的论文

    Returns:
        {'stems': set[str], 'by_pdf': {...}, 'by_note': {...}, 'by_daily': {...},
         'months': {stem: 'YYYY-MM'},
         'reference': {stem: {'title': str, 'source': str, 'path': str}}, 'dirs': {...}}
    """
    result = {'stems': set(), 'by_pdf': {}, 'by_note': {}, 'by_daily': {},
              'months': {}, 'reference': {}, 'dirs': {}}

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

    for label, directory, entries, source in (
        ('by_pdf', papers_dir, scan_paper_entries(papers_dir, kind='file', suffix='.pdf'), 'pdf'),
        ('by_note', notes_dir, scan_paper_entries(notes_dir, kind='dir'), 'note'),
    ):
        if not directory.is_dir():
            if log:
                log("%s 不存在，跳过：%s", label, directory)
            continue
        for entry in entries:
            stem = clean_stem(entry['stem'])
            if not stem:
                continue
            result[label][stem] = str(entry['path'])
            result['stems'].add(stem)
            if entry.get('month'):
                result['months'][stem] = entry['month']
            # 文件夹/文件名去掉扩展名就是可读标题（下划线换空格更好读）
            remember(stem, entry['stem'].replace('_', ' ').strip(), source, str(entry['path']))

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
