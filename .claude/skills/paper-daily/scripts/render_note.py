#!/usr/bin/env python3
"""Render a paper-daily Markdown note from search results.

The script owns the repeatable layout: topic sections, score ordering, the
full-paper table, top-paper links, and the landing-file table. Human-written
overview and top-paper analysis can be supplied through ``--editorial-json``.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_config import (  # noqa: E402
    load_config,
    month_of,
    paper_dir,
    paper_path,
    rel_link,
    resolve_workspace_path,
    resolve_workspace_subdir,
    sanitize_paper_title,
)


def workspace_dirs(config: dict, workspace: Path):
    """返回 (papers_dir, notes_dir)：01-raw / 03-notes 的实际路径（支持月份目录）。"""
    return (resolve_workspace_subdir(config, 'papers_dir', '01-raw', workspace),
            resolve_workspace_subdir(config, 'notes_dir', '03-notes', workspace))


def daily_note_dir(config: dict, workspace: Path, date: str) -> Path:
    """日报所在目录：<daily_dir>/<日期>/。"""
    return resolve_workspace_subdir(config, 'daily_dir', '08-daily', workspace) / str(date)


def local_pdf(papers_dir: Path, stem: str):
    """按主干定位已归档 PDF（月份目录优先），找不到返回 None。"""
    return paper_path(papers_dir, stem, '.pdf') if stem else None


def local_note(notes_dir: Path, stem: str):
    """按主干定位精读笔记（月份目录优先），找不到返回 None。"""
    if not stem:
        return None
    folder = paper_dir(notes_dir, stem)
    if not folder:
        return None
    note = folder / '精读.md'
    return note if note.is_file() else None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def cell(value) -> str:
    return str(value or '--').replace('|', '\\|').replace('\n', ' ')


def abstract_first_sentence(value: str) -> str:
    """Use a source excerpt when an editorial one-line assessment is absent."""
    text = ' '.join(str(value or '').split())
    return re.split(r'(?<=[.!?])\s+(?=[A-Z\"“(])', text, maxsplit=1)[0] if text else '--'


def score(value) -> str:
    return f'{value:.2f}' if isinstance(value, (int, float)) else '--'


def authors(paper: dict) -> str:
    names = []
    for author in paper.get('authors') or []:
        names.append(author.get('name', '') if isinstance(author, dict) else str(author))
    return ', '.join(name for name in names if name) or '--'


def related_available(item: dict, workspace: Path, config: dict = None) -> bool:
    """相关论文是否可核验：按主干在当前目录结构里能找到笔记/PDF，或记录路径仍存在。

    不直接信任 `search_result.json` 里记录的历史路径——目录结构变更后它会过期，
    因此先按主干做月份感知查找，再回落到记录路径。
    """
    if not item:
        return False
    stem = paper_link_stem(item.get('title') or '')
    papers_dir, notes_dir = workspace_dirs(config, workspace) if config else (
        workspace / '01-raw', workspace / '03-notes')
    if local_note(notes_dir, stem) or local_pdf(papers_dir, stem):
        return True
    path = Path(item.get('path') or '')
    return bool(path.is_file())


def related_title(paper: dict, workspace: Path = None, config: dict = None) -> str:
    for item in paper.get('related_papers') or []:
        if workspace is None or related_available(item, workspace, config):
            return cell(item.get('title'))
    return '--'


def paper_link_stem(title: str) -> str:
    return sanitize_paper_title(title)


def status(paper: dict) -> str:
    if paper.get('already_known'):
        return f"已在库（{paper.get('kb_source') or 'unknown'}）"
    return '推荐'


def subdomain_specs(config: dict, domain: str) -> dict:
    domain_config = (config.get('research_domains') or {}).get(domain) or {}
    return domain_config.get('subdomains') or {}


def subdomain_for(paper: dict, config: dict) -> str:
    explicit = paper.get('matched_subdomain')
    if explicit:
        return explicit
    domain = paper.get('matched_domain') or '未分类'
    keywords = {str(item).lower() for item in (paper.get('matched_keywords') or [])}
    title_summary = f"{paper.get('title', '')} {paper.get('summary', '')}".lower()
    for name, spec in subdomain_specs(config, domain).items():
        for keyword in spec.get('keywords') or []:
            text = str(keyword).lower()
            if text in keywords or text in title_summary:
                return name
    non_category = [
        str(item) for item in (paper.get('matched_keywords') or [])
        if not str(item).startswith(('cs.', 'math.', 'stat.'))
    ]
    return non_category[0] if non_category else '未细分'


def access_links(paper: dict, stem: str = '', workspace: Path = None,
                 config: dict = None, note_dir: Path = None) -> str:
    """访问链接：本地 PDF 使用**相对日报文件**的路径，未归档才给远程原文。"""
    links = []
    if paper.get('pdf_url'):
        remote_pdf = str(paper['pdf_url'])
        if remote_pdf.startswith('/'):
            remote_pdf = f'https://openreview.net{remote_pdf}'
        local = None
        if workspace and config and stem:
            papers_dir, _ = workspace_dirs(config, workspace)
            local = local_pdf(papers_dir, stem)
        if local and note_dir:
            links.append(f'[PDF]({rel_link(local, note_dir)})')
        else:
            links.append(f'[PDF原文]({remote_pdf})')
    if paper.get('url'):
        links.append(f'[原文]({paper["url"]})')
    return ' | '.join(links) or '--'


def table_access_links(paper: dict, stem: str = '', workspace: Path = None,
                       config: dict = None, note_dir: Path = None) -> str:
    return access_links(paper, stem, workspace, config, note_dir).replace(' | ', '<br>')


def render_overview(data: dict, config: dict, editorial: dict) -> None:
    domains = list((config.get('research_domains') or {}).keys())
    overview = editorial.get('overview') or {}
    total = data.get('total_unique', 0)
    sources = data.get('papers_by_source') or {}
    top_count = len(data.get('top_papers') or [])
    known = data.get('total_known', 0)
    print('## 今日概览\n')
    print(
        f"本次检索 {total} 篇唯一论文（"
        f"{' / '.join(f'{name} {count}' for name, count in sources.items())}），"
        f"新增推荐 {top_count} 篇，知识库命中 {known} 篇。\n"
    )
    print('- **方向聚焦**：' + '、'.join(f'**{domain}**' for domain in domains) + '\n')
    print(f"- **总体趋势**：{overview.get('trend') or '待补充'}\n")
    hotspots = overview.get('hotspots') or []
    if hotspots:
        print('- **研究热点**：')
        for hotspot in hotspots:
            if isinstance(hotspot, dict):
                title = hotspot.get('title') or '未命名热点'
                text = hotspot.get('text') or '--'
            else:
                title, text = '热点', str(hotspot)
            print(f'  - **{title}**：{text}')
        print()
    elif overview.get('summary'):
        print(f"- **本日摘要**：{overview['summary']}\n")

    print('### 各研究大方向汇总\n')
    all_papers = data.get('all_papers') or []
    top_titles = {paper.get('title') for paper in data.get('top_papers') or []}
    domain_summaries = editorial.get('domain_summaries') or {}
    for domain in domains:
        group = [paper for paper in all_papers if (paper.get('matched_domain') or '未分类') == domain]
        if not group:
            continue
        recommendations = sum(1 for paper in group if paper.get('title') in top_titles)
        print(f'#### {domain}\n')
        print(f'- **论文概况**：检索 {len(group)} 篇，进入今日推荐 {recommendations} 篇。')
        print(f"- **方向汇总**：{(domain_summaries.get(domain) or {}).get('summary') or '今日结果主要覆盖以下细分领域。'}")
        print('- **具体细分领域**：')
        by_subdomain = {}
        for paper in group:
            by_subdomain.setdefault(subdomain_for(paper, config), []).append(paper)
        provided = (domain_summaries.get(domain) or {}).get('subfields') or {}
        for name, subpapers in sorted(by_subdomain.items(), key=lambda item: (-len(item[1]), item[0])):
            detail = provided.get(name) or '代表论文：' + '；'.join(paper.get('title', '--') for paper in subpapers[:3])
            print(f'  - **{name}**（{len(subpapers)} 篇）：{detail}')
        print()


def render_appendix(data: dict, config: dict, workspace: Path, note_dir: Path) -> None:
    domains = list((config.get('research_domains') or {}).keys())
    papers = data.get('all_papers') or []
    by_domain = {}
    for paper in papers:
        by_domain.setdefault(paper.get('matched_domain') or '未分类', []).append(paper)
    for group in by_domain.values():
        group.sort(key=lambda item: item.get('score') if item.get('score') is not None else -1, reverse=True)

    print('## 附录：本次检索列表\n')
    print('按研究大方向分节；每节内部按研究优先级从高到低。该分数用于候选预筛，不代表全文质量。序号跨主题连续。\n')
    row_number = 1
    for domain in domains + ['未分类']:
        group = by_domain.get(domain, [])
        if not group:
            continue
        print(f'### {domain}（{len(group)} 篇）\n')
        print('| # | 题目 | 来源 | 研究优先级 | 状态 | 研究大方向 | 具体细分领域 | 同脉络 | 访问链接 |')
        print('|---|---|---|---|---|---|---|---|---|')
        for paper in group:
            stem = paper.get('paper_stem') or paper.get('note_filename') or paper_link_stem(paper.get('title'))
            print(
                f"| {row_number} | {cell(paper.get('title'))} | {cell(paper.get('source'))} | "
                f"{score(paper.get('score'))} | {status(paper)} | {cell(paper.get('matched_domain'))} | "
                f"{cell(subdomain_for(paper, config))} | {related_title(paper, workspace, config)} | "
                f"{table_access_links(paper, stem, workspace, config, note_dir)} |"
            )
            row_number += 1
        print()


def related_link(item: dict, workspace: Path, config: dict = None, note_dir: Path = None) -> str:
    """相关论文链接：优先本地精读笔记，其次本地 PDF，最后历史日报。

    所有本地路径都相对于日报文件所在目录（`note_dir`），保证在任意 Markdown
    渲染器中都能解析。
    """
    title = item.get('title') or '--'
    stem = paper_link_stem(title)
    papers_dir, notes_dir = workspace_dirs(config, workspace) if config else (
        workspace / '01-raw', workspace / '03-notes')
    base = note_dir or workspace

    note = local_note(notes_dir, stem)
    if note:
        return f'[{title}]({rel_link(note, base)})'
    pdf = local_pdf(papers_dir, stem)
    if pdf:
        return f'[{title}]({rel_link(pdf, base)})'
    source_path = Path(item.get('path') or '')
    try:
        relative_source = source_path.relative_to(workspace) if source_path.is_file() else None
    except ValueError:
        relative_source = None
    if relative_source:
        if source_path.name == 'search_result.json':
            daily_note = source_path.parent / '今日检索.md'
            if daily_note.is_file():
                return f'[{title}]({rel_link(daily_note, base)})'
        else:
            return f'[{title}]({rel_link(source_path, base)})'
    return ''


def render_top_papers(data: dict, config: dict, editorial: dict, workspace: Path, note_dir: Path) -> None:
    print('## 今日推荐论文\n')
    top = data.get('top_papers') or []
    analyses = editorial.get('top_papers') or {}
    for number, paper in enumerate(top[:3], 1):
        title = paper.get('title') or '--'
        analysis = analyses.get(title) or {}
        stem = paper.get('paper_stem') or paper.get('note_filename') or paper_link_stem(title)
        print(f'### {number}. {title} — 研究优先级 {score((paper.get("scores") or {}).get("recommendation"))}\n')
        print(f"- **来源**：{paper.get('source') or '--'}")
        print(f"- **主题（大方向）**：{cell(paper.get('matched_domain'))}")
        print(f"- **细分领域**：{cell(subdomain_for(paper, config))}")
        print(f"- **题目中文翻译**：{analysis.get('title_zh') or '--'}")
        print(f"- **作者**：{cell(authors(paper))}")
        print(f"- **英文摘要**：{cell(paper.get('summary'))}")
        print(f"- **中文摘要**：{analysis.get('abstract_zh') or '--'}")
        print(f"- **一句话总结**：{analysis.get('summary') or cell(paper.get('summary'))}")
        print('- **摘要声称的核心贡献**：')
        contributions = analysis.get('contributions') or []
        for item in contributions or ['--']:
            print(f'  - {item}')
        print(f"- **方法思路**：{analysis.get('method') or '--'}")
        print(f"- **摘要报告的主要结果**：{analysis.get('result') or '--'}")
        print(f"- **选择理由**：{analysis.get('selection_reason') or '--'}")
        print(f"- **证据边界与待核验**：{analysis.get('evidence_limit') or '当前仅按题目与摘要初评，待全文核验。'}")
        print(f"- **阅读决策**：{analysis.get('reading_decision') or '跟踪'}")
        print('- **相关论文**：')
        rendered = set()
        for item in paper.get('related_papers') or []:
            title = item.get('title') or '--'
            link = related_link(item, workspace, config, note_dir)
            if not link:
                continue
            shared = ', '.join(item.get('shared') or item.get('shared_terms') or []) or '--'
            print(f'  - {link} — 共同点：`{shared}`')
            rendered.add(title)
        for item in analysis.get('related_papers') or []:
            title = item.get('title') or '--'
            url = item.get('url') or ''
            if not item.get('verified') or not url or title in rendered:
                continue
            print(f'  - [{title}]({url}) — 来源：已核验外部链接')
            rendered.add(title)
        if not rendered:
            print('  - 当前工作区没有可核验的相关论文资产；未生成未经核验的文献。')
        print(f"- **访问链接**：{access_links(paper, stem, workspace, config, note_dir)}")
        print()


def render_rest(data: dict, config: dict, editorial: dict, workspace: Path, note_dir: Path) -> None:
    remaining = max(0, len(data.get('top_papers') or []) - 3)
    print(f'## 其余 {remaining} 篇推荐\n')
    print('| # | 题目 | 研究优先级 | 主题 | 细分领域 | 一句话初评或摘要首句 | 访问链接 |')
    print('|---|---|---|---|---|---|---|')
    analyses = editorial.get('top_papers') or {}
    for number, paper in enumerate((data.get('top_papers') or [])[3:], 4):
        stem = paper.get('paper_stem') or paper.get('note_filename') or paper_link_stem(paper.get('title'))
        analysis = analyses.get(paper.get('title') or '') or {}
        initial = analysis.get('summary') or '摘要首句（原文）：' + abstract_first_sentence(paper.get('summary'))
        print(
            f"| {number} | {cell(paper.get('title'))} | "
            f"{score((paper.get('scores') or {}).get('recommendation'))} | "
            f"{cell(paper.get('matched_domain'))} | {cell(subdomain_for(paper, config))} | "
            f"{cell(initial)} | "
            f"{table_access_links(paper, stem, workspace, config, note_dir)} |"
        )
    print()


def render_files(data: dict, date: str, workspace: Path, config: dict, note_dir: Path) -> None:
    print('## 今日落盘\n')
    print('| 类型 | 路径 | 状态 |')
    print('|---|---|---|')
    papers_dir, _ = workspace_dirs(config, workspace)
    for paper in (data.get('top_papers') or [])[:3]:
        stem = paper.get('paper_stem') or paper.get('note_filename') or paper_link_stem(paper.get('title'))
        path = local_pdf(papers_dir, stem)
        if path:
            state = '新下载' if paper.get('archive_status') == 'downloaded' else '已存在跳过'
            shown = rel_link(path, note_dir)
        elif paper.get('archive_status') == 'failed':
            state = f"下载失败：{paper.get('archive_detail') or '原因未记录'}"
            shown = paper.get('pdf_url') or '--'
        else:
            state = '待归档'
            month = month_of(date)
            shown = rel_link(papers_dir / month / f'{stem}.pdf', note_dir)
        print(f'| PDF | {shown} | {state} |')
    daily_dir = daily_note_dir(config, workspace, date)
    print(f'| 检索结果 | {rel_link(daily_dir / "search_result.json", note_dir)} | 本次生成 |')
    print(f'| 日报 | {rel_link(daily_dir / "今日检索.md", note_dir)} | 本次生成 |')
    print(f'| 索引 | {rel_link(daily_dir / "_index.json", note_dir)} | 随日报生成 |')
    print(f'| 编辑稿 | {rel_link(daily_dir / "daily-editorial.json", note_dir)} | 可选保留，便于重渲染 |')


def main() -> int:
    parser = argparse.ArgumentParser(description='Render a paper-daily Markdown note')
    parser.add_argument('--papers-json', required=True, help='search_arxiv.py output JSON')
    parser.add_argument('--date', required=True, help='Target date, YYYY-MM-DD')
    parser.add_argument('--config', default=None, help='Path to shared config.yaml')
    parser.add_argument('--workspace', default=None, help='Paper workspace root')
    parser.add_argument('--daily-dir', default=None,
                        help='Override the daily directory (default: config daily_dir)')
    parser.add_argument('--editorial-json', default=None, help='Optional human-written overview/top-paper JSON')
    args = parser.parse_args()

    config = load_config(args.config)
    workspace = resolve_workspace_path(config, args.workspace)
    data = load_json(Path(args.papers_json))
    editorial = load_json(Path(args.editorial_json)) if args.editorial_json else {}
    if args.daily_dir:
        base = Path(args.daily_dir).expanduser()
        note_dir = (base if base.is_absolute() else workspace / base) / args.date
    else:
        note_dir = daily_note_dir(config, workspace, args.date)
    render_overview(data, config, editorial)
    render_top_papers(data, config, editorial, workspace, note_dir)
    render_rest(data, config, editorial, workspace, note_dir)
    render_files(data, args.date, workspace, config, note_dir)
    render_appendix(data, config, workspace, note_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
