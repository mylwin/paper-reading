#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sync_indexes —— 论文工作区索引与状态维护

职责（规范见工作区 .AGENT.md）：

1. 以 `01-raw` 为全集构建**统一论文登记表**：
   paper_stem / archive_date / archive_month / source / raw_path / markdown_path /
   notes_path / translation_path / parse_status / reading_status / translation_status
2. 生成四份根索引：`01-raw/index.md`、`02-markdown/index.md`、`03-notes/index.md`、
   `06-translation/index.md`。
3. 生成各阶段 `YYYY-MM/README.md` 的 `<!-- INDEX:BEGIN -->` 表格区（表格外内容保留；
   文件不存在时写入完整模板）。
4. `--check-links`：校验所有文件相对链接、月份一致性、月份 README 完整性。

人工可编辑的单元格（`来源` / `失败原因` / `优先级` / `精读中` / `翻译中`）在刷新时
按 paper_stem 从上一版表格中保留，不会被脚本清空。

用法：
    python sync_indexes.py                      # 刷新索引与月份 README 表格
    python sync_indexes.py --dry-run --json     # 只打印登记表，不写文件
    python sync_indexes.py --check-links        # 只校验链接与一致性（退出码 0/1）
    python sync_indexes.py --check-links --json # 校验结果以 JSON 输出
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_config import (  # noqa: E402
    clean_stem,
    is_month_dir,
    iter_month_dirs,
    load_config,
    month_of,
    parse_markdown_table,
    read_archive_dates,
    rel_link,
    resolve_workspace_path,
    resolve_workspace_subdir,
    scan_paper_entries,
    stem_from_cell,
    table_cell,
)

INDEX_BEGIN = '<!-- INDEX:BEGIN -->'
INDEX_END = '<!-- INDEX:END -->'

PAPER_LABEL = '论文标题'
STEM_LABEL = 'paper_stem'

STATUS_PARSED = ('未解析', '已解析', '解析失败')
STATUS_READING = ('未精读', '精读中', '已精读')
STATUS_TRANSLATION = ('未翻译', '翻译中', '已翻译')


def _force_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


def _write_if_changed(path: Path, content: str, dry_run: bool = False) -> bool:
    """内容有变化才写盘，保证幂等（重复运行不产生 diff）。"""
    path = Path(path)
    old = path.read_text(encoding='utf-8-sig') if path.is_file() else None
    if old == content:
        return False
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    return True


def replace_marked_block(text: str, block: str, begin: str = INDEX_BEGIN, end: str = INDEX_END):
    """替换标记内的内容；没有标记时返回 (None, False) 表示调用方需要写完整模板。"""
    if begin not in text or end not in text:
        return None, False
    head, _, rest = text.partition(begin)
    _, _, tail = rest.partition(end)
    return '%s%s\n%s%s' % (head, begin, block.rstrip('\n') + '\n', end + tail), True


def display_title(stem: str) -> str:
    return str(stem).replace('_', ' ').strip()


# ---------------------------------------------------------------------------
# 上一版表格：保留人工可编辑单元格
# ---------------------------------------------------------------------------

def previous_rows(path: Path, required=()) -> dict:
    """读取旧表格并按规范化主干索引，用于保留人工填写的单元格。"""
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding='utf-8-sig')
    except OSError:
        return {}
    rows = parse_markdown_table(text, required=required)
    result = {}
    for row in rows:
        stem = table_cell(row, STEM_LABEL, '主干') or stem_from_cell(
            table_cell(row, 'PDF') or table_cell(row, '解析文件') or table_cell(row, '精读文件') or table_cell(row, '目录')
        )
        key = clean_stem(stem)
        if key and key not in result:
            result[key] = row
    return result


# ---------------------------------------------------------------------------
# 登记表
# ---------------------------------------------------------------------------

def build_registry(workspace: Path, config: dict, overrides: dict = None) -> list:
    """构建统一论文登记表（按月份、标题排序）。

    `overrides`：`{paper_stem: {'date','month','title','source'}}`，用于刚下载、
    尚未写入月份 README 的论文（见 fetch_pdfs.py）。
    """
    papers_dir = resolve_workspace_subdir(config, 'papers_dir', '01-raw', workspace)
    markdown_dir = resolve_workspace_subdir(config, 'markdown_dir', '02-markdown', workspace)
    notes_dir = resolve_workspace_subdir(config, 'notes_dir', '03-notes', workspace)
    translation_dir = resolve_workspace_subdir(config, 'translation_dir', '06-translation', workspace)

    archive = read_archive_dates(workspace, config)
    for stem, meta in (overrides or {}).items():
        item = archive.setdefault(stem, {'month': '', 'date': '', 'title': '', 'source': ''})
        for key in ('month', 'date', 'title', 'source'):
            if meta.get(key) and not item.get(key):
                item[key] = meta[key]

    prev_raw = previous_rows(papers_dir / 'index.md', required=[PAPER_LABEL])
    prev_md = previous_rows(markdown_dir / 'index.md', required=[PAPER_LABEL])
    prev_notes = previous_rows(notes_dir / 'index.md', required=[PAPER_LABEL])
    prev_translation = previous_rows(translation_dir / 'index.md', required=[PAPER_LABEL])

    entries = {}   # norm -> item

    def slot(stem: str):
        key = clean_stem(stem)
        if not key:
            return None
        item = entries.setdefault(key, {
            'paper_stem': stem, 'title': '', 'archive_date': '', 'archive_month': '',
            'source': '', 'raw_path': None, 'markdown_path': None,
            'notes_path': None, 'translation_path': None,
            'parse_status': '', 'reading_status': '', 'translation_status': '',
            'reading_start': '', 'reading_finish': '', 'priority': '',
            'translation_forms': [], 'parse_error': '', 'months_seen': set(),
        })
        # 文件名主干优先取真实文件名（比规范化键可读）
        if stem and len(str(stem)) > len(str(item['paper_stem'])):
            item['paper_stem'] = stem
        return item

    for entry in scan_paper_entries(papers_dir, kind='file', suffix='.pdf'):
        item = slot(entry['stem'])
        if not item:
            continue
        item['raw_path'] = entry['path']
        if entry['month']:
            item['months_seen'].add(entry['month'])
    for entry in scan_paper_entries(markdown_dir, kind='file', suffix='.md'):
        item = slot(entry['stem'])
        if not item:
            continue
        item['markdown_path'] = entry['path']
        if entry['month']:
            item['months_seen'].add(entry['month'])
    for entry in scan_paper_entries(notes_dir, kind='dir'):
        item = slot(entry['stem'])
        if not item:
            continue
        item['notes_path'] = entry['path']
        if entry['month']:
            item['months_seen'].add(entry['month'])
    for entry in scan_paper_entries(translation_dir, kind='dir'):
        item = slot(entry['stem'])
        if not item:
            continue
        item['translation_path'] = entry['path']
        if entry['month']:
            item['months_seen'].add(entry['month'])

    # 已知但四个阶段都没有文件的论文（例如解析失败后临时删除）也登记进来看不出来，跳过
    for stem, meta in archive.items():
        item = slot(stem)
        if not item:
            continue
        if meta.get('month') and not item['months_seen']:
            item['months_seen'].add(meta['month'])

    result = []
    for key in sorted(entries):
        item = entries[key]
        meta = archive.get(item['paper_stem']) or {}
        if not meta:
            for stem, value in archive.items():
                if clean_stem(stem) == key:
                    meta = value
                    break

        # 月份：入库日期 > 文件名/目录所在月份 > 目录结构
        month = month_of(meta.get('date') or '') or meta.get('month') or ''
        if not month and item['months_seen']:
            month = sorted(item['months_seen'])[0]
        item['archive_month'] = month
        item['archive_date'] = meta.get('date') or ''
        item['title'] = (meta.get('title') or '').strip() or display_title(item['paper_stem'])

        # 来源：保留上一版人工填写值，其次用调用方传入的本次来源
        prev_source = table_cell(prev_raw.get(key, {}), '来源', 'source')
        if prev_source in ('', '--'):
            prev_source = (meta.get('source') or '').strip()
        item['source'] = prev_source if prev_source and prev_source != '--' else '--'
        if item['source'] in ('', '--'):
            for readme in ([papers_dir / month / 'README.md'] if month else []):
                rows = previous_rows(readme, required=['入库日期'])
                item['source'] = table_cell(rows.get(key, {}), '来源', 'source') or '--'
                break

        # 解析状态：磁盘为准，人工填写的「解析失败」保留
        marker = item['markdown_path'].is_file() if item['markdown_path'] else False
        prev_md_row = prev_md.get(key, {})
        if marker:
            item['parse_status'] = '已解析'
        elif table_cell(prev_md_row, '解析状态') == '解析失败':
            item['parse_status'] = '解析失败'
        else:
            item['parse_status'] = '未解析'
        item['parse_error'] = table_cell(prev_md_row, '失败原因') or '--'

        # 精读状态 + 日期
        note_file = (item['notes_path'] / '精读.md') if item['notes_path'] else None
        prev_note_row = prev_notes.get(key, {})
        if note_file and note_file.is_file() and note_file.stat().st_size > 0:
            item['reading_status'] = '已精读'
        elif table_cell(prev_note_row, '精读状态') == '精读中':
            item['reading_status'] = '精读中'
        else:
            item['reading_status'] = '未精读'
        item['priority'] = table_cell(prev_note_row, '优先级') or '--'
        if item['notes_path']:
            note_date = frontmatter_date(item['notes_path'] / '精读.md')
            first, last = _git_commit_range(item['notes_path'], workspace)
            # 日期口径：笔记 frontmatter 的 date 最贴近实际（笔记生成日），
            # 其次才是 git 历史与文件时间——避免目录迁移的提交日期被误当成精读日期。
            item['reading_start'] = note_date or first or _mtime_date(item['notes_path'], 'dir')
            item['reading_finish'] = (note_date or last) if item['reading_status'] == '已精读' else ''
        # 脚本推不出日期时，保留上一版人工填写的开始/完成日期（例如标记为「精读中」的论文）
        prev_start = table_cell(prev_note_row, '精读开始')
        prev_finish = table_cell(prev_note_row, '精读完成')
        if not item['reading_start'] and prev_start not in ('', '--'):
            item['reading_start'] = prev_start
        if not item['reading_finish'] and prev_finish not in ('', '--'):
            item['reading_finish'] = prev_finish
        item['reading_start'] = item['reading_start'] or '--'
        item['reading_finish'] = item['reading_finish'] or '--'

        # 翻译状态 + 形式
        tr_files = []
        if item['translation_path'] and item['translation_path'].is_dir():
            tr_files = sorted(
                p for p in item['translation_path'].rglob('*')
                if p.is_file() and not p.name.startswith('.')
            )
        prev_tr_row = prev_translation.get(key, {})
        if tr_files:
            item['translation_status'] = '已翻译'
            item['translation_forms'] = sorted({p.stem for p in tr_files})
            item['translation_finish'] = (
                _git_commit_range(item['translation_path'], workspace)[1]
                or _mtime_date(item['translation_path'], 'dir') or '--'
            )
        else:
            item['translation_status'] = (
                '翻译中' if table_cell(prev_tr_row, '翻译状态') == '翻译中' else '未翻译'
            )
            item['translation_finish'] = '--'

        item['months_seen'] = sorted(item['months_seen'])
        result.append(item)

    result.sort(key=lambda x: (x['archive_month'] or '9999-99', x['archive_date'] or '', x['paper_stem']))
    return result


def _mtime_date(path: Path, kind: str = 'file') -> str:
    candidates = [path] if kind == 'file' else sorted(
        (p for p in path.rglob('*') if p.is_file() and not p.name.startswith('.')),
        key=lambda p: p.name,
    )
    for candidate in candidates:
        try:
            return datetime.fromtimestamp(candidate.stat().st_mtime).strftime('%Y-%m-%d')
        except OSError:
            continue
    return ''


def _git_commit_range(path: Path, workspace: Path):
    """返回 (首次提交日期, 最后提交日期)；查询失败时返回 ('', '')。

    文件路径加 `--follow` 以便跨改名/移动追踪历史（目录不支持 `--follow`）。
    """
    import subprocess

    cmd = ['git', '-C', str(workspace), '-c', 'core.quotePath=false', 'log',
           '--format=%ad', '--date=short']
    if Path(path).is_file():
        cmd.append('--follow')
    cmd += ['--', str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return '', ''
    if proc.returncode != 0:
        return '', ''
    dates = [line.strip() for line in proc.stdout.decode('utf-8', 'replace').splitlines() if line.strip()]
    if not dates:
        return '', ''
    return dates[-1], dates[0]


def frontmatter_date(path: Path) -> str:
    """读 Markdown frontmatter 的 `date:`（YYYY-MM-DD）；读不到返回空串。

    用于「精读完成日期」这类无法从目录结构推断、但笔记自身记录了的日期。
    """
    path = Path(path)
    if not path.is_file():
        return ''
    try:
        text = path.read_text(encoding='utf-8-sig', errors='replace')
    except OSError:
        return ''
    match = re.match(r'^---\s*\n(.*?)^---\s*\n', text, re.MULTILINE | re.DOTALL)
    if not match:
        return ''
    found = re.search(r'^date:\s*["\']?(\d{4}-\d{2}-\d{2})', match.group(1), re.MULTILINE)
    return found.group(1) if found else ''


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def _md_link(text: str, target: Path, origin_dir: Path) -> str:
    return '[%s](%s)' % (text, rel_link(target, origin_dir))


def render_raw_rows(registry: list, origin_dir: Path) -> str:
    """01-raw 月份 README 的表格行（PDF 链接相对该 README）。"""
    lines = []
    for item in registry:
        pdf_cell = _md_link('PDF', item['raw_path'], origin_dir) if item['raw_path'] else '未入库'
        cells = [item['archive_date'] or '--', item['title'], item['paper_stem'],
                 item['source'] or '--', pdf_cell]
        lines.append('| ' + ' | '.join(str(cell) for cell in cells) + ' |')
    return '\n'.join(lines)


def render_status_rows(registry: list, origin_dir: Path, stage: str) -> str:
    lines = []
    for item in registry:
        if stage == 'markdown':
            file_cell = _md_link('解析文件', item['markdown_path'], origin_dir) if item['markdown_path'] else '--'
            cells = [item['archive_month'] or '--', item['title'], item['paper_stem'],
                     item['parse_status'], file_cell, item['archive_date'] or '--', item['parse_error']]
        elif stage == 'notes':
            file_cell = '--'
            if item['notes_path']:
                note = item['notes_path'] / '精读.md'
                if note.is_file():
                    file_cell = _md_link('精读.md', note, origin_dir)
                else:
                    file_cell = '--'
            cells = [item['archive_month'] or '--', item['title'], item['paper_stem'],
                     item['reading_status'], item['reading_start'], item['reading_finish'],
                     item['priority'], file_cell]
        else:
            dir_cell = '--'
            if item['translation_path'] and item['translation_path'].is_dir():
                dir_cell = _md_link('目录', item['translation_path'], origin_dir)
            forms = '、'.join(item['translation_forms']) or (
                '（待补充形式）' if item['translation_status'] == '翻译中' else '--'
            )
            cells = [item['archive_month'] or '--', item['title'], item['paper_stem'],
                     item['translation_status'], forms, item['translation_finish'], dir_cell]
        lines.append('| ' + ' | '.join(str(cell) for cell in cells) + ' |')
    return '\n'.join(lines)


def summary_lines(registry: list, stage: str = '') -> str:
    total = len(registry)
    months = {}
    for item in registry:
        months[item['archive_month'] or '未知'] = months.get(item['archive_month'] or '未知', 0) + 1
    lines = ['## 汇总', '']
    lines.append('- 论文总数：%d 篇（%s）' % (
        total, '、'.join('%s %d 篇' % (m, c) for m, c in sorted(months.items())) or '--'))
    if stage in ('', 'raw', 'markdown'):
        parsed = sum(1 for i in registry if i['parse_status'] == '已解析')
        failed = [i for i in registry if i['parse_status'] == '解析失败']
        unparsed = [i for i in registry if i['parse_status'] == '未解析']
        lines.append('- 解析状态：已解析 %d / 未解析 %d / 解析失败 %d' % (parsed, len(unparsed), len(failed)))
        if unparsed:
            lines.append('- **未解析论文**（需补解析）：' + '、'.join(i['paper_stem'] for i in unparsed))
        if failed:
            lines.append('- **解析失败论文**：' + '、'.join(
                '%s（%s）' % (i['paper_stem'], i['parse_error']) for i in failed))
    if stage in ('', 'raw', 'notes'):
        read = sum(1 for i in registry if i['reading_status'] == '已精读')
        doing = [i for i in registry if i['reading_status'] == '精读中']
        todo = [i for i in registry if i['reading_status'] == '未精读']
        lines.append('- 精读状态：已精读 %d / 精读中 %d / 未精读 %d' % (read, len(doing), len(todo)))
        if doing:
            lines.append('- **精读中**：' + '、'.join(
                '%s（开始 %s）' % (i['paper_stem'], i['reading_start']) for i in doing))
        if todo:
            lines.append('- **未精读论文**：' + '、'.join(i['paper_stem'] for i in todo))
    if stage in ('', 'raw', 'translation'):
        done = sum(1 for i in registry if i['translation_status'] == '已翻译')
        doing = [i for i in registry if i['translation_status'] == '翻译中']
        todo = [i for i in registry if i['translation_status'] == '未翻译']
        lines.append('- 翻译状态：已翻译 %d / 翻译中 %d / 未翻译 %d' % (done, len(doing), len(todo)))
        if doing:
            lines.append('- **翻译中**：' + '、'.join(i['paper_stem'] for i in doing))
        if todo:
            lines.append('- **未翻译论文**：' + '、'.join(i['paper_stem'] for i in todo))
    return '\n'.join(lines) + '\n'


def render_month_readme(root_name: str, month: str, registry: list, stage: str,
                        origin_dir: Path) -> str:
    """月份 README 全文模板：正文说明写在表格区之外，表格区格式与 _month_block 完全一致。"""
    papers = [i for i in registry if (i['archive_month'] or '') == month]
    if stage == 'raw':
        title = '# %s 原始论文' % month
        intro = (
            '本目录保存 **%s** 首次入库的论文原始 PDF，共 %d 篇。\n'
            '文件名主干与 `02-markdown/%s/`、`03-notes/%s/`、`06-translation/%s/` 中同名论文一致。\n'
            % (month, len(papers), month, month, month)
        )
    elif stage == 'markdown':
        title = '# %s 解析结果' % month
        intro = (
            '本目录保存 **%s** 入库论文的 PDF 解析 Markdown，共 %d 篇。\n'
            '文件名主干与 `01-raw/%s/` 中对应 PDF 完全一致。\n' % (month, len(papers), month)
        )
    elif stage == 'notes':
        title = '# %s 精读笔记' % month
        intro = (
            '本目录保存 **%s** 入库论文的精读笔记，按论文标题建立目录，AI 精读固定为 `精读.md`。\n'
            % month
        )
    else:
        title = '# %s 翻译结果' % month
        intro = (
            '本目录保存 **%s** 入库论文的翻译结果，按论文标题建立目录，文件按翻译形式命名。\n'
            '尚未翻译的论文只在此处登记，不预建空目录。\n' % month
        )

    skeleton = (
        '%s\n\n%s\n## 清单\n\n%s\n%s\n\n## 说明\n\n'
        '- 表格由 `sync_indexes.py` 维护（`%s` 内请勿手工编辑，其他内容可自由补充）。\n'
        '- 月份归属依据论文**首次进入 `01-raw` 的日期**；同一篇论文在四个阶段月份与主干必须一致。\n'
        % (title, intro, INDEX_BEGIN, INDEX_END, INDEX_BEGIN)
    )
    updated, ok = replace_marked_block(skeleton, _month_block(stage, month, registry, origin_dir))
    return updated if ok else skeleton


def render_markdown_month_rows(papers: list, origin_dir: Path) -> str:
    lines = []
    for item in papers:
        file_cell = _md_link('解析文件', item['markdown_path'], origin_dir) if item['markdown_path'] else '--'
        cells = [item['parse_status'], item['title'], item['paper_stem'], file_cell,
                 item['archive_date'] or '--', item['parse_error']]
        lines.append('| ' + ' | '.join(str(c) for c in cells) + ' |')
    return '\n'.join(lines)


def render_notes_month_rows(papers: list, origin_dir: Path) -> str:
    lines = []
    for item in papers:
        file_cell = '--'
        if item['notes_path']:
            note = item['notes_path'] / '精读.md'
            if note.is_file():
                file_cell = _md_link('精读.md', note, origin_dir)
        cells = [item['reading_status'], item['title'], item['paper_stem'],
                 item['reading_start'], item['reading_finish'], item['priority'], file_cell]
        lines.append('| ' + ' | '.join(str(c) for c in cells) + ' |')
    return '\n'.join(lines)


def render_translation_month_rows(papers: list, origin_dir: Path) -> str:
    lines = []
    for item in papers:
        dir_cell = '--'
        if item['translation_path'] and item['translation_path'].is_dir():
            dir_cell = _md_link('目录', item['translation_path'], origin_dir)
        forms = '、'.join(item['translation_forms']) or (
            '（待补充形式）' if item['translation_status'] == '翻译中' else '--'
        )
        cells = [item['translation_status'], item['title'], item['paper_stem'],
                 forms, item['translation_finish'], dir_cell]
        lines.append('| ' + ' | '.join(str(c) for c in cells) + ' |')
    return '\n'.join(lines)


def render_root_index(root_name: str, registry: list, stage: str, origin_dir: Path) -> str:
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    if stage == 'raw':
        title = '# 论文原文索引（统一论文登记表）'
        intro = (
            '本目录保存原始论文 PDF，按**首次入库月份**分目录：`01-raw/YYYY-MM/<论文标题>.pdf`。\n'
            '本表是跨阶段统一登记表：月份与入库日期由 `01-raw/YYYY-MM/README.md` 维护，'
            '解析/精读/翻译状态由 `sync_indexes.py` 依据磁盘实际文件生成。\n'
        )
        header = '| 月份 | 入库日期 | 论文标题 | paper_stem | 来源 | PDF | 解析 | 精读 | 翻译 |\n|---|---|---|---|---|---|---|---|---|'
        rows = render_raw_index_rows(registry, origin_dir)
    elif stage == 'markdown':
        title = '# 论文解析结果索引'
        intro = (
            '本目录保存从 `01-raw/YYYY-MM/` 中论文 PDF 解析得到的 Markdown。\n'
            '`未解析` / `解析失败`（含失败原因）必须在本表登记；解析文件路径始终指向真实文件。\n'
        )
        header = '| 月份 | 论文标题 | paper_stem | 解析状态 | 解析文件 | 入库日期 | 失败原因 |\n|---|---|---|---|---|---|---|'
        rows = render_status_rows(registry, origin_dir, 'markdown')
    elif stage == 'notes':
        title = '# 精读笔记索引'
        intro = (
            '本目录按 `YYYY-MM/<论文标题>/` 组织精读笔记，AI 精读固定为 `精读.md`。\n'
            '`未精读` / `精读中` 论文必须在本表登记，并记录开始与完成日期。\n'
        )
        header = '| 月份 | 论文标题 | paper_stem | 精读状态 | 精读开始 | 精读完成 | 优先级 | 精读文件 |\n|---|---|---|---|---|---|---|---|'
        rows = render_status_rows(registry, origin_dir, 'notes')
    else:
        title = '# 翻译结果索引'
        intro = (
            '本目录按 `YYYY-MM/<论文标题>/` 组织翻译结果，文件按翻译形式命名。\n'
            '尚未翻译的论文只在本表登记，不预建空目录。\n'
        )
        header = '| 月份 | 论文标题 | paper_stem | 翻译状态 | 翻译形式 | 完成日期 | 目录 |\n|---|---|---|---|---|---|---|'
        rows = render_status_rows(registry, origin_dir, 'translation')

    block = '\n'.join([header, rows]).rstrip('\n')
    return (
        '%s\n\n%s\n<!-- 本文件由 sync_indexes.py 生成，最后更新：%s -->\n\n%s\n%s\n%s\n\n%s'
        % (title, intro, now, INDEX_BEGIN, block, INDEX_END, summary_lines(registry, stage))
    )


def render_raw_index_rows(registry: list, origin_dir: Path) -> str:
    lines = []
    for item in registry:
        pdf_cell = _md_link('PDF', item['raw_path'], origin_dir) if item['raw_path'] else '未入库'
        cells = [item['archive_month'] or '--', item['archive_date'] or '--', item['title'],
                 item['paper_stem'], item['source'] or '--', pdf_cell,
                 item['parse_status'], item['reading_status'], item['translation_status']]
        lines.append('| ' + ' | '.join(str(c) for c in cells) + ' |')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 刷新
# ---------------------------------------------------------------------------

def refresh(workspace: Path, config: dict, dry_run: bool = False, overrides: dict = None) -> dict:
    workspace = Path(workspace)
    registry = build_registry(workspace, config, overrides=overrides)
    layout = config.get('layout') or {}
    month_dirs = layout.get('month_dirs') or ['01-raw', '02-markdown', '03-notes', '06-translation']
    stages = {
        'papers_dir': ('01-raw', 'raw'),
        'markdown_dir': ('02-markdown', 'markdown'),
        'notes_dir': ('03-notes', 'notes'),
        'translation_dir': ('06-translation', 'translation'),
    }

    changed = []
    months = sorted({i['archive_month'] for i in registry if i['archive_month']})

    for key, (default_name, stage) in stages.items():
        root = resolve_workspace_subdir(config, key, default_name, workspace)
        root.mkdir(parents=True, exist_ok=True)
        is_month_stage = default_name in month_dirs
        # 1) 根索引
        index_path = root / 'index.md'
        content = render_root_index(default_name, registry, stage, root)
        if _write_if_changed(index_path, content, dry_run):
            changed.append(str(index_path))
        # 2) 月份 README
        if is_month_stage:
            for month in months:
                month_dir = root / month
                readme = month_dir / 'README.md'
                block = _month_block(stage, month, registry, month_dir)
                if readme.is_file():
                    text = readme.read_text(encoding='utf-8-sig')
                    updated, ok = replace_marked_block(text, block)
                    if not ok:
                        updated = render_month_readme(default_name, month, registry, stage, month_dir)
                else:
                    updated = render_month_readme(default_name, month, registry, stage, month_dir)
                if _write_if_changed(readme, updated, dry_run):
                    changed.append(str(readme))
        # 3) 月份目录缺 README 的情况（例如历史遗留）由上面循环覆盖

    return {'workspace': str(workspace), 'papers': len(registry), 'months': months,
            'changed': changed, 'dry_run': dry_run, 'registry': registry}


def _month_block(stage: str, month: str, registry: list, origin_dir: Path) -> str:
    papers = [i for i in registry if (i['archive_month'] or '') == month]
    if stage == 'raw':
        header = '| 入库日期 | 论文标题 | paper_stem | 来源 | PDF |\n|---|---|---|---|---|'
        body = render_raw_rows(papers, origin_dir)
    elif stage == 'markdown':
        header = '| 解析状态 | 论文标题 | paper_stem | 解析文件 | 入库日期 | 失败原因 |\n|---|---|---|---|---|---|'
        body = render_markdown_month_rows(papers, origin_dir)
    elif stage == 'notes':
        header = '| 精读状态 | 论文标题 | paper_stem | 精读开始 | 精读完成 | 优先级 | 精读文件 |\n|---|---|---|---|---|---|---|'
        body = render_notes_month_rows(papers, origin_dir)
    else:
        header = '| 翻译状态 | 论文标题 | paper_stem | 翻译形式 | 完成日期 | 目录 |\n|---|---|---|---|---|---|'
        body = render_translation_month_rows(papers, origin_dir)
    block = '\n'.join([header, body]).rstrip('\n')
    if papers:
        block += '\n\n' + summary_lines(papers, stage)
    return block


# ---------------------------------------------------------------------------
# 链接与一致性校验
# ---------------------------------------------------------------------------

LINK_RE = re.compile(r'!?\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
SKIP_DIRS = {'.git', '.claude', '.agents', '.dsh', '.pencode', '.qoder', '.trae', 'node_modules'}


def _iter_markdown(workspace: Path):
    for path in sorted(workspace.rglob('*.md')):
        parts = set(path.relative_to(workspace).parts[:-1])
        if parts & SKIP_DIRS:
            continue
        yield path


def check_links(workspace: Path, config: dict) -> dict:
    workspace = Path(workspace)
    dangling = []
    for path in _iter_markdown(workspace):
        try:
            text = path.read_text(encoding='utf-8-sig', errors='replace')
        except OSError:
            continue
        in_fence = False
        for number, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith('```') or stripped.startswith('~~~'):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for match in LINK_RE.finditer(line):
                target = match.group(1).strip().strip('<>')
                if not target or target.startswith(('http://', 'https://', 'mailto:', '#', 'data:')):
                    continue
                target = target.split('#', 1)[0]
                if not target:
                    continue
                if not (path.parent / target).exists():
                    dangling.append({
                        'file': str(path.relative_to(workspace)),
                        'line': number,
                        'target': target,
                    })

    layout = config.get('layout') or {}
    month_dirs = layout.get('month_dirs') or []
    missing_readme = []
    errors = []

    registry = build_registry(workspace, config)
    for item in registry:
        stems = {clean_stem(item['paper_stem'])}
        month = item['archive_month']
        if not month:
            errors.append('无法确定月份：%s' % item['paper_stem'])
            continue
        for key, name, suffix in (
            ('raw_path', 'papers_dir', '.pdf'),
            ('markdown_path', 'markdown_dir', '.md'),
            ('notes_path', 'notes_dir', ''),
            ('translation_path', 'translation_dir', ''),
        ):
            found = item[key]
            if not found:
                continue
            parent_month = found.parent.name
            if is_month_dir(parent_month) and parent_month != month:
                errors.append('月份错位：%s 的 %s 位于 %s，但入库月份为 %s'
                              % (item['paper_stem'], name, parent_month, month))

    for key, default_name in (('papers_dir', '01-raw'), ('markdown_dir', '02-markdown'),
                              ('notes_dir', '03-notes'), ('translation_dir', '06-translation'),
                              ('equation_dir', '04-equation_problem'), ('daily_dir', '08-daily'),
                              ('reading_dir', '08-reading')):
        root = resolve_workspace_subdir(config, key, default_name, workspace)
        if not root.is_dir():
            continue
        if default_name in month_dirs:
            for month_dir in iter_month_dirs(root):
                if not (month_dir / 'README.md').is_file():
                    missing_readme.append(str(month_dir.relative_to(workspace)))
        if not (root / 'README.md').is_file():
            missing_readme.append(str(root.relative_to(workspace)))

    problems = len(dangling) + len(missing_readme) + len(errors)
    return {
        'workspace': str(workspace),
        'dangling': dangling,
        'dangling_count': len(dangling),
        'missing_readme': missing_readme,
        'consistency_errors': errors,
        'problems': problems,
    }


def main():
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description='刷新论文工作区索引与状态清单，或校验链接')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--workspace', type=str, default=None)
    parser.add_argument('--check-links', action='store_true', help='只校验链接与一致性，不写文件')
    parser.add_argument('--dry-run', action='store_true', help='只报告将要写入的文件')
    parser.add_argument('--json', action='store_true', help='以 JSON 输出（含完整登记表）')
    args = parser.parse_args()

    config = load_config(args.config)
    workspace = resolve_workspace_path(config, args.workspace)
    if not workspace.is_dir():
        print('ERROR: workspace 不存在：%s' % workspace, file=sys.stderr)
        return 1

    if args.check_links:
        result = check_links(workspace, config)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            if result['dangling']:
                print('悬空链接 %d 处：' % result['dangling_count'])
                for item in result['dangling']:
                    print('  - %s:%s -> %s' % (item['file'], item['line'], item['target']))
            if result['missing_readme']:
                print('缺少 README.md：')
                for path in result['missing_readme']:
                    print('  - %s' % path)
            if result['consistency_errors']:
                print('一致性错误：')
                for text in result['consistency_errors']:
                    print('  - %s' % text)
            print('检查完成：问题 %d 处（悬空链接 %d / 缺 README %d / 一致性 %d）'
                  % (result['problems'], result['dangling_count'],
                     len(result['missing_readme']), len(result['consistency_errors'])))
        return 1 if result['problems'] else 0

    result = refresh(workspace, config, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print('论文总数：%d；月份：%s' % (result['papers'], '、'.join(result['months']) or '--'))
        for path in result['changed']:
            print('[%s] %s' % ('dry-run' if args.dry_run else '更新', path))
        if not result['changed']:
            print('索引已是最新，无文件变化。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
