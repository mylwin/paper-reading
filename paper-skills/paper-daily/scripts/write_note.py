#!/usr/bin/env python3
"""
paper-daily 笔记落盘脚本

把 agent 写好的推荐笔记内容写入按日期隔离的输出目录，并更新该目录的运行索引。

输出结构（daily_dir 默认 "daily"，相对 skill 目录；也可在 config.yaml 改为绝对路径）：

    daily/2026-09-16/
    ├── 2026-09-16论文推荐.md      # 本次推荐笔记
    ├── search_result.json         # 本次检索原始结果（若传入 --papers-json）
    └── _index.json                # 该日期目录的元信息索引

用法：
    python write_note.py --date 2026-09-16 --papers-json search_result.json < note.md
    python write_note.py --date 2026-09-16 --stdin-file note.md --config /path/config.yaml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_config import (  # noqa: E402
    find_config_path,
    load_config,
    resolve_workspace_path,
    resolve_workspace_subdir,
)


def _force_utf8_stdio():
    """Windows 控制台默认 GBK，打印含中文的路径/摘要会抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


_force_utf8_stdio()


def _build_index_entry(date, note_path, papers_json, args):
    """构造 _index.json 的内容。"""
    entry = {
        'date': date,
        'note_file': note_path.name,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'skill': 'paper-daily',
    }

    if not papers_json or not Path(papers_json).is_file():
        return entry

    try:
        with open(papers_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return entry

    entry['sources_searched'] = data.get('sources_searched', [])
    entry['source_counts'] = data.get('source_counts', {})
    entry['date_windows'] = data.get('date_windows', {})
    entry['total_unique'] = data.get('total_unique', 0)
    entry['top_papers'] = [
        {
            'title': p.get('title', ''),
            'source': p.get('source', ''),
            'arxiv_id': p.get('arxiv_id', ''),
            'url': p.get('url', ''),
            'note_filename': p.get('note_filename', ''),
            'matched_domain': p.get('matched_domain', ''),
            'score': (p.get('scores') or {}).get('recommendation'),
        }
        for p in data.get('top_papers', [])
    ]

    # 复制一份检索结果到日期目录，便于回溯当天到底搜到了什么
    try:
        target = Path(papers_json).resolve()
        if target.parent.resolve() != Path(note_path).resolve().parent:
            (Path(note_path).parent / 'search_result.json').write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8',
            )
            entry['search_result_file'] = 'search_result.json'
    except OSError:
        pass

    return entry


def main():
    parser = argparse.ArgumentParser(description='Write paper-daily note into a dated output directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml (default: skill config.yaml, then shared config.yaml)')
    parser.add_argument('--workspace', type=str, default=None,
                        help='Markdown workspace path (default: config workspace_path or PAPER_WORKSPACE_PATH)')
    parser.add_argument('--date', type=str, required=True,
                        help='Target date, YYYY-MM-DD; becomes the output subdirectory name')
    parser.add_argument('--daily-dir', type=str, default=None,
                        help='Override output base directory (relative paths resolve against the skill dir)')
    parser.add_argument('--note-suffix', type=str, default=None,
                        help='Note filename suffix (default: daily_note_suffix in config)')
    parser.add_argument('--filename', type=str, default=None,
                        help='Explicit note filename, overrides --note-suffix')
    parser.add_argument('--title', type=str, default=None,
                        help='Note title recorded in _index.json')
    parser.add_argument('--papers-json', type=str, default=None,
                        help='search_arxiv.py output JSON; copied into the dated dir and summarised in _index.json')
    parser.add_argument('--stdin-file', type=str, default=None,
                        help='Read note content from this file instead of stdin')
    parser.add_argument('--dry-run', action='store_true',
                        help='Resolve and print paths without writing anything')

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        config_path = find_config_path(args.config)
    except FileNotFoundError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1

    try:
        workspace = resolve_workspace_path(config, args.workspace)
    except ValueError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1

    if not workspace.exists():
        print("ERROR: workspace 路径不存在：%s" % workspace, file=sys.stderr)
        return 1

    language = str(config.get('language', 'zh')).lower()
    daily_note_name = (config.get('daily_note_name') or '').strip()
    if daily_note_name:
        # paperread 约定：日期已经在文件夹名上，文件名固定为 今日检索.md
        filename = args.filename or ("%s.md" % daily_note_name)
    else:
        default_suffix = config.get('daily_note_suffix') or (
            'paper-recommendations' if language == 'en' else '论文推荐'
        )
        note_suffix = args.note_suffix or default_suffix
        filename = args.filename or ("%s%s.md" % (args.date, note_suffix))

    # 输出目录：<workspace>/<daily_dir>/<日期>/
    base = resolve_workspace_subdir(config, 'daily_dir', '08-daily', workspace)
    if args.daily_dir:
        override = Path(str(args.daily_dir)).expanduser()
        base = override if override.is_absolute() else (workspace / override)
    out_dir = base / args.date
    note_path = out_dir / filename

    if args.dry_run:
        print(json.dumps({
            'config_path': str(config_path),
            'workspace': str(workspace),
            'output_dir': str(out_dir),
            'note_path': str(note_path),
            'language': language,
        }, ensure_ascii=False, indent=2))
        return 0

    # 读取笔记正文
    if args.stdin_file:
        try:
            content = Path(args.stdin_file).read_text(encoding='utf-8-sig')
        except OSError as e:
            print("ERROR: 无法读取 %s: %s" % (args.stdin_file, e), file=sys.stderr)
            return 1
    else:
        content = sys.stdin.read().lstrip('\ufeff')

    content = content.strip('\n')
    if not content:
        print("ERROR: 笔记内容为空（stdin 未提供内容）。", file=sys.stderr)
        return 1

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        note_path.write_text(content + '\n', encoding='utf-8')
    except OSError as e:
        print("ERROR: 写入失败 %s: %s" % (note_path, e), file=sys.stderr)
        return 1

    index = _build_index_entry(args.date, note_path, args.papers_json, args)
    index_path = out_dir / '_index.json'
    try:
        index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8'
        )
    except OSError as e:
        print("WARNING: 笔记已写入，但索引更新失败：%s" % e, file=sys.stderr)

    print(json.dumps({
        'note_path': str(note_path),
        'index_path': str(index_path),
        'output_dir': str(out_dir),
        'bytes': len(content.encode('utf-8')),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
