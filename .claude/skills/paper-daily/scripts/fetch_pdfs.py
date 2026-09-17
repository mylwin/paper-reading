#!/usr/bin/env python3
"""
paper-daily PDF 归档脚本

把每日推荐的前 N 篇论文的原始 PDF 保存到 `papers_dir`（默认 01-raw），
文件名使用稳定主干 `<论文标题>`（规则见 paperread 工作区 AGENT.md）。

硬性规则：
  * **目标文件已存在 -> 立即停止**，不下载、不覆盖、不改名覆盖
  * 下载内容必须是真 PDF（校验 %PDF 魔术字），否则删除临时文件并记为失败
  * 只写 `papers_dir` 与其中的 `index.md`，不碰 02-markdown / 03-notes（用户自己的流程负责）

用法：
    python fetch_pdfs.py --papers-json search_result.json --date 2026-09-16 --top-k 3
    python fetch_pdfs.py --papers-json search_result.json --dry-run
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_config import (  # noqa: E402
    clean_stem,
    find_config_path,
    load_config,
    resolve_vault_path,
    resolve_vault_subdir,
    sanitize_paper_title,
)

USER_AGENT = "paper-skills/0.1 (paper-daily; +local research workspace)"
TIMEOUT = 120
PDF_MAGIC = b"%PDF"


def index_existing_pdfs(papers_dir: Path) -> dict:
    """建立"规范化主干 -> 实际文件"索引。

    手工命名的历史论文与自动生成的名称可能只差大小写、连字符或下划线
    （例如 `lp-norm` vs `lp_norm`），按字符串精确比较会误判为"不存在"从而重复下载。
    这里统一按 `clean_stem()` 归一化后比对。
    """
    index = {}
    if not papers_dir.is_dir():
        return index
    for path in papers_dir.glob('*.pdf'):
        key = clean_stem(path.stem)
        if key:
            index.setdefault(key, path)
    return index


def _force_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


def download_pdf(url: str, dest: Path, timeout: int = TIMEOUT) -> tuple:
    """下载 PDF 到 dest。返回 (ok, message)。

    先写 .part 临时文件，校验魔术字后再原子改名，避免半截文件污染 01-raw。
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    part = dest.with_suffix(dest.suffix + '.part')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            head = resp.read(5)
            if not head.startswith(PDF_MAGIC):
                ctype = resp.headers.get('Content-Type', '')
                return False, "响应不是 PDF（Content-Type: %s）" % (ctype or 'unknown')
            with open(part, 'wb') as f:
                f.write(head)
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
    except urllib.error.HTTPError as e:
        return False, "HTTP %s" % e.code
    except Exception as e:
        part.unlink(missing_ok=True)
        return False, "%s: %s" % (type(e).__name__, e)

    size = part.stat().st_size
    if size < 1024:
        part.unlink(missing_ok=True)
        return False, "文件过小（%d 字节），疑似占位响应" % size

    part.replace(dest)
    return True, "%.1f KB" % (size / 1024.0)


def update_index(index_path: Path, entries: list, date: str):
    """维护 `papers_dir/index.md`：只记录题目（+ 落盘日期），已存在的条目不重复写。"""
    existing = index_path.read_text(encoding='utf-8') if index_path.is_file() else ''
    existing_lines = existing.splitlines()

    def already(title: str) -> bool:
        return any(title in line for line in existing_lines)

    new_lines = []
    for entry in entries:
        if already(entry['title']):
            continue
        suffix = '（已存在）' if entry['status'] == 'exists' else ''
        new_lines.append("- %s —— %s%s" % (entry['title'], date, suffix))

    if not new_lines:
        return False

    if not existing:
        header = [
            "# 论文原文索引",
            "",
            "本目录保存原始论文 PDF，文件名主干 `<论文标题>` 与 `02-markdown`、`03-notes` 一致。",
            "",
        ]
        existing_lines = header

    content = '\n'.join(existing_lines).rstrip('\n') + '\n' + '\n'.join(new_lines) + '\n'
    index_path.write_text(content, encoding='utf-8')
    return True


def main():
    _force_utf8_stdio()

    parser = argparse.ArgumentParser(description='Archive recommended paper PDFs into papers_dir')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--vault', type=str, default=None)
    parser.add_argument('--papers-json', type=str, required=True,
                        help='search_arxiv.py output JSON')
    parser.add_argument('--date', type=str, default=None,
                        help='Date for the index entry (default: target_date in JSON or today)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='How many top papers to archive (default: paper_daily.pdf_top_k or 3)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would happen without downloading')
    parser.add_argument('--json', action='store_true', help='Emit machine-readable JSON only')

    args = parser.parse_args()

    config = load_config(args.config)
    vault = resolve_vault_path(config, args.vault)
    if not vault.is_dir():
        print("ERROR: vault 路径不存在：%s" % vault, file=sys.stderr)
        return 1

    papers_dir = resolve_vault_subdir(config, 'papers_dir', '01-raw', vault)
    top_k = args.top_k if args.top_k is not None else int(config.get('pdf_top_k', 3) or 3)

    json_path = Path(args.papers_json)
    if not json_path.is_file():
        print("ERROR: 找不到检索结果文件：%s" % json_path, file=sys.stderr)
        return 1
    try:
        data = json.loads(json_path.read_text(encoding='utf-8-sig'))
    except json.JSONDecodeError as e:
        print("ERROR: 检索结果不是合法 JSON：%s" % e, file=sys.stderr)
        return 1

    date = args.date or data.get('target_date') or datetime.now().strftime('%Y-%m-%d')
    top_papers = data.get('top_papers') or []
    if not top_papers:
        print(json.dumps({'papers_dir': str(papers_dir), 'date': date, 'archived': [],
                          'skipped': 0, 'failed': 0, 'note': 'top_papers 为空'},
                         ensure_ascii=False, indent=2))
        return 0

    # 已归档索引：按主干归一化，避免命名变体导致重复下载
    existing_index = index_existing_pdfs(papers_dir)

    results = []
    for paper in top_papers[:top_k]:
        title = paper.get('title', '')
        stem = paper.get('paper_stem') or sanitize_paper_title(title)
        pdf_url = paper.get('pdf_url') or ''
        entry = {
            'title': title,
            'paper_stem': stem,
            'pdf_url': pdf_url,
            'path': str(papers_dir / (stem + '.pdf')),
            'status': '',
            'detail': '',
        }

        if not stem:
            entry.update(status='failed', detail='标题无法生成有效文件名')
            results.append(entry)
            continue
        if not pdf_url:
            entry.update(status='failed', detail='该来源没有可用的 PDF 链接')
            results.append(entry)
            continue

        dest = papers_dir / (stem + '.pdf')
        # 已存在就停止：不下载、不覆盖。
        # 用主干归一化后的索引判断，避免因命名变体（大小写/连字符/下划线）重复下载。
        existing = existing_index.get(clean_stem(stem))
        if existing is not None:
            entry.update(status='exists', path=str(existing),
                         detail='已存在，跳过下载：%s' % existing.name)
            results.append(entry)
            if not args.json:
                print("[跳过] 已存在：%s" % existing.name, file=sys.stderr)
            continue

        if args.dry_run:
            entry.update(status='would_download', detail=pdf_url)
            results.append(entry)
            continue

        papers_dir.mkdir(parents=True, exist_ok=True)
        ok, message = download_pdf(pdf_url, dest)
        if ok:
            entry.update(status='downloaded', detail=message)
        else:
            entry.update(status='failed', detail=message)
        if not args.json:
            print("[%s] %s —— %s" % (entry['status'], dest.name, message), file=sys.stderr)
        results.append(entry)

    archived = [r for r in results if r['status'] == 'downloaded']
    existed = [r for r in results if r['status'] == 'exists']
    failed = [r for r in results if r['status'] == 'failed']

    index_updated = False
    if archived or existed:
        if args.dry_run:
            index_updated = False
        else:
            index_updated = update_index(papers_dir / 'index.md', results, date)

    summary = {
        'papers_dir': str(papers_dir),
        'date': date,
        'requested': len(top_papers[:top_k]),
        'downloaded': len(archived),
        'skipped_existing': len(existed),
        'failed': len(failed),
        'index_updated': index_updated,
        'archived': results,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failed else 0  # 部分失败不影响日报生成，交由上层标注


if __name__ == '__main__':
    sys.exit(main())
