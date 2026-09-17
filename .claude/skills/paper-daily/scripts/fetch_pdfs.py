#!/usr/bin/env python3
"""
paper-daily PDF 归档脚本

把每日推荐的前 N 篇论文的原始 PDF 保存到 `papers_dir`（默认 01-raw）的**入库月份目录**：
`01-raw/YYYY-MM/<论文标题>.pdf`（规则见工作区 .AGENT.md）。

硬性规则：
  * **目标文件已存在 -> 立即停止**，不下载、不覆盖、不改名覆盖
  * 下载内容必须是真 PDF（校验 %PDF 魔术字），否则删除临时文件并记为失败
  * 只写 `papers_dir`、其月份目录 README 与 `index.md`，不碰 02-markdown / 03-notes（用户自己的流程负责）
  * 入库月份 = `--date`（入库日期）所在月；同一篇论文四个阶段月份保持一致

用法：
    python fetch_pdfs.py --papers-json search_result.json --date 2026-09-16 --top-k 3
    python fetch_pdfs.py --papers-json search_result.json --date 2026-09-16 --month 2026-09
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
    load_config,
    month_of,
    month_paper_path,
    resolve_workspace_path,
    resolve_workspace_subdir,
    sanitize_paper_title,
    scan_paper_entries,
)

from sync_indexes import refresh  # noqa: E402

USER_AGENT = "paper-skills/0.1 (paper-daily; +local research workspace)"
TIMEOUT = 120
PDF_MAGIC = b"%PDF"


def index_existing_pdfs(papers_dir: Path) -> dict:
    """建立"规范化主干 -> 实际文件"索引（递归月份目录 + 根部平铺）。

    手工命名的历史论文与自动生成的名称可能只差大小写、连字符或下划线
    （例如 `lp-norm` vs `lp_norm`），按字符串精确比较会误判为"不存在"从而重复下载。
    这里统一按 `clean_stem()` 归一化后比对。
    """
    index = {}
    for entry in scan_paper_entries(papers_dir, kind='file', suffix='.pdf'):
        key = clean_stem(entry['stem'])
        if key:
            index.setdefault(key, entry['path'])
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


def main():
    _force_utf8_stdio()

    parser = argparse.ArgumentParser(description='Archive recommended paper PDFs into papers_dir/YYYY-MM/')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--workspace', type=str, default=None)
    parser.add_argument('--papers-json', type=str, required=True,
                        help='search_arxiv.py output JSON')
    parser.add_argument('--date', type=str, default=None,
                        help='入库日期 YYYY-MM-DD（默认取 JSON 的 target_date 或今天）')
    parser.add_argument('--month', type=str, default=None,
                        help='入库月份 YYYY-MM（默认取 --date 所在月）')
    parser.add_argument('--top-k', type=int, default=None,
                        help='How many top papers to archive (default: paper_daily.pdf_top_k or 3)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would happen without downloading')
    parser.add_argument('--json', action='store_true', help='Emit machine-readable JSON only')

    args = parser.parse_args()

    config = load_config(args.config)
    workspace = resolve_workspace_path(config, args.workspace)
    if not workspace.is_dir():
        print("ERROR: workspace 路径不存在：%s" % workspace, file=sys.stderr)
        return 1

    papers_dir = resolve_workspace_subdir(config, 'papers_dir', '01-raw', workspace)
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
    month = args.month or month_of(date)
    if not month:
        print("ERROR: 无法从 --date/--month 解析入库月份（需要 YYYY-MM-DD 或 YYYY-MM）：%s" % date,
              file=sys.stderr)
        return 1

    top_papers = data.get('top_papers') or []
    if not top_papers:
        print(json.dumps({'papers_dir': str(papers_dir), 'month': month, 'date': date, 'archived': [],
                          'skipped': 0, 'failed': 0, 'note': 'top_papers 为空'},
                         ensure_ascii=False, indent=2))
        return 0

    month_dir = papers_dir / month
    # 已归档索引：按主干归一化、递归月份目录，避免命名变体导致重复下载
    existing_index = index_existing_pdfs(papers_dir)

    results = []
    for paper in top_papers[:top_k]:
        title = paper.get('title', '')
        stem = paper.get('paper_stem') or sanitize_paper_title(title)
        pdf_url = paper.get('pdf_url') or ''
        dest = month_paper_path(papers_dir, month, stem, '.pdf') if stem else None
        entry = {
            'title': title,
            'paper_stem': stem,
            'month': month,
            'pdf_url': pdf_url,
            'path': str(dest) if dest else '',
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

        # 已存在就停止：不下载、不覆盖。
        # 用主干归一化后的索引判断，避免因命名变体（大小写/连字符/下划线）重复下载。
        existing = existing_index.get(clean_stem(stem))
        if existing is not None:
            entry.update(status='exists', path=str(existing),
                         detail='已存在，跳过下载：%s' % existing.relative_to(papers_dir))
            results.append(entry)
            if not args.json:
                print("[跳过] 已存在：%s" % existing.relative_to(papers_dir), file=sys.stderr)
            continue

        if args.dry_run:
            entry.update(status='would_download', detail=pdf_url)
            results.append(entry)
            continue

        month_dir.mkdir(parents=True, exist_ok=True)
        ok, message = download_pdf(pdf_url, dest)
        entry.update(status='downloaded' if ok else 'failed', detail=message)
        if not args.json:
            print("[%s] %s —— %s" % (entry['status'], dest.relative_to(papers_dir), message),
                  file=sys.stderr)
        results.append(entry)

    archived = [r for r in results if r['status'] == 'downloaded']
    existed = [r for r in results if r['status'] == 'exists']
    failed = [r for r in results if r['status'] == 'failed']

    # 刷新索引：新下载的论文把入库日期/来源写进登记表（01-raw/YYYY-MM/README.md 为权威记录）
    overrides = {}
    for item in archived:
        overrides[item['paper_stem']] = {'date': date, 'month': month,
                                         'title': item['title'], 'source': item.get('source') or '--'}
    index_updated = []
    if (archived or existed) and not args.dry_run:
        report = refresh(workspace, config, overrides=overrides)
        index_updated = report['changed']

    summary = {
        'papers_dir': str(papers_dir),
        'month_dir': str(month_dir),
        'month': month,
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
