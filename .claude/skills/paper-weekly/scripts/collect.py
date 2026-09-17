#!/usr/bin/env python3
"""
paper-weekly 素材收集与周期调度

三部分职责：

1. **周期判断**：从 `weekly.start_date`（首次调用日）起算，距 `weekly.last_run`
   满 `interval_days` 天才算 due。`--mark-run` 在分析成功后写回日期。
2. **素材清单**：汇总时间窗口内的
   - `weekly.inputs`（默认 03-notes > 02-markdown）的论文素材
   - `08-daily` 期间的每日检索结果
   - `weekly.context_dirs`（06-translation / 04-equation_problem）的**资产存在性**（只标注，不读内容）
   - **git 提交记录**判定"本周新增了哪些论文"（比 mtime 可靠）
   输出一个**小清单 JSON**；agent 读清单后再按需打开正文。
3. **周报命名**：按 `weekly.note_pattern` 生成 `<周起始日>-第N周周报.md`，N 从 start_date 起算（第 1 周即 start_date 所在周）。

用法：
    python collect.py                                  # 看是否到期 + 输出清单
    python collect.py --mark-run                       # 分析完成后写回日期
    python collect.py --force                          # 忽略间隔，强制收集
    python collect.py --date 2026-09-14                # 以指定日期为基准
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'paper-daily' / 'scripts'))

from paper_config import (  # noqa: E402
    clean_stem,
    find_config_path,
    load_config,
    resolve_vault_path,
    resolve_vault_subdir,
)


def _force_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


def weekly_config(config: dict) -> dict:
    return config.get('weekly') or {}


def compute_schedule(config: dict, today: date):
    """计算周期状态与周序号。"""
    weekly = weekly_config(config)
    interval = int(weekly.get('interval_days', 7) or 7)
    start_raw = str(weekly.get('start_date') or '').strip()
    last_raw = str(weekly.get('last_run') or '').strip()

    start = _parse_date(start_raw)
    last = _parse_date(last_raw)

    # 从首次调用日起算：没有 last_run 时以 start_date 为基准
    anchor = last or start
    if anchor is None:
        due = True
        next_due = today
    else:
        next_due = anchor + timedelta(days=interval)
        due = today >= next_due

    return {
        'enabled': bool(weekly.get('enabled', True)),
        'interval_days': interval,
        'start_date': start.isoformat() if start else '',
        'last_run': last.isoformat() if last else '',
        'is_first_run': start is None,
        'due': due,
        'next_due': next_due.isoformat(),
        'days_since_last': (today - last).days if last else None,
    }


def compute_week_no(start, target):
    """第 N 周：start 所在周为第 1 周，每满 7 天进一位。"""
    if start is None:
        return 1
    offset = (target - start).days
    if offset < 0:
        return 1
    return offset // 7 + 1


def week_context(config: dict, today: date):
    """给出本次周报的周序号与建议文件名。"""
    weekly = weekly_config(config)
    interval = int(weekly.get('interval_days', 7) or 7)
    start = _parse_date(str(weekly.get('start_date') or '')) or today
    week_no = compute_week_no(start, today)
    week_start = start + timedelta(days=(week_no - 1) * interval)
    pattern = weekly.get('note_pattern') or '{week_start}-第{week_no}周周报.md'
    filename = (pattern
                .replace('{week_start}', week_start.isoformat())
                .replace('{week_no}', str(week_no))
                .replace('{week_no_cn}', '第%d周' % week_no)
                .replace('{date}', today.isoformat()))
    return {
        'week_no': week_no,
        'week_start': week_start.isoformat(),
        'week_end': (week_start + timedelta(days=interval - 1)).isoformat(),
        'suggested_filename': filename,
    }


def _parse_date(value: str):
    if not value:
        return None
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y%m%d'):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# git：本周具体提交了哪些论文
# ---------------------------------------------------------------------------

def stem_from_repo_path(line: str) -> str:
    """从 git 给出的仓库相对路径求论文主干。

    与 `_paper_stem` 同一套规则：`01-raw/<标题>.pdf` 用文件名，
    `03-notes/<标题>/<任意笔记>.md` 用**文件夹名**（否则 `精读.md` 会被当成一篇论文）。
    """
    parts = Path(line).parts
    if not parts:
        return ''
    if len(parts) >= 3 and parts[0] in ('03-notes', '04-equation_problem', '06-translation'):
        return clean_stem(parts[1])
    return clean_stem(Path(parts[-1]).stem)


def git_added_papers(vault: Path, since: date, until: date):
    """用 git 提交记录判定"本周新增了哪些论文"，比文件 mtime 可靠。

    只看 `--diff-filter=A`（新增文件），因此批量解析、clone、同步都不会污染结果。

    Returns:
        {'available': bool, 'error': str|None, 'commits': int, 'files': [...],
         'stems': [...], 'roots': {root: count}}
    """
    result = {'available': False, 'error': None, 'commits': 0,
              'files': [], 'stems': [], 'roots': {}}
    if not (vault / '.git').is_dir():
        result['error'] = 'vault 不是 git 仓库'
        return result

    # 取 --since 前一天的 00:00，避免时区/时间边界漏掉当天提交
    since_dt = datetime.combine(since - timedelta(days=1), datetime.min.time())
    # core.quotePath=false：否则中文路径（如 精读.md）会被 git 转义成 \347\262\276...
    cmd = [
        'git', '-C', str(vault), '-c', 'core.quotePath=false', 'log',
        '--diff-filter=A', '--name-only',
        '--date=short', '--pretty=format:@@%ad\t%s', '--no-merges',
        '--since=%s' % since_dt.strftime('%Y-%m-%d %H:%M:%S'),
    ]
    try:
        # commit message 是仓库里的原始字节，按 utf-8 宽松解码，避免 Windows 上 GBK 报错
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        result['error'] = '%s: %s' % (type(e).__name__, e)
        return result
    if proc.returncode != 0:
        err = (proc.stderr or b'').decode('utf-8', 'replace').strip()[:200]
        result['error'] = err or 'git log 失败'
        return result

    result['available'] = True
    commit_date, commit_subject = None, ''
    seen = {}
    for raw in proc.stdout.decode('utf-8', 'replace').splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('@@'):
            head = line[2:]
            commit_date, _, commit_subject = head.partition('\t')
            commit_date = commit_date.strip()
            result['commits'] += 1
            continue
        if commit_date:
            d = _parse_date(commit_date)
            if d and not (since <= d <= until):
                continue
        # git 可能对含特殊字符的路径加引号
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        # 只关心论文相关的真实文件
        if not line.lower().endswith(('.pdf', '.md', '.markdown')):
            continue
        if '/images/' in line or Path(line).name.upper() == 'README.MD':
            continue
        top = Path(line).parts[0] if Path(line).parts else ''
        if top not in ('01-raw', '02-markdown', '03-notes'):
            # 04/06/08-reading 属于过程与成果资产，不算"本周新增论文"
            continue

        name = Path(line).stem if Path(line).suffix.lower() != '.pdf' else Path(line).stem
        key = stem_from_repo_path(line)
        if not key:
            continue
        entry = seen.setdefault(key, {
            'stem': key,
            'files': [],
            'first_commit_date': commit_date,
            'commit_subject': commit_subject,
        })
        entry['files'].append({'path': line, 'root': top})
        result['roots'][top] = result['roots'].get(top, 0) + 1

    result['files'] = sorted(seen.values(), key=lambda x: x['first_commit_date'] or '')
    result['stems'] = [e['stem'] for e in result['files']]
    return result


# ---------------------------------------------------------------------------
# 素材扫描
# ---------------------------------------------------------------------------

def _recent_files(directory: Path, since: date, until: date, tolerance_days: int = 1):
    """按修改时间筛出窗口内的文件。

    mtime 与论文日期可能因时区/解析时间差 1 天，这里留 `tolerance_days`
    容差，宁可多给素材也不要漏掉刚解析完的论文。
    """
    low = since - timedelta(days=tolerance_days)
    high = until + timedelta(days=tolerance_days)
    found = []
    if not directory.is_dir():
        return found
    for path in directory.rglob('*'):
        if not path.is_file():
            continue
        if path.name.startswith('.') or path.name.startswith('_'):
            continue
        # 目录说明文件不是论文素材
        if path.name.lower() in ('readme.md', 'index.md'):
            continue
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
        except OSError:
            continue
        if low <= mtime <= high:
            found.append((path, mtime))
    return found


def _paper_stem(path: Path, root: Path) -> str:
    """求文件所属论文的稳定主干。

    `03-notes` 的约定是 `03-notes/<论文标题>/<任意笔记>.md`，所以当文件位于
    论文子目录内时，主干取**文件夹名**，否则取文件自己的名字。
    这样 `02-markdown/<标题>.md` 与 `03-notes/<标题>/精读.md` 才能归到同一篇。
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return clean_stem(path.stem)

    parts = rel.parts
    if len(parts) >= 2:
        # 位于某个论文子目录内 -> 用子目录名（相对 root 的第一层）
        return clean_stem(parts[0])
    return clean_stem(path.stem)


DAILY_INPUT = '08-daily'


def _resolve_input_dir(config: dict, vault: Path, rel: str) -> Path:
    """inputs 里可能混有 08-daily 与普通素材目录。"""
    if rel == DAILY_INPUT or rel == (config.get('daily_dir') or DAILY_INPUT):
        return resolve_vault_subdir(config, 'daily_dir', DAILY_INPUT, vault)
    return resolve_vault_subdir(config, rel, rel, vault)


def _daily_topics(daily: list, limit: int = 30):
    """把窗口内所有逐日检索结果聚成"本周课题全景"。

    这是宏观阶段成本最低的入口：只看题目与领域，不读正文。
    """
    domain_counts = {}
    by_domain = {}
    seen_titles = set()
    scored = []

    for record in daily:
        path = record.get('search_result')
        if not path:
            continue
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            continue
        for item in (data.get('all_papers') or []):
            title = (item.get('title') or '').strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            domain = item.get('matched_domain') or '未分类'
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            by_domain.setdefault(domain, []).append({
                'title': title,
                'score': item.get('score'),
                'source': item.get('source', ''),
                'already_known': bool(item.get('already_known')),
            })
            scored.append((item.get('score') or 0, title, domain))

    scored.sort(reverse=True)
    return {
        'titles_total': len(seen_titles),
        'domain_counts': dict(sorted(domain_counts.items(), key=lambda kv: -kv[1])),
        'by_domain': {k: v[:limit] for k, v in by_domain.items()},
        'top_scored': [{'title': t, 'score': s, 'domain': d} for s, t, d in scored[:limit]],
    }


def _read_text_head(path: Path, max_chars: int = 1200) -> str:
    """读文件开头一小段，用于给 agent 判断"要不要深读"，避免整文件灌进上下文。"""
    try:
        raw = path.read_text(encoding='utf-8-sig', errors='replace')
    except OSError:
        return ''
    head = raw[:max_chars]
    if len(raw) > max_chars:
        head += '\n...(已截断)'
    return head


def extract_equation_questions(child: Path, stem: str):
    """04-equation_problem/<论文>/ 的针对性提取。

    只挑**用户疑问**类文件（文件名含 疑问/问题/不懂/待解决）并给标题骨架，
    不读 24–48KB 的推导正文。
    """
    question_markers = ('疑问', '问题', '不懂', '待解决', '困惑', 'question')
    out = {'files': [], 'question_files': [], 'headings': []}
    for p in sorted(child.rglob('*')):
        if not p.is_file():
            continue
        out['files'].append(p.name)
        if any(m in p.name.lower() for m in question_markers):
            out['question_files'].append({
                'path': str(p),
                'head': _read_text_head(p, 1500),
            })
        elif p.suffix.lower() in ('.md', '.markdown'):
            try:
                text = p.read_text(encoding='utf-8-sig', errors='replace')
            except OSError:
                continue
            out['headings'].append({
                'file': p.name,
                'headings': [ln.strip() for ln in text.splitlines()
                             if ln.startswith('#')][:25],
            })
    return out


def extract_reading_qa(child: Path):
    """08-reading/<论文>/ 的针对性提取：问答记录——判断卡点与掌握程度的最佳信号。"""
    out = {'files': [], 'qa_heads': [], 'headings': []}
    for p in sorted(child.rglob('*')):
        if not p.is_file():
            continue
        out['files'].append(p.name)
        if p.suffix.lower() not in ('.md', '.markdown'):
            continue
        try:
            text = p.read_text(encoding='utf-8-sig', errors='replace')
        except OSError:
            continue
        heads = [ln.strip() for ln in text.splitlines() if ln.startswith('#')]
        out['headings'].append({'file': p.name, 'headings': heads[:30],
                                'kb': round(len(text.encode('utf-8')) / 1024, 1)})
        # 问答类文件给出开头，便于 agent 判断是否要深读
        if any(m in p.name for m in ('精读笔记', '问答', 'qa', '过程', '思考')):
            out['qa_heads'].append({'path': str(p), 'head': _read_text_head(p, 1500)})
    return out


def scan_context_assets(vault: Path, config: dict, stems):
    """扫描 context_dirs 的资产与提取片段。

    - 04-equation_problem：`extract=question_files_only` —— 疑问文件 + 标题骨架
    - 08-reading：`extract=qa_log` —— 问答记录
    - 06-translation：`extract=existence_only` —— 只列文件名
    """
    weekly = weekly_config(config)
    dirs = weekly.get('context_dirs') or []
    result = {}
    wanted = set(stems)
    for rel in dirs:
        directory = resolve_vault_subdir(config, rel, rel, vault)
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue
            stem = clean_stem(child.name)
            if wanted and stem not in wanted:
                continue
            entry = result.setdefault(stem, {})
            files = sorted(p.name for p in child.rglob('*') if p.is_file())
            if not files:
                continue
            slot = entry.setdefault(rel, {'extract': 'existence_only', 'files': files})
            if rel == config.get('equation_dir', '04-equation_problem'):
                slot['extract'] = 'question_files_only'
                slot.update(extract_equation_questions(child, stem))
            elif rel == config.get('reading_dir', '08-reading'):
                slot['extract'] = 'qa_log'
                slot.update(extract_reading_qa(child))
    return result


def collect_materials(config: dict, vault: Path, since: date, until: date, history_size: int):
    """汇总窗口内的检索日报、论文素材、git 新增论文与上下文资产。"""
    daily_dir = resolve_vault_subdir(config, 'daily_dir', '08-daily', vault)
    inputs = weekly_config(config).get('inputs') or ['02-markdown', '03-notes']

    # 1) 每日检索（目录名即日期）
    daily = []
    if daily_dir.is_dir():
        for folder in sorted(daily_dir.iterdir()):
            if not folder.is_dir():
                continue
            folder_date = _parse_date(folder.name)
            if not folder_date or not (since <= folder_date <= until):
                continue
            record = {'date': folder_date.isoformat(), 'dir': str(folder)}
            for name in ('search_result.json', config.get('daily_note_name', '今日检索') + '.md'):
                candidate = folder / name
                if candidate.is_file():
                    key = 'search_result' if name.endswith('.json') else 'daily_note'
                    record[key] = str(candidate)
            if 'search_result' in record:
                try:
                    data = json.loads(Path(record['search_result']).read_text(encoding='utf-8-sig'))
                    record['total_unique'] = data.get('total_unique', 0)
                    record['titles'] = [p.get('title', '') for p in (data.get('all_papers') or [])][:50]
                except Exception:
                    pass
            daily.append(record)

    # 2) 用户产物：03-notes 精读笔记 + 02-markdown 解析结果（按 inputs 优先级标注）
    matched = {}
    for order, rel in enumerate(inputs):
        if rel == DAILY_INPUT or rel == (config.get('daily_dir') or DAILY_INPUT):
            continue  # 日报已由上面 daily 处理
        directory = _resolve_input_dir(config, vault, rel)
        for path, mtime in _recent_files(directory, since, until):
            stem = _paper_stem(path, directory)
            if not stem:
                continue
            entry = matched.setdefault(stem, {
                'stem': stem, 'files': [], 'mtime': '',
                'roots': [], 'priority': len(inputs) + 1,
            })
            entry['files'].append({
                'path': str(path), 'kind': path.suffix.lstrip('.'), 'root': rel,
            })
            entry['mtime'] = max(entry['mtime'], mtime.isoformat())
            if rel not in entry['roots']:
                entry['roots'].append(rel)
            entry['priority'] = min(entry['priority'], order + 1)

    # 3) git：本周具体新增了哪些论文
    git_info = {'available': False, 'error': 'disabled'}
    weekly = weekly_config(config)
    if weekly.get('use_git_history', True):
        git_info = git_added_papers(vault, since, until)        # 用 git 结果补充/标注素材（git 能捞到 mtime 已被污染的旧文件）
        for item in git_info.get('files', []):
            stem = item['stem']
            entry = matched.setdefault(stem, {
                'stem': stem, 'files': [], 'mtime': '', 'roots': [],
                'priority': len(inputs) + 1,
            })
            entry['git_added'] = item['first_commit_date']
            entry['git_subject'] = item['commit_subject']
            for f in item['files']:
                entry['files'].append({'path': f['path'], 'kind': 'git', 'root': f['root'],
                                       'git_added': item['first_commit_date']})
                if f['root'] not in entry['roots']:
                    entry['roots'].append(f['root'])

    # 4) 上下文资产与提取片段（04 疑问 / 08-reading 问答 / 06 翻译存在性）
    assets = scan_context_assets(vault, config, matched.keys())
    for stem, entry in matched.items():
        if stem in assets:
            entry['assets'] = assets[stem]

    # 5) 学习进展信号：本周你在哪些论文上做了互动式精读
    reading_dir_name = config.get('reading_dir', '08-reading')
    reading_log = []
    for stem, entry in matched.items():
        slot = (entry.get('assets') or {}).get(reading_dir_name)
        if slot:
            reading_log.append({
                'stem': stem,
                'files': slot.get('files', []),
                'files_with_headings': slot.get('headings', []),
                'qa_samples': slot.get('qa_heads', []),
            })

    papers = sorted(
        matched.values(),
        key=lambda x: (x.get('git_added') or x['mtime'] or '', x['priority']),
        reverse=True,
    )

    return {
        'window': {'since': since.isoformat(), 'until': until.isoformat()},
        'vault': str(vault),
        'daily_dirs': daily,
        # 第一阶段入口：本周课题全景（只看题目/领域，不读正文）
        'daily_topics': _daily_topics(daily),
        'papers': papers[:history_size],
        'papers_total': len(papers),
        'daily_total': len(daily),
        # 第二阶段入口：你不理解的地方（04 疑问 + 08-reading 问答）
        'reading_log': reading_log,
        'git_added': {
            'available': git_info.get('available', False),
            'error': git_info.get('error'),
            'commits': git_info.get('commits', 0),
            'stems': git_info.get('stems', []),
            'roots': git_info.get('roots', {}),
            'files': git_info.get('files', []),
        },
    }


def _set_yaml_key(text: str, dotted: str, value: str) -> str:
    """在保留注释与排版的前提下，更新 `weekly:` 段下的单个标量键。"""
    section, key = dotted.split('.', 1)
    lines = text.split('\n')
    in_section = False
    for i, line in enumerate(lines):
        if re.match(r'^%s:\s*$' % re.escape(section), line):
            in_section = True
            continue
        if in_section:
            if line and not line.startswith(' ') and not line.startswith('#'):
                break  # 进入下一个顶层段落
            m = re.match(r'^(\s+)(%s):(\s*)(.*)$' % re.escape(key), line)
            if m:
                lines[i] = '%s%s:%s"%s"' % (m.group(1), key, ' ' if not m.group(3) else m.group(3), value)
                return '\n'.join(lines)
    # 段落存在但键缺失 -> 追加到段落末尾
    if in_section:
        for i, line in enumerate(lines):
            if re.match(r'^%s:\s*$' % re.escape(section), line):
                lines.insert(i + 1, '  %s: "%s"' % (key, value))
                return '\n'.join(lines)
    return text


def mark_run(config_path: Path, today: date, schedule: dict):
    """写回 weekly.start_date / last_run。"""
    text = config_path.read_text(encoding='utf-8-sig')
    updated = text
    if not schedule['start_date']:
        updated = _set_yaml_key(updated, 'weekly.start_date', today.isoformat())
    updated = _set_yaml_key(updated, 'weekly.last_run', today.isoformat())

    try:
        import yaml
        parsed = yaml.safe_load(updated) or {}
        weekly = parsed.get('weekly') or {}
        if weekly.get('last_run') != today.isoformat():
            raise ValueError('last_run 未写入')
        if not weekly.get('start_date'):
            raise ValueError('start_date 缺失')
    except Exception as e:
        raise RuntimeError('写回 weekly 日期失败，配置未修改：%s' % e)

    backup = config_path.with_suffix(config_path.suffix + '.bak-%s' % datetime.now().strftime('%Y%m%d-%H%M%S'))
    shutil.copy2(config_path, backup)
    config_path.write_text(updated, encoding='utf-8')
    return backup


def main():
    _force_utf8_stdio()

    parser = argparse.ArgumentParser(description='paper-weekly: schedule check + material collection')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--vault', type=str, default=None)
    parser.add_argument('--date', type=str, default=None, help='基准日期 YYYY-MM-DD（默认今天）')
    parser.add_argument('--force', action='store_true', help='忽略间隔，直接收集')
    parser.add_argument('--mark-run', action='store_true', help='分析完成后写回 start_date / last_run')
    parser.add_argument('--output', type=str, default='weekly_manifest.json')
    parser.add_argument('--json', action='store_true', help='只输出 JSON')

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        config_path = find_config_path(args.config)
    except FileNotFoundError as e:
        print('ERROR: %s' % e, file=sys.stderr)
        return 1

    today = _parse_date(args.date) if args.date else date.today()
    if today is None:
        print('ERROR: --date 需要 YYYY-MM-DD 格式', file=sys.stderr)
        return 1

    schedule = compute_schedule(config, today)
    week = week_context(config, today)

    if args.mark_run:
        try:
            backup = mark_run(config_path, today, schedule)
        except RuntimeError as e:
            print('ERROR: %s' % e, file=sys.stderr)
            return 1
        print(json.dumps({'marked': True, 'date': today.isoformat(), 'backup': str(backup),
                          'config_path': str(config_path)}, ensure_ascii=False, indent=2))
        return 0

    if not schedule['enabled']:
        print(json.dumps({'enabled': False, 'message': 'weekly.enabled 为 false，跳过周期分析'},
                         ensure_ascii=False, indent=2))
        return 0

    if not schedule['due'] and not args.force:
        print(json.dumps({
            'due': False,
            'schedule': schedule,
            'week': week,
            'message': '未到期：距上次分析 %s 天（间隔 %s 天），下次 %s'
                       % (schedule['days_since_last'], schedule['interval_days'], schedule['next_due']),
        }, ensure_ascii=False, indent=2))
        return 0

    try:
        vault = resolve_vault_path(config, args.vault)
    except ValueError as e:
        print('ERROR: %s' % e, file=sys.stderr)
        return 1
    if not vault.is_dir():
        print('ERROR: vault 路径不存在：%s' % vault, file=sys.stderr)
        return 1

    weekly = weekly_config(config)
    window_days = int(weekly.get('window_days', 7) or 7)
    history_size = int(weekly.get('history_size', 40) or 40)
    since = today - timedelta(days=window_days - 1)

    materials = collect_materials(config, vault, since, today, history_size)
    research_dir = resolve_vault_subdir(config, 'research_dir', '07-research', vault)
    manifest = {
        'skill': 'paper-weekly',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'schedule': schedule,
        'week': week,
        'theme_first': bool(weekly.get('theme_first', True)),
        'config_path': str(config_path),
        'research_dir': str(research_dir),
        'report_path': str(research_dir / week['suggested_filename']),
        'domains': list((config.get('research_domains') or {}).keys()),
        'domain_keywords': {
            name: (cfg.get('keywords') or [])[:12]
            for name, cfg in (config.get('research_domains') or {}).items()
        },
        **materials,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    else:
        print('周期状态：%s' % ('到期，可以分析' if schedule['due'] else '未到期（--force 可强制）'))
        print('本次为第 %d 周（%s ~ %s）' % (week['week_no'], week['week_start'], week['week_end']))
        print('建议报告名：%s' % week['suggested_filename'])
        print('时间窗口：%s ~ %s' % (since.isoformat(), today.isoformat()))
        print('检索日报：%d 天' % materials['daily_total'])
        print('论文素材：%d 篇（清单上限 %d）' % (materials['papers_total'], history_size))
        git_added = materials['git_added']
        if git_added['available']:
            print('git 本周新增论文：%d 篇（来自 %d 个提交）'
                  % (len(git_added['stems']), git_added['commits']))
        else:
            print('git 判定不可用：%s' % git_added.get('error'))
        print('素材清单：%s' % out_path)
        if not materials['papers'] and not materials['daily_dirs']:
            print('提示：窗口内没有找到素材。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
