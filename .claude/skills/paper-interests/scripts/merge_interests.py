#!/usr/bin/env python3
"""
paper-interests 配置合并脚本

把确认后的研究领域写入共享 config.yaml 的 research_domains。

设计取舍：语义工作（理解用户兴趣、扩散出关键词）由 agent 完成，
本脚本只负责安全地合并配置——YAML 结构与注释尽量保留、关键词去重、
写入前自动备份。

用法：
    # 预览（不写盘）：stdin 传入 JSON
    python merge_interests.py --preview < payload.json

    # 写入
    python merge_interests.py --apply < payload.json

    # 从文件读取 payload
    python merge_interests.py --apply --payload payload.json

payload 格式（单个领域或领域数组均可）：
    {
      "domain_name": "世界模型",
      "keywords": ["world model", "video prediction"],
      "arxiv_categories": ["cs.CV", "cs.AI"],
      "priority": 5,
      "replace": false          # true = 覆盖同名领域的已有关键词
    }

输出：JSON 摘要（新增/更新的领域、新增关键词数、备份文件路径、配置文件路径）。
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# paper-interests/scripts -> .claude/skills -> paper-daily/scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'paper-daily' / 'scripts'))

from paper_config import find_config_path, load_config  # noqa: E402


def _force_utf8_stdio():
    """Windows 控制台默认 GBK，打印中文领域名会抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


_force_utf8_stdio()


def _normalize_payloads(payload):
    """payload 可以是单个 dict，也可以是 dict 数组。"""
    if isinstance(payload, dict):
        if 'domains' in payload and isinstance(payload['domains'], list):
            return payload['domains']
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError("payload 必须是对象或对象数组")


def _validate_domain(item):
    name = str(item.get('domain_name') or '').strip()
    if not name:
        raise ValueError("每个领域都需要 domain_name")

    keywords = []
    seen = set()
    for kw in item.get('keywords') or []:
        kw_clean = str(kw).strip()
        key = kw_clean.lower()
        if kw_clean and key not in seen:
            seen.add(key)
            keywords.append(kw_clean)
    if not keywords:
        raise ValueError("领域 '%s' 没有有效关键词" % name)

    categories = [str(c).strip() for c in (item.get('arxiv_categories') or []) if str(c).strip()]

    try:
        priority = int(item.get('priority', 5))
    except (TypeError, ValueError):
        priority = 5
    priority = max(1, min(10, priority))

    return {
        'domain_name': name,
        'keywords': keywords,
        'arxiv_categories': categories,
        'priority': priority,
        'replace': bool(item.get('replace', False)),
    }


def _render_domain_block(domain, indent='  '):
    """把一个领域渲染成 YAML 文本块（保持与手写配置一致的排版）。"""
    lines = ['%s%s:' % (indent, domain['domain_name'])]
    lines.append('%s  keywords:' % indent)
    for kw in domain['keywords']:
        lines.append('%s    - %s' % (indent, _yaml_scalar(kw)))
    if domain['arxiv_categories']:
        lines.append('%s  arxiv_categories:' % indent)
        for cat in domain['arxiv_categories']:
            lines.append('%s    - %s' % (indent, _yaml_scalar(cat)))
    lines.append('%s  priority: %d' % (indent, domain['priority']))
    return lines


def _yaml_scalar(value):
    """按需给字符串加引号，避免中文/冒号/特殊字符破坏 YAML。"""
    text = str(value)
    if text == '' or text != text.strip():
        return json.dumps(text, ensure_ascii=False)
    special = set(':#{}[]&*!|>%@`",\'')
    if any(ch in special for ch in text) or text.lower() in ('true', 'false', 'null', 'yes', 'no'):
        return json.dumps(text, ensure_ascii=False)
    return text


def _format_diff(domains, existing):
    """生成人类可读的变更预览。"""
    lines = []
    for d in domains:
        name = d['domain_name']
        old = existing.get(name)
        if old is None:
            lines.append("+ 新增领域「%s」：%d 个关键词，priority=%d"
                         % (name, len(d['keywords']), d['priority']))
            if d['arxiv_categories']:
                lines.append("    分类：%s" % ', '.join(d['arxiv_categories']))
            for kw in d['keywords']:
                lines.append("      + %s" % kw)
            continue

        old_keywords = [str(k) for k in (old.get('keywords') or [])]
        old_lower = {k.lower() for k in old_keywords}
        additions = [k for k in d['keywords'] if k.lower() not in old_lower]

        if d['replace']:
            lines.append("~ 覆盖领域「%s」：%d -> %d 个关键词"
                         % (name, len(old_keywords), len(d['keywords'])))
        else:
            lines.append("~ 更新领域「%s」：新增 %d 个关键词（原有 %d 个保留）"
                         % (name, len(additions), len(old_keywords)))
        for kw in additions:
            lines.append("      + %s" % kw)
        if old.get('priority') != d['priority']:
            lines.append("      priority: %s -> %s" % (old.get('priority'), d['priority']))
    return lines


def _merge_into_text(text, domains):
    """把领域块合并进配置文本，尽量保留原有注释与排版。

    已存在的领域：替换其整块内容（从 "<indent>名称:" 到下一条同缩进 key）。
    新领域：追加到 research_domains 段末尾。
    """
    lines = text.split('\n')
    out = list(lines)

    for domain in domains:
        name = domain['domain_name']
        block = _render_domain_block(domain)

        # 定位 research_domains: 起始行
        start = None
        for i, line in enumerate(out):
            if line.startswith('research_domains:'):
                start = i
                break

        if start is None:
            # 配置里没有 research_domains，追加整段
            out.append('')
            out.append('research_domains:')
            out.extend(block)
            continue

        # 目标领域是否已存在
        target = None
        for i in range(start + 1, len(out)):
            line = out[i]
            if not line.strip():
                continue
            if line.startswith('  ') and not line.startswith('    ') and line.rstrip().endswith(':'):
                if line.strip()[:-1] == name:
                    target = i
                    break
                continue

        if target is None:
            # 追加到 research_domains 段末尾（遇到下一个顶层 key 前插入）
            insert_at = len(out)
            for i in range(start + 1, len(out)):
                line = out[i]
                if line and not line.startswith(' ') and not line.startswith('#'):
                    insert_at = i
                    break
            # 回退掉段末尾的空行，插入后再补一个空行分隔
            while insert_at > start + 1 and out[insert_at - 1].strip() == '':
                insert_at -= 1
            out[insert_at:insert_at] = block + ['']
            continue

        # 替换已有领域块
        end = len(out)
        for i in range(target + 1, len(out)):
            line = out[i]
            if line and not line.startswith('    ') and line.strip():
                end = i
                break
        while end > target + 1 and out[end - 1].strip() == '':
            end -= 1
        out[target:end] = block

    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description='Merge research interests into the shared config.yaml')
    parser.add_argument('--config', type=str, default=None,
                        help='Config path (default: skill config.yaml, then shared config.yaml)')
    parser.add_argument('--payload', type=str, default=None,
                        help='JSON payload file (default: read from stdin)')
    parser.add_argument('--preview', action='store_true',
                        help='Show what would change without writing')
    parser.add_argument('--apply', action='store_true',
                        help='Write changes to config.yaml (makes a timestamped backup)')
    parser.add_argument('--json', action='store_true',
                        help='Emit machine-readable JSON summary')

    args = parser.parse_args()

    if not args.preview and not args.apply:
        parser.error("需要指定 --preview 或 --apply")

    # 读取 payload（utf-8-sig 兼容 PowerShell 等工具写出的 BOM）
    if args.payload:
        raw_text = Path(args.payload).read_text(encoding='utf-8-sig')
    else:
        raw_text = sys.stdin.read().lstrip('\ufeff')
    if not raw_text.strip():
        print("ERROR: payload 为空。", file=sys.stderr)
        return 1

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print("ERROR: payload 不是合法 JSON: %s" % e, file=sys.stderr)
        return 1

    try:
        domains = [_validate_domain(item) for item in _normalize_payloads(payload)]
    except ValueError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1

    try:
        config_path = find_config_path(args.config)
    except FileNotFoundError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1

    existing = (load_config(args.config).get('research_domains') or {})

    summary = {
        'config_path': str(config_path),
        'mode': 'apply' if args.apply else 'preview',
        'domains': [
            {
                'domain_name': d['domain_name'],
                'keyword_count': len(d['keywords']),
                'priority': d['priority'],
                'is_new': d['domain_name'] not in existing,
                'replace': d['replace'],
            }
            for d in domains
        ],
        'changes': _format_diff(domains, existing),
    }

    if not args.apply:
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print("配置文件：%s" % config_path)
            print("当前领域：%s" % ', '.join(existing.keys()) if existing else "当前领域：（空）")
            print("")
            print("将要执行的变更：")
            for line in summary['changes']:
                print(line)
            print("")
            print("（预览模式，未写入。确认后加 --apply 生效）")
        return 0

    # 写入
    original = config_path.read_text(encoding='utf-8-sig')
    updated = _merge_into_text(original, domains)

    # 写前校验：合并结果必须是合法 YAML
    try:
        import yaml
        parsed = yaml.safe_load(updated)
        merged_domains = (parsed or {}).get('research_domains') or {}
        for d in domains:
            if d['domain_name'] not in merged_domains:
                raise ValueError("合并后缺少领域 '%s'" % d['domain_name'])
    except Exception as e:
        print("ERROR: 合并结果校验失败，未写入：%s" % e, file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    backup_path = config_path.with_suffix(config_path.suffix + '.bak-%s' % timestamp)
    shutil.copy2(config_path, backup_path)

    try:
        config_path.write_text(updated, encoding='utf-8')
    except OSError as e:
        print("ERROR: 写入失败 %s: %s" % (config_path, e), file=sys.stderr)
        return 1

    summary['backup_path'] = str(backup_path)
    summary['domains_after'] = list(merged_domains.keys())

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("已写入：%s" % config_path)
        print("已备份：%s" % backup_path)
        print("当前领域：%s" % ', '.join(merged_domains.keys()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
