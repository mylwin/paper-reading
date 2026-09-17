#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""localize_markdown_images —— 把 `02-markdown` 里解析服务（如 MinerU）的远程图片落到本地

背景：PDF 解析服务（MinerU 等）输出的 Markdown 里，图片往往是解析服务的 CDN 链接
（例如 `https://cdn-mineru.openxlab.org.cn/result/.../*.jpg`）。这类链接一旦过期或
离线就全部失效，也不符合工作区「跨目录引用使用指向真实文件的相对路径」的规范。

本脚本把 `markdown_dir`（默认 `02-markdown`）下 Markdown 中**允许来源的远程图片**
下载到本地并改写为相对路径：

```text
02-markdown/
└── YYYY-MM/
    ├── <论文标题>.md                     # 链接改写为 images/<论文标题>/fig1.jpg
    └── images/
        └── <论文标题>/
            ├── _sources.md               # 本地文件名 -> 原始 URL（可追溯）
            ├── fig1.jpg
            └── fig2.jpg
```

规则：
  * 只处理 `image_localize_hosts` 允许的来源（默认 MinerU CDN），不盲抓任意远程图片
  * 按 Markdown 中出现顺序命名为 `fig1.<ext>`、`fig2.<ext>`…，扩展名按**文件魔数**判定
  * 已存在同名文件则跳过（幂等，不重复下载）；下载失败保留原远程链接，不破坏文档
  * 只读写 `markdown_dir`，不碰 `01-raw` / `03-notes` / `06-translation`

用法：
    python localize_markdown_images.py --dry-run
    python localize_markdown_images.py
    python localize_markdown_images.py --force        # 重新下载已存在的图片
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_config import (  # noqa: E402
    load_config,
    rel_link,
    resolve_workspace_path,
    resolve_workspace_subdir,
)

USER_AGENT = "paper-skills/0.1 (localize-markdown-images; +local research workspace)"
DEFAULT_HOSTS = ('cdn-mineru.openxlab.org.cn',)
IMAGE_LINK_RE = re.compile(r'(!\[[^\]]*\]\()(?P<url>https?://[^)\s]+)(\))')

MAGIC = (
    (b'\xff\xd8\xff', '.jpg'),
    (b'\x89PNG\r\n\x1a\n', '.png'),
    (b'GIF87a', '.gif'),
    (b'GIF89a', '.gif'),
    (b'BM', '.bmp'),
)
SKIP_NAMES = ('readme.md', 'index.md')


def _force_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


def detect_extension(data: bytes, url: str) -> str:
    """按文件魔数判断图片扩展名，失败时回落到 URL 后缀。"""
    for magic, ext in MAGIC:
        if data.startswith(magic):
            return ext
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return '.webp'
    suffix = Path(url.split('?', 1)[0]).suffix.lower()
    if suffix in ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'):
        return '.jpg' if suffix == '.jpeg' else suffix
    return '.jpg'


def allowed_host(url: str, hosts) -> bool:
    match = re.match(r'https?://([^/]+)/', url)
    if not match:
        return False
    host = match.group(1).lower()
    return any(host == h.lower() or host.endswith('.' + h.lower()) for h in hosts)


def download(url: str, timeout: int) -> tuple:
    """返回 (ok, data_or_message)。"""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except urllib.error.HTTPError as e:
        return False, "HTTP %s" % e.code
    except Exception as e:                                        # noqa: BLE001
        return False, "%s: %s" % (type(e).__name__, e)
    if len(data) < 128:
        return False, "响应过小（%d 字节），疑似占位符" % len(data)
    if detect_extension(data, url) == '.jpg' and not data.startswith(b'\xff\xd8'):
        # 既不是已知图片魔数、也不是 jpeg：拒绝写入
        if not (data[:4] == b'RIFF' and data[8:12] == b'WEBP'):
            return False, "响应不是可识别的图片格式"
    return True, data


def expired_note(path: Path, url: str, reason: str, notes_images: Path, workspace: Path) -> str:
    """图片源已失效时的替代文本：不留断链，并指向 03-notes 下已提取的插图（如有）。"""
    text = '*（插图未本地化：解析服务原链接已失效 %s）*' % reason
    if notes_images and notes_images.is_dir():
        first = sorted(p for p in notes_images.iterdir() if p.is_file() and not p.name.startswith('.'))
        if first:
            text = ('*（插图未本地化：解析服务原链接已失效 %s；本篇已提取插图见 '
                    '[03-notes 图片](%s)）*' % (reason, rel_link(notes_images, path.parent)))
    return text


def localize_file(path: Path, hosts, timeout: int, force: bool, dry_run: bool,
                  replace_failed: bool = False, notes_root: Path = None,
                  workspace: Path = None) -> dict:
    """本地化单个 Markdown 文件中的远程图片，返回统计。"""
    text = path.read_text(encoding='utf-8-sig')
    images_dir = path.parent / 'images' / path.stem
    occurrences = list(IMAGE_LINK_RE.finditer(text))
    report = {'file': str(path), 'total': len(occurrences), 'downloaded': 0,
              'skipped': 0, 'failed': 0, 'ignored': 0, 'images': []}
    if not occurrences:
        return report

    sources = {}
    index = 0
    pieces = []
    cursor = 0
    for match in occurrences:
        url = match.group('url')
        pieces.append(text[cursor:match.start()])
        cursor = match.end()
        if not allowed_host(url, hosts):
            report['ignored'] += 1
            pieces.append(match.group(0))
            continue
        index += 1
        target_dir = images_dir
        existing = sorted(target_dir.glob('fig%d.*' % index)) if target_dir.is_dir() else []
        if existing and not force:
            local = existing[0]
            report['skipped'] += 1
        else:
            if dry_run:
                report['downloaded'] += 1
                pieces.append(match.group(0))
                print('[dry-run] 将下载 %s -> %s' % (url, images_dir / ('fig%d.<ext>' % index)),
                      file=sys.stderr)
                continue
            ok, payload = download(url, timeout)
            if not ok:
                report['failed'] += 1
                report.setdefault('errors', []).append({'url': url, 'reason': payload})
                if replace_failed:
                    notes_images = None
                    if notes_root and notes_root.is_dir() and path.parent.name:
                        notes_images = notes_root / path.parent.name / path.stem / 'images'
                    pieces.append(expired_note(path, url, payload, notes_images, workspace or path.parent))
                else:
                    pieces.append(match.group(0))
                continue
            ext = detect_extension(payload, url)
            target_dir.mkdir(parents=True, exist_ok=True)
            for stale in target_dir.glob('fig%d.*' % index):
                stale.unlink()
            local = target_dir / ('fig%d%s' % (index, ext))
            local.write_bytes(payload)
            report['downloaded'] += 1
        sources[local.name] = url
        report['images'].append({'name': local.name, 'url': url, 'path': str(local)})
        pieces.append('%s%s%s' % (match.group(1), rel_link(local, path.parent), match.group(3)))
    pieces.append(text[cursor:])
    new_text = ''.join(pieces)

    if not dry_run and new_text != text:
        path.write_text(new_text, encoding='utf-8')
    errors = report.get('errors') or []
    if not dry_run and (sources or errors):
        images_dir.mkdir(parents=True, exist_ok=True)
        # 本文件位于 images/<论文主干>/ 下，回到解析结果 Markdown 需要两级
        md_link = rel_link(path, images_dir)
        lines = ['# 图片原始来源', '',
                 '本目录图片由 `localize_markdown_images.py` 从解析服务的远程链接下载而来，',
                 '与 [%s](%s) 中的 `images/%s/figN.*` 一一对应。' % (path.name, md_link, path.stem),
                 '', '| 本地文件 | 原始 URL |', '|---|---|']
        lines += ['| `%s` | %s |' % (name, url) for name, url in sorted(sources.items())]
        if errors:
            lines += ['', '## 未能本地化（源已失效或被拒绝）', '',
                      '| 原始 URL | 失败原因 |', '|---|---|']
            lines += ['| %s | %s |' % (e['url'], e['reason']) for e in errors]
        (images_dir / '_sources.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def main():
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description='Localize remote parser images in markdown_dir')
    parser.add_argument('--config', default=None)
    parser.add_argument('--workspace', default=None)
    parser.add_argument('--hosts', default=None,
                        help='允许本地化的图片主机，逗号分隔（默认取 image_localize_hosts 或 MinerU CDN）')
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--force', action='store_true', help='重新下载已存在的图片')
    parser.add_argument('--replace-failed', action='store_true',
                        help='下载失败时把远程图片语法换成「未本地化」说明文字（避免留下断链）')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    workspace = resolve_workspace_path(config, args.workspace)
    if not workspace.is_dir():
        print('ERROR: workspace 不存在：%s' % workspace, file=sys.stderr)
        return 1

    hosts = [h.strip() for h in (args.hosts or '').split(',') if h.strip()]
    if not hosts:
        hosts = [str(h).strip() for h in (config.get('image_localize_hosts') or []) if str(h).strip()]
    if not hosts:
        hosts = list(DEFAULT_HOSTS)

    markdown_dir = resolve_workspace_subdir(config, 'markdown_dir', '02-markdown', workspace)
    if not markdown_dir.is_dir():
        print('ERROR: markdown_dir 不存在：%s' % markdown_dir, file=sys.stderr)
        return 1

    notes_root = resolve_workspace_subdir(config, 'notes_dir', '03-notes', workspace)
    reports = []
    for path in sorted(markdown_dir.rglob('*.md')):
        if path.name.lower() in SKIP_NAMES or '_sources' in path.name.lower():
            continue
        if 'images' in {part.lower() for part in path.relative_to(markdown_dir).parts}:
            continue
        report = localize_file(path, hosts, args.timeout, args.force, args.dry_run,
                               replace_failed=args.replace_failed, notes_root=notes_root,
                               workspace=workspace)
        if report['total']:
            reports.append(report)

    summary = {
        'workspace': str(workspace),
        'markdown_dir': str(markdown_dir),
        'hosts': hosts,
        'files': len(reports),
        'images': sum(r['total'] for r in reports),
        'downloaded': sum(r['downloaded'] for r in reports),
        'skipped': sum(r['skipped'] for r in reports),
        'failed': sum(r['failed'] for r in reports),
        'ignored': sum(r['ignored'] for r in reports),
        'detail': reports,
        'dry_run': args.dry_run,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for report in reports:
            print('%-70s 共 %d 张：下载 %d / 已存在 %d / 失败 %d / 忽略 %d'
                  % (rel_link(report['file'], workspace), report['total'], report['downloaded'],
                     report['skipped'], report['failed'], report['ignored']))
        print('合计：%d 个文件、%d 张图片（下载 %d / 已存在 %d / 失败 %d / 忽略 %d）%s'
              % (summary['files'], summary['images'], summary['downloaded'], summary['skipped'],
                 summary['failed'], summary['ignored'], '（dry-run）' if args.dry_run else ''))
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
