#!/usr/bin/env python3
"""
paper-sgd-reading 的路径解析与笔记落盘

**自包含**：本脚本不依赖其他 skill 的代码，可独立复制到任何位置运行。

解耦要点：目录与文件名一律从 `config.yaml` 读取，不在 SKILL.md 里写死路径。

  * 过程材料（问答记录、全局推理草稿） -> `reading_dir`（默认 08-reading）/<论文标题>/
  * 定稿（全局推理、公式伴读等）        -> `equation_dir`（默认 04-equation_problem）/<论文标题>/

配置查找顺序（第一个存在的生效）：
  1. `--config` 显式指定
  2. `$PAPER_SKILLS_CONFIG`
  3. `$PAPER_READING_CONFIG`
  4. `<skill 目录>/config.yaml`
  5. 从 `$PAPER_WORKSPACE_PATH` 查找 `<workspace>/.claude/skills/config.yaml`
  6. 从当前工作目录逐级向上找 `.claude/skills/config.yaml`

workspace 路径：`--workspace` -> `$PAPER_WORKSPACE_PATH` -> 配置的 `workspace_path`
-> 自动识别当前项目根目录。

用法：
    python outputs.py --title "<论文标题>" --kind process
    python outputs.py --title "<论文标题>" --kind final --name 全局推理 --prepare
    python outputs.py --title "<论文标题>" --kind process --append-file chunk.md
    python outputs.py --title "<论文标题>" --emit-asset-links --from 03-notes/<标题>/精读.md
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]   # <skill-dir>
PAPER_SKILLS_ROOT = SKILL_ROOT.parent              # <project>/.claude/skills
WORKSPACE_MARKERS = ('01-raw', '02-markdown', '03-notes', '04-equation_problem')

DEFAULT_DIRS = {
    'equation_dir': '04-equation_problem',
    'translation_dir': '06-translation',
    'reading_dir': '08-reading',
    'notes_dir': '03-notes',
    'papers_dir': '01-raw',
    'markdown_dir': '02-markdown',
    'research_dir': '07-research',
    'daily_dir': '08-daily',
}

# 固定的产出文件名（保持不变）
PROCESS_NOTE_SUFFIX = '_精读笔记.md'          # 交互模式问答记录
PROCESS_DRAFT_SUFFIX = '_全局推理_过程.md'     # 全局推理中间草稿
DEFAULT_FINAL_NAME = '全局推理'                # 定稿默认名（对齐 04 现有命名）
ALLOWED_FINAL_NAMES = ('全局推理', '公式伴读', '附录完整推导')


def _force_utf8_stdio():
    """Windows 控制台默认 GBK，打印中文路径会抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


_force_utf8_stdio()


# ---------------------------------------------------------------------------
# 配置与路径解析（自包含实现）
# ---------------------------------------------------------------------------

def _config_candidates(explicit=None):
    """按优先级列出候选配置。

    顺序要点：**眼下安装位置旁边的配置优先于靠 workspace 反推的配置**，
    这样 skill 放在 `.claude/skills/` 时会用 `.claude/skills/config.yaml`，
    而不会去命中机器上别处的另一份副本。
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for env in ('PAPER_SKILLS_CONFIG', 'PAPER_READING_CONFIG'):
        value = os.environ.get(env)
        if value:
            candidates.append(Path(value).expanduser())

    # 1) 配置与 skill 同级（推荐安装形态：.claude/skills/config.yaml）
    candidates.append(SKILL_ROOT.parent / 'config.yaml')
    # 2) skill 目录内自带配置
    candidates.append(SKILL_ROOT / 'config.yaml')

    # 3) 从 workspace 查找：<workspace>/.claude/skills/config.yaml
    workspace_env = os.environ.get('PAPER_WORKSPACE_PATH')
    if workspace_env:
        candidates.append(Path(workspace_env).expanduser() / '.claude' / 'skills' / 'config.yaml')

    # 4) 从当前工作目录逐级向上找
    here = Path.cwd()
    for parent in [here, *here.parents]:
        candidates.append(parent / '.claude' / 'skills' / 'config.yaml')
    return candidates


def find_config_path(explicit=None):
    for path in _config_candidates(explicit):
        if path.is_file():
            return path
    raise FileNotFoundError(
        '未找到配置文件，已尝试：%s' % ', '.join(str(p) for p in _config_candidates(explicit))
    )


def load_config(explicit=None):
    import yaml

    path = find_config_path(explicit)
    with open(path, 'r', encoding='utf-8-sig') as f:
        return yaml.safe_load(f) or {}


def resolve_workspace_path(config, workspace=None):
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


def sanitize_paper_title(title, max_length=120):
    """论文稳定主干：空白与 ASCII 连字符转下划线，与工作区命名风格一致。"""
    text = str(title or '').strip()
    text = re.sub(r'[\\/:*?"<>|\t\n\r]+', '_', text)
    text = re.sub(r'[,\u3001;；]+', '_', text)
    text = re.sub(r'[\s\-]+', '_', text)
    text = re.sub(r'_{2,}', '_', text).strip(' ._')
    if len(text) > max_length:
        text = text[:max_length].rstrip(' ._')
    return text


def resolve_workspace_subdir(config, key, workspace):
    """目录名来自配置；缺失时用该目录的标准名。"""
    raw = (config or {}).get(key) or DEFAULT_DIRS.get(key)
    if not raw:
        raise ValueError('未知目录键：%s' % key)
    path = Path(str(raw).strip())
    return path if path.is_absolute() else workspace / path


# ---------------------------------------------------------------------------
# 目标解析与写盘
# ---------------------------------------------------------------------------

def resolve_targets(config, workspace, title, kind, name=None):
    stem = sanitize_paper_title(title)
    if not stem:
        raise ValueError('论文标题无法生成有效主干')

    if kind == 'process':
        base = resolve_workspace_subdir(config, 'reading_dir', workspace)
        return {
            'kind': 'process',
            'stem': stem,
            'dir': base / stem,
            'note_path': base / stem / (stem + PROCESS_NOTE_SUFFIX),
            'draft_path': base / stem / (stem + PROCESS_DRAFT_SUFFIX),
        }

    base = resolve_workspace_subdir(config, 'equation_dir', workspace)
    final_name = (name or DEFAULT_FINAL_NAME).strip()
    if final_name.endswith('.md'):
        final_name = final_name[:-3]
    return {
        'kind': 'final',
        'stem': stem,
        'dir': base / stem,
        'final_path': base / stem / (final_name + '.md'),
        'final_name': final_name,
    }


def _target_file(targets, mode):
    if targets['kind'] == 'process':
        return targets['note_path'] if mode != 'draft' else targets['draft_path']
    return targets['final_path']


def prepare(targets, mode):
    """创建目录与骨架文件；已存在则不覆盖。"""
    targets['dir'].mkdir(parents=True, exist_ok=True)
    wanted = _target_file(targets, mode)
    if wanted.exists():
        return {'created': False, 'path': str(wanted), 'reason': '已存在，未覆盖'}
    wanted.write_text('', encoding='utf-8')
    return {'created': True, 'path': str(wanted), 'created_files': [str(wanted)]}


def append(targets, mode, text):
    """追加内容——边讲边记的标准动作。"""
    path = _target_file(targets, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, 'a', encoding='utf-8') as f:
        if is_new:
            f.write('# %s\n\n' % path.stem)
            f.write('- 生成时间：%s\n\n---\n\n' % datetime.now().strftime('%Y-%m-%d %H:%M'))
        f.write(text.rstrip('\n') + '\n\n')
    return {'appended_to': str(path), 'bytes': len(text.encode('utf-8'))}


def collect_assets(config, workspace, title):
    """找出这篇论文在各目录下的资产（04 定稿 / 06 翻译 / 08-reading 过程 / 03-notes 精读）。"""
    stem = sanitize_paper_title(title)
    found = {}
    for key, label in (('equation_dir', '公式推理'), ('translation_dir', '翻译'),
                       ('reading_dir', '思考过程'), ('notes_dir', '精读笔记')):
        folder = resolve_workspace_subdir(config, key, workspace) / stem
        if not folder.is_dir():
            continue
        # 跳过 .gitkeep 等占位/隐藏文件
        files = sorted(p for p in folder.rglob('*')
                       if p.is_file() and not p.name.startswith('.'))
        if files:
            found[key] = {'label': label, 'dir': folder, 'files': files}
    return stem, found


def emit_asset_links(config, workspace, title, from_path=None):
    """生成可粘贴的"论文资产"小节（标准 Markdown 相对链接）。只输出文本，不改文件。"""
    stem, found = collect_assets(config, workspace, title)
    if not found:
        return '', stem

    origin = Path(from_path).parent if from_path else workspace
    lines = ['## 论文资产', '']
    for key in ('equation_dir', 'translation_dir', 'reading_dir', 'notes_dir'):
        item = found.get(key)
        if not item:
            continue
        links = []
        for f in item['files']:
            try:
                rel = os.path.relpath(f, origin).replace('\\', '/')
            except ValueError:
                rel = f.as_posix()
            links.append('[%s](%s)' % (f.name, rel))
        lines.append('- **%s**：%s' % (item['label'], '、'.join(links)))
    lines.append('')
    return '\n'.join(lines), stem


def main():
    parser = argparse.ArgumentParser(description='paper-sgd-reading: resolve paths and write session notes')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--workspace', type=str, default=None)
    parser.add_argument('--title', type=str, required=True, help='论文标题（用于生成稳定主干）')
    parser.add_argument('--kind', choices=['process', 'final'], default='process',
                        help='process=过程材料（reading_dir）；final=定稿（equation_dir）')
    parser.add_argument('--name', type=str, default=None,
                        help='定稿文件名（默认 全局推理；常用 公式伴读 / 附录完整推导）')
    parser.add_argument('--mode', choices=['note', 'draft'], default='note',
                        help='process 模式下写问答记录(note)还是全局推理草稿(draft)')
    parser.add_argument('--prepare', action='store_true', help='创建目录与骨架（不覆盖已有文件）')
    parser.add_argument('--append', action='store_true', help='把 stdin 内容追加到目标文件')
    parser.add_argument('--append-file', type=str, default=None, help='把该文件内容追加到目标文件')
    parser.add_argument('--emit-asset-links', action='store_true',
                        help='输出可粘贴的"论文资产"相对链接小节（不修改任何文件）')
    parser.add_argument('--from', dest='from_path', type=str, default=None,
                        help='相对链接的基准文件（如 03-notes/<标题>/精读.md）')
    parser.add_argument('--json', action='store_true', help='只输出 JSON')

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        config_path = find_config_path(args.config)
    except FileNotFoundError as e:
        print('ERROR: %s' % e, file=sys.stderr)
        return 1

    try:
        workspace = resolve_workspace_path(config, args.workspace)
    except ValueError as e:
        print('ERROR: %s' % e, file=sys.stderr)
        return 1
    if not workspace.is_dir():
        print('ERROR: 论文工作区路径不存在：%s' % workspace, file=sys.stderr)
        return 1

    if args.kind == 'final' and args.name:
        base = args.name[:-3] if args.name.endswith('.md') else args.name
        if base not in ALLOWED_FINAL_NAMES:
            print('提示：定稿名 %r 不在常规命名 %s 内；若确有需要可继续使用。'
                  % (base, list(ALLOWED_FINAL_NAMES)), file=sys.stderr)

    try:
        targets = resolve_targets(config, workspace, args.title, args.kind, args.name)
    except ValueError as e:
        print('ERROR: %s' % e, file=sys.stderr)
        return 1

    if args.emit_asset_links:
        block, stem = emit_asset_links(config, workspace, args.title, args.from_path)
        if args.json:
            print(json.dumps({'stem': stem, 'block': block}, ensure_ascii=False, indent=2))
        elif block:
            print(block)
        else:
            print('（这篇论文在各目录下还没有任何资产）', file=sys.stderr)
        return 0

    if args.append or args.append_file:
        if args.append_file:
            try:
                text = Path(args.append_file).read_text(encoding='utf-8-sig')
            except OSError as e:
                print('ERROR: 无法读取 %s: %s' % (args.append_file, e), file=sys.stderr)
                return 1
        else:
            text = sys.stdin.read().lstrip('\ufeff')
        if not text.strip():
            print('ERROR: 要追加的内容为空。', file=sys.stderr)
            return 1
        result = append(targets, args.mode, text)
        result.update({'config_path': str(config_path), 'workspace': str(workspace)})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.prepare:
        result = prepare(targets, args.mode)
        result.update({'config_path': str(config_path), 'workspace': str(workspace),
                       'targets': {k: str(v) for k, v in targets.items()}})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(json.dumps({
        'config_path': str(config_path),
        'workspace': str(workspace),
        'targets': {k: str(v) for k, v in targets.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
