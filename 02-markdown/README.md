# 论文解析结果

`02-markdown/` 用于保存从 `01-raw/` 中的论文 PDF 解析得到的 Markdown 文本。

## 目录规则

```text
02-markdown/
├── README.md
├── index.md
└── YYYY-MM/
    ├── README.md
    ├── <论文标题>.md
    └── images/
        └── <论文标题>/
            ├── _sources.md
            ├── fig1.jpg
            └── fig2.png
```

- 月份与文件名主干必须与对应 `01-raw/YYYY-MM/<论文标题>.pdf` 完全一致（月份 = 论文入库月份）。
- 内容应尽量保持论文原文结构，作为阅读和后续整理的内容来源。
- 不在解析文本中混入个人笔记、批注或 AI 分析；相关内容请放入 `03-notes/YYYY-MM/<论文标题>/`。
- 尚未解析的论文不创建空文件，而是在月份 `README.md` 与 `index.md` 中登记为 `未解析`；解析失败登记为 `解析失败` 并写明原因。

## 图片本地化（MinerU 等解析服务的插图）

解析服务输出的 Markdown 里，插图默认是**解析服务的远程 CDN 链接**
（例如 `https://cdn-mineru.openxlab.org.cn/result/<批次>/<hash>.jpg`）。
这类链接会随解析批次被回收而失效，也违反工作区「跨目录引用指向真实文件」的规范，
因此必须本地化到本目录：

| 项 | 规则 |
|---|---|
| 存放位置 | `02-markdown/<YYYY-MM>/images/<论文标题>/figN.<扩展名>` |
| 正文引用 | 相对解析结果文件：`![说明](images/<论文标题>/fig1.jpg)` |
| 命名 | 按图片在 Markdown 中出现的顺序编号 `fig1`、`fig2`…；扩展名按**文件魔数**判定（`.jpg`/`.png`/`.gif`/`.webp`/`.bmp`），不信任 URL 后缀 |
| 溯源 | 同目录 `_sources.md` 记录「本地文件 → 原始 URL」，以及未能本地化的条目与失败原因 |
| 允许来源 | 只处理配置 `image_localize_hosts` 中列出的主机（默认 MinerU CDN），不盲抓任意远程图片 |
| 源已失效 | 用 `--replace-failed` 把图片语法替换为「插图未本地化」说明（不留断链），并在有本地插图时指向 `03-notes/<YYYY-MM>/<论文标题>/images/` |

```bash
# 预览（不写盘）
python .claude/skills/paper-daily/scripts/localize_markdown_images.py --dry-run
# 下载并改写链接；已存在的图片会跳过（幂等，不重复下载）
python .claude/skills/paper-daily/scripts/localize_markdown_images.py
# 解析源已失效时，明确替换为说明文字而不是留下死链
python .claude/skills/paper-daily/scripts/localize_markdown_images.py --replace-failed
```

- 本目录的图片 **只属于解析结果**；精读笔记自己的插图（含引用文献图片）在 `03-notes/<YYYY-MM>/<论文标题>/images/`，两处不要混用或互相复制。
- 解析批次失效后无法再取回图片时，属于**源侧数据丢失**：脚本会如实登记在 `_sources.md` 的「未能本地化」表中，不要伪造图片。

## 说明文件

- `index.md`：解析总索引，含状态、解析文件路径、入库日期与失败原因。
- `YYYY-MM/README.md`：该月论文的解析清单与未解析提醒。

状态由 `sync_indexes.py` 依据磁盘文件刷新，人工填写的失败原因会保留：

```bash
python .claude/skills/paper-daily/scripts/sync_indexes.py
python .claude/skills/paper-daily/scripts/sync_indexes.py --check-links   # 含图片链接校验
```

完整规范见工作区 `.AGENT.md`。
