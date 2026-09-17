---
name: paper-interests
description: 对话式维护论文推荐的研究主题——询问用户最近关注的方向，扩散补充成完整关键词集合，用户确认后写入 paper-skills 的配置文件。Use when the user wants to change what papers get recommended, says their research interests changed, or wants to add/remove a research topic.
---

# paper-interests —— 研究主题维护

本 skill 只做一件事：**把用户口头描述的研究兴趣，变成配置文件里可用的检索关键词。**

它不搜索论文、不生成推荐笔记。检索由 `paper-daily` / `paper-conf` 负责，周期分析由 `paper-weekly` 负责——**它们三者的研究主题全部来自同一份 `config.yaml` 的 `research_domains`**，所以改这里就等于同时改变了每日推荐、顶会推荐和周报的分析骨架。

**写目标只有一处**：`research_domains`。不要改任何 SKILL.md、不要改脚本里的默认值（脚本里刻意不内置主题）。

## 配置位置

按以下顺序找到第一个存在的配置文件（`paper_config.py` 的查找规则）：

1. `--config` 显式指定的路径
2. `$PAPER_SKILLS_CONFIG` 环境变量
3. `<paper-daily 目录>/config.yaml`
4. `<paper-skills 根目录>/config.yaml` ← 默认生效的就是这一份

要改的是其中的 `research_domains` 段。

## 执行流程

### 第 0 步：读取现状（静默）

先看现有配置，后面的追问和预览都要基于它：

```bash
# 直接读共享配置里的 research_domains 段
python -c "import sys,yaml; d=yaml.safe_load(open(r'<config_path>',encoding='utf-8')); [print(k, '->', len(v.get('keywords',[])), 'keywords') for k,v in (d.get('research_domains') or {}).items()]"
```

**不要跳过这一步**——否则会把用户已有的方向覆盖掉，或重复添加同义领域。

### 第 1 步：询问用户

用一句话提问，不要一次抛多个问题：

> 你最近关注的研究领域是什么？（可以说得口语化，比如"具身智能的抓取"、"扩散模型加速"）

如果用户已经在上文说清楚了自己的方向（例如"我最近在看世界模型"），就**不要再问**，直接进入第 2 步。

### 第 2 步：扩散补充

把用户的回答扩散成结构化的研究领域。每条包含：

- **领域名**：用中文短语，4-10 字，作为配置里 `research_domains` 的键
- **keywords**：英文检索关键词，8-15 个
- **arxiv_categories**：1-5 个 arXiv 分类
- **priority**：1-10（默认 5）

关键词的写法要求：

1. **用 arXiv 上的实际表述**，不要生造术语。
2. **覆盖同义词与缩写**：`vision-language-action` / `VLA` / `vision language action` 都要有。
3. **包含代表性方法或模型名**：该方向公认的模型/方法（如 `OpenVLA`、`flow matching`）能显著提高召回。
4. **细分任务词**：把大方向拆成 2-4 个子任务词（如 `dexterous manipulation`、`in-hand manipulation`）。
5. **只写英文**：arXiv / DBLP / OpenReview 的检索都是英文匹配，中文关键词几乎不命中。
6. **避免过于宽泛的词**：`learning`、`model`、`network` 这类词会引入大量无关论文；确实需要时，用更具体的限定形式（`world model` 而不是 `model`）。
7. 不要在 keywords 里放排除词——排除词走 `excluded_keywords`。

如果用户的方向能自然拆成多个独立领域（例如"具身智能"和"世界模型"），拆开成 2-3 条分别给关键词，比揉成一条更准。

### 第 3 步：把扩散结果讲给用户听

**这一步不能省。** 用户要看到具体会写进配置什么。用清单形式展示，并说明判断依据：

```markdown
我理解你关注的是 **{方向概括}**，我扩散成 {N} 个研究领域：

### 1. {领域名}
- **关键词**（{n} 个）：{keyword1}、{keyword2}、...
- **arXiv 分类**：{cs.XX, cs.YY}
- **优先级**：{priority}

{一句话说明这个领域会捞到什么类型的论文、为什么这么拆}

### 2. ...
```

然后明确告知影响范围，并询问确认：

```markdown
写入后，`paper-daily` 和 `paper-conf` 下次检索就会用这套领域。
现有领域「{X}」「{Y}」**保留不变**（如需替换/删除请说明）。

确认写入吗？也可以直接告诉我要增删的关键词。
```

### 第 4 步：按用户反馈迭代

用户的反馈通常是这几种，对应不同处理：

| 用户说 | 处理 |
|---|---|
| "可以 / 确认 / 写入" | 进入第 5 步 |
| "再加 XX 关键词" | 补充关键词后重新展示第 3 步 |
| "XX 领域不要了" | 询问是删除还是降低优先级；删除需要用户明确确认 |
| "改成只关注 XX" | 说明会替换掉其它方向，得到确认后再写 |
| "换个说法叫 XX" | 调整领域名（注意：改配置键等于新增领域，旧领域仍在，需一并删除） |

### 第 5 步：写入配置

先用预览确认变更内容，再执行写入：

```bash
# 预览（不写盘）
python "<skill-dir>/scripts/merge_interests.py" --preview --payload "<payload.json>"

# 确认无误后写入（会自动生成带时间戳的备份）
python "<skill-dir>/scripts/merge_interests.py" --apply --payload "<payload.json>"
```

payload 格式（单个领域或数组）：

```json
{
  "domain_name": "世界模型",
  "keywords": ["world model", "video prediction", "world simulator", "dreamer"],
  "arxiv_categories": ["cs.CV", "cs.AI", "cs.LG"],
  "priority": 6,
  "replace": false
}
```

**重要**：
- `replace: false`（默认）是**追加关键词**到同名领域，已有词保留 —— 用户说"补充"时用这个。
- `replace: true` 是**覆盖**同名领域的全部关键词 —— 只有用户明确要求"重做这个方向"时才用。
- 脚本会先校验合并结果是不是合法 YAML，校验失败就不写盘，并把原有配置备份成 `config.yaml.bak-<时间戳>`。
- 加了 BOM 的 JSON payload 也能正常读取。

### 第 6 步：回报结果

写入后告诉用户：

1. 配置文件路径
2. 备份文件路径
3. 当前所有领域列表
4. 建议的下一步：跑一次 `paper-daily` 验证新领域能不能召回到论文

```markdown
已写入 `{config_path}`（备份：`{backup_path}`）。

当前研究领域：{领域1}、{领域2}、{领域3}

建议跑一次 paper-daily 看看新方向召回效果，如果捞到的东西不对，回来告诉我调整关键词。
```

## 注意

- **不要编造 arXiv 分类**。只用真实存在的分类（cs.AI/cs.LG/cs.CL/cs.CV/cs.RO/cs.MM/cs.MA/cs.GR/cs.NE/cs.SY/cs.CY/stat.ML/eess.IV/eess.SY/q-bio.QM 等）。
- **不要因为用户说"我不懂技术"就跳过确认**。配置直接决定推荐质量，必须让用户看到写入内容。
- **删除领域是破坏性操作**：`merge_interests.py` 只做新增/更新。需要删除时，先向用户复述要删的领域和关键词，得到明确同意后，用编辑工具从配置文件里移除对应段落。
- 改完配置**不需要**改动 `paper-daily` 或 `paper-conf` 的代码，它们每次运行都会重新读取配置。
