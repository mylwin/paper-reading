# Parameter-free Clipped Gradient Descent Meets Polyak

## 代码跑通与论文实验执行计划

## 1. 这份计划要完成什么

目标不是重新设计项目，也不是先做一套独立工程，而是按论文作者的补充代码把四组实验实际跑起来，并保存可以和论文 Figure 1、Figure 2、Figure 3、Figure 4 对照的结果：

1. Synthetic function：生成 Figure 1 的六类收敛曲线。
2. LSTM：在 Penn Treebank 上运行论文比较的优化器。
3. Nano-GPT：在 Shakespeare 上运行论文比较的优化器。
4. T5：在 C4 上运行论文比较的优化器。

论文实验中的方法为：

- SGD；
- Clipped SGD；
- Polyak；
- DecSPS；
- AdaSPS；
- DoG；
- Inexact Polyak。

其中 SGD 和 Clipped SGD 需要网格搜索步长/裁剪阈值，其他方法按代码自动计算步长。论文最终结果使用三个随机种子取平均；作者没有在论文中给出完整 seed 列表，因此第一次先用代码默认 seed `2137` 跑通，服务器正式实验时再固定三个 seed 并记录。

## 2. 硬件边界

### 2.1 3060 Desktop 6 GB：只负责本地跑通

3060 阶段的目标是确认：环境、依赖、数据读取、模型前向/反向、优化器更新、日志和 checkpoint 都能工作。

本地必须完成：

- Synthetic function 完整运行；
- LSTM、Nano-GPT、T5 的单种子短跑；
- T5 先用 `google/t5-v1_1-small` 跑通完整 T5 代码路径；
- 再尝试论文同系列的 `google/t5-v1_1-base`（约 248M 参数）短跑；
- 220M 与 3B 的配置只做启动前检查或极短初始化，不在 6 GB 卡上做正式训练。

3060 阶段不宣称复现论文最终 loss，只宣称“代码跑通”。

### 2.2 服务器：负责论文实验结果

服务器阶段才运行：

- LSTM、Nano-GPT、T5 的完整训练周期；
- SGD/Clipped SGD 网格搜索；
- 三个 seed 的正式实验；
- Figure 2、Figure 3、Figure 4 所需的 loss 和 final test loss；
- T5 的论文同规格模型 `google/t5-v1_1-base`；
- 之后再运行 `google/t5-v1_1-xl`（约 3B）扩展实验。

注意：作者 nanoT5 默认是 `google/t5-v1_1-base`，参数量约 248M，不是严格 220M。报告中把它写成“论文同规格 base”，不要写成严格 220M。若需要严格 220M，单独增加 `t5-base`，并标注为变体。

## 3. 实验目录和执行原则

不要把官方代码拆到新的 `src/` 目录。补充包中的脚本使用相对路径和导入路径，直接在官方代码根目录运行最稳妥。

建议目录：

```text
Parameter_Free_Clipped_Gradient_Descent_Meets_Polyak/
├── 复现规划.md
├── official/
│   ├── code/                 # 补充包解压后的原始代码
│   └── official-supplement.zip
├── patches/                  # 为了跑通官方代码的最小修补记录
├── runs/
│   ├── synthetic/
│   ├── lstm/
│   ├── nanogpt/
│   └── nanot5/
└── results/
    ├── figures/
    ├── metrics/
    └── summaries/
```

每一次运行都保存：

```text
command.txt
environment.txt
hardware.txt
stdout.log
metrics 或 wandb 离线目录
最终 checkpoint
最终 test loss
```

不要先做完整重构。只有遇到实际阻塞运行的问题，才做最小修改，并把修改记录到 `patches/`。

## 4. 第一步：固化代码和环境

### 4.1 准备官方补充包

补充包 SHA-256：

```text
d938b0426c2f59f6a7e5cc26b98697f4cdfc85cc2d9fb4a7a34f48e52bf5ad55
```

在目标目录执行：

```bash
cd /Users/myl/Documents/research-code/reproduction/Parameter_Free_Clipped_Gradient_Descent_Meets_Polyak
mkdir -p official runs results/{figures,metrics,summaries} patches
cp /tmp/reproduction-plan-parameter-free-polyak/supplement.zip official/official-supplement.zip
unzip -q official/official-supplement.zip -d official
```

最终确认以下文件存在：

```text
official/code/polyak.py
official/code/nanoT5/nanoT5/main.py
official/code/nanogpt/train.py
official/code/lstm/main.py
official/code/.ipynb_checkpoints/Synthetic-FIgure-checkpoint.ipynb
```

### 4.2 建立 GPU 环境

在 3060 或服务器上使用 Linux + NVIDIA CUDA 环境，不要在当前无 NVIDIA 的 Mac 环境上判定 GPU 实验失败。

```bash
conda create -n polyak-repro python=3.10 -y
conda activate polyak-repro
python -m pip install --upgrade pip
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r official/code/nanoT5/requirements.txt
pip install wandb tqdm matplotlib jupyter
```

安装后立即记录环境：

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    print('memory_gb', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY
nvidia-smi > runs/nvidia-smi.txt
pip freeze > runs/pip-freeze.txt
```

`torch.cuda.is_available()` 必须为 `True` 后，才进入 3060/T5 步骤。

## 5. 第二步：只做能够阻塞运行的最小修补

补充包当前不能直接完成全部实验，原因是几个脚本在模块顶层导入了补充包未提供的 `parameterfree.py`、`dowg.py` 和 `dadaptation.py`，同时还写死了作者机器的 W&B 路径。先做以下最小处理：

1. 保留 `official/code/` 作为未修改原始快照。
2. 复制一份为实际运行目录，所有修改只发生在运行副本。
3. 把 `official/code/lstm/dog.py` 放到三个脚本能够导入的位置，保证 DoG 可用。
4. 将 `parameterfree`、`dowg`、`dadaptation` 改为只在对应 optimizer 分支内导入；本论文主实验不需要 COCOB、DoWG、DAdaptSGD。
5. 把 `wandb.init(..., dir="/work/YamadaU/...")` 改成当前运行目录，并设置 `WANDB_MODE=offline`。
6. nanoT5 的 checkpoint 路径改为当前运行目录，不能使用 `/work/YamadaU/takezawa/...`。
7. T5 短跑关闭 `model.compile`；正式服务器实验再单独测试 compile。

这一步不改论文算法公式，不替换优化器，不新增分布式算法。每处修改记录：文件、原代码、修改原因、影响的实验。

修补完成后先执行导入检查：

```bash
cd official/code
python -c "import polyak; print('polyak import ok')"
python -c "import sys; sys.path.insert(0, 'nanogpt'); import model; print('nanogpt import ok')"
python -c "import sys; sys.path.insert(0, 'nanoT5'); import nanoT5.main; print('nanoT5 import ok')"
```

如果这里失败，不进入训练；先解决导入错误。

## 6. 第三步：先跑 Synthetic function

这是最先完成的实验，因为不需要下载数据，也不需要 GPU。

### 6.1 实验设置

严格使用论文设置：

- `L0 = 1`；
- `x0 = 5`；
- `f* = 1`；
- `l* = 0`；
- `L1 = 1, 10, 100, 1000`；
- GD、Clipped GD、Polyak 各 500 步；
- DecSPS、AdaSPS 各 10000 步；
- Inexact Polyak 使用不同 `T` 绘制最终最小 loss 曲线。

从 notebook 提取脚本时必须修复一个运行问题：Inexact Polyak 的 `model` 要在每一个 `total_iteration` 外层循环内重新初始化，不能复用上一个 `T` 的模型状态。否则图不是论文实验的定义。

### 6.2 执行顺序

1. 从 notebook 复制 Synthetic function 的代码到 `runs/synthetic/run_synthetic.py`。
2. 创建 `runs/synthetic/pic/`。
3. 先运行 1 个 `L1`，确认 loss 为有限值。
4. 再运行四个 `L1` 的完整实验。
5. 输出 `gradient_descent.pdf`、`clipped_gradient_descent.pdf`、`polyak.pdf`、`decsps.pdf`、`adasps.pdf`、`inexact_polyak.pdf`。
6. 同时保存每个方法的 loss 数组，不只保存图片。

验收：

- GD 在 `L1` 增大时明显变慢；
- Polyak 和 Clipped GD 的收敛趋势对 `L1` 不敏感；
- DecSPS/AdaSPS 随 `L1` 增大变差；
- Inexact Polyak 的趋势接近 Polyak，不出现 NaN/Inf。

## 7. 第四步：先用三类神经网络做短跑

短跑顺序固定为：LSTM → Nano-GPT → T5。每个模型先只跑 `seed=2137`，每个方法 20 至 100 个更新，确认完整路径后再做网格搜索。

统一方法顺序：

```text
sgd
clipped sgd（sgd + grad_clip）
polyak
decsps
adasps
dog
inexact_polyak
```

每个短跑都必须记录：开始 loss、结束 loss、是否 NaN/Inf、最后 test loss、checkpoint 路径、实际运行命令。

## 8. 第五步：LSTM / Penn Treebank

### 8.1 准备数据

补充包没有完整 Penn Treebank 数据。使用论文引用的 AWD-LSTM 数据准备方式，将生成的 `train.txt`、`valid.txt`、`test.txt` 放到：

```text
official/code/lstm/data/penn/
```

确认目录至少包含这三个文件后再运行：

```bash
cd official/code/lstm
ls data/penn/train.txt data/penn/valid.txt data/penn/test.txt
```

### 8.2 3060 短跑

注意：代码中的 `--cuda` 是 `store_false`，默认就是使用 CUDA；在 3060 上不要加 `--cuda`。

先用论文模型结构、缩短 epoch：

```bash
cd official/code/lstm
WANDB_MODE=offline python main.py \
  --data data/penn \
  --method inexact_polyak \
  --epochs 1 \
  --batch_size 80 \
  --bptt 70 \
  --seed 2137 \
  --save ../../../runs/lstm/inexact_polyak_seed2137.pt \
  2>&1 | tee ../../../runs/lstm/inexact_polyak_seed2137.log
```

然后依次替换 `--method` 为 `sgd`、`polyak`、`decsps`、`adasps`、`dog`。Clipped SGD 仍使用 `--method sgd`，再增加 `--grad_clip`。

### 8.3 服务器正式实验

模型和 batch 使用论文配置：

- embedding size：400；
- hidden size：1150；
- layers：3；
- batch size：80；
- `bptt=70`；
- 训练 200 epochs；
- SGD 网格：`100, 50, 10, 1, 0.1, 0.01`；
- clipped SGD 的 clip 网格：`0.5, 1, ..., 5, inf`。

网格搜索只对 SGD 和 clipped SGD 做。选出验证集最优配置后，再用相同 seed 运行全部方法，记录每个 epoch 的 valid/test loss，生成 LSTM 曲线和 final test loss。

## 9. 第六步：Nano-GPT / Shakespeare

### 9.1 数据

补充包已包含 `official/code/nanogpt/data/shakespeare_char/` 的 `train.bin`、`val.bin`、`test.bin`，不需要重新下载。

### 9.2 3060 短跑

先用小 batch 和短序列证明代码路径。Nano-GPT 的模型宽度在 `train.py` 中是常量，若显存不足，只在运行副本中把层数、头数和 hidden size 缩小，并在日志中标记“smoke model”。

```bash
cd official/code/nanogpt
WANDB_MODE=offline python train.py \
  --method inexact_polyak \
  --batch_size 4 \
  --block_size 64 \
  --max_iters 20 \
  --seed 2137 \
  2>&1 | tee ../../../runs/nanogpt/inexact_polyak_seed2137.log
```

依次跑 `sgd`、`polyak`、`decsps`、`adasps`、`dog`。Clipped SGD 通过 `--method sgd --grad_clip 10` 验证。

### 9.3 服务器正式实验

恢复论文配置：

- batch size：64；
- block size：256；
- 论文代码默认 GPT 配置；
- SGD 学习率网格：`1, 0.5, 0.1, ..., 0.0005, 0.0001`；
- clipped SGD clip 网格：`1, 2, ..., 10, inf`；
- 训练步数使用代码的正式 `max_iters`，不要把 20 步短跑结果当正式结果。

保存每 10 step 的 train loss、每 100 step 的 valid/test loss，最终输出与论文 Figure 3、Figure 4 对应的曲线。

## 10. 第七步：T5 / C4

### 10.1 模型选择

运行顺序：

1. `google/t5-v1_1-small`：3060 本地完整路径短跑。
2. `google/t5-v1_1-base`：服务器论文主线，约 248M 参数。
3. `t5-base`：只有需要严格 220M 时才增加，作为变体。
4. `google/t5-v1_1-xl`：服务器 3B 扩展，不属于论文原始规模结果。

### 10.2 3060 本地 T5-small

在 `official/code/nanoT5` 目录运行，使用随机初始化，避免先下载预训练权重：

```bash
cd official/code/nanoT5
WANDB_MODE=offline python -m nanoT5.main \
  model.name=google/t5-v1_1-small \
  model.random_init=true \
  model.compile=false \
  precision=no \
  data.input_length=128 \
  data.num_workers=0 \
  optim.batch_size=1 \
  optim.grad_acc=1 \
  optim.total_steps=20 \
  optim.warmup_steps=0 \
  optim.lr_scheduler=constant \
  eval.every_steps=10 \
  checkpoint.every_steps=20 \
  2>&1 | tee ../../../runs/nanot5/t5_small_inexact_seed2137.log
```

上述命令默认使用配置中的 optimizer；为了验证论文主线，分别运行：

```text
optim.name=sgd
optim.name=decsps
optim.name=adasps
optim.name=polyak
optim.name=inexact_polyak
optim.name=dog
```

Clipped SGD 使用 `optim.name=sgd optim.grad_clip=2`。T5 的自适应优化器必须使用 `optim.lr_scheduler=constant`，不要叠加 cosine 或 warmup，否则测到的是“优化器 + 外部调度器”，不是论文算法。

如果 T5-small 仍 OOM，先把 `data.input_length` 降到 64；不要在 3060 上继续尝试完整 3B。

### 10.3 服务器 T5-base 正式实验

恢复论文 nanoT5 配置：

- `model.name=google/t5-v1_1-base`；
- `model.random_init=true`；
- `data.input_length=512`；
- `data.mlm_probability=0.15`；
- `data.mean_noise_span_length=3`；
- global batch size：128；
- total steps：论文代码默认 65536；
- C4 English streaming；
- 训练/验证 loss 按代码记录。

正式运行分三层：

1. 20 steps：确认 C4、tokenizer、模型、优化器、评估和 checkpoint 全部工作。
2. 200 至 1000 steps：确认 loss 曲线方向与论文描述一致，筛查发散。
3. 65536 steps：只对通过前两层的方法运行完整实验。

先完成单种子，再运行三个 seed。SGD 和 Clipped SGD 先按论文网格搜索：

- learning rate：`5, 1, 0.5, 0.1, 0.05`；
- gradient clipping：`1, 2, 3, inf`；
- batch size：128。

选出验证集最优的组合后，再固定配置运行所有方法。每个正式运行保存：训练 loss、test loss、实际步长、梯度范数、最终 checkpoint、峰值显存、吞吐量和总耗时。

### 10.4 3B T5 扩展

3B 不与论文 base 结果混在一张表里。服务器上按以下门禁执行：

1. 加载 `google/t5-v1_1-xl` 配置并打印参数量。
2. 用 `batch=1`、短序列完成单卡/多卡初始化。
3. 运行 20 steps，确认无 OOM、NaN、rank 间 loss 分歧。
4. 运行 200 steps，确认 checkpoint 可恢复。
5. 运行 1000 steps，和 base 级实验比较趋势。
6. 资源允许后再决定是否长跑。

3B 首选 BF16、activation checkpointing、`use_cache=false`、分片 checkpoint。不要在 3060 上初始化完整参数。

## 11. 第八步：生成论文对应结果

每个模型的正式结果必须整理成同一张表：

```text
model, method, seed, best_valid_loss, final_test_loss,
steps, batch_size, learning_rate, grad_clip,
nan_or_inf, peak_memory, elapsed_seconds
```

输出顺序：

1. Synthetic：六张方法图，组成 Figure 1 对照。
2. LSTM、Nano-GPT、T5：每个方法的训练/测试 loss 曲线，对照 Figure 3。
3. 各方法 final test loss 和超参数扫描结果，对照 Figure 2。
4. LSTM、Nano-GPT 的完整 Polyak 曲线单独保存，对照补充材料 Figure 4。
5. 三个 seed 计算均值和标准差；如果只完成一个 seed，明确标注为 short run，不填入正式平均结果。

不要用论文图片反推数值，也不要把 T5-small 或 20-step 结果写成论文复现结果。

## 12. 每一步的停止条件

按以下门禁推进，前一步失败就停在当前步骤：

| 门禁 | 必须看到的证据 |
|---|---|
| 环境 | CUDA 可用，GPU 名称和显存已记录 |
| 导入 | polyak、LSTM、Nano-GPT、nanoT5 均可导入 |
| Synthetic | 六类曲线和 loss 数据文件生成 |
| LSTM smoke | 至少一个 epoch 内完成前向、反向和 optimizer.step |
| Nano-GPT smoke | 20 steps 完成，val/test loss 可计算 |
| T5-small smoke | 20 steps 完成，loss、grad norm、checkpoint 均存在 |
| T5-base server smoke | 20 steps 完成，C4 流式数据和评估正常 |
| 正式实验 | 三 seed 的曲线、最终 test loss 和运行配置齐全 |
| 3B 扩展 | 20/200 steps 无 OOM、NaN、rank 分歧 |

## 13. 当前实际执行顺序

1. 在 GPU Linux 主机建立环境并保存硬件信息。
2. 固化官方补充包，保留原始快照。
3. 完成缺失导入和硬编码路径的最小修补。
4. 先运行 Synthetic function，确认优化器公式和图形输出。
5. 运行 LSTM 单种子短跑。
6. 运行 Nano-GPT 单种子短跑。
7. 运行 T5-small 单种子 20 steps。
8. 在 3060 上尝试 T5-v1.1-base 1 至 3 steps；OOM 就停止，不继续消耗时间。
9. 在服务器上运行 T5-v1.1-base 的 20/200/1000 steps 门禁。
10. 完成 LSTM、Nano-GPT、T5 的 SGD/Clipped SGD 网格搜索。
11. 固定每个模型的超参数，运行 Polyak、DecSPS、AdaSPS、DoG、Inexact Polyak。
12. 每个方法运行三个 seed，整理 Figure 2/3/4 对应结果。
13. 最后再在服务器运行 T5-v1.1-xl 3B 的初始化、20、200、1000 steps 门禁。

这份计划的完成标准是“作者实验链路真正跑起来并留下结果”，不是“设计一套更复杂的新训练框架”。

## 14. 依据

- 论文第 6 节：Synthetic function、LSTM、Nano-GPT、T5 实验目标。
- 论文附录 D：数据集、batch size、网格搜索范围和选定超参数。
- NeurIPS Supplemental：`polyak.py`、`lstm/`、`nanogpt/`、`nanoT5/`。
- nanoT5 README：T5-v1.1-base 约 248M、C4 预训练、65536 steps 的默认设置。
