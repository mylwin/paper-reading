# ICML写作/润色情境练习

全部为本次同一主代理按已加载规则生成并人工核对的合成材料；不是独立模型基准，也没有运行旧规则对照组。原始给定事实、原稿和输出见同名JSON。

## 01-draft-empirical — paper-writing

从给定事实起草摘要、引言、结果和一句局限；不用新增实验或引用

Abstract. Optimizer state can limit the memory available for model training. We introduce StepScale, which shares one scale per block to reduce stored optimizer state. Across two tasks with matched training steps and hardware, StepScale reduces optimizer-state memory from 8 to 6 GB while matching the reported mean accuracies. Mean training time increases from 100 to 105 seconds.

Introduction. Optimizer-state storage competes with the rest of the training workload for device memory. StepScale addresses this storage cost by sharing a scale within each block. This design targets state memory, and its evaluation measures the resulting memory and training-time tradeoff.

Results. The comparison uses five seeds for each task. StepScale reduces optimizer-state memory by 25%. Both methods achieve mean accuracies of 80% on Task A and 78% on Task B; StepScale takes 5% longer to train. The measurements support a reduction in state storage at the reported mean accuracy levels.

Limitations. The evaluation covers two tasks on the tested hardware; distributed training remains untested.

人工核对：贡献、机制与内存/耗时交换直接可辨。8→6是25%下降，100→105是5%增加；只称均值相同，不虚构等效检验。不同章节各完成自身论证，不逐句重复未复现/无显著性。

## 02-draft-theory — paper-writing

起草理论结果及含义；材料只有给定定理，不补证明或实验

Under L-smoothness, a lower bound on the objective, and unbiased stochastic gradients with variance at most sigma^2, the iterates satisfy min_{0<=t<T} E||grad f(x_t)||^2 <= C/sqrt(T), where C depends on L, sigma, and the initial objective gap. This bound controls the smallest expected squared gradient norm among the first T iterates and quantifies progress toward stationarity.

人工核对：完整保留三个前提、min与期望平方梯度及C依赖。没有把最小平稳性界改成最后迭代点或全局最优点距离。用户只要结果解释，输出无需实验节或缺证明免责声明。

## 03-draft-exploratory — paper-writing

用两句总结探索性研究，不扩大证据

Language-guided program search recovered competitive models in three runs on each of two toy modeling problems. These results suggest that language feedback is a promising search signal for this setting.

人工核对：首句明确观测，次句suggest/promising匹配探索性判断；不把它当弱词强删，也不升格为普遍建模能力。

## 04-polish-overhedged — paper-polishing

仅润色一句，保留证据力度

StepScale reduces optimizer-state memory from 8 to 6 GB under the reported setup.

人工核对：数据确定时直接陈述；保留一次setup范围，删除叠加可能性和无具体缘由的谨慎语。不追加整篇待办。

## 05-polish-overclaim — paper-polishing

修正给定结果段的过强主张

Across the two evaluated tasks, StepScale reduces optimizer-state memory by 25% and matches the baseline mean accuracies, with a 5% increase in training time.

人工核对：移除无来源all/SOTA/加速，仍明确保留实测收益和真实交换；未将所有结果弱化成may。

## 06-polish-accurate — paper-polishing

轻量润色；准确通顺可原样保留

StepScale reduces optimizer-state memory from 8 to 6 GB on the two evaluated tasks.

人工核对：原句已有对象、数值和范围；无需为体现修改而添加副词、免责声明或新内容。

## 07-polish-hypothesis — paper-polishing

改顺句子，不能把相关性写成机制证明

Across four tested configurations, lower gradient kurtosis accompanies higher returns. This association suggests a possible mechanism for the observed gains.

人工核对：保留观测的确定性、解释的假说语气；possible不冗余，不能从相关观察生成因果证明。

## 08-polish-conflict — paper-polishing

修正存在冲突的提升数字；没有权威源时不要静默选值

The method improves Dice over the baseline.

人工核对：暂用两来源均支持的方向性结果，具体幅度不擅自定稿。作者核对项单独列出，不在正文插入审计过程。此案例数字定稿仍pending，不计为完整成稿。

作者待办：待作者确认：摘要20.09与表中96.11−79.02=17.09个百分点不一致；请确定正确来源后恢复具体提升量。

## 09-chinese-to-english — paper-polishing

中文材料转两句英文；明确时间口径

On the two evaluated tasks using the same hardware, the method reaches the reference quality threshold in 80 hours, compared with 100 hours for the baseline. This corresponds to a 20% reduction in time to the threshold.

人工核对：证明改为实测结果陈述，20%用100为分母；不虚构显存改善，也不声称广泛全面高效。无需正文强调没有显存数据。

## 10-compress-theory — paper-polishing

压缩重复叙述，保留公式与条件

For fixed 0<gamma<1/2, the method reaches the expected stationarity criterion for the smoothed objective in O(epsilon^(-2/(1-2gamma))) iterations.

人工核对：删除重复而保留严格开区间、固定参数、同一复杂度指数及smoothed对象。不改成epsilon^-2或全局最优点距离。

## 11-draft-analysis — paper-writing

起草贡献段；这是理解现象的实证论文，不提出新算法

We characterize how parallel data collection changes learning under a fixed interaction budget. Across four tasks, larger parallel batches reduce returns in three tasks and leave one task unchanged. Gradient-kurtosis measurements identify an accompanying change in the update distribution, motivating a hypothesis about the learning dynamics.

人工核对：贡献是受控理解，无需制造算法名称/理论保证。重要例外留在同句，更新分布测量只用于机制假说。

## 12-polish-no-material — paper-polishing

只润色已有背景一句，不要求全文材料或强加venue

Optimizer states consume memory during model training.

人工核对：轻量任务直接完成通用润色，不追问实验/目标会议，不生成LaTeX工程或整篇诊断清单。
