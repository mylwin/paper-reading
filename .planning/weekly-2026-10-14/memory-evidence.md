# 低状态预算优化器证据补查（2026-10-16）

读取范围：依据 `manifest-through-2026-10-16.json` 定位 MoARa 与 Rounding in Preconditioner Space 的本地归档条目；本次只定点读取官方 HTML 的有关章节、表与作者代码，不改本地 PDF、笔记、配置或 git。网络使用 web-access 指引的一手静态网页路径；Chrome 检查未连接，公开页面可直接读取，故不依赖浏览器登录。

## 核验结果与边界

- **ZIP-SR / ZE-EDEN**：[原论文 v1](https://arxiv.org/html/2610.12444v1)，§3.1、§4.2、§5.1 Table 3、§5.2 Table 4。这是 AdamW 状态量化工作。固定码本的舍入空间对比与整套配方收益必须分开。Table 3 的最大单步改进来自一阶矩 SDyn4→NF4；ZIP-SR 分支同时改变二阶矩码本和舍入，且该消融仅单次运行，不能把总收益全部归给舍入。§4.2 用匹配 checkpoint 干预定位 LM-head 后期确定性舍入问题；最终配方仅最后 10% 训练切换其一阶矩至 SR。Table 4 的完整预训练比较使用三配对种子；这些结果尚不能直接推出 Muon 有同样的模块敏感性。
- **MoARa**：[原论文 v1](https://arxiv.org/html/2609.15037v1)，§4.2、Table 2、Table 3、Appendix E.3。7B 少 37% 步、少 34% 时间来自短上下文 seq=256；seq=2048 的验证只到 350M。相同总投影 rank 并不等于相同状态字节：7B 持久优化器状态增加 1,250 MiB（17.4%），而编译下峰值 reserved 仅增 0.2%，两种内存指标不可互换。Table 3 中单独重分配 rank 在六种 host 都改善，但 decomposition 在 LDAdam、SubTrack++ 等 host 上会退化，不能当通用插件。
- **GefenMuon 已存在**：[作者仓库](https://github.com/ndvbd/Gefen)、[原始代码 gefen_muon.py](https://raw.githubusercontent.com/ndvbd/Gefen/main/gefen_muon.py)。README 已明确 8bit 一阶矩 Muon 变体，相对 FP32 动量声称约 4 倍持久状态压缩。源码 `_fused_quantized_momentum_update`（160–198 行）通过 `m_codebook` / `m_magnitude` 调用 `automatic_gefen_muon_momentum_cuda`，并生成工作动量；199–255 行检查二维梯度并初始化分区状态。这里核验的是确有量化实现路径；没有复现其性能，也没有读取 fused kernel 全部数值细节。**“将 Muon 动量改为 8bit”已不是创新点。** 代码链接跟随 main，需实验时固定 commit。
- **低秩近邻一**：[No Subspace to Track v2](https://arxiv.org/abs/2607.05872v2)，本次只核验官方摘要。摘要报告 minibatch 间子空间差异几乎等于时间刷新差异，且 LDAdam 已做一阶矩坐标传输；1B/40k/3 种子下 LDAdam PPL=18.7，优于调过 beta2 的未传输 GaLore=19.3。不能将简单坐标传输或缩短 beta2 直接包装为新方向；理论适用假设待全文核验。
- **低秩近邻二（外部补查，不计入本周本地归档）**：[Memory-Efficient Subspace Optimization with Residual Correction: From Momentum SGD to Muon](https://arxiv.org/abs/2610.19844v1)，2026-10-14。官方摘要已覆盖 SMRC/PMRC、子空间状态+正交残差更新、周期基刷新和 boundary correction，并声明 O(T^-1/4) 率。只读摘要，未审核证明与实验。不能将“低秩 Muon+残差+周期刷新”作为已确认空白。

## 建议：固定状态字节下，以实际更新失真分配模块精度（低成本先证伪）

**研究假设（我们的推断，未证实新颖性）**：Muon 正交化会改变动量误差到参数更新误差的映射。在同一总状态字节约束下，根据真实更新偏差分配 4/8/16bit 模块精度，比按动量重建 MSE 或固定全 8bit 分配更可靠。收益应体现在 loss 或到目标所需时间，而不只在单步误差。潜在增量是“预算约束+正交化之后的失真目标+模块分配”；不是量化 Muon 本身。需要再查已有 quantized Muon 的分配准则才可宣称新颖。

**与至少两项最强近邻的区别**：GefenMuon 已实现 8bit 动量和结构分区；本建议只检验额外的预算分配准则是否有净收益。ZIP-SR 已提出按更新相关空间设计舍入，且已有 LM-head 定向 SR；本建议也不能声称首次关注更新误差，而是针对 Muon 的非线性矩阵映射做受控分配实验。MoARa 已证明模块异质性可用于 rank 分配，但其同 rank 实验不是同字节基准。若扩展到低秩，还必须与 LDAdam、No Subspace to Track、PMRC 比较；当前建议不依赖低秩刷新机制。

**定义目标**：对同一历史解码状态和同一当前梯度，分别计算参考工作动量与候选量化状态递推后的工作动量，经完全相同的 Nesterov、Newton–Schulz 步数、尺度规则、矩阵形状与 weight decay 后，比较真正参数更新 `u_ref` 与 `u_quant`：相对误差 `||u_quant-u_ref||F/(||u_ref||F+eps)`、方向夹角及径向误差。量化在本步更新后写回时，应评价下一步或短序列递推，不能把存储后量化状态错误当作当前步已使用量化状态。首先固定码本、块大小、舍入规则，只改精度分配；SR 与 RTN 的因果比较留作独立消融，不同时混改。

**最小验证与预算**：

1. 先做 350M 模型的离线梯度重放：早/中/晚各 100 个连续 step，固定梯度序列和初始状态，在相同 NS 配置下比较全 8bit、按 MSE 分配、随机分配、按 update 误差分配。4/8/16bit 分配必须满足 `sum(n_l*b_l/8 + scale_bytes_l + metadata_bytes_l) <= uniform8_total_bytes`；16bit 的保护必须由其他模块降至 4bit 支付，不能额外保留全量残差/影子动量。全精度参考仅离线诊断使用，不计入生产方案。若 4bit 实现尚不可用，先用解码重放评估必要性，再投入 kernel 开发。
2. 只在离线结果通过后，做 350M、seq=2048、固定 1B tokens 的预训练先导，每组 3 个配对种子；全 8bit GefenMuon、MSE 分配与 update 分配使用同一代码路径/调参预算/数据顺序/全局 batch/学习率搜索次数；FP32 Muon 是质量参考。embedding、head、norm 的辅助优化器配置须完全相同，持久状态字节与峰值显存计全模型。预算包括分配 profiling、重分配、同步与额外 NS 计算；同时给固定 token 的 loss 和到固定 loss 的端到端时间。
3. 先固定一次 profiling 得出的分配，避免在线逐步额外 NS；若需要在线策略，其成本必须单独报告。实验 GPU 时数由实测吞吐估算，不预先承诺 7B 能受益。通过后才做 1B；最后再考虑 7B。

**预注册继续/停止条件（研究设计，非文献结果）**：离线 update 分配相对 MSE 分配的中位实际更新误差至少下降 20%，且尾部 p99 不恶化，才进入训练。训练以配对种子 loss 差和置信区间为主，目标平均至少改善 0.005 nats/token 或到同一预注册 loss 的总时间减少 5%，额外总时间开销不超过 3%。若只降 MSE 而更新误差无差异、实际字节超预算、收益仅来自额外高精度状态、收益小于种子波动，或 profiling 成本抵消到目标节约，则停止该分配路线，不升 7B。

**明确的风险**：局部单步误差不是长期 loss 的充分条件；4bit 降精度可能抵消局部16bit保护；静态分配未必跨训练阶段迁移；未来搜索若发现相同预算与 update 目标已被 quantized Muon 使用，课题需重定位为复现/机制检验。
