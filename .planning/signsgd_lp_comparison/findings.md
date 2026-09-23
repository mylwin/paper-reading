# 证据摘录

- 主论文：`01-raw/2026-09/When_and_Why_SignSGD_Outperforms_SGD_A_Theoretical_Study_Based_on_l1_norm_Lower_Bounds.pdf`，25 页，含附录。解析稿同名位于 `02-markdown/2026-09/`。
- 本地对照笔记：PowerStep、Softmax 均在 `03-notes/2026-09/`，已有精读。
- 论文 arXiv:2605.06615v1，2026-05-07，作者 Hongyi Tao、Dingzhi Yu、Lijun Zhang；理论算法 + 小规模实证。
- 定理 1：SignSGD 在 Assumptions 1a–3a 下给出平均梯度的 $\ell_1$ 上界，查询数 $N=BT$；定理 2 给出匹配的算法下界；定理 3 给出固定常步长、$B=1$ 的 vanilla SGD 下界，借助 Jiang et al. 2025a 的坐标光滑结果。
- 复杂度比较属同一问题类的最坏情况比较：SignSGD 的统一上界 vs SGD 的存在性困难实例下界，不能解释为每个具体目标函数都快。噪声密度 $\phi=\|\sigma\|_1^2/(d\|\sigma\|_2^2)$；均匀噪声时等于 1，随机项无严格优势；零噪声时该比值未定义。
- 第 5 节 Muon 在谱范数光滑、核范数驻点的矩阵几何中做约化；正文 Theorem 5 的公式误写大 O，附录 D.2 收尾为大 Omega，需标注排版错误。
- PowerStep：用动量后保号幂变换，参数 beta=0 对应带动量 sign 更新，beta=1 对应动量 SGD；$ℓ_p$ 更新几何，论文主定理证明已有本地笔记指出缺口。不能把当前 SignSGD 界直接转给 PowerStep。
- Softmax：对象是 logit 到概率映射的同范数 Lipschitz 常数，不是参数空间目标函数的光滑常数；本地笔记明确指出同范数算子界不单独推出优化器收敛。
- 最近邻包括 Jiang et al. 2025a（COLT 原文，AdaGrad 相对 SGD 的坐标几何分离）、Balles et al. 2020（sign 几何）、Bernstein et al. 2018（SignSGD）。
- 作者实验：图 1 为 nanoGPT-124M/C4 的 SGD vs SignSGD 损失曲线，图 2 噪声密度轨迹，图 3 各向异性曲率与稀疏噪声 toy；附录 E.1 四张 NVIDIA Pro 6000，10k 步、全局 batch 512、序列长 512、学习率网格 1e-2/1e-3/1e-4、10% warmup。代码仓库公开，但 README 有占位与目录命名不一致，复现仍需手工核对。
