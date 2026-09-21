# Robust and Efficient Zeroth-Order LLM Fine-Tuning via Adaptive Bayesian Subspace Optimizer

Jian Feng <sup>1</sup> Zhihong Huang 1 * 

## Abstract

Fine-tuning large language models (LLMs) with zeroth-order (ZO) optimization reduces memory by approximating gradients through function evaluations. However, existing methods essentially perform updates in a one-dimensional space, and suffer from collapse or substantial performance degradation under low-precision training. We introduce BSZO, an adaptive Bayesian Subspace Zeroth-Order Optimizer, which applies Kalman filtering to combine finite-difference information across multiple perturbation directions within a subspace. By treating each finite-difference measurement as a noisy observation, BSZO builds a posterior distribution over the subspace-projected gradient and updates it through Bayesian inference, with a residual-based adaptive mechanism to adapt to noise variations. Theoretical analysis shows that BSZO improves the convergence rate by a factor of k/γ compared to standard ZO methods. Experiments on RoBERTa, Mistral, and OPT models show that BSZO outperforms the baselines across various tasks, achieving up to 6.67% absolute average improvement on OPT-13B while remaining robust under fp16/bf16 precision and keeping memory usage close to inference-only baselines (1.00×–1.08× of MeZO). 

## 1. Introduction

Large language models (LLMs) are getting increasingly important in natural language understanding and generation (Devlin et al., 2019; Brown et al., 2020; Touvron et al., 2023). However, adapting these models to downstream tasks through fine-tuning remains challenging due to their large scale. The standard approach, using first-order optimizers like Adam, requires consuming a large amount of GPU memory. For a 13B-parameter model, this translates to over 

100GB of GPU memory, roughly 10× the cost of inference alone (Malladi et al., 2023). Such requirements put full finetuning out of reach for most people, no matter in academia or industry. 

Several strategies have been proposed to reduce memory burden. Parameter-efficient fine-tuning (PEFT) methods, including LoRA (Hu et al., 2022) and Adapters (Houlsby et al., 2019), freeze the base model and only update a small set of additional parameters. But these methods still rely on backpropagation and may underperform full fine-tuning on difficult tasks. An alternative direction is zeroth-order (ZO) optimization, which estimates gradients using only forward passes. MeZO (Malladi et al., 2023) demonstrated that this approach can match the memory footprint of inference, while achieving reasonable accuracy. The catch? ZO methods converge slowly and require significantly more iterations than their first-order counterparts, due to the high variance inherent in finite-difference gradient estimates. 

This raises a question: how can we achieve a better tradeoff between convergence speed and memory usage? We observe that the existing ZO methods have three main weaknesses. First, most existing ZO optimizers essentially perform updates along a single random direction within each batch. Even with increased forward passes and perturbation directions, they process each perturbation in isolation, simply averaging or using them independently—throwing away information about how these measurements relate to each other. Second, the noise level in ZO estimates varies significantly during training, yet most methods do not account for this effect. This rigidity leads to poor adaptation: updates may oscillate wildly around local minima, jump out of the basin, and finally cause training collapse. Moreover, reduced-precision training (fp16/bf16) can cause these methods to collapse or suffer substantial performance degradation, as we show in Figure 1 and Table 3. 

We propose Bayesian Subspace Zeroth-order Optimization (BSZO) to address these limitations. The main idea is to treat gradient estimation as an inference problem. At each step, we sample k random directions to form a lowdimensional subspace (Zhang, 2025) and model the projected gradient as a latent variable. Instead of treating each finite-difference query as providing an independent estimate, we use Kalman filtering to aggregate observations—essentially asking: given what we have measured so far, what is our best guess of the true gradient? This Bayesian formulation accounts for measurement noise and produces more accurate estimates from the same number of forward passes. We further introduce an adaptive mechanism that tracks prediction residuals and adjusts the noise variance on the fly, allowing the algorithm to respond to changing curvature conditions during training. 

![image](images/Robust_and_Efficient_Zeroth_Order_LLM_Fine_Tuning_via_Adaptive_Bayesian_Subspace_Optimizer/fig1.jpg)


![image](images/Robust_and_Efficient_Zeroth_Order_LLM_Fine_Tuning_via_Adaptive_Bayesian_Subspace_Optimizer/fig2.jpg)


![image](images/Robust_and_Efficient_Zeroth_Order_LLM_Fine_Tuning_via_Adaptive_Bayesian_Subspace_Optimizer/fig3.jpg)



Figure 1. Training loss on SST-2 with OPT-13B under bf16 precision. (a)–(b) Existing ZO methods exhibit erratic loss curves across different learning rates, with some runs failing to converge or even diverging. BSZO achieves smooth and steady convergence. (c) Comparison under each method’s best-tuned learning rate.


## Our contributions can be summarized as follows:

1. We propose BSZO, a zeroth-order optimizer that uses Bayesian inference to aggregate gradient information across multiple perturbation directions within a subspace. To our knowledge, this is the first application of Bayesian inference and Kalman filtering to ZO optimization for LLMs. 

2. We design a residual-based adaptive scheme that enables BSZO to adjust the parameter update scale adaptively without manual tuning. 

3. We analyze the convergence of BSZO and show that the rate improves by a factor of $k / \gamma$ compared to standard ZO methods. 

4. Experiments on multiple LLMs and benchmarks show that BSZO achieves strong performance across diverse tasks while remaining robust under low-precision training and maintaining memory consumption comparable to MeZO. 

## 2. Related Work

Zeroth-Order Optimization for LLMs. Classical derivative-free methods achieve strong sample efficiency via surrogate modeling, but their per-iteration cost grows rapidly with dimension, making them impractical at LLM scale (Zhang, 2025). The SPSA estimator (Spall, 1992) offers a scalable alternative by approximating gradients through random perturbations. Building on this, MeZO (Malladi et al., 2023) introduced memory-efficient ZO fine-tuning for LLMs, matching inference-time memory by regenerating perturbations from random seeds. Follow-up methods target different bottlenecks: Sparse-MeZO (Liu et al., 2024) restricts updates to influential parameters, HiZOO (Zhao et al., 2025) leverages diagonal Hessian estimates for adaptive preconditioning, LOZO (Chen et al., 2024) exploits low-rank gradient structure, and TeZO (Sun et al., 2025) captures temporal correlations across iterations. Despite these advances, most methods adhere to the “one batch, one update” paradigm, overlooking the possibility that multiple function evaluations within a batch could support multiple parameter updates. Moreover, some of these methods incur substantial memory overhead; while still lower than full fine-tuning, this conflicts with the original motivation of ZO optimization—minimizing memory consumption. Since low-precision fine-tuning is essential in memory-constrained scenarios, the robustness of these methods also warrants further evaluation. 

Population-Based Gradient Estimation. An alternative strategy evaluates multiple perturbations per iteration and aggregates them into a single update. Evolution Strategies (Salimans et al., 2017) and Augmented Random Search (Mania et al., 2018) popularized this paradigm in reinforcement learning. However, these methods typically require a large number of function evaluations per batch to obtain reliable gradient estimates. Given that each forward pass through an LLM is already computationally expensive, such sample-intensive approaches become impractical for language model fine-tuning. This raises a natural question: how can we extract more information from a limited number of function evaluations? Our work addresses this by treating finite-difference measurements as noisy linear observations of the underlying gradient and applying Bayesian inference to fuse information across directions. 

Bayesian Inference for Optimization. Bayesian methods provide a principled way to integrate observations with prior knowledge while quantifying uncertainty. Kalman filtering (Kalman, 1960) is the canonical example: it sequentially updates a Gaussian belief over a hidden state as new measurements arrive. Gaussian processes extend this idea to function-space modeling and underpin Bayesian optimization (Shahriari et al., 2016; Rasmussen & Williams, 2006). Our work adapts the Kalman perspective to ZO gradient estimation: we model the projected gradient as a hidden state, interpret each perturbation query as a noisy linear measurement, and update a posterior that pools information across all sampled directions within an iteration. Leveraging the flexibility of the Bayesian framework, we further design an adaptive residual mechanism that effectively fuses both historical and current-batch information. This yields improved gradient estimates without additional memory overhead. 

## 3. Method

In this section, we present the Bayesian Subspace Zerothorder Optimization (BSZO) algorithm, which controls the step size of subspace by the Bayesian method. 

## 3.1. Preliminaries

We consider the stochastic optimization problem: 

$$
\min _ {\theta \in \mathbb {R} ^ {n}} \mathcal {L} (\theta) := \mathbb {E} _ {\xi \sim \mathcal {D}} [ \mathcal {L} (\theta ; \xi) ],\tag{1}
$$

where $\theta \in \mathbb { R } ^ { n }$ denotes the model parameters, D is the training dataset, and $\mathcal { L } ( \boldsymbol { \theta } ; \boldsymbol { \xi } )$ is the loss on a minibatch $\xi .$ We denote the optimal value by $\mathcal { L } ^ { * } : = \operatorname* { m i n } _ { \theta } \mathcal { L } ( \theta )$ 

Assumption 3.1. The function L is L-smooth, i.e., there exists $L > 0$ such that for all $\theta , \theta ^ { \prime } \in \mathbb { R } ^ { n }$ ， 

$$
\left\| \mathcal {L} (\theta) - \mathcal {L} \left(\theta^ {\prime}\right) \right\| \leq L \| \theta - \theta^ {\prime} \|.\tag{2}
$$

Equivalently, 

$$
\mathcal {L} (\theta) \leq \mathcal {L} \left(\theta^ {\prime}\right) + \nabla \mathcal {L} \left(\theta^ {\prime}\right) ^ {\top} \left(\theta - \theta^ {\prime}\right) + \frac {L}{2} \| \theta - \theta^ {\prime} \| ^ {2}.\tag{3}
$$

Assumption 3.2. The stochastic gradient $\nabla { \mathcal { L } } ( \theta , \xi )$ has bounded variance, i.e., there exists $\bar { \sigma } _ { g } ^ { 2 } \geq 0$ such that: 

$$
\mathbb {E} _ {\xi} [ \| \nabla \mathcal {L} (\theta ; \xi) - \nabla \mathcal {L} (\theta) \| ^ {2} ] \leq \sigma_ {g} ^ {2}, \quad \forall \theta \in \mathbb {R} ^ {n}\tag{4}
$$

Definition 3.3. Given a set of k perturbation vectors $\{ z _ { 1 } , z _ { 2 } , \dots , z _ { k } \}$ , where $z _ { i } \in \mathbb { R } ^ { n }$ is from Gaussian distribution $\textstyle { \mathcal { N } } ( 0 , I _ { n } )$ , define the subspace basis matrix $B =$ $[ z _ { 1 } , z _ { 2 } , \ldots , z _ { k } ] \in \mathbb { R } ^ { n \times k }$ 

Algorithm 1 Bayesian Subspace Zeroth-Order Optimization (BSZO)

Input: parameters $\theta$ , learning rate $\eta$ , perturbation scale $\varepsilon$ , subspace dimension $k$ , sampling steps $m$ , prior variance $\sigma_p^2$ , noise variance $\sigma_e^2$ , smoothing factor $\alpha$ , max step $T$ for $t = 1$ to $T$ do

Sample $k$ random seeds $\{s_i\}_{i=1}^k$ Initialize $\mu \leftarrow \mathbf{0}_k$ , $\Sigma \leftarrow \sigma_p^2 I_k$ , $f_0 \leftarrow \mathcal{L}(\theta)$ Initialize cache $Y \leftarrow \{\}$ for $\tau = 1$ to $m$ do

if $\tau \leq k$ then $d \leftarrow e_\tau$ $\theta \leftarrow \theta + \varepsilon \cdot \text{RANDN}(n, s_\tau)$ $y \leftarrow (\mathcal{L}(\theta) - f_0)/\varepsilon$ $\theta \leftarrow \theta - \varepsilon \cdot \text{RANDN}(n, s_\tau)$ $Y[\tau] \leftarrow y$ ▷ Cache directional derivative

else $r \leftarrow (y - d^\top \mu)/\|d\|$ , $\sigma_e^2 \leftarrow (1 - \alpha) \sigma_e^2 + \alpha r^2$ $j \leftarrow \arg \max_i \Sigma_{ii}$ ▷ Find max uncertainty axis $d \leftarrow e_j$ $y \leftarrow Y[j]$ ▷ Reuse cached value

end if $K \leftarrow \Sigma d/(d^\top \Sigma d + \sigma_e^2)$ $\mu \leftarrow \mu + K(y - d^\top \mu)$ , $\Sigma \leftarrow \Sigma - K d^\top \Sigma$ end for

for $i = 1$ to $k$ do $\theta \leftarrow \theta - \eta \cdot \mu_i \cdot \text{RANDN}(n, s_i)$ end for

return $\theta$ RANDN(n, s): returns $n$ -dim Gaussian vector seeded by $s$ 

Definition 3.4. The one-side difference of L along the displacement $d \in \mathbb { R } ^ { k }$ in subspace B on minibatch ξ is defined as follows: 

$$
\hat {y} (\theta ; \xi , d) = \frac {\mathcal {L} (\theta + \varepsilon B d ; \xi) - \mathcal {L} (\theta ; \xi)}{\varepsilon},\tag{5}
$$

where $\varepsilon > 0$ is a small constant. 

## 3.2. Bayesian Gradient Estimation

For the $\mathcal { L } ( \boldsymbol { \theta } + \varepsilon B d )$ , the subspace gradient can be obtained through the chain rule: 

$$
g _ {d} := \nabla_ {d} \mathcal {L} (\theta + \varepsilon B d \mid d = 0) = \varepsilon B ^ {T} g,\tag{6}
$$

where $g : = \nabla { \mathcal { L } }$ is the real gradient of ${ \mathcal { L } } .$ In order to keep numerical accuracy controllable, we introduce the concept of normalized subspace gradient as $\begin{array} { r } { \tilde { g } : = B ^ { \top } g = \frac { g _ { s } } { \varepsilon } \in \mathbb { R } ^ { \bar { k } } } \end{array}$ 

Lemma 3.5. For any direction d $\in \mathbb { R } ^ { k }$ of subspace $B ,$ , the expectation of one-side difference ${ \hat { y } } ( d )$ satisfies: 

$$
\mathbb {E} [ \hat {y} (d) ] = d ^ {\top} B ^ {\top} g + O (\varepsilon L) \approx d ^ {\top} \tilde {g}.\tag{7}
$$


Table 1. Test accuracy (%) on RoBERTa-large (355M). We report the mean±std over 5 runs. The top two results are highlighted in bold. BSZO-B is the baseline version of BSZO without caching optimization.


<table><tr><td>METHOD</td><td>SST-2</td><td>RTE</td><td>CB</td><td>WIC</td><td>TREC</td><td>AVG</td></tr><tr><td>MEZO</td><td>92.22 (±0.42)</td><td>66.35 (±3.06)</td><td>86.07 (±5.56)</td><td>55.20 (±3.73)</td><td>85.36 (±2.33)</td><td>77.04</td></tr><tr><td>MEZO-ADAM</td><td>92.34 (±0.50)</td><td>63.61 (±1.41)</td><td>81.07 (±2.71)</td><td>52.85 (±4.19)</td><td>78.80 (±5.76)</td><td>73.73</td></tr><tr><td>HiZOO</td><td>91.44 (±0.45)</td><td>59.21 (±2.46)</td><td>76.43 (±1.96)</td><td>53.60 (±2.93)</td><td>63.44 (±2.61)</td><td>68.82</td></tr><tr><td>LOZO</td><td>91.83 (±0.30)</td><td>62.60 (±2.31)</td><td>84.29 (±3.87)</td><td>54.20 (±1.32)</td><td>77.76 (±2.15)</td><td>74.14</td></tr><tr><td>BSZO</td><td>92.66 (±0.21)</td><td>67.80 (±1.52)</td><td>85.71 (±1.79)</td><td>56.05 (±1.47)</td><td>84.16 (±0.54)</td><td>77.28</td></tr><tr><td>BSZO-B</td><td>92.27 (±0.41)</td><td>68.38 (±1.94)</td><td>84.29 (±1.49)</td><td>57.21 (±0.98)</td><td>84.80 (±1.57)</td><td>77.39</td></tr></table>

Based on Lemma3.5, we can model the one-side difference yˆ(d) as a linear observation of the normalized subspace gradient g˜ with Gaussian noise: 

$$
\hat {y} (d) = d ^ {\top} \tilde {g} + \nu , \quad \nu \sim \mathcal {N} (0, \sigma_ {e} ^ {2} \| d \| ^ {2}),\tag{8}
$$

where ν represents comprehensive noise term. The justification of the variance definition is provided in Appendix B.2. Then, we adopt a Bayesian approach by placing a Gaussian prior on $\widetilde { g } , \mathrm { i } . \mathrm { e } . , \widetilde { g } \sim \mathcal { N } ( 0 , \sigma _ { p } ^ { 2 } I _ { k } )$ which make the posterior computable in closed-form (Kalman, 1960). 

## 3.3. Posterior Update In Subspace

According to the standard Bayesian linear regression theory (Rasmussen & Williams, 2006), after m perturbations and observations $( d ^ { ( 1 ) } , \hat { y } ^ { ( 1 ) } ) , \dots , ( d ^ { ( m ) } , \hat { y } ^ { ( m ) } )$ , the posterior $\tilde { g } | Y \sim \mathcal { N } ( \mu ^ { ( m ) } , \Sigma ^ { ( \bar { m } ) } )$ is also a Gaussian distribution, where 

$$
\begin{array}{l} \Sigma^ {(m)} = \left(\sigma_ {p} ^ {- 2} I _ {k} + D ^ {\top} R ^ {- 1} D\right) ^ {- 1}, \\ \mu^ {(m)} = \Sigma^ {(m)} D ^ {\top} R ^ {- 1} Y. \end{array}\tag{9}
$$

Here, $D = [ d ^ { ( 1 ) } , \ldots , d ^ { ( m ) } ] ^ { \top } \in \mathbb { R } ^ { m \times k }$ is the design matrix, $Y = [ \hat { y } ^ { ( 1 ) } , \ldots , \hat { y } ^ { ( m ) } ] ^ { \top } \in \bar { \mathbb { R } } ^ { m }$ is the observation vector, and $R = \bar { \mathrm { d i a g } } ( \sigma _ { e } ^ { 2 } \| \dot { d } ^ { ( 1 ) } \| ^ { \bar { 2 } } , \dots , \sigma _ { e } ^ { 2 } \| d ^ { ( m ) } \| ^ { 2 } )$ is the noise covariance matrix. When m > k or Σ is already full-rank, we set the new sampling direction to the principal eigenvector of the covariance matrix, i.e., $d ^ { ( j ) } = \dot { v } _ { \mathrm { m a x } } ( \dot { \Sigma } ^ { ( j - 1 \bar { ) } } )$ 

After getting the posterior mean $\mu ^ { ( m ) }$ , we can use it as the final displacement in subspace $B ,$ which means the parameters updated by: 

$$
\Delta \theta = - \eta B \mu^ {(k)},\tag{10}
$$

where $\eta > 0$ is learning rate. In this way, we can use the finite k forward passes to update the parameters k times, with $\mu ^ { ( k ) }$ controlling the step size in subspace. This means that, for the same batch, the parameters move along a ”diagonal” direction rather than a single direction. 

Corollary 3.6. Under coordinate-axis sampling, i.e., $m =$ k and $\boldsymbol { d } ^ { ( i ) } = \boldsymbol { e } _ { i }$ (the i-th standard basis vector), then the posterior mean and covariance reduce to: 

$$
\begin{array}{c} \Sigma^ {(k)} = \gamma I _ {k}, \\ \mu^ {(k)} = \gamma Y, \end{array}\tag{11}
$$

where $\begin{array} { r } { \gamma : = \frac { \sigma _ { p } ^ { 2 } } { \sigma _ { p } ^ { 2 } + \sigma _ { e } ^ { 2 } } \in ( 0 , 1 ) } \end{array}$ is the shrinkage factor. Corollary3.6 simplifies the form of the posterior distribution, thereby making the analysis and update easier. Thus, we adopt coordinate-axis sampling as the default sampling strategy in BSZO (for the first k sampling directions). 

Theorem 3.7. Let $\Delta \theta = - \eta B \mu ^ { ( k ) }$ . Under Assumptions 3.1 and Assumption3.2, we have: 

$$
\mathbb {E} [ \Delta \theta ] = - \eta \gamma k \cdot \nabla \mathcal {L} (\theta) + O (\varepsilon^ {3})\tag{12}
$$

The above theorem shows that the expected update direction aligns with the negative gradient under coordinate-axis sampling. Furthermore, the analysis of the expected direction under adaptive sampling is provided in Theorem B.6 (Appendix B.5). 

## 3.4. Algorithm

Clearly, the choice of γ is crucial. We observe that the norm of the projected gradient estimated via finite differences remains stable during the early and middle stages of optimization, but tends to grow in later stages due to numerical precision limitations, which restricts the achievable convergence accuracy. To this end, we design a residual-based mechanism that adaptively adjusts $\sigma _ { e }$ after the τ -th sample: 

$$
\begin{array}{c} r _ {\tau} := \frac {\hat {y} ^ {(\tau)} - d ^ {(\tau) ^ {\top}} \mu^ {(\tau - 1)}}{\| d ^ {(\tau)} \|}, \\ (\sigma_ {e} ^ {(\tau)}) ^ {2} = (1 - \alpha) (\sigma_ {e} ^ {(\tau - 1)}) ^ {2} + \alpha r _ {\tau} ^ {2}, \end{array}\tag{13}
$$

where $\alpha \in ( 0 , 1 )$ is the smoothing factor. 

Corollary 3.6 shows that under coordinate-axis sampling, the posterior covariance Σ degenerates into a diagonal matrix with a single distinct eigenvalue, implying that any axis-aligned direction may serve as the adaptive sampling direction when $j > k .$ The residual-based adaptation breaks this degeneracy by differentiating the diagonal entries of Σ, thereby producing a meaningful adaptive sampling direction. However, the diagonal structure implies that the adaptive sampling direction always coincides with one of the coordinate axes, which can lead to redundant computation. To address this, we cache the $( d , y )$ pairs from the first k samples within each batch. When $j > k ,$ , we directly reuse the cached pair corresponding to the largest diagonal entry of Σ, eliminating the need for an additional forward pass. This extra sample leverages the updated residual to more precisely correct the step size along the direction of greatest uncertainty. In practice, we set $m = k + 1$ by default. 


Table 2. Memory complexity comparison of different methods.


<table><tr><td>Method</td><td>Memory</td><td>Additional Space</td></tr><tr><td>MeZO-SGD</td><td>O(n)</td><td>O(1)</td></tr><tr><td>MeZO-Adam</td><td>O(n)</td><td>O(n)</td></tr><tr><td>HiZOO</td><td>O(n)</td><td>O(n)</td></tr><tr><td>LOZO</td><td>O(n)</td><td>O(1)</td></tr><tr><td>BSZO (Ours)</td><td>O(n)</td><td>O(k2)</td></tr></table>

The main procedure of BSZO is summarized in Algorithm 1. Following MeZO (Malladi et al., 2023), we store perturbation vectors via random seeds rather than explicitly, requiring only $O ( k ^ { 2 } )$ additional space. A basic version without caching is provided in Algorithm 2 (Appendix A), which supports arbitrary initial sampling directions and additional adaptive sampling steps. In this version, the adaptive sampling performs extra forward passes to obtain new function values. Typically, the result of this forward pass coincides with the cached value. However, under reduced precision (fp16 or bf16), certain GPU operations use nondeterministic algorithms (PyTorch Team, 2024), causing function evaluations to differ across calls even with identical inputs and random seeds. Moreover, due to numerical errors, parameters do not fully recover after perturbation and restoration. As a result, the extra forward pass in the basic version yields a value different from the cached one, better reflecting the local landscape at the perturbed point and leading to improved performance (as confirmed in Section 5.3). To examine this effect, we include the coordinate-axis sampling variant of Algorithm 2 as an experimental baseline (denoted as BSZO-B). Table 2 compares the memory complexity of different methods, showing that BSZO is also memory-efficient. We analyze the convergence properties of BSZO in the next section. 

## 4. Theoretical Analysis

Definition 4.1. Let $\Sigma = \operatorname { C o v } ( \zeta )$ be the covariance matrix of the gradient noise, the effective noise $\sigma _ { e } ^ { 2 }$ can be decomposed 

![image](images/Robust_and_Efficient_Zeroth_Order_LLM_Fine_Tuning_via_Adaptive_Bayesian_Subspace_Optimizer/fig4.jpg)



Figure 2. GPU memory usage comparison across different models. BSZO maintains memory consumption comparable to MeZO, while MeZO-Adam and HiZOO require significantly more memory due to storing optimizer states or Hessian estimates.


as: 

$$
\sigma_ {e} ^ {2} = \sigma_ {\varepsilon} ^ {2} + \mathrm{tr} (\Sigma),\tag{14}
$$

where $\operatorname { t r } ( \Sigma ) \leq \sigma _ { g } ^ { 2 }$ (Assumption 3.2). The justification for this definition is provided by Lemma B.5 in Appendix B.2. For analytical tractability, we assume that $\sigma _ { e }$ is fixed (taking the worst-case noise across batches gives the same result). The convergence of BSZO is characterized by the following theorem: 

Theorem 4.2. Under Assumptions 3.1 and 3.2, let $\tilde { n } =$ $n + k + 1$ be effective dimension. Suppose $m = k$ and $\begin{array} { r } { \eta < \frac { 2 } { L \gamma \tilde { n } } } \end{array}$ . Then, after T iterations, the following inequality holds: 

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} [ \| \nabla \mathcal {L} (\theta_ {t}) \| ^ {2} ] \leq \frac {\mathcal {L} (\theta_ {0}) - \mathcal {L} ^ {*}}{\beta (\eta) \eta \gamma k T} + \frac {L \eta \gamma (\tilde {n} \cdot t r (\Sigma) + n \sigma_ {\varepsilon} ^ {2})}{2 \beta (\eta)},
$$

where $\begin{array} { r } { \beta ( \eta ) : = 1 - \frac { L \eta \gamma \tilde { n } } { 2 } } \end{array}$ and $\sigma _ { e } ^ { 2 } = \sigma _ { \varepsilon } ^ { 2 } + t r ( \Sigma )$ 

(15) 

Corollary 4.3. Let $\begin{array} { r } { \eta = \frac { 1 } { L \gamma \tilde { n } } } \end{array}$ , then $\beta = 1 / 2 ,$ , which simplifies Theorem 4.2 to: 

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} [ \| \nabla \mathcal {L} (\theta_ {t}) \| ^ {2} ] \leq \frac {2 L \gamma \tilde {n} \Delta_ {0}}{k T} + t r (\Sigma) + \frac {n}{\tilde {n}} \sigma_ {\varepsilon} ^ {2},\tag{16}
$$

where $\Delta _ { 0 } : = \mathcal { L } ( \theta _ { 0 } ) - \mathcal { L } ^ { * }$ 

According to Corollary 4.3, the convergence rate of BSZO is improved by the factor of subspace dimension k. Although γ slightly reduces the convergence rate, it is crucial for training stability. We also analyze the convergence under adaptive sampling in Theorem B.7 (Appendix B.6). 

## 5. Experiments

In this section, we evaluate the performance of BSZO and BSZO-B (Section 3.4) on various fine-tuning tasks in different language models, comparing them with several baselines: MeZO (Malladi et al., 2023), MeZO-Adam (Malladi et al., 2023), HiZOO (Zhao et al., 2025), and LOZO (Chen et al., 2024). Our experiments show that both variants achieve excellent robustness and strong accuracy across most scenarios, requiring only the GPU memory needed for forward propagation, making them more cost-effective than HiZOO and MeZO-Adam. 


Table 3. Experiments on three different models (OPT-1.3B, Mistral-7B, OPT-13B). We show the test accuracy (%) of MeZO, MeZO-Adam, HiZOO, LOZO, BSZO, and BSZO-B on them, with the top two results highlighted in bold. BSZO-B is the baseline version of BSZO. Since fp16 can cause training crashes with Adam, we did not record the results of ZO-Adam for Mistral-7B. * indicates training collapse due to numerical overflow under fp16 precision.


<table><tr><td rowspan="2">MODEL</td><td rowspan="2">METHOD</td><td>SST-2</td><td>RTE</td><td>COPA</td><td>WIC</td><td>WSC</td><td>TREC</td><td rowspan="2">AVG</td></tr><tr><td>SENTIMENT</td><td>NLI</td><td colspan="3">REASONING</td><td>TOPIC</td></tr><tr><td>OPT-1.3B</td><td>MEZO</td><td>91.74</td><td>64.98</td><td>76.0</td><td>58.78</td><td>59.62</td><td>80.6</td><td>71.95</td></tr><tr><td>OPT-1.3B</td><td>MEZO-ADAM</td><td>93.35</td><td>60.29</td><td>75.0</td><td>56.58</td><td>62.50</td><td>79.4</td><td>71.19</td></tr><tr><td>OPT-1.3B</td><td>HiZOO</td><td>91.51</td><td>62.09</td><td>77.0</td><td>56.58</td><td>63.46</td><td>66.2</td><td>69.48</td></tr><tr><td>OPT-1.3B</td><td>LOZO</td><td>92.66</td><td>63.18</td><td>75.0</td><td>56.58</td><td>57.69</td><td>75.8</td><td>70.15</td></tr><tr><td>OPT-1.3B</td><td>BSZO</td><td>92.43</td><td>66.79</td><td>79.0</td><td>59.88</td><td>64.42</td><td>87.0</td><td>74.92</td></tr><tr><td>OPT-1.3B</td><td>BSZO-B</td><td>93.01</td><td>64.98</td><td>81.0</td><td>59.09</td><td>61.54</td><td>87.4</td><td>74.50</td></tr><tr><td>MISTRAL-7B</td><td>MEZO</td><td>90.94</td><td>64.26</td><td>88.0</td><td>56.58</td><td>63.46</td><td>88.6</td><td>75.31</td></tr><tr><td>MISTRAL-7B</td><td>HiZOO</td><td>93.01</td><td>63.90</td><td>90.0</td><td>55.64</td><td>63.46</td><td>*</td><td>73.20</td></tr><tr><td>MISTRAL-7B</td><td>LOZO</td><td>92.43</td><td>61.37</td><td>86.0</td><td>57.83</td><td>63.46</td><td>*</td><td>72.22</td></tr><tr><td>MISTRAL-7B</td><td>BSZO</td><td>94.50</td><td>75.81</td><td>87.0</td><td>60.03</td><td>59.62</td><td>90.0</td><td>77.83</td></tr><tr><td>MISTRAL-7B</td><td>BSZO-B</td><td>94.04</td><td>78.70</td><td>87.0</td><td>59.72</td><td>60.58</td><td>91.0</td><td>78.51</td></tr><tr><td>OPT-13B</td><td>MEZO</td><td>85.89</td><td>62.09</td><td>80.0</td><td>54.55</td><td>60.58</td><td>59.4</td><td>67.09</td></tr><tr><td>OPT-13B</td><td>MEZO-ADAM</td><td>79.82</td><td>61.73</td><td>81.0</td><td>54.39</td><td>57.69</td><td>62.2</td><td>66.14</td></tr><tr><td>OPT-13B</td><td>HiZOO</td><td>72.71</td><td>62.46</td><td>80.0</td><td>52.35</td><td>46.15</td><td>19.8</td><td>55.58</td></tr><tr><td>OPT-13B</td><td>LOZO</td><td>86.12</td><td>57.04</td><td>80.0</td><td>55.96</td><td>59.62</td><td>60.4</td><td>66.52</td></tr><tr><td>OPT-13B</td><td>BSZO</td><td>93.23</td><td>69.31</td><td>83.0</td><td>56.27</td><td>61.54</td><td>79.2</td><td>73.76</td></tr><tr><td>OPT-13B</td><td>BSZO-B</td><td>91.86</td><td>71.84</td><td>85.0</td><td>53.14</td><td>64.42</td><td>80.8</td><td>74.51</td></tr></table>

## 5.1. Experimental Setup

Language Models. The experiments in this paper center on two categories of models: masked Language Models (mLMs) and decoder-only Large Language Models (LLMs). For mLMs, we adopt RoBERTa-large (355M) (Liu et al., 2019) as the backbone model. For decoder-only LLMs, we select OPT-1.3B and OPT-13B (Zhang et al., 2022), as well as Mistral-7B (Jiang et al., 2023). 

Datasets. We full fine-tune the above models on tasks from the GLUE (Wang et al., 2018), SuperGLUE (Wang et al., 2019) and TREC (Li & Roth, 2002) benchmarks, including Stanford Sentiment Treebank (SST-2), Boolean Questions (BoolQ) (Clark et al., 2019), Recognizing Textual Entailment (RTE) (Dagan et al., 2005), Choice of Plausible Alternatives (COPA) (Roemmele et al., 2011), Word-in-Context (WIC) (Pilehvar & Camacho-Collados, 2019), Winograd Schema Challenge (WSC) (Levesque et al., 2012), CommitmentBank (CB) (De Marneffe et al., 

2019), and TREC. Following HiZOO (Zhao et al., 2025), we use the first $n _ { 1 }$ samples (up to 1000) from the training set for training and the next $n _ { 2 }$ samples for validation. The original validation set serves as the test set. See Table 6 in Appendix C for specific values of $n _ { 1 }$ and $n _ { 2 }$ 

Hyperparameters. For BSZO and BSZO-B, we set the default subspace dimension k = 2 and the number of samples $m = k + 1$ . This results in 3 forward passes per step for BSZO (with caching) and 4 for BSZO-B (without caching). BSZO matches HiZOO’s forward pass count. As discussed in Section 3.4, we report results for both BSZO and BSZO-B across all models, with particular focus on comparing them under reduced precision (Mistral-7B in fp16 and OPT-13B in bf16) to examine the caching effect. Other methods use their default hyperparameters. Given the slower convergence of zeroth-order methods, all experiments are trained for up to 20,000 steps (Zhang et al., 2024), with early stopping applied when validation performance does not improve for 8 evaluations (4,000 steps). For every experiment, we set the perturbation scale $\mathbf { t o } \varepsilon = 1 0 ^ { - 4 }$ and the batch size to 16. Hyperparameters are tuned via grid search. We select the best configuration based on validation performance and report its test accuracy. Due to memory constraints, we load OPT-13B in bf16 precision and Mistral-7B in fp16 precision, while other models use fp32. All experiments are conducted on a single H200 GPU. More details are provided in Appendix C. 

## 5.2. Performance in Masked Language Models

BSZO achieves stable and competitive performance on mLMs. As shown in Table 1, BSZO-B reaches 77.39% average accuracy on RoBERTa-large, surpassing MeZO (77.04%, +0.35%), MeZO-Adam (73.73%, +3.66%), Hi-ZOO (68.82%, +8.57%), and LOZO (74.14%, +3.25%). BSZO achieves 77.28% average accuracy, second only to BSZO-B. BSZO secures top result on SST-2 (92.66%), while BSZO-B excels on RTE (68.38%) and WIC (57.21%). Moreover, BSZO exhibits notably lower variance across tasks (see Table 11 for raw results). Both variants demonstrate strong and consistent performance across all tasks. 

## 5.3. Performance in decoder-only models

BSZO performs well on larger LLMs. Table 3 shows that BSZO outperforms baselines on decoder-only models, with gains increasing as model size grows. BSZO-B typically maintains a small lead over BSZO. 

OPT-1.3B. BSZO achieves 74.92% average accuracy, the highest among all methods, beating MeZO (71.95%, +2.97%), MeZO-Adam (71.19%, +3.73%), HiZOO (69.48%, +5.44%), and LOZO (70.15%, +4.77%). BSZO-B reaches 74.50% average accuracy. BSZO secures top results on RTE (66.79%), WIC (59.88%), and WSC (64.42%), while BSZO-B excels on COPA (81.0%) and TREC (87.4%). Both variants perform well across most tasks. 

Mistral-7B (fp16). BSZO reaches 77.83% on average, ahead of MeZO (75.31%, +2.52%), HiZOO (73.20%, +4.63%), and LOZO (72.22%, +5.61%). It also achieves the best results on SST-2 (94.50%, +1.49% vs HiZOO) and WIC (60.03%, +3.45% vs MeZO). BSZO-B reaches 78.51% on average, excelling on RTE (78.70%) and TREC (91.0%). The small 0.68% gap shows that the two variants perform very similarly. 

OPT-13B (bf16). The gains grow larger here. BSZO reaches 73.76% on average, up 6.67% over MeZO (67.09%), 7.62% over MeZO-Adam (66.14%), and 7.24% over LOZO (66.52%). BSZO achieves strong results across tasks, including top performance on WIC (56.27%), with particularly notable gains on SST-2 (93.23%, +7.34% vs MeZO) and TREC (79.2%, +19.8% vs MeZO). BSZO-B reaches 74.51% on average (+7.42% vs MeZO), with stronger balance across tasks. BSZO-B maintains a slight edge with one additional forward pass, though the gap in average accuracy remains very small (0.75%). 

Robustness. Reduced precision exposes fragility in several baselines (Table 3) and Figure 1. HiZOO and LOZO are particularly affected: on Mistral-7B (fp16), both methods suffer from TREC training overflow (*). On OPT-13B (bf16), all baseline methods show varying degrees of performance degradation compared to OPT-1.3B, with HiZOO being especially severe—its average accuracy drops from 69.48% to 55.58%, with TREC collapsing to 19.8% and 


Table 4. Memory usage (GB) and per-step time (ms) across different models.


<table><tr><td rowspan="2">Method</td><td colspan="2">OPT-1.3B</td><td colspan="2">Mistral-7B</td><td colspan="2">OPT-13B</td></tr><tr><td>Mem</td><td>Time</td><td>Mem</td><td>Time</td><td>Mem</td><td>Time</td></tr><tr><td>MeZO</td><td>9.1</td><td>109.7</td><td>18.3</td><td>283.9</td><td>30.0</td><td>464.2</td></tr><tr><td>MeZO-Adam</td><td>19.7</td><td>135.1</td><td>47.6</td><td>373.1</td><td>82.1</td><td>614.5</td></tr><tr><td>HiZOO</td><td>15.7</td><td>188.0</td><td>34.3</td><td>540.2</td><td>58.9</td><td>877.1</td></tr><tr><td>LOZO</td><td>9.3</td><td>102.0</td><td>18.3</td><td>274.2</td><td>30.0</td><td>452.0</td></tr><tr><td>BSZO</td><td>9.8</td><td>97.0</td><td>18.8</td><td>275.7</td><td>30.1</td><td>440.5</td></tr></table>

WSC to 46.15%. We suspect H200’s default TF32 mode introduces errors in Hessian-based estimates. In contrast, BSZO and BSZO-B remain stable throughout all precision settings, with BSZO-B even maintaining performance (from 74.50% to 74.51%). 

## 5.4. Memory and Time Efficiency

BSZO keeps memory usage low. As shown in Figure 2 and Table 4, BSZO’s memory footprint stays close to MeZO across three model scales—ranging from 1.00× to 1.08× of MeZO’s usage. In contrast, HiZOO and MeZO-Adam need 1.73×–1.96× and 2.16×–2.74× more memory because they store additional optimizer states (momentum, Hessian estimates). BSZO avoids this overhead by using only O(k<sup>2</sup>) extra space for the posterior covariance and adaptive noise estimation. 

BSZO runs fast. Table 4 also reports per-step time. BSZO and LOZO are the fastest—both under 100ms per step on OPT-1.3B. HiZOO is roughly 2× slower due to Hessian estimation, and MeZO-Adam incurs extra cost from momentum updates. 

## 5.5. Ablation Study

Table 5 shows ablation results on OPT-1.3B for two design choices of BSZO: subspace dimension k and sample count m. In Table 5(a), when m = k, RTE accuracy climbs from 60.29% (k = 1) to 67.51% (k = 4), while SST-2 peaks at k = 8 (93.23%), suggesting that increasing k generally improves performance. In Table 5(b), with extra refinement (m = k + 1), RTE performance improves consistently. Comparing to Table 5(a), m = k + 1 boosts RTE by 1-2% at most k levels (e.g., from 64.26% to 66.79% at k = 2, from 66.07% to 68.59% at k = 8). This confirms that the adaptive sampling step refines the posterior estimate (see Table 12 for more details). Table 5(c) investigates adaptive noise under bf16 precision on OPT-1.3B. As k grows, the gap between w/ and w/o adaptive noise becomes more pronounced: at k = 8, the adaptive variant leads by 8.67% on RTE, indicating that adaptive noise yields substantial gains in low-precision settings. Table 5(d) validates this on 


Table 5. Ablation studies. (a) Effect of subspace dimension k with m = k on OPT-1.3B. (b) Effect of $m = k ^ { - } + 1$ on OPT-1.3B. (c) Effect of adaptive noise on OPT-1.3B (bf16). (d) Effect of adaptive noise on OPT-13B (bf16). Best results in bold. Full results in fp32 are in Table 12.


<table><tr><td colspan="3">(a) Effect of k</td><td colspan="3">(b) Effect of m</td><td colspan="3">(c) Adaptive Noise</td></tr><tr><td>k</td><td>SST-2</td><td>RTE</td><td>k</td><td>SST-2</td><td>RTE</td><td>k</td><td>w/</td><td>w/o</td></tr><tr><td>1</td><td>92.32</td><td>60.29</td><td>1</td><td>91.74</td><td>61.37</td><td>1</td><td>54.15</td><td>55.24</td></tr><tr><td>2</td><td>92.78</td><td>64.26</td><td>2</td><td>92.43</td><td>66.79</td><td>2</td><td>57.76</td><td>56.32</td></tr><tr><td>4</td><td>92.66</td><td>67.51</td><td>4</td><td>93.58</td><td>66.43</td><td>4</td><td>61.73</td><td>56.32</td></tr><tr><td>8</td><td>93.23</td><td>66.07</td><td>8</td><td>93.23</td><td>68.59</td><td>8</td><td>66.43</td><td>57.76</td></tr></table>

<table><tr><td colspan="7">(d) Adaptive Noise on OPT-13B (bf16)</td></tr><tr><td>Method</td><td>SST-2</td><td>RTE</td><td>WSC</td><td>COPA</td><td>TREC</td><td>WIC</td></tr><tr><td>w/ adaptive</td><td>93.23</td><td>69.31</td><td>61.54</td><td>83.00</td><td>79.20</td><td>56.27</td></tr><tr><td>w/o adaptive</td><td>91.97</td><td>63.18</td><td>58.65</td><td>85.00</td><td>75.00</td><td>54.70</td></tr></table>

OPT-13B (bf16), where adaptive noise brings improvements on 5 out of 6 tasks, with RTE gaining 6.13%. 

## 6. Conclusion

In this work, we introduce BSZO, which is the first zerothorder optimizer that applies Kalman filtering to aggregate gradient information across multiple perturbation directions for LLM fine-tuning. By treating finite-difference measurements as noisy observations of the true gradient, BSZO builds a posterior distribution over the projected gradient and refines it through Bayesian updates. We design a residual-based adaptive mechanism to adjust the perturbation scale adaptively without manual tuning. Our theoretical analysis shows that BSZO improves the convergence rate by a factor of k/γ over standard ZO methods. Experiments on RoBERTa, Mistral, and OPT show that BSZO achieves strong accuracy across various tasks, remains stable under fp16/bf16 precision where existing methods often collapse, and keeps memory usage close to inference-only baselines. 

## Software and Data

Our implementation is available at https://github. com/AeonianQuill/BSZO. Datasets used in this work (GLUE, SuperGLUE, TREC) are publicly accessible and should be downloaded separately. Pre-trained models can also be obtained from Hugging Face. 

## Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. 

## References



Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33: 1877–1901, 2020. 





Chen, Y., Zhang, Y., Cao, L., Yuan, K., and Wen, Z. Enhancing zeroth-order fine-tuning for language models with low-rank structures. arXiv preprint arXiv:2410.07698, 2024. 





Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of NAACL-HLT, pp. 2924–2936, 2019. 





Dagan, I., Glickman, O., and Magnini, B. The PASCAL recognising textual entailment challenge. In Machine Learning Challenges Workshop, pp. 177–190. Springer, 2005. 





De Marneffe, M.-C., Simons, M., and Tonhauser, J. The CommitmentBank: Investigating projection in naturally occurring discourse. In Proceedings of Sinn und Bedeutung, volume 23, pp. 107–124, 2019. 





Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT, pp. 4171–4186, 2019. 





Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., and Gelly, S. Parameter-efficient transfer learning for NLP. In International Conference on Machine Learning, pp. 2790–2799, 2019. 





Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 





Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de Las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.-A., Stock, P., Le Scao, T., Lavril, T., Wang, T., Lacroix, T., and El Sayed, W. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. 





Kalman, R. E. A new approach to linear filtering and prediction problems. Journal of Basic Engineering, 82(1): 35–45, 1960. 





Levesque, H. J., Davis, E., and Morgenstern, L. The Winograd schema challenge. In Proceedings of the Thirteenth International Conference on Principles of Knowledge Representation and Reasoning, 2012. 





Li, X. and Roth, D. Learning question classifiers. In COL-ING 2002: The 19th International Conference on Computational Linguistics, 2002. 





Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., and Stoyanov, V. RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692, 2019. 





Liu, Y., Zhu, Z., Gong, C., Cheng, M., Hsieh, C.-J., and You, Y. Sparse MeZO: Less parameters for better performance in zeroth-order LLM fine-tuning. arXiv preprint arXiv:2402.15751, 2024. 





Malladi, S., Gao, T., Nichani, E., Damian, A., Lee, J. D., Chen, D., and Arora, S. Fine-tuning language models with just forward passes. Advances in Neural Information Processing Systems, 36:53038–53075, 2023. 





Mania, H., Guy, A., and Recht, B. Simple random search of static linear policies is competitive for reinforcement learning. In Advances in Neural Information Processing Systems, volume 31, 2018. 





Pilehvar, M. T. and Camacho-Collados, J. WiC: the wordin-context dataset for evaluating context-sensitive meaning representations. In Proceedings of NAACL-HLT, pp. 1267–1273, 2019. 





PyTorch Team. Reproducibility. https://pytorch. org/docs/stable/notes/randomness.html, 2024. Accessed: 2026-01-02. 





Rasmussen, C. E. and Williams, C. K. I. Gaussian Processes for Machine Learning. MIT Press, 2006. 





Roemmele, M., Bejan, C. A., and Gordon, A. S. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In AAAI Spring Symposium on Logical Formalizations ofCommonsense Reasoning, 2011. 





Salimans, T., Ho, J., Chen, X., Sidor, S., and Sutskever, I. Evolution strategies as a scalable alternative to reinforcement learning. arXiv preprint arXiv:1703.03864, 2017. 





Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., and De Freitas, N. Taking the human out of the loop: A review of Bayesian optimization. Proceedings of the IEEE, 104(1):148–175, 2016. 





Spall, J. C. Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. IEEE Transactions on Automatic Control, 37(3):332–341, 1992. 





Sun, Y., Huang, T., Ding, L., Shen, L., and Tao, D. TeZO: Empowering the low-rankness on the temporal dimension in the zeroth-order optimization for fine-tuning LLMs. arXiv preprint arXiv:2501.19057, 2025. 





Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E.,` Azhar, F., et al. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 





Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP, pp. 353–355, 2018. 





Wang, A., Pruksachatkun, Y., Nangia, N., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. SuperGLUE: A stickier benchmark for general-purpose language understanding systems. In Advances in Neural Information Processing Systems, volume 32, 2019. 





Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., Mihaylov, T., Ott, M., Shleifer, S., Shuster, K., Simig, D., Koura, P. S., Sridhar, A., Wang, T., and Zettlemoyer, L. OPT: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022. 





Zhang, Y., Li, P., Hong, J., Li, J., Zhang, Y., Zheng, W., Chen, P.-Y., Lee, J. D., Yin, W., Hong, M., Wang, Z., Liu, S., and Chen, T. Revisiting zeroth-order optimization for memory-efficient LLM fine-tuning: A benchmark. In International Conference on Machine Learning, 2024. 





Zhang, Z. Scalable derivative-free optimization algorithms with low-dimensional subspace techniques. arXiv preprint arXiv:2501.04536, 2025. 





Zhao, Y., Dang, S., Ye, H., Dai, G., Qian, Y., and Tsang, I. W. Second-order fine-tuning without pain for LLMs: A Hessian informed zeroth-order optimizer. In International Conference on Learning Representations, 2025. 



## A. BSZO Basic Algorithm

We provide a basic version of BSZO without caching optimization, which supports arbitrary sampling directions when m $> k$ 

Algorithm 2 Bayesian Subspace Zeroth-Order Optimization (Basic Version)

Input: parameters $\theta$ , learning rate $\eta$ , perturbation scale $\varepsilon$ , subspace dimension $k$ , sampling steps $m$ , prior variance $\sigma_p^2$ , noise variance $\sigma_e^2$ , smoothing factor $\alpha$ , max step $T$ for $t = 1$ to $T$ do

Sample $k$ random seeds $\{s_i\}_{i=1}^k$ Initialize $\mu \leftarrow 0_k, \Sigma \leftarrow \sigma_p^2 I_k, f_0 \leftarrow \mathcal{L}(\theta)$ for $\tau = 1$ to $m$ do $d \leftarrow d_\tau$ if $\tau \leq k$ , else $d \leftarrow \arg \max_{\|v\|=1} v^\top \Sigma v$ for $i = 1$ to $k$ do $\theta \leftarrow \theta + \varepsilon \cdot d_i \cdot \text{RANDN}(n, s_i)$ if $d_i > 10^{-10}$ end for $y \leftarrow (\mathcal{L}(\theta) - f_0)/\varepsilon$ for $i = 1$ to $k$ do $\theta \leftarrow \theta - \varepsilon \cdot d_i \cdot \text{RANDN}(n, s_i)$ if $d_i > 10^{-10}$ end for $r \leftarrow (y - d^\top \mu)/\|d\|$ , $\sigma_e^2 \leftarrow (1 - \alpha) \sigma_e^2 + \alpha r^2$ $K \leftarrow \Sigma d/(d^\top \Sigma d + \sigma_e^2)$ $\mu \leftarrow \mu + K(y - d^\top \mu), \quad \Sigma \leftarrow \Sigma - K d^\top \Sigma$ end for

for $i = 1$ to $k$ do $\theta \leftarrow \theta - \eta \cdot \mu_i \cdot \text{RANDN}(n, s_i)$ end for

end for

return $\theta$ RANDN(n, s): returns $n$ -dim Gaussian vector seeded by $s$ 

## B. Theoretical Proofs

## B.1. Auxiliary Lemmas

Lemma B.1 (Expectation of Direction Derivative). Under Assumption 3.1, the one-sided difference satisfies: 

$$
\mathbb {E} [ \hat {y} (d) ] = d ^ {\top} B ^ {\top} g + O (\varepsilon L)\tag{17}
$$

Proof. By $\mathbb { E } [ \mathcal { L } ( \theta ; \xi ) ] = \mathcal { L } ( \theta )$ 

$$
\mathbb {E} [ \hat {y} (d) ] = \frac {\mathcal {L} (\theta_ {0} + \varepsilon B d) - \mathcal {L} (\theta_ {0})}{\varepsilon}\tag{18}
$$

By Taylor expansion at $\theta _ { 0 }$ : 

$$
\mathcal {L} (\theta_ {0} + \varepsilon B d) = \mathcal {L} (\theta_ {0}) + \varepsilon \langle \nabla \mathcal {L} (\theta_ {0}), B d \rangle + \frac {\varepsilon^ {2}}{2} (B d) ^ {\top} H (B d) + O (\varepsilon^ {3})\tag{19}
$$

where $H = \nabla ^ { 2 } \mathcal { L } ( \theta _ { 0 } )$ is the Hessian. 

Substituting: 

$$
\mathbb {E} [ \hat {y} (d) ] = \frac {\varepsilon d ^ {\top} B ^ {\top} g + \frac {\varepsilon^ {2}}{2} d ^ {\top} B ^ {\top} H B d + O (\varepsilon^ {3})}{\varepsilon} = d ^ {\top} B ^ {\top} g + \frac {\varepsilon}{2} d ^ {\top} B ^ {\top} H B d + O (\varepsilon^ {2})\tag{20}
$$

By Assumption 3.1, $\| H \| \leq L ,$ so $\vert d ^ { \top } B ^ { \top } H B d \vert \leq L \Vert B d \Vert ^ { 2 }$ . Thus the bias is $O ( \varepsilon L )$ 

Lemma B.2 (Variance of Direction Derivative). Let $\Sigma = C o \nu ( \zeta )$ where $\zeta = \nabla \mathcal { L } ( \theta ; \xi ) - \nabla \mathcal { L } ( \theta )$ . When using the same mini-batch ξ for bothfunction evaluations: 

(a) Conditional variance: Va $r ( { \hat { y } } ( d ) | B ) = ( B d ) ^ { \top } \Sigma ( B d ) + O ( \varepsilon ^ { 4 } )$ 

(b) Unconditional variance: $\mathbb { E } _ { B } [ V a r ( \hat { y } ( d ) | B ) ] = t r ( \Sigma ) + O ( \varepsilon ^ { 4 } )$ 

Proof. Key insight: Using the same mini-batch $\xi$ for both evaluations causes the noise to be correlated, not independent. (a) Conditional variance derivation: 

For fixed ξ, Taylor expand the random loss $\mathcal { L } ( \theta ; \xi )$ at $\theta _ { 0 }$ : 

$$
\mathcal {L} \left(\theta_ {0} + \varepsilon B d; \xi\right) = \mathcal {L} \left(\theta_ {0}; \xi\right) + \varepsilon \langle \nabla \mathcal {L} \left(\theta_ {0}; \xi\right), B d \rangle + O \left(\varepsilon^ {2}\right)\tag{21}
$$

Since both evaluations use the same $\xi ,$ the base term $\mathcal { L } ( \theta _ { 0 } ; \xi )$ cancels: 

$$
\hat {y} (d) = \frac {\mathcal {L} (\theta_ {0} + \varepsilon B d ; \xi) - \mathcal {L} (\theta_ {0} ; \xi)}{\varepsilon} = \langle \nabla \mathcal {L} (\theta_ {0}; \xi), B d \rangle + O (\varepsilon)\tag{22}
$$

Let $\nabla \mathcal { L } ( \theta _ { 0 } ; \xi ) = \nabla \mathcal { L } ( \theta _ { 0 } ) + \xi$ where $\zeta$ is zero-mean noise with $\begin{array} { r } { \mathbf { C o v } ( \zeta ) = \Sigma } \end{array}$ . Given $B \colon$ 

$$
\operatorname{Var} (\hat {y} (d) | B) = \operatorname{Var} (\langle \zeta , B d \rangle | B) = (B d) ^ {\top} \operatorname{Cov} (\zeta) (B d) = (B d) ^ {\top} \Sigma (B d) + O (\varepsilon^ {2})\tag{23}
$$

(b) Unconditional variance derivation: 

For coordinate-axis sampling $d = e _ { i }$ , we have $B d = z _ { i } \sim \mathcal { N } ( 0 , I _ { n } )$ 

Taking expectation over B: 

$$
\mathbb {E} _ {B} [ (B d) ^ {\top} \Sigma (B d) ] = \mathbb {E} [ z _ {i} ^ {\top} \Sigma z _ {i} ]\tag{24}
$$

By the trace trick: 

$$
\mathbb {E} [ z _ {i} ^ {\top} \Sigma z _ {i} ] = \mathbb {E} [ \mathrm{tr} (\Sigma z _ {i} z _ {i} ^ {\top}) ] = \mathrm{tr} (\Sigma \cdot \mathbb {E} [ z _ {i} z _ {i} ^ {\top} ]) = \mathrm{tr} (\Sigma \cdot I _ {n}) = \mathrm{tr} (\Sigma)\tag{25}
$$

By Assumption 3.2, $\operatorname { t r } ( \Sigma ) = \mathbb { E } [ \| \zeta \| ^ { 2 } ] \leq \sigma _ { g } ^ { 2 } .$ 

Lemma B.3 (High-Dimensional Approximate Orthogonality). Let $z _ { 1 } , \ldots , z _ { k } \stackrel { i i d } { \sim } { \mathcal { N } } ( 0 , I _ { n } )$ . When $n \gg k ^ { 2 } .$ 

(a) $\| z _ { i } \| ^ { 2 } = n \pm O ( \sqrt { n } )$ 

(b) $\begin{array} { r } { F o r i \ne j \colon \frac { z _ { i } ^ { \top } z _ { j } } { \| z _ { i } \| \| z _ { j } \| } = O ( 1 / \sqrt { n } ) } \end{array}$ 

Proof. (a) Norm concentration: 

Since $\begin{array} { r } { \| z _ { i } \| ^ { 2 } = \sum _ { j = 1 } ^ { n } z _ { i j } ^ { 2 } \sim \chi ^ { 2 } ( n ) } \end{array}$ , we have: 

$$
\mathbb {E} [ \| z _ {i} \| ^ {2} ] = n, \quad \operatorname{Var} (\| z _ {i} \| ^ {2}) = 2 n\tag{26}
$$

By Chebyshev inequality or sub-Gaussian concentration: 

$$
\mathbb {P} \left(\left| \left\| z _ {i} \right\| ^ {2} - n \right| > t \sqrt {n}\right) \leq 2 e ^ {- c t ^ {2}}\tag{27}
$$

Thus $\| z _ { i } \| ^ { 2 } = n \pm O ( \sqrt { n } )$ with high probability. 

(b) Approximate orthogonality: 

For independent $z _ { i } , z _ { j } \sim \mathcal { N } ( 0 , I _ { n } )$ , the inner product $\begin{array} { r } { z _ { i } ^ { \top } z _ { j } = \sum _ { l = 1 } ^ { n } z _ { i l } z _ { j l } } \end{array}$ is a sum of n independent random variables with: 

$$
\mathbb {E} \left[ z _ {i} ^ {\top} z _ {j} \right] = 0, \quad \operatorname{Var} \left(z _ {i} ^ {\top} z _ {j}\right) = \sum_ {l = 1} ^ {n} \operatorname{Var} \left(z _ {i l} z _ {j l}\right) = n\tag{28}
$$

Thus $z _ { i } ^ { \top } z _ { j } = O ( { \sqrt { n } } )$ with high probability. Since $\| z _ { i } \| \| z _ { j } \| = O ( n )$ 

$$
\cos \theta_ {i j} = \frac {z _ {i} ^ {\top} z _ {j}}{\| z _ {i} \| \| z _ {j} \|} = O \left(\frac {\sqrt {n}}{n}\right) = O \left(\frac {1}{\sqrt {n}}\right)\rightarrow 0\tag{29}
$$

This shows that random Gaussian vectors are approximately orthogonal in high dimensions. 

Lemma B.4 (Isserlis’ Theorem Application). For $z \sim \mathcal { N } ( 0 , I _ { n } )$ and symmetric matrices $A , B .$ 

$$
\mathbb {E} [ (z ^ {\top} A z) (z ^ {\top} B z) ] = t r (A) t r (B) + 2 t r (A B)\tag{30}
$$

In particular, for $A = I _ { n }$ and $B = \Sigma$ 

$$
\mathbb {E} [ \| z \| ^ {2} \cdot z ^ {\top} \Sigma z ] = (n + 2) t r (\Sigma)\tag{31}
$$

Proof. By Isserlis’ theorem (Wick’s theorem), for $z \sim \mathcal { N } ( 0 , I _ { n } )$ ): 

$$
\mathbb {E} [ z _ {i} z _ {j} z _ {k} z _ {l} ] = \delta_ {i j} \delta_ {k l} + \delta_ {i k} \delta_ {j l} + \delta_ {i l} \delta_ {j k}\tag{32}
$$

Expanding the quadratic forms: 

$$
(z ^ {\top} A z) (z ^ {\top} B z) = \sum_ {i, j, k, l} A _ {i j} B _ {k l} z _ {i} z _ {j} z _ {k} z _ {l}\tag{33}
$$

Taking expectation: 

$$
\mathbb {E} [ (z ^ {\top} A z) (z ^ {\top} B z) ] = \sum_ {i, j, k, l} A _ {i j} B _ {k l} (\delta_ {i j} \delta_ {k l} + \delta_ {i k} \delta_ {j l} + \delta_ {i l} \delta_ {j k})\tag{34}
$$

$$
= \sum_ {i, k} A _ {i i} B _ {k k} + \sum_ {i, j} A _ {i j} B _ {i j} + \sum_ {i, j} A _ {i j} B _ {j i}\tag{35}
$$

$$
= \operatorname{tr} (A) \operatorname{tr} (B) + \operatorname{tr} (A B) + \operatorname{tr} (A B ^ {\top})\tag{36}
$$

For symmetric $A , B \colon \operatorname { t r } ( A B ^ { \top } ) = \operatorname { t r } ( A B )$ , thus: 

$$
\mathbb {E} \left[ \left(z ^ {\top} A z\right) \left(z ^ {\top} B z\right) \right] = \operatorname{tr} (A) \operatorname{tr} (B) + 2 \operatorname{tr} (A B)\tag{37}
$$

Setting $A = I _ { n } , B = \Sigma \colon$ 

$$
\mathbb {E} [ \| z \| ^ {2} \cdot z ^ {\top} \Sigma z ] = n \cdot \operatorname{tr} (\Sigma) + 2 \operatorname{tr} (\Sigma) = (n + 2) \operatorname{tr} (\Sigma)\tag{38}
$$

## B.2. Noise Variance Justification

The observation model $\hat { y } ( d ) = d ^ { \top } \tilde { g } + \nu$ with $\nu \sim \mathcal { N } ( 0 , \sigma _ { e } ^ { 2 } \| d \| ^ { 2 } )$ is justified as follows. 

Lemma B.5 (Effective Noise Decomposition). The effective noise variance decomposes as: 

$$
\sigma_ {e} ^ {2} = \sigma_ {\varepsilon} ^ {2} + t r (\Sigma)\tag{39}
$$

where $\sigma _ { \varepsilon } ^ { 2 }$ is the finite-difference approximation error and $t r ( \Sigma )$ is the gradient noise variance. 

## Proof. Step 1: Decomposition of the observation.

For coordinate-axis sampling with $d = e _ { i }$ , the direction in parameter space is $z _ { i } = B d = B e _ { i }$ (the i-th column of $B )$ , where $z _ { i } \sim \mathcal { N } ( 0 , I _ { n } )$ 

The observation can be decomposed as: 

$$
y _ {i} = \underbrace {z _ {i} ^ {\top} g} _ {\text { true   signal }} + \underbrace {z _ {i} ^ {\top} \zeta} _ {\text { gradient   noise }} + \underbrace {\epsilon_ {i}} _ {\text { finite - diff   error }}\tag{40}
$$

where: 

$g = \nabla \mathcal { L } ( \theta )$ is the true gradient 

$\zeta = \nabla \mathcal { L } ( \theta ; \xi ) - \nabla \mathcal { L } ( \theta )$ is the stochastic gradient noise with $\mathbb { E } [ \zeta ] = 0$ and $\mathrm { C o v } ( \zeta ) = \Sigma$ 

$\epsilon _ { i } \sim \mathcal { N } ( 0 , \sigma _ { \varepsilon } ^ { 2 } )$ is the finite-difference truncation error, independent of $\zeta$ and $z _ { i }$ 

## Step 2: Identifying the noise term.

The observation noise is defined as $\nu _ { i } : = y _ { i } - z _ { i } ^ { \top } g = z _ { i } ^ { \top } \zeta + \epsilon _ { i } .$ 

Since $\mathbb { E } [ \zeta ] = 0$ and $\mathbb { E } [ \epsilon _ { i } ] = 0$ , we have $\mathbb { E } [ \nu _ { i } | z _ { i } ] = 0$ 

Step 3: Conditional variance (given $z _ { i } )$ . 

Since $\zeta$ and $\epsilon _ { i }$ are independent: 

$$
\operatorname{Var} \left(\nu_ {i} \mid z _ {i}\right) = \operatorname{Var} \left(z _ {i} ^ {\top} \zeta \mid z _ {i}\right) + \operatorname{Var} \left(\epsilon_ {i}\right) = z _ {i} ^ {\top} \Sigma z _ {i} + \sigma_ {\varepsilon} ^ {2}\tag{41}
$$

## Step 4: Unconditional variance (taking expectation over $z _ { i } )$ .

Using the trace trick from Lemma B.2(b): 

$$
\mathbb {E} _ {z _ {i}} [ z _ {i} ^ {\top} \Sigma z _ {i} ] = \mathbb {E} [ \mathrm{tr} (\Sigma z _ {i} z _ {i} ^ {\top}) ] = \mathrm{tr} (\Sigma \cdot \mathbb {E} [ z _ {i} z _ {i} ^ {\top} ]) = \mathrm{tr} (\Sigma \cdot I _ {n}) = \mathrm{tr} (\Sigma)\tag{42}
$$

Therefore, the effective noise variance is: 

$$
\sigma_ {e} ^ {2} := \mathbb {E} _ {z _ {i}} [ \mathrm{Var} (\nu_ {i} | z _ {i}) ] = \mathrm{tr} (\Sigma) + \sigma_ {\varepsilon} ^ {2}\tag{43}
$$

By Assumption 3.2, $\operatorname { t r } ( \Sigma ) = \mathbb { E } [ \| \zeta \| ^ { 2 } ] \leq \sigma _ { g } ^ { 2 } ,$ , so $\sigma _ { e } ^ { 2 } \le \sigma _ { g } ^ { 2 } + \sigma _ { \varepsilon } ^ { 2 }$ 

## B.3. Proof of Main Convergence Theorem

ProofofTheorem 4.2. Step 1: Single-step descent. 

By Assumption 3.1 (L-smoothness): 

$$
\mathcal {L} \left(\theta_ {t + 1}\right) \leq \mathcal {L} \left(\theta_ {t}\right) + \left\langle g _ {t}, \Delta \theta_ {t} \right\rangle + \frac {L}{2} \| \Delta \theta_ {t} \| ^ {2}\tag{44}
$$

where $g _ { t } = \nabla \mathcal { L } ( \theta _ { t } )$ and $\Delta \theta _ { t } = - \eta B _ { t } \mu _ { t } ^ { ( k ) }$ 

Step 2: Inner product term. 

By Lemma B.2, the observation model is $y _ { i } = z _ { i } ^ { \top } g _ { t } + z _ { i } ^ { \top } \zeta + \epsilon _ { i }$ , where $\zeta$ is gradient noise with $\mathbf { C o v } ( \zeta ) = \Sigma ,$ and $\epsilon _ { i } \sim \mathcal { N } ( 0 , \sigma _ { \varepsilon } ^ { 2 } )$ is finite-difference error. 

Let $\tilde { g } = g _ { t } + \zeta$ and $\boldsymbol { \epsilon } = [ \epsilon _ { 1 } , \dots , \epsilon _ { k } ] ^ { \intercal }$ . Then: 

$$
Y = B _ {t} ^ {\top} \tilde {g} + \epsilon , \quad \mu_ {t} ^ {(k)} = \gamma Y = \gamma (B _ {t} ^ {\top} \tilde {g} + \epsilon)\tag{45}
$$

The parameter update becomes: 

$$
B _ {t} \mu_ {t} ^ {(k)} = \gamma B _ {t} B _ {t} ^ {\top} (g _ {t} + \zeta) + \gamma \sum_ {i = 1} ^ {k} z _ {i} \epsilon_ {i}\tag{46}
$$

Computing the expectation of the inner product. Since $\mathbb { E } [ \zeta ] = 0 , \mathbb { E } [ \epsilon _ { i } ] = 0$ , and $\zeta , \epsilon _ { i }$ are independent of $B _ { t }$ : 

$$
\mathbb {E} [ \langle g _ {t}, B _ {t} \mu_ {t} ^ {(k)} \rangle | \theta_ {t} ] = \gamma \mathbb {E} [ g _ {t} ^ {\top} B _ {t} B _ {t} ^ {\top} g _ {t} ] + \gamma \mathbb {E} [ g _ {t} ^ {\top} B _ {t} B _ {t} ^ {\top} \zeta ] + \gamma \sum_ {i = 1} ^ {k} \mathbb {E} [ (g _ {t} ^ {\top} z _ {i}) \epsilon_ {i} ]\tag{47}
$$

The second term: $\begin{array} { r } { \mathbb { E } [ g _ { t } ^ { \top } B _ { t } B _ { t } ^ { \top } \zeta ] = g _ { t } ^ { \top } \mathbb { E } [ B _ { t } B _ { t } ^ { \top } ] \mathbb { E } [ \zeta ] = 0 . } \end{array}$ 

The third term: $\mathbb { E } [ ( g _ { t } ^ { \top } z _ { i } ) \epsilon _ { i } ] = \mathbb { E } [ g _ { t } ^ { \top } z _ { i } ] \mathbb { E } [ \epsilon _ { i } ] = 0$ (independence). 

The first term: $\begin{array} { r } { \mathbb { E } [ g _ { t } ^ { \top } B _ { t } B _ { t } ^ { \top } g _ { t } ] = \mathbb { E } [ \| B _ { t } ^ { \top } g _ { t } \| ^ { 2 } ] = \sum _ { i = 1 } ^ { k } \mathbb { E } [ ( z _ { i } ^ { \top } g _ { t } ) ^ { 2 } ] = k \| g _ { t } \| ^ { 2 } . } \end{array}$ 

Therefore: 

$$
\mathbb {E} [ \langle g _ {t}, \Delta \theta_ {t} \rangle | \theta_ {t} ] = - \eta \gamma k \| g _ {t} \| ^ {2}\tag{48}
$$

## Step 3: Second moment (detailed computation).

From Step 2, $\begin{array} { r } { B _ { t } \mu _ { t } ^ { ( k ) } = \gamma B _ { t } B _ { t } ^ { \top } \tilde { g } + \gamma \sum _ { i } z _ { i } \epsilon _ { i } . } \end{array}$ . Thus: 

$$
\| B _ {t} \mu_ {t} ^ {(k)} \| ^ {2} = \gamma^ {2} \| B _ {t} B _ {t} ^ {\top} \tilde {g} \| ^ {2} + \gamma^ {2} \left\| \sum_ {i} z _ {i} \epsilon_ {i} \right\| ^ {2} + 2 \gamma^ {2} \left\langle B _ {t} B _ {t} ^ {\top} \tilde {g}, \sum_ {i} z _ {i} \epsilon_ {i} \right\rangle\tag{49}
$$

Cross term vanishes: Since $\epsilon _ { i }$ is independent of $B _ { t }$ and ${ \tilde { g } } ,$ and $\mathbb { E } [ \epsilon _ { i } ] = 0 \mathrm { : }$ 

$$
\mathbb {E} \left[ \left\langle B _ {t} B _ {t} ^ {\top} \tilde {g}, \sum_ {i} z _ {i} \epsilon_ {i} \right\rangle \right] = \sum_ {i} \mathbb {E} [ \epsilon_ {i} ] \cdot \mathbb {E} [ \langle B _ {t} B _ {t} ^ {\top} \tilde {g}, z _ {i} \rangle ] = 0\tag{50}
$$

First term: We compute $\mathbb { E } [ \| B _ { t } B _ { t } ^ { \top } \tilde { g } \| ^ { 2 } ]$ by first conditioning on $B _ { t }$ , then taking expectation over $B _ { t }$ 

(A) Given $B _ { t }$ , taking expectation over $\zeta$ (using $\mathbb { E } [ \zeta ] = 0 \rangle$ : 

$$
\mathbb {E} [ \| B _ {t} B _ {t} ^ {\top} \tilde {g} \| ^ {2} | B _ {t} ] = \| B _ {t} B _ {t} ^ {\top} g _ {t} \| ^ {2} + \mathbb {E} [ \| B _ {t} B _ {t} ^ {\top} \zeta \| ^ {2} | B _ {t} ]\tag{51}
$$

where $\mathbb { E } [ \| B _ { t } B _ { t } ^ { \top } \zeta \| ^ { 2 } | B _ { t } ] = \operatorname { t r } ( ( B _ { t } B _ { t } ^ { \top } ) ^ { 2 } \Sigma )$ 

(B) Taking expectation over $B _ { t }$ . For $\mathbb { E } [ \| B _ { t } B _ { t } ^ { \top } g \| ^ { 2 } ]$ 

$$
\| B _ {t} B _ {t} ^ {\top} g \| ^ {2} = \sum_ {i, j = 1} ^ {k} (z _ {i} ^ {\top} g) (z _ {j} ^ {\top} g) (z _ {i} ^ {\top} z _ {j})\tag{52}
$$

Diagonal terms $( i = j ) \colon \mathbb { E } [ ( z _ { i } ^ { \top } g ) ^ { 2 } \| z _ { i } \| ^ { 2 } ] = ( n + 2 ) \| g \| ^ { 2 }$ (by Lemma B.4). 

Off-diagonal terms $( i \neq j ) \colon$ By independence of $z _ { i }$ and $z _ { j } { \mathrm { : } }$ 

$$
\mathbb {E} [ (z _ {i} ^ {\top} g) (z _ {j} ^ {\top} g) (z _ {i} ^ {\top} z _ {j}) ] = \sum_ {a, b, c} g _ {a} g _ {b} \mathbb {E} [ (z _ {i}) _ {a} (z _ {i}) _ {c} ] \mathbb {E} [ (z _ {j}) _ {b} (z _ {j}) _ {c} ]\tag{53}
$$

$$
= \sum_ {a, b, c} g _ {a} g _ {b} \delta_ {a c} \delta_ {b c} = \sum_ {c} g _ {c} ^ {2} = \| g \| ^ {2}\tag{54}
$$

Thus: 

$$
\mathbb {E} [ \| B _ {t} B _ {t} ^ {\top} g \| ^ {2} ] = k (n + 2) \| g \| ^ {2} + k (k - 1) \| g \| ^ {2} = k (n + k + 1) \| g \| ^ {2} = k \tilde {n} \| g \| ^ {2}
$$

(55) 

For $\mathbb { E } [ { \bf t r } ( ( B _ { t } B _ { t } ^ { \top } ) ^ { 2 } \Sigma ) ]$ ], let $P = B _ { t } B _ { t } ^ { \top }$ 

$$
\mathbf {t r} (P ^ {2} \Sigma) = \sum_ {i, j} (z _ {i} ^ {\top} z _ {j}) (z _ {i} ^ {\top} \Sigma z _ {j})\tag{56}
$$

Diagonal $( i = j ) \colon$ By Lemma B.4, $\mathbb { E } [ \| z \| ^ { 2 } \cdot z ^ { \top } \Sigma z ] = ( n + 2 ) \mathrm { t r } ( \Sigma )$ 

Off-diagonal $( i \neq j ) \colon$ By independence of $z _ { i }$ and $z _ { j } { \mathrm { : } }$ 

$$
\mathbb {E} [ (z _ {i} ^ {\top} z _ {j}) (z _ {i} ^ {\top} \Sigma z _ {j}) ] = \sum_ {a, b, c} \Sigma_ {b c} \mathbb {E} [ (z _ {i}) _ {a} (z _ {i}) _ {b} ] \mathbb {E} [ (z _ {j}) _ {a} (z _ {j}) _ {c} ]\tag{57}
$$

$$
= \sum_ {a, b, c} \Sigma_ {b c} \delta_ {a b} \delta_ {a c} = \sum_ {a} \Sigma_ {a a} = \operatorname{tr} (\Sigma)\tag{58}
$$

Thus: 

$$
\mathbb {E} [ \mathrm{tr} (P ^ {2} \Sigma) ] = k (n + 2) \mathrm{tr} (\Sigma) + k (k - 1) \mathrm{tr} (\Sigma) = k \tilde {n} \cdot \mathrm{tr} (\Sigma)\tag{59}
$$

Second term: For the finite-difference noise: 

$$
\mathbb {E} \left[ \left\| \sum_ {i = 1} ^ {k} z _ {i} \epsilon_ {i} \right\| ^ {2} \right] = \sum_ {i, j} \mathbb {E} [ \epsilon_ {i} \epsilon_ {j} ] \mathbb {E} [ z _ {i} ^ {\top} z _ {j} ] = \sum_ {i} \sigma_ {\varepsilon} ^ {2} \cdot n = k n \sigma_ {\varepsilon} ^ {2}\tag{60}
$$

Total second moment: 

$$
\mathbb {E} [ \| B _ {t} \mu_ {t} ^ {(k)} \| ^ {2} ] = \gamma^ {2} k \left(\tilde {n} (\| g _ {t} \| ^ {2} + \mathrm{tr} (\Sigma)) + n \sigma_ {\varepsilon} ^ {2}\right)\tag{61}
$$

Step 4: Combining. 

Substituting into the descent inequality: 

$$
\mathbb {E} [ \| \Delta \theta_ {t} \| ^ {2} ] = \eta^ {2} \gamma^ {2} k \tilde {n} \| g _ {t} \| ^ {2} + \eta^ {2} \gamma^ {2} k (\tilde {n} \cdot \mathrm{tr} (\Sigma) + n \sigma_ {\varepsilon} ^ {2})\tag{62}
$$

Thus: 

$$
\mathbb {E} [ \mathcal {L} (\theta_ {t + 1}) ] \leq \mathbb {E} [ \mathcal {L} (\theta_ {t}) ] - \eta \gamma k \| g _ {t} \| ^ {2} + \frac {L \eta^ {2} \gamma^ {2} k}{2} \left[ \tilde {n} \| g _ {t} \| ^ {2} + \tilde {n} \cdot \mathrm{tr} (\Sigma) + n \sigma_ {\varepsilon} ^ {2} \right]\tag{63}
$$

Collecting terms in $\| g _ { t } \| ^ { 2 }$ 

$$
\mathbb {E} [ \mathcal {L} (\theta_ {t + 1}) ] \leq \mathbb {E} [ \mathcal {L} (\theta_ {t}) ] - \eta \gamma k \underbrace {\left(1 - \frac {L \eta \gamma \tilde {n}}{2}\right)} _ {:= \beta (\eta)} \| g _ {t} \| ^ {2} + \frac {L \eta^ {2} \gamma^ {2} k (\tilde {n} \cdot \operatorname{tr} (\Sigma) + n \sigma_ {\varepsilon} ^ {2})}{2}\tag{64}
$$

Step 5: Telescoping sum. 

When $\begin{array} { r } { \eta < \frac { 2 } { L \gamma \tilde { n } } } \end{array}$ , we have $\beta ( \eta ) > 0$ . Rearranging: 

$$
\left\| g _ {t} \right\| ^ {2} \leq \frac {1}{\beta (\eta) \eta \gamma k} \left(\mathcal {L} \left(\theta_ {t}\right) - \mathcal {L} \left(\theta_ {t + 1}\right)\right) + \frac {L \eta \gamma \left(\tilde {n} \cdot \operatorname{tr} (\Sigma) + n \sigma_ {\varepsilon} ^ {2}\right)}{2 \beta (\eta)}\tag{65}
$$

Summing over $t = 0 , \ldots , T - 1$ and dividing by $T _ { \mathbf { \delta } }$ : 

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} [ \| g _ {t} \| ^ {2} ] \leq \frac {\mathcal {L} (\theta_ {0}) - \mathcal {L} ^ {*}}{\beta (\eta) \eta \gamma k T} + \frac {L \eta \gamma (\tilde {n} \cdot \operatorname{tr} (\Sigma) + n \sigma_ {\varepsilon} ^ {2})}{2 \beta (\eta)}\tag{66}
$$

## B.4. Proof of Expected Update Direction

ProofofTheorem 3.7. Step 1: Posterior mean unbiasedness. 

By Corollary 3.6, for coordinate-axis sampling $( d ^ { ( i ) } = e _ { i } )$ , the posterior mean is: 

$$
\mu^ {(k)} = \gamma Y = \gamma [ y ^ {(1)}, \dots , y ^ {(k)} ] ^ {\top}\tag{67}
$$

where $\begin{array} { r } { \gamma = \frac { \sigma _ { p } ^ { 2 } } { \sigma _ { p } ^ { 2 } + \sigma _ { e } ^ { 2 } } } \end{array}$ 

Each observation satisfies $y ^ { ( i ) } = e _ { i } ^ { \top } \tilde { g } + \nu ^ { ( i ) } = \tilde { g } _ { i } + \nu ^ { ( i ) }$ , where $\tilde { g } = B ^ { \top } g$ is the true normalized subspace gradient and $\nu ^ { ( i ) }$ is zero-mean noise. 

Taking conditional expectation given B and g (so $\tilde { g } ^ { * } = B ^ { \top } g$ is fixed): 

$$
\mathbb {E} [ y ^ {(i)} | B, g ] = \tilde {g} _ {i} ^ {*} + \mathbb {E} [ \nu^ {(i)} ] = \tilde {g} _ {i} ^ {*}\tag{68}
$$

Thus: 

$$
\mathbb {E} [ \mu^ {(k)} | B, g ] = \gamma \mathbb {E} [ Y | B, g ] = \gamma \tilde {g} ^ {*} = \gamma B ^ {\top} g\tag{69}
$$

## Step 2: Conditional expectation of update.

The parameter update is $\Delta \theta = - \eta B \mu ^ { ( k ) }$ . Taking conditional expectation: 

$$
\mathbb {E} [ \Delta \theta | B ] = - \eta B \mathbb {E} [ \mu^ {(k)} | B ] = - \eta \gamma B B ^ {\top} g\tag{70}
$$

Step 3: Expectation over subspace basis. 

Taking expectation over $B = [ z _ { 1 } , \ldots , z _ { k } ]$ where $z _ { i } \overset { i i d } { \sim } \mathcal { N } ( 0 , I _ { n } )$ 

$$
\mathbb {E} [ \Delta \theta ] = - \eta \gamma \mathbb {E} [ B B ^ {\top} ] g\tag{71}
$$

Computing $\Sigma [ B B ^ { \top } ]$ 

$$
B B ^ {\top} = \sum_ {i = 1} ^ {k} z _ {i} z _ {i} ^ {\top}\tag{72}
$$

$$
\mathbb {E} [ B B ^ {\top} ] = \sum_ {i = 1} ^ {k} \mathbb {E} [ z _ {i} z _ {i} ^ {\top} ] = \sum_ {i = 1} ^ {k} I _ {n} = k I _ {n}\tag{73}
$$

Therefore: 

$$
\mathbb {E} [ \Delta \theta ] = - \eta \gamma k \cdot I _ {n} \cdot g = - \eta \gamma k \cdot \nabla \mathcal {L} (\theta_ {0})\tag{74}
$$

## Step 4: Higher-order bias.

By Lemma B.1, the finite-difference estimator has $O ( \varepsilon L )$ bias. After multiplication by ε in the update, this becomes $O ( \varepsilon ^ { 2 } L )$ Since ε is typically small $( \sim 1 0 ^ { - 3 } )$ , we write: 

$$
\mathbb {E} [ \Delta \theta ] = - \eta \gamma k \cdot \nabla \mathcal {L} (\theta_ {0}) + O (\varepsilon^ {3})\tag{75}
$$

This proves that the expected update direction aligns with the negative gradient, with effective learning rate $\eta _ { e f f } = \eta \gamma k$ . 

## B.5. Adaptive Sampling Analysis

Theorem B.6 (Conditional Unbiasedness of Posterior Mean under Adaptive Sampling). Let $\mu ^ { ( m ) }$ denote the posterior mean after m adaptive sampling steps. Given the subspace basis B and the true gradient g, for any adaptive sampling strategy π (where $d ^ { ( j ) }$ is $\mathcal { D } _ { j - 1 }$ -measurable), we have: 

$$
\mathbb {E} [ \mu^ {(m)} | B, g ] = \mathbb {E} \left[ \Sigma^ {(m)} D _ {m} ^ {\top} R _ {m} ^ {- 1} D _ {m} \mid B \right] \tilde {g} ^ {*} = \mathbb {E} [ \Gamma_ {m} | B ] \cdot \tilde {g} ^ {*}\tag{76}
$$

In particular, $i f \Sigma ^ { ( m ) }$ is deterministic given $\textit { B } ( e . g .$ ., coordinate-axis sampling or any strategy that depends only on $\mathcal { D } _ { m - 1 } ) ,$ then: 

$$
\mathbb {E} [ \mu^ {(m)} | B, g, \mathcal {D} _ {m} ] = \Gamma_ {m} \cdot \tilde {g} ^ {*}\tag{77}
$$

where $\Gamma _ { m } : = I _ { k } - \sigma _ { p } ^ { - 2 } \Sigma ^ { ( m ) }$ is the shrinkage matrix. 

## Proof. Step 1: Expression for the posterior mean.

By the standard Bayesian linear regression formula: 

$$
\mu^ {(m)} = \Sigma^ {(m)} D _ {m} ^ {\top} R _ {m} ^ {- 1} Y _ {m}\tag{78}
$$

where $Y _ { m } = [ y ^ { ( 1 ) } , \dots , y ^ { ( m ) } ] ^ { \intercal }$ 

Step 2: Computing the conditional expectation. 

Note that $\Sigma ^ { ( m ) }$ and $D _ { m }$ are both $\mathcal { D } _ { m }$ -measurable. The key is to compute $\mathbb { E } [ Y _ { m } | B , g , \mathcal { D } _ { m } ]$ 

$$
y ^ {(j)}
$$

$$
\mathbb {E} [ y ^ {(j)} | B, g, \mathcal {D} _ {m} ] = \mathbb {E} [ y ^ {(j)} | B, g, d ^ {(j)} ] = d ^ {(j) \top} \tilde {g} ^ {*}\tag{79}
$$

The first equality holds because given $d ^ { ( j ) } , y ^ { ( j ) }$ is conditionally independent of other $d ^ { ( i ) } ( i \neq j )$ 

Therefore: 

$$
\mathbb {E} [ Y _ {m} | B, g, \mathcal {D} _ {m} ] = D _ {m} \tilde {g} ^ {*}\tag{80}
$$

Step 3: Substituting into the posterior mean. 

$$
\mathbb {E} [ \mu^ {(m)} | B, g, \mathcal {D} _ {m} ] = \Sigma^ {(m)} D _ {m} ^ {\top} R _ {m} ^ {- 1} \mathbb {E} [ Y _ {m} | B, g, \mathcal {D} _ {m} ] = \Sigma^ {(m)} D _ {m} ^ {\top} R _ {m} ^ {- 1} D _ {m} \tilde {g} ^ {*}\tag{81}
$$

Step 4: Simplifying the shrinkage matrix. 

By the definition of $\Sigma ^ { ( m ) }$ 

$$
(\Sigma^ {(m)}) ^ {- 1} = \sigma_ {p} ^ {- 2} I _ {k} + D _ {m} ^ {\top} R _ {m} ^ {- 1} D _ {m}\tag{82}
$$

Therefore: 

$$
D _ {m} ^ {\top} R _ {m} ^ {- 1} D _ {m} = (\Sigma^ {(m)}) ^ {- 1} - \sigma_ {p} ^ {- 2} I _ {k}\tag{83}
$$

Substituting: 

$$
\mathbb {E} [ \mu^ {(m)} | B, g, \mathcal {D} _ {m} ] = \Sigma^ {(m)} \left[ (\Sigma^ {(m)}) ^ {- 1} - \sigma_ {p} ^ {- 2} I _ {k} \right] \tilde {g} ^ {*} = \left(I _ {k} - \sigma_ {p} ^ {- 2} \Sigma^ {(m)}\right) \tilde {g} ^ {*}\tag{84}
$$

Defining the shrinkage matrix $\Gamma _ { m } : = I _ { k } - \sigma _ { p } ^ { - 2 } \Sigma ^ { ( m ) }$ , we obtain: 

$$
\mathbb {E} [ \mu^ {(m)} | B, g, \mathcal {D} _ {m} ] = \Gamma_ {m} \tilde {g} ^ {*}\tag{85}
$$

## B.6. Convergence Rate under Adaptive Sampling

Theorem B.7 (Convergence Rate under Adaptive Sampling). Under Assumptions 3.1, 3.2, and isotropic noise, consider the BSZO algorithm with adaptive sampling (m samples, where the first k samples use coordinate-axis sampling). Let $\tilde { n } = n + k + 1$ be the effective dimension. Suppose $\begin{array} { r } { \eta < \frac { 2 } { L \bar { \gamma } \tilde { n } } } \end{array}$ , and define $\begin{array} { r } { \beta ( \eta ) : = 1 - \frac { L \eta \bar { \gamma } \tilde { n } } { 2 } } \end{array}$ . Then, after T iterations, the following inequality holds: 

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} [ \| \nabla \mathcal {L} (\theta_ {t}) \| ^ {2} ] \leq \frac {\Delta_ {0}}{\beta (\eta) \eta \bar {\gamma} k T} + \frac {L \eta \bar {\gamma} (\tilde {n} \sigma_ {g} ^ {2} + n \sigma_ {n} ^ {2})}{2 \beta (\eta)},\tag{86}
$$

where: 

• γ¯ := min<sub>t</sub> $\bar { \gamma } _ { t } \geq \gamma$ is the minimum effective shrinkage factor, 

• $\sigma _ { g } ^ { 2 }$ is the gradient noise variance, $\sigma _ { n } ^ { 2 }$ is thefinite-difference approximation noise variance, 

$\Delta _ { 0 } : = \mathcal { L } ( \theta _ { 0 } ) - \mathcal { L } ^ { * } .$ 

Corollary B.8. Let $\begin{array} { r } { \eta = \frac { 1 } { L \bar { \gamma } \tilde { n } } } \end{array}$ . Then $\beta = 1 / 2 ,$ , and the convergence bound simplifies to: 

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} [ \| \nabla \mathcal {L} (\theta_ {t}) \| ^ {2} ] \leq \frac {2 L \bar {\gamma} \tilde {n} \Delta_ {0}}{k T} + \sigma_ {g} ^ {2} + \frac {n}{\tilde {n}} \sigma_ {n} ^ {2}.\tag{87}
$$

Remark B.9. When $n \gg k .$ , we have $\tilde { n } \approx n$ , so the noise floor $\begin{array} { r } { \sigma _ { g } ^ { 2 } + \frac { n } { \tilde { n } } \sigma _ { n } ^ { 2 } \approx \sigma _ { e } ^ { 2 } } \end{array}$ becomes decoupled from the dimension n. 

Proof. The proof follows the same structure as Theorem 4.2, with the fixed γ replaced by the adaptive effective shrinkage factor $\bar { \gamma } _ { t }$ 

Step 1: Single-step descent. 

By Assumption 3.1 (L-smoothness): 

$$
\mathcal {L} \left(\theta_ {t + 1}\right) \leq \mathcal {L} \left(\theta_ {t}\right) + \left\langle g _ {t}, \Delta \theta_ {t} \right\rangle + \frac {L}{2} \| \Delta \theta_ {t} \| ^ {2}\tag{88}
$$

## Step 2: Inner product term under adaptive sampling.

By the adaptive sampling theorem (Theorem B.6), the expected update direction satisfies: 

$$
\mathbb {E} [ \langle g _ {t}, \Delta \theta_ {t} \rangle | \theta_ {t} ] = - \eta \mathbb {E} [ \mathrm{tr} (\Gamma_ {m} ^ {(t)}) ] \| g _ {t} \| ^ {2} = - \eta \bar {\gamma} _ {t} k \| g _ {t} \| ^ {2}\tag{89}
$$

where $\begin{array} { r } { \bar { \gamma } _ { t } = \frac { 1 } { k } \mathrm { t r } ( \Gamma _ { m } ^ { ( t ) } ) = 1 - \frac { U _ { m } ^ { ( t ) } } { k \sigma _ { p } ^ { 2 } } } \end{array}$ is the effective shrinkage factor at iteration t. 

## Step 3: Second moment (same structure as main theorem).

Following the same derivation as Theorem 4.2, with γ replaced by $\bar { \gamma } _ { t } \colon$ 

$$
\mathbb {E} [ \| \Delta \theta_ {t} \| ^ {2} | \theta_ {t} ] = \eta^ {2} \bar {\gamma} _ {t} ^ {2} k \tilde {n} \| g _ {t} \| ^ {2} + \eta^ {2} \bar {\gamma} _ {t} ^ {2} k (\tilde {n} \sigma_ {g} ^ {2} + n \sigma_ {n} ^ {2})\tag{90}
$$

The key observation is that the second moment structure remains unchanged because: 

• The gradient noise $\sigma _ { g } ^ { 2 }$ interacts with $B _ { t }$ to produce the n˜ factor 

• The finite-difference noise $\sigma _ { n } ^ { 2 }$ is independent of $B _ { t }$ , producing only the n factor 

## Step 4: Combining and bounding.

Substituting into the descent inequality: 

$$
\mathbb {E} [ \mathcal {L} (\theta_ {t + 1}) ] \leq \mathbb {E} [ \mathcal {L} (\theta_ {t}) ] - \eta \bar {\gamma} _ {t} k \left(1 - \frac {L \eta \bar {\gamma} _ {t} \tilde {n}}{2}\right) \mathbb {E} [ \| g _ {t} \| ^ {2} ] + \frac {L \eta^ {2} \bar {\gamma} _ {t} ^ {2} k (\tilde {n} \sigma_ {g} ^ {2} + n \sigma_ {n} ^ {2})}{2}\tag{91}
$$

Since $\bar { \gamma } _ { t } \geq \bar { \gamma } : =$ min<sub>t</sub> $\bar { \gamma } _ { t } \geq \gamma$ (by Lemma in Theorem B.6), and assuming $\begin{array} { r } { \eta < \frac { 2 } { L \bar { \gamma } \tilde { n } } } \end{array}$ , we define $\begin{array} { r } { \beta ( \eta ) = 1 - \frac { L \eta \bar { \gamma } \tilde { n } } { 2 } > 0 } \end{array}$ Rearranging: 

$$
\mathbb {E} [ \| g _ {t} \| ^ {2} ] \leq \frac {1}{\beta (\eta) \eta \bar {\gamma} k} \left(\mathbb {E} [ \mathcal {L} (\theta_ {t}) ] - \mathbb {E} [ \mathcal {L} (\theta_ {t + 1}) ]\right) + \frac {L \eta \bar {\gamma} (\tilde {n} \sigma_ {g} ^ {2} + n \sigma_ {n} ^ {2})}{2 \beta (\eta)}\tag{92}
$$

Step 5: Telescoping sum. 

Summing over $t = 0 , \ldots , T - 1$ and dividing by T: 

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} [ \| \nabla \mathcal {L} (\theta_ {t}) \| ^ {2} ] \leq \frac {\Delta_ {0}}{\beta (\eta) \eta \bar {\gamma} k T} + \frac {L \eta \bar {\gamma} (\tilde {n} \sigma_ {g} ^ {2} + n \sigma_ {n} ^ {2})}{2 \beta (\eta)}\tag{93}
$$

For the special learning rate $\begin{array} { r } { \eta = \frac { 1 } { L \bar { \gamma } \tilde { n } } } \end{array}$ , we have $\beta = 1 / 2$ , and the bound simplifies to: 

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} [ \| \nabla \mathcal {L} (\theta_ {t}) \| ^ {2} ] \leq \frac {2 L \bar {\gamma} \tilde {n} \Delta_ {0}}{k T} + \sigma_ {g} ^ {2} + \frac {n}{\tilde {n}} \sigma_ {n} ^ {2}\tag{94}
$$

When $n \gg k ,$ we have $\tilde { n } \approx n ,$ , so the noise floor $\begin{array} { r } { \sigma _ { g } ^ { 2 } + \frac { n } { \tilde { n } } \sigma _ { n } ^ { 2 } \approx \sigma _ { e } ^ { 2 } } \end{array}$ becomes decoupled from dimension n. 

## C. Experiment Details


Table 6. Number of training and validation samples for each dataset.


<table><tr><td>SPLIT</td><td>SST-2</td><td>BOOLQ</td><td>RTE</td><td>COPA</td><td>WIC</td><td>WSC</td><td>CB</td><td>TREC</td></tr><tr><td>TRAINING</td><td>1000</td><td>1000</td><td>1000</td><td>300</td><td>1000</td><td>450</td><td>200</td><td>1000</td></tr><tr><td>VALIDATION</td><td>500</td><td>500</td><td>500</td><td>100</td><td>500</td><td>100</td><td>50</td><td>500</td></tr></table>


Table 7. Hyperparameter configurations for fine-tuning RoBERTa-large.


<table><tr><td>Algorithm</td><td>Hyperparameter</td><td>Values</td></tr><tr><td rowspan="3">MeZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{ 1 \times {10}^{-4},1 \times {10}^{-5},1 \times {10}^{-6},1 \times {10}^{-7},1 \times {10}^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td rowspan="3">MeZO-Adam</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{ 1 \times {10}^{-4},1 \times {10}^{-5},1 \times {10}^{-6},1 \times {10}^{-7},1 \times {10}^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td rowspan="3">HiZOO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{ 1 \times {10}^{-4},1 \times {10}^{-5},1 \times {10}^{-6},1 \times {10}^{-7},1 \times {10}^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td rowspan="6">LOZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{ 1 \times {10}^{-4},1 \times {10}^{-5},1 \times {10}^{-6},1 \times {10}^{-7},1 \times {10}^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td>Rank</td><td></td></tr><tr><td>Interval</td><td>2</td></tr><tr><td></td><td>50</td></tr><tr><td rowspan="5">BSZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{ 1 \times {10}^{-4},1 \times {10}^{-5},1 \times {10}^{-6},1 \times {10}^{-7},1 \times {10}^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td>k(Subspace dim)</td><td>2</td></tr><tr><td>m(Samples)</td><td>3</td></tr><tr><td rowspan="5">BSZO-B</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{ 1 \times {10}^{-4},1 \times {10}^{-5},1 \times {10}^{-6},1 \times {10}^{-7},1 \times {10}^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td>k(Subspace dim)</td><td>2</td></tr><tr><td></td><td>3</td></tr><tr><td>All Methods</td><td>Early stopping patience</td><td>4,000</td></tr></table>


Table 8. Hyperparameter configurations for fine-tuning OPT-1.3B.


<table><tr><td>Algorithm</td><td>Hyperparameter</td><td>Values</td></tr><tr><td rowspan="3">MeZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-7}, 5 \times 10^{-8}, 1 \times 10^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td rowspan="3">MeZO-Adam</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{1 \times 10^{-4}, 5 \times 10^{-5}, 1 \times 10^{-5}, 5 \times 10^{-6}, 1 \times 10^{-6}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td rowspan="3">HiZOO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{1 \times 10^{-5}, 5 \times 10^{-6}, 1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-7}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td rowspan="5">LOZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-7}, 5 \times 10^{-8}, 1 \times 10^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td>Rank</td><td>2</td></tr><tr><td>Interval</td><td>50</td></tr><tr><td rowspan="5">BSZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{1 \times 10^{-5}, 5 \times 10^{-6}, 1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-7}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td>k (Subspace dim)</td><td>2</td></tr><tr><td>m (Samples)</td><td>3</td></tr><tr><td rowspan="5">BSZO-B</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{1 \times 10^{-5}, 5 \times 10^{-6}, 1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-7}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td>k (Subspace dim)</td><td>2</td></tr><tr><td>n (Samples)</td><td>3</td></tr><tr><td>All Methods</td><td>Early stopping patience</td><td>4,000</td></tr></table>


Table 9. Hyperparameter configurations for fine-tuning Mistral-7B.


<table><tr><td>Algorithm</td><td>Hyperparameter</td><td>Values</td></tr><tr><td rowspan="3">MeZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{5 \times 10^{-7}, 1 \times 10^{-7}, 5 \times 10^{-8}, 1 \times 10^{-8}, 5 \times 10^{-9}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td rowspan="3">HiZOO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-6}, 5 \times 10^{-8}, 1 \times 10^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td rowspan="5">LOZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{5 \times 10^{-7}, 1 \times 10^{-7}, 5 \times 10^{-8}, 1 \times 10^{-8}, 5 \times 10^{-9}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td>Rank</td><td>2</td></tr><tr><td>Interval</td><td>50</td></tr><tr><td rowspan="5">BSZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-7}, 5 \times 10^{-8}, 1 \times 10^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td>k (Subspace dim)</td><td>2</td></tr><tr><td>m (Samples)</td><td>3</td></tr><tr><td rowspan="5">BSZO-B</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-7}, 5 \times 10^{-8}, 1 \times 10^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td>k (Subspace dim)</td><td>2</td></tr><tr><td>n (Samples)</td><td>3</td></tr><tr><td>All Methods</td><td>Early stopping patience</td><td>4,000</td></tr></table>


Table 10. Hyperparameter configurations for fine-tuning OPT-13B.


<table><tr><td>Algorithm</td><td>Hyperparameter</td><td>Values</td></tr><tr><td rowspan="3">MeZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{5 \times 10^{-6}, 1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-7}, 5 \times 10^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td rowspan="3">MeZO-Adam</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{1 \times 10^{-4}, 5 \times 10^{-5}, 1 \times 10^{-5}, 5 \times 10^{-6}, 1 \times 10^{-6}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td rowspan="3">HiZOO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{1 \times 10^{-5}, 5 \times 10^{-6}, 1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-7}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td rowspan="5">LOZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{5 \times 10^{-6}, 1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-7}, 5 \times 10^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td>Rank</td><td>2</td></tr><tr><td>Interval</td><td>50</td></tr><tr><td rowspan="5">BSZO</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{5 \times 10^{-6}, 1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-7}, 5 \times 10^{-8}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td>k (Subspace dim)</td><td>2</td></tr><tr><td>m (Samples)</td><td>3</td></tr><tr><td rowspan="5">BSZO-B</td><td>Batch size</td><td>16</td></tr><tr><td>Learning rate</td><td><eq>\{1 \times 10^{-5}, 5 \times 10^{-6}, 1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times 10^{-7}\}</eq></td></tr><tr><td>ε</td><td><eq>10^{-4}</eq></td></tr><tr><td>k (Subspace dim)</td><td>2</td></tr><tr><td>m (Samples)</td><td>3</td></tr><tr><td>All Methods</td><td>Early stopping patience</td><td>4,000</td></tr></table>

## D. Raw Experimental Results

We provide the complete raw results of 5 independent runs for each method on RoBERTa-large in Table 11. The mean and standard deviation reported in Table 1 are computed from these results. 


Table 11. Raw test accuracy (%) of 5 runs on RoBERTa-large (355M).


<table><tr><td>DATASET</td><td>METHOD</td><td>RUN 1</td><td>RUN 2</td><td>RUN 3</td><td>RUN 4</td><td>RUN 5</td></tr><tr><td rowspan="6">SST-2</td><td>MEZO</td><td>92.43</td><td>92.32</td><td>91.74</td><td>92.78</td><td>91.86</td></tr><tr><td>MEZO-ADAM</td><td>92.32</td><td>92.66</td><td>91.51</td><td>92.43</td><td>92.78</td></tr><tr><td>HIZOO</td><td>91.97</td><td>91.86</td><td>91.28</td><td>91.17</td><td>90.94</td></tr><tr><td>LOZO</td><td>91.63</td><td>92.09</td><td>91.74</td><td>91.51</td><td>92.20</td></tr><tr><td>BSZO</td><td>92.89</td><td>92.43</td><td>92.78</td><td>92.43</td><td>92.78</td></tr><tr><td>BSZO-B</td><td>92.66</td><td>91.74</td><td>92.32</td><td>91.97</td><td>92.66</td></tr><tr><td rowspan="6">RTE</td><td>MEZO</td><td>69.68</td><td>68.95</td><td>65.70</td><td>65.34</td><td>62.09</td></tr><tr><td>MEZO-ADAM</td><td>62.09</td><td>64.26</td><td>64.62</td><td>64.98</td><td>62.09</td></tr><tr><td>HIZOO</td><td>59.21</td><td>63.18</td><td>57.76</td><td>56.68</td><td>59.21</td></tr><tr><td>LOZO</td><td>59.57</td><td>63.90</td><td>61.01</td><td>65.34</td><td>63.18</td></tr><tr><td>BSZO</td><td>68.59</td><td>68.23</td><td>69.68</td><td>66.07</td><td>66.43</td></tr><tr><td>BSZO-B</td><td>67.87</td><td>70.04</td><td>70.76</td><td>66.43</td><td>66.79</td></tr><tr><td rowspan="6">CB</td><td>MEZO</td><td>87.50</td><td>85.71</td><td>91.07</td><td>76.79</td><td>89.29</td></tr><tr><td>MEZO-ADAM</td><td>82.14</td><td>83.93</td><td>80.36</td><td>82.14</td><td>76.79</td></tr><tr><td>HIZOO</td><td>78.57</td><td>75.00</td><td>75.00</td><td>78.57</td><td>75.00</td></tr><tr><td>LOZO</td><td>87.50</td><td>82.14</td><td>82.14</td><td>89.29</td><td>80.36</td></tr><tr><td>BSZO</td><td>83.93</td><td>85.71</td><td>87.50</td><td>87.50</td><td>83.93</td></tr><tr><td>BSZO-B</td><td>85.71</td><td>83.93</td><td>82.14</td><td>85.71</td><td>83.93</td></tr><tr><td rowspan="6">WIC</td><td>MEZO</td><td>49.69</td><td>58.31</td><td>52.98</td><td>57.52</td><td>57.52</td></tr><tr><td>MEZO-ADAM</td><td>54.39</td><td>51.10</td><td>54.70</td><td>46.55</td><td>57.52</td></tr><tr><td>HIZOO</td><td>50.31</td><td>54.39</td><td>51.10</td><td>54.70</td><td>57.52</td></tr><tr><td>LOZO</td><td>54.55</td><td>54.55</td><td>51.88</td><td>55.17</td><td>54.86</td></tr><tr><td>BSZO</td><td>57.68</td><td>57.52</td><td>55.64</td><td>54.70</td><td>54.70</td></tr><tr><td>BSZO-B</td><td>57.99</td><td>57.99</td><td>55.80</td><td>56.58</td><td>57.68</td></tr><tr><td rowspan="6">TREC</td><td>MEZO</td><td>81.20</td><td>86.40</td><td>86.40</td><td>86.20</td><td>86.60</td></tr><tr><td>MEZO-ADAM</td><td>84.80</td><td>74.20</td><td>71.40</td><td>83.00</td><td>80.60</td></tr><tr><td>HIZOO</td><td>65.20</td><td>65.20</td><td>65.20</td><td>62.20</td><td>59.40</td></tr><tr><td>LOZO</td><td>80.40</td><td>74.80</td><td>77.20</td><td>79.20</td><td>77.20</td></tr><tr><td>BSZO</td><td>83.40</td><td>84.60</td><td>84.40</td><td>83.80</td><td>84.60</td></tr><tr><td>BSZO-B</td><td>85.80</td><td>84.60</td><td>85.20</td><td>82.20</td><td>86.20</td></tr></table>


Table 12. Full ablation studies on OPT-1.3B (fp32). (a) Effect of subspace dimension k with $m = k .$ (b) Effect of observation count m with $m = k + 1 .$ . (c) Noise-free adaptive sampling. Best per row in bold.


<table><tr><td colspan="3">(a) Effect of k</td><td colspan="3">(b) Effect of m</td><td colspan="3">(c) NF-Adaptive</td></tr><tr><td>k</td><td>SST-2</td><td>RTE</td><td>k</td><td>SST-2</td><td>RTE</td><td>k</td><td>SST-2</td><td>RTE</td></tr><tr><td>1</td><td>92.32</td><td>60.29</td><td>1</td><td>91.74</td><td>61.37</td><td>1</td><td>91.28</td><td>63.58</td></tr><tr><td>2</td><td>92.78</td><td>64.26</td><td>2</td><td>92.43</td><td>66.79</td><td>2</td><td>92.43</td><td>65.34</td></tr><tr><td>4</td><td>92.66</td><td>67.51</td><td>4</td><td>93.58</td><td>66.43</td><td>4</td><td>93.12</td><td>66.07</td></tr><tr><td>8</td><td>93.23</td><td>66.07</td><td>8</td><td>93.23</td><td>68.59</td><td>8</td><td>93.35</td><td>69.31</td></tr></table>