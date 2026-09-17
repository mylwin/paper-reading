# PowerStep: Memory-Efficient Adaptive Optimization via ℓ<sub>p</sub>-Norm Steepest Descent

Yao Lu<sup>1∗</sup> Dengdong Fan<sup>1</sup> Shixun Zhang<sup>1</sup> Yonghong Tian<sup>1,2</sup> 

Pengcheng Laboratory<sup>1</sup> Peking University<sup>2</sup> 

## Abstract

Adaptive optimizers, most notably Adam, have become the default standard for training large-scale neural networks such as Transformers. These methods maintain running estimates of gradient first and second moments, incurring substantial memory overhead. We introduce PowerStep, a memory-efficient optimizer that achieves coordinate-wise adaptivity without storing second-moment statistics. Motivated by steepest descent under an $\ell _ { p } .$ -norm geometry, we show that applying a nonlinear transform directly to a momentum buffer yields coordinate-wise adaptivity. We prove that PowerStep converges at the optimal $O ( 1 / \sqrt { T } )$ rate for non-convex stochastic optimization. Extensive experiments on Transformer models ranging from 124M to 235B parameters demonstrate that PowerStep matches Adam’s convergence speed while halving optimizer memory. Furthermore, when combined with aggressive int8 quantization, PowerStep remains numerically stable and reduces optimizer memory by ∼ 8× compared to full-precision Adam. PowerStep thus provides a principled, scalable and resource-efficient alternative for large-scale training. Code is available at https://github.com/yaolubrain/PowerStep. 

## 1 Introduction

Neural networks are notoriously difficult to train, owing in large part to the vanishing and exploding gradient problem [Hochreiter, 1991; Bengio et al., 1994] and the ill-conditioned curvature of the loss landscape [LeCun et al., 1998; Dauphin et al., 2014; Choromanska et al., 2015], which can cause stochastic gradient descent (SGD) to converge slowly or even diverge. Adaptive gradient methods [Duchi et al., 2011; Tieleman and Hinton, 2012; Kingma and Ba, 2015; Loshchilov and Hutter, 2019] alleviate these issues by preconditioning gradient updates with accumulated second-moment statistics. Among these methods, Adam [Kingma and Ba, 2015] and its decoupled weight decay variant AdamW [Loshchilov and Hutter, 2019] have become the default optimizers for training large-scale neural networks, particularly Transformers [Vaswani et al., 2017]. 

Consider the optimization of an objective function $f ( \pmb \theta ) : \mathbb R ^ { d }  \mathbb R$ . Let $\mathbf { g } _ { t }$ denote the stochastic gradient at step t, while $\mathbf { m } _ { t }$ and $\mathbf { v } _ { t }$ represent the exponential moving averages (EMA) of the first and second moments of $\mathbf { g } _ { t } ,$ respectively. The update rule for Adam [Kingma and Ba, 2015] is defined as 

$$
\boldsymbol {\theta} _ {t} = \boldsymbol {\theta} _ {t - 1} - \eta_ {t} \cdot \mathbf {m} _ {t} / (\sqrt {\mathbf {v} _ {t}} + \epsilon),\tag{1}
$$

where $\eta _ { t }$ is the learning rate and ϵ is a small constant. Intuitively, $\mathbf { v } _ { t }$ acts as a coordinate-wise scaling factor: by dividing the first moment by the square root of the second moment, Adam dampens parameters with large historical gradients and amplifies those with small ones. Equivalently, this applies a diagonal preconditioner $\mathbf { D } _ { t } = \mathrm { d i a g } ( ( \sqrt { \mathbf { v } _ { t } } + \epsilon ) ^ { - 1 } )$ to the momentum direction. This adaptivity enables Adam to converge substantially faster than SGD, a phenomenon whose underlying mechanisms are still under active investigation [Balles and Hennig, 2018; Pan and Li, 2022; Kunstner et al., 2023; Zhang et al., 2024; Tomihari and Sato, 2025; Zhao et al., 2025; Jin et al., 2026]. 

However, the reliance on the second-moment accumulator $\mathbf { v } _ { t }$ incurs a significant memory overhead. Since $\mathbf { v } _ { t }$ shares the same dimensionality as the model parameters $\theta _ { t }$ and typically requires fp32 precision to maintain numerical stability, it doubles the optimizer state footprint relative to SGD with momentum. This “memory wall” has motivated a growing body of work aimed at reducing optimizer memory. One line of work approximates the second-moment matrix through factorization [Shazeer and Stern, 2018] or blockwise sharing [Zheng and Kwok, 2019; Zhang et al., 2025b]. Another applies low-precision quantization to the optimizer states [Dettmers et al., 2022; Li et al., 2024a; Han et al., 2025]. Alternatively, several optimizers dispense with the second moment buffer entirely [Bernstein et al., 2018; Balles and Hennig, 2018; Chen et al., 2024; Zhang et al., 2025a]. 

Orthogonal to the adaptive optimization methods, a parallel line of work explores optimization under an $\ell _ { p } { \ - } \mathrm { n o r m }$ geometry. Early work by Grove et al. [Grove et al., 2001] and Gentile [Gentile, 2003] established convergence guarantees for $\ell _ { p } .$ -norm algorithms in linear classification and regression. More recently, this geometric perspective has been adapted to accelerate gradient-based optimization in broader model classes. The Powerball method [Yuan et al., 2019] leverages $\ell _ { p }$ -norm steepest descent to update parameters directly, while pbSGDM [Zhou et al., 2021] integrates this geometry with momentum accumulation by applying a signed power transform before momentum. Stacey [Luo et al., 2025] further extends the approach, applying the transform to a momentum buffer within a primal-dual interpolation framework to accelerate SGD in non-convex settings. 

In this paper, we establish a direct connection between adaptive optimization and $\ell _ { p } .$ -norm steepest descent. Our contributions are as follows: 

• Algorithm design. We derive PowerStep from $\ell _ { p }$ -norm steepest descent principles and show that applying a signed power transform directly to a heavy-ball momentum buffer yields coordinate-wise adaptivity without storing any second-moment statistics. 

• Theoretical analysis. We establish an optima $O ( 1 / \sqrt { T } )$ convergence rate for PowerStep in non-convex stochastic optimization. 

• Empirical validation. We evaluate PowerStep on Transformer models ranging from 124M to 235B parameters, spanning both dense and mixture-of-experts (MoE) architectures. PowerStep matches the convergence speed of AdamW while using only half the optimizer state memory. Furthermore, PowerStep remains numerically stable under aggressive int8 quantization, reducing optimizer memory by $\sim 8 \times$ compared to full-precision AdamW. 

## 2 Algorithm

In this section, we present PowerStep, a memory-efficient optimizer that achieves adaptive convergence without second-moment estimation. We derive its update rule from first principles and provide a geometric interpretation of its behavior. 

## 2.1 Steepest descent on $\ell _ { p }$ -norm geometry

The goal of first-order optimization is to identify an update direction v that maximally decreases the objective $f ( \pmb \theta )$ . In the framework of steepest descent in general normed spaces [Boyd and Vandenberghe, 2004], the optimal direction is defined as 

$$
\mathbf {v} ^ {*} = \arg \min _ {\mathbf {v} \in \mathbb {R} ^ {d}} \langle \nabla f (\boldsymbol {\theta}), \mathbf {v} \rangle \quad \text { s.t. } \| \mathbf {v} \| \leq 1.\tag{2}
$$

The geometry of the trust region is dictated by the choice of norm. By adopting the Minkowski $\ell _ { p }$ norm, $\begin{array} { r } { \| \mathbf { v } \| _ { p } = ( \sum _ { i = 1 } ^ { d } | v _ { i } | ^ { p } ) ^ { 1 / p } } \end{array}$ , we define a continuum of geometries. To derive the analytical update, we form the Lagrangian using the p-th power of the norm constraint 

$$
\mathcal {L} (\mathbf {v}, \mu) = \sum_ {i = 1} ^ {d} g _ {i} v _ {i} + \mu \left(\sum_ {i = 1} ^ {d} | v _ {i} | ^ {p} - 1\right),\tag{3}
$$

where $g _ { i }$ is the i-th component of the gradient $\mathbf { g } = \nabla f ( \pmb { \theta } )$ and $\mu > 0$ is the Lagrange multiplier. Setting $\begin{array} { r } { \frac { \partial \mathcal { L } } { \partial v _ { i } } = 0 } \end{array}$ yields the coordinate-wise update rule 

$$
v _ {i} ^ {*} \propto - \mathrm{sign} (g _ {i}) \cdot | g _ {i} | ^ {\frac {1}{p - 1}}.\tag{4}
$$

Updating the parameters directly along this direction in a Euclidean manifold leads to $\ell _ { p }$ -norm steepest descent, also known as the Powerball method [Yuan et al., 2019]. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/00316063fc5f5002e084d34fcebb0883aa89e0947c5677bd795c6d7694e4026b.jpg)



Figure 1: Signed power transform $\Phi _ { \beta } ( x )$ for different $\beta$ values


## 2.2 Adaptivity via $\ell _ { p }$ -norm steepest descent

Motivated by steepest descent direction $\mathbf { v } ^ { * }$ in (4), we define the signed power transform as 

$$
\Phi_ {\beta} (\mathbf {x}) = \mathrm{sign} (\mathbf {x}) \odot | \mathbf {x} | ^ {\beta},\tag{5}
$$

where $\beta = 1 / ( p - 1 ) \in [ 0 , 1 ] , \odot$ denotes elementwise multiplication and all other operations $( \mathrm { s i g n } ( \cdot )$ $| \cdot |$ and $( \cdot ) ^ { \beta } )$ are applied elementwise. As illustrated in Figure 1, for $\beta < 1$ , the transform nonlinearly dampens large magnitudes while amplifying small ones. 

We propose PowerStep, which applies this nonlinear transform to the heavy-ball momentum [Polyak, 1964]. Unlike Powerball that transforms raw gradients, PowerStep accumulates gradients linearly and applies $\Phi _ { \beta }$ after momentum integration, allowing temporal smoothing of the noisy gradients, 

$$
\mathbf {m} _ {t} = \gamma \mathbf {m} _ {t - 1} + \mathbf {g} _ {t},\tag{6}
$$

$$
\boldsymbol {\theta} _ {t} = \boldsymbol {\theta} _ {t - 1} - \eta_ {t} \Phi_ {\beta} (\mathbf {m} _ {t}).\tag{7}
$$

The complete procedure is stated in Algorithm 1. 

Equivalently, the PowerStep update (7) can be expressed as a preconditioned momentum step, 

$$
\pmb {\theta} _ {t} = \pmb {\theta} _ {t - 1} - \eta_ {t} \mathbf {D} _ {t} \mathbf {m} _ {t},\tag{8}
$$

$$
\mathbf {D} _ {t} = \mathrm{diag} \big (| \mathbf {m} _ {t} | ^ {\beta - 1} \big),\tag{9}
$$

where $\mathbf { D } _ { t }$ is a diagonal preconditioner whose i-th entry is $| m _ { t , i } | ^ { \beta - 1 }$ , with $m _ { t , i }$ denoting the i-th component of $\mathbf { m } _ { t }$ . This formulation reveals the source of PowerStep’s adaptivity. For $\beta \in \bar { ( 0 , 1 ) }$ ), the exponent $\beta - 1$ is strictly negative, so each $| m _ { t , i } | ^ { \beta - 1 }$ is inversely related to the local momentum magnitude: when $| m _ { t , i } |$ is small (a flat direction), the preconditioner amplifies the step to accelerate progress; when $| m _ { t , i } |$ is large (a steep direction), it dampens the update to prevent overshooting. Crucially, this coordinate-wise scaling is computed instantaneously from $\mathbf { m } _ { t }$ without maintaining a second-moment estimator. 

At the extremes, PowerStep recovers classical methods: $\beta = 0$ yields $\Phi _ { 0 } ( { \bf m } _ { t } ) = \mathrm { s i g n } ( { \bf m } _ { t } )$ , reducing to SignSGD with momentum [Bernstein et al., 2018; Balles and Hennig, 2018]; $\beta = 1$ yields $\Phi _ { 1 } ( \mathbf { m } _ { t } ) = \mathbf { m } _ { t } .$ , recovering standard SGD with momentum [Polyak, 1964]. Between these limits, PowerStep interpolates between sign-based and linear updates, with $\beta = 0 .$ 1 providing a practically effective trade-off (Section 5.3). 

Algorithm 1 PowerStep
Require: $\theta_0 \in \mathbb{R}^d$ (initial parameters), $\{\eta_t\}_{t=1}^T$ (learning rate schedule)
Require: $\gamma \in [0,1)$ (momentum coefficient), $\beta \in [0,1]$ (power exponent), $\lambda \geq 0$ (weight decay)
1: $\mathbf{m}_0 \leftarrow \mathbf{0}$ 2: for $t = 1$ to $T$ do
3: $\mathbf{g}_t \leftarrow \nabla f_t(\boldsymbol{\theta}_{t-1})$ ▷ Compute stochastic gradient
4: $\mathbf{m}_t \leftarrow \gamma \mathbf{m}_{t-1} + \mathbf{g}_t$ ▷ Update momentum
5: $\mathbf{u}_t \leftarrow \text{sign}(\mathbf{m}_t) \odot |\mathbf{m}_t|^{\beta}$ ▷ Apply signed power transform
6: $\boldsymbol{\theta}_t \leftarrow \boldsymbol{\theta}_{t-1} - \eta_t(\mathbf{u}_t + \lambda \boldsymbol{\theta}_{t-1})$ ▷ Update with weight decay
7: end for 

## 2.3 Relation to prior methods

Powerball [Yuan et al., 2019] applies the signed power transform $\Phi _ { \beta } ( \mathbf { x } )$ to raw gradients, providing no temporal smoothing. pbSGDM [Zhou et al., 2021] applies $\Phi _ { \beta }$ before momentum accumulation, which can introduce a temporal mismatch when the gradient distribution shifts. PowerStep instead applies $\Phi _ { \beta }$ after momentum integration, ensuring the nonlinear scaling reflects the fully accumulated state. We provide empirical comparisons between PowerStep and pbSGDM in Section 5. 

Stacey [Luo et al., 2025] applies a stabilized transform $\Phi _ { \beta } ^ { \epsilon } ( \mathbf { x } ) = \mathrm { s i g n } ( \mathbf { x } ) \odot ( | \mathbf { x } | + \epsilon ) ^ { \beta }$ to an EMA momentum buffer within a primal-dual framework, making it the closest algorithmic relative to PowerStep. PowerStep differs in three respects. First, heavy-ball momentum [Polyak, 1964] replaces EMA, removing the zero-initialization bias and being more robust to quantization (see Appendix E). Second, the primal-dual auxiliary variables and the ϵ stabilization are eliminated, leaving only a single buffer with the exact transform $\Phi _ { \beta } ( \mathbf { x } )$ . Third, the design target shifts from accelerated convergence to memory efficiency. We show that this simplified, single-buffer variant suffices to match Adam’s adaptivity at scale, a finding bolstered by direct comparisons with Stacey in Appendix C. 

PowerStep can also be interpreted through mirror descent [Nemirovsky and Yudin, 1983; Beck and Teboulle, 2003]: the momentum buffer m acts as an accumulated dual state and $\Phi _ { \beta } ( \cdot )$ serves as the inverse mirror map, projecting the dual momentum back onto the primal manifold. 

## 3 Convergence analysis

In this section, we establish the convergence rate of PowerStep for non-convex stochastic optimization. We show that under standard regularity conditions, the algorithm achieves the optimal $O ( 1 / \sqrt { T } )$ convergence rate to a stationary point. For clarity, we analyze the unregularized setting $( \lambda = 0$ in Algorithm 1). All proofs are deferred to Appendix F. 

Following the standard setup for non-convex stochastic optimization [Ghadimi and Lan, 2013], we make the following assumptions on the objective function $f : \mathbb { R } ^ { d }  \bar { \mathbb { R } }$ 

Assumption 1 (L-Smoothness). f is continuously differentiable and there exists a constant $L > 0$ such that, for all $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d }$ 

$$
\left\| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \right\| _ {2} \leq L \| \mathbf {x} - \mathbf {y} \| _ {2}.\tag{10}
$$

Assumption 2 (Bounded Below). There exists $f ^ { * } \in \mathbb { R }$ such that $f ( \pmb { \theta } ) \geq f ^ { * }$ for all $\pmb \theta \in \mathbb { R } ^ { d }$ 

Assumption 3 (Bounded Gradient). There exists a constant $G > 0$ such that $\| \nabla f ( \pmb \theta ) \| _ { 2 } \le G$ for all $\pmb \theta \in \mathbb { R } ^ { d } .$ 

Assumption 4 (Unbiased Gradient). At each iteration t, the stochastic gradient g<sub>t</sub> satisfies 

$$
\mathbb {E} \left[ \mathbf {g} _ {t} \mid \boldsymbol {\theta} _ {t - 1} \right] = \nabla f (\boldsymbol {\theta} _ {t - 1}).\tag{11}
$$

Assumption 5 (Bounded Variance). There exists a constant $\sigma > 0$ such that, for all $t \geq 1$ 

$$
\mathbb {E} \big [ \| \mathbf {g} _ {t} - \nabla f (\pmb {\theta} _ {t - 1}) \| _ {2} ^ {2} | \pmb {\theta} _ {t - 1} \big ] \leq \sigma^ {2}.\tag{12}
$$

Assumption 3 is standard in the analysis of adaptive methods [Reddi et al., 2018; Chen et al., 2019; Luo et al., 2019; Défossez et al., 2022] and simplifies the control of the momentum buffer. We note that this condition is stronger than necessary for many practical objectives and has been removed in recent work on convergence of Adam [Zhang et al., 2022; Li et al., 2024b; Wang et al., 2024]. Extending our analysis to relax the global bounded gradient assumption is an important direction for future work. 

The signed power transform $\Phi _ { \beta } ( \mathbf { x } ) = \mathrm { s i g n } ( \mathbf { x } ) \odot | \mathbf { x } | ^ { \beta }$ is the core operation distinguishing PowerStep from standard momentum methods. The following three lemmas characterize its essential properties. Lemma 1 (Induced Norm Structure). For any vector m $\in \mathbb { R } ^ { d }$ and $\beta \in ( 0 , 1 ]$ 

$$
\langle \mathbf {m}, \Phi_ {\beta} (\mathbf {m}) \rangle = \| \mathbf {m} \| _ {1 + \beta} ^ {1 + \beta}.\tag{13}
$$

Lemma 1 shows that stepping along $- \Phi _ { \beta } ( \mathbf { m } )$ decreases the local linear approximation of f when m is aligned with the gradient, generalizing the identity $\langle \mathbf { m } , \mathbf { m } \rangle = \| \mathbf { m } \| _ { 2 } ^ { 2 }$ 

Lemma 2 (Norm Relationship). For any m $\in \mathbb { R } ^ { d }$ and $\beta \in ( 0 , 1 ]$ 

$$
\| \Phi_ {\beta} (\mathbf {m}) \| _ {2} ^ {2} = \| \mathbf {m} \| _ {2 \beta} ^ {2 \beta} \leq d ^ {1 - \beta} \| \mathbf {m} \| _ {2} ^ {2 \beta}.\tag{14}
$$

Lemma 2 bounds the $\ell _ { 2 } \cdot$ -norm of the transformed update. When $\beta < 1$ , the dependence on $\| \mathbf { m } \| _ { 2 }$ is sub-quadratic, providing an implicit gradient clipping effect. 

Lemma 3 (Hölder Continuity of $\Phi _ { \beta } ) _ { }$ . For any $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d }$ and $\beta \in ( 0 , 1 ]$ 

$$
\| \Phi_ {\beta} (\mathbf {x}) - \Phi_ {\beta} (\mathbf {y}) \| _ {1 + \beta} \leq C _ {\beta} \| \mathbf {x} - \mathbf {y} \| _ {1 + \beta} ^ {\beta},\tag{15}
$$

where $C _ { \beta } = 2 ^ { 1 - \beta } d ^ { ( 1 - \beta ) / ( 1 + \beta ) } \leq 2 d .$ 

Lemma 3 establishes Hölder continuity of order $\beta ,$ , which is critical for controlling the error when the momentum deviates from the true gradient. 

The next three lemmas form the backbone of the convergence proof. 

Lemma 4 (Momentum Bound). Under Assumptions 3–5, for PowerStep with $\gamma \in [ 0 , 1 )$ and any $t \geq 1$ 

$$
\mathbb {E} \left[ \| \mathbf {m} _ {t} \| _ {2} ^ {2} \right] \leq \frac {2 (G ^ {2} + \sigma^ {2})}{(1 - \gamma) ^ {2}}.\tag{16}
$$

Consequently, for any $\beta \in ( 0 , 1 ]$ 

$$
\mathbb {E} \left[ \| \Phi_ {\beta} (\mathbf {m} _ {t}) \| _ {2} ^ {2} \right] \leq d ^ {1 - \beta} 2 ^ {\beta} \left(\frac {G ^ {2} + \sigma^ {2}}{(1 - \gamma) ^ {2}}\right) ^ {\beta} =: M _ {\beta}.\tag{17}
$$

Lemma 4 provides a uniform bound on the second moment of the momentum and the transformed update. The bound depends only on $G , \sigma$ and $\gamma ,$ not on the learning rate or iteration count. 

Lemma 5 (Descent Inequality). Under Assumption $^ { l , }$ the iterates ofPowerStep with learning rate $\eta _ { t }$ satisfy 

$$
\mathbb {E} [ f (\pmb {\theta} _ {t}) ] \leq \mathbb {E} [ f (\pmb {\theta} _ {t - 1}) ] - \eta_ {t} \mathbb {E} \big [ \langle \nabla f (\pmb {\theta} _ {t - 1}), \Phi_ {\beta} (\mathbf {m} _ {t}) \rangle \big ] + \frac {L \eta_ {t} ^ {2}}{2} \mathbb {E} \big [ \| \Phi_ {\beta} (\mathbf {m} _ {t}) \| _ {2} ^ {2} \big ].\tag{18}
$$

Lemma 5 follows directly from L-smoothness and bounds the per-iteration decrease in function value. 

Lemma 6 (Gradient Alignment). Under Assumptions 1–3,for PowerStep with learning rate $\eta _ { t }$ and $\gamma \in [ 0 , 1 )$ , there exists a constant $C _ { 0 } > 0$ depending on $L , \gamma , G , \sigma ,$ d and $\beta$ such that for all $t \geq 1$ 

$$
\mathbb {E} \left[ \langle \nabla f (\boldsymbol {\theta} _ {t - 1}), \Phi_ {\beta} (\mathbf {m} _ {t}) \rangle \right] \geq \mathbb {E} \left[ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {1 + \beta} ^ {1 + \beta} \right] - C _ {0} (1 + \eta_ {t} ^ {\beta}).\tag{19}
$$

Lemma 6 lower-bounds the expected inner product $\left. \nabla f ( \pmb { \theta } _ { t - 1 } ) , \Phi _ { \beta } ( \mathbf { m } _ { t } ) \right.$ by the $( 1 + \beta )$ -power norm of the gradient, minus a bias term consisting of a constant noise floor from stochastic gradient variance and an $O ( \eta _ { t } ^ { \beta } )$ drift penalty from stale gradients. The drift term follows from L-smoothness and the Hölder continuity of $\Phi _ { \beta }$ (Lemma 3). Crucially, the additive separation of noise and drift enables the convergence proof: the decreasing learning rate $\eta _ { t } = \eta / \sqrt { t }$ shrinks the drift to zero while the noise floor is absorbed in the telescoping sum. For $\beta = 1$ , the bound recovers the standard heavy-ball momentum analysis. 

We now state the main convergence result. 

Theorem 1 (Convergence Rate). Under Assumptions $I { - } 5 ,$ let $\{ \pmb { \theta } _ { t } \} _ { t = 1 } ^ { T }$ be generated by PowerStep with learning rate $\eta _ { t } = \eta / \sqrt { t } f o r$ some $\eta > 0$ and momentum coefficient $\gamma \in [ 0 , 1 )$ . Thenfor any $\beta \in ( 0 , 1 ]$ 

$$
\min _ {t \in [ T ]} \mathbb {E} \left[ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {2} ^ {2} \right] = O \left(\frac {1}{\sqrt {T}}\right).\tag{20}
$$

Thus, PowerStep achieves the $O ( 1 / \sqrt { T } )$ convergence rate for non-convex stochastic optimization under standard assumptions, attaining the optimal rate bound for this problem class [Arjevani et al., 2023]. Their result states that any stochastic first-order method requires at least $\epsilon ^ { - 4 }$ gradient queries to find an ϵ-stationary point. Inverting this relationship, $T = \Omega ( \epsilon ^ { - 4 } )$ implies $\epsilon = \overset { \overline { { } } } { O } ( T ^ { - 1 / \overline { { 4 } } } )$ and therefore $\| \nabla f \| ^ { 2 } \leq \epsilon ^ { \bar { 2 } } = O ( T ^ { - 1 / 2 } )$ . Thus, the query complexity lower bound of $\epsilon ^ { - 4 }$ is equivalent to an $\Omega ( 1 / \sqrt { T } )$ convergence rate for the squared gradient norm, confirming that PowerStep’s rate is optimal for this problem class. 

## 4 Low-precision training

PowerStep’s single-buffer design naturally lends itself to aggressive quantization for further memory savings. We advocate the following implementation that reuses the gradient buffer $\mathbf { g } _ { t }$ for in-place computation. 

$$
\mathbf {g} _ {t} \leftarrow \nabla f _ {t} (\boldsymbol {\theta} _ {t - 1}),\tag{21}
$$

$$
\mathbf {g} _ {t} \leftarrow \mathbf {g} _ {t} + \gamma \cdot \text { dequantize } (\mathbf {m} _ {t - 1}),\tag{22}
$$

$$
\mathbf {m} _ {t} \leftarrow \operatorname{quantize} (\mathbf {g} _ {t}),\tag{23}
$$

$$
\mathbf {g} _ {t} \leftarrow \mathrm{sign} (\mathbf {g} _ {t}) \odot | \mathbf {g} _ {t} | ^ {\beta},\tag{24}
$$

$$
\boldsymbol {\theta} _ {t} \leftarrow \boldsymbol {\theta} _ {t - 1} - \eta_ {t} (\mathbf {g} _ {t} + \lambda \boldsymbol {\theta} _ {t - 1}).\tag{25}
$$

Crucially, only the momentum buffer $\mathbf { m } _ { t }$ is stored in low precision; both the signed power transform and the parameter update are then computed in full precision. We compress m<sub>t</sub> from fp32 to int8 using the blockwise quantization technique of Dettmers et al. [2022]. As shown in Section 5.4, this configuration preserves both convergence speed and numerical stability while reducing the optimizer memory footprint by approximately 8× compared to full-precision Adam. In contrast, Adam diverges under the same int8 quantization. This failure stems from the high precision-sensitivity of its second-moment estimator [Li et al., 2024a; Han et al., 2025; Tang et al., 2026], a vulnerability that PowerStep eliminates entirely by design. We provide a detailed analysis in Appendix E. 

## 5 Experiments

## 5.1 Setup

Models. We evaluate PowerStep on a diverse suite of Transformer language models for pretraining, spanning 124M to 235B parameters. The suite covers both dense and MoE architectures. Dense models include the GPT-2 series [Radford et al., 2019] (124M and 350M) and the Qwen3 series [Yang et al., 2025] (0.6B, 1.7B, 4B, 8B and 32B). MoE models include DeepSeek-V2-Lite [DeepSeek-AI et al., 2024] (16B total, 2.4B active) and two Qwen3-MoE variants (30B-A3B and 235B-A22B) [Yang et al., 2025]. All model parameters are kept in bf16 precision. 

Datasets. GPT-2 models are trained on the OpenWebText corpus [Gokaslan et al., 2019]. All Qwen3 and DeepSeek-V2 models are trained on the C4 dataset [Raffel et al., 2020]. 

Optimizers. We compare PowerStep against AdamW [Loshchilov and Hutter, 2019] and several closely related optimizers that also completely eliminate the full second-moment buffer: pbSGDM [Zhou et al., 2021], SignSGD with momentum [Bernstein et al., 2018; Balles and Hennig, 2018] and AdamS [Zhang et al., 2025a]. All optimizer states are maintained in fp32 precision for the smallscale experiments (Sections 5.2–5.3). For the quantization and large-scale experiments (Sections 5.4 and 5.5), precision is indicated explicitly in the text and figures. 

Hyperparameters. For AdamW and AdamS, we set $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ and $\epsilon = 1 0 ^ { - 8 }$ . For PowerStep and pbSGDM, we use momentum coefficient $\gamma = 0 . 9$ and power exponent $\beta = 0 .$ .1; an ablation study of both hyperparameters is provided in Section 5.3. The learning rate ranges from $6 \times 1 0 ^ { - 4 } \ \mathrm { t o } \ \dot { 2 } \times 1 0 ^ { - 4 }$ depending on model size (see Table 2 in Appendix A). All runs employ decoupled weight decay $\lambda = 0 . 1$ , global gradient norm clipping at 1.0 and a 2000-step linear warmup followed by cosine decay. For MoE models, an auxiliary load-balancing loss with coefficient $1 \times 1 0 ^ { - \bar { 3 } }$ is applied. For GPT-2, we use a sequence length of 1024 and a global batch size of 480. For all other models, we use a sequence length of 2048 and a global batch size of 256. More details are provided in Appendix A. 

Infrastructure. All experiments are conducted on Huawei Ascend 910C NPU clusters. GPT-2 models are trained using the nanoGPT codebase<sup>2</sup>. Billion-parameter models are trained with Megatron-Core<sup>3</sup> and MindSpeed-LLM<sup>4</sup> frameworks. 

## 5.2 Comparison with prior methods

We evaluate PowerStep against AdamW, AdamS, pbSGDM and SignSGD on small-scale models, ranging from 124M to 8B parameters. Figure 2 reports training loss trajectories. Across all model scales, PowerStep matches the convergence speed of AdamW. In contrast, the related memoryefficient optimizers, pbSGDM, SignSGD and AdamS, exhibit slower convergence or, for the larger models, catastrophic training instability. The validation loss can be found in Appendix D. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/0f3c3790e2d2dc8a5d085c3eb01994aa492edd9d41522b653892a4e60832a4c0.jpg)



(a) GPT-2-Small (124M)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/8d9602c8b00c4501887eb4fd9d31e78672cd9c4f226e25bf3f9e20c02d8b4f2a.jpg)



(b) GPT-2-Medium (350M)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/d56f797462dce54597395bcce07eeccf75f177b89585f8ff8a52976657426c0e.jpg)



(c) Qwen3-0.6B


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/e1cfe95aa707461495eb19dbc1e9ca76cad899543ea702a18e3231be4012cf51.jpg)



(d) Qwen3-1.7B


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/e9d08708619ee28c81bfff245eea4c3b967d92db5d26ab15dc58cf46f1542bff.jpg)



(e) Qwen3-4B


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/8ab4870c1d0d3f3ad207f0650d5e18baaf0bc68ce8f23ec452fbcaeb4553b493.jpg)



(f) Qwen3-8B



Figure 2: Training loss comparison across model scales. PowerStep matches the convergence speed and stability of AdamW while using half the optimizer memory. Other memory-efficient optimizers, pbSGDM, SignSGD and AdamS, exhibit slower convergence or instability on larger models.


## 5.3 Hyperparameter sensitivity

We analyze the sensitivity of PowerStep to its two key hyperparameters: power exponent $\beta$ and momentum coefficient γ. We conduct an ablation on GPT-2-Medium (350M), varying $\bar { \beta } \in \{ 0 , 0 . 1 , 0 . 2 \}$ and $\gamma \in \{ 0 . 8 5 , 0 . 9 , 0 . 9 5 \}$ . Figure 3(a) shows that $\beta = 0 . 1$ provides the best trade-off. A value of $\beta = 0 . 0$ (equivalent to SignSGD with momentum) leads to initial rapid progress but ultimately collapses, underscoring the necessity of retaining some magnitude information. Conversely, $\beta = 0 . \dot { 2 }$ converges more slowly due to insufficient nonlinearity. As shown in Figure 3(b), the method is robust to the momentum coefficient $\gamma$ in the range [0.85, 0.95]. We also provide an ablation study on learning rates in Appendix B. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/453be8a103cbe64dea953a40da5c607b8b7158f8d86dbefab76ed7bf54ea1799.jpg)



(a) Varying power exponent β


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/36e5b32c4a8d3f6ebb66c7a00c7cd50597ee8b0a5437a232f74e330c4f25aa95.jpg)



(b) Varying momentum coefficient γ



Figure 3: Hyperparameter sensitivity on GPT-2-Medium (350M). (a) Power exponent $\beta \colon \beta = 0 . 1$ balances rapid early progress with long-run stability; $\beta = 0$ (SignSGD) collapses, while $\beta = 0 . 2$ converges slowly. (b) Momentum coefficient γ: performance is robust across $\gamma \in [ 0 . 8 5 , 0 . 9 5 ]$ , with $\gamma = 0 . 9$ providing a reliable default.


## 5.4 Quantization

A key advantage of PowerStep’s single-buffer design is its amenability to aggressive quantization. We compare AdamW and PowerStep under a naive int8 quantization of optimizer states on GPT-2-Small and GPT-2-Medium with blockwise dynamic quantization [Dettmers et al., 2022] of block size 128, without sophisticated techniques such as stochastic rounding, for all layers (including embedding). As shown in Figure 4, AdamW training collapses immediately under the int8 quantization, a known failure mode caused by the catastrophic accumulation of quantization error in the second-moment estimator [Li et al., 2024a; Han et al., 2025; Tang et al., 2026]. In contrast, PowerStep maintains stability and convergence speed, matching its full-precision counterpart (see Appendix E for further analysis). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/1f6c0c2d0ac2c2a5240f5b001a03e1b3d16ac688888dfbacf59b2c4218e60250.jpg)



(a) GPT-2-Small


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/02337b6913286ea978fce525bca791ebf4e48fd29f41eb3add6eb8d67a2c848c.jpg)



(b) GPT-2-Medium



Figure 4: Training loss under int8 optimizer state quantization. AdamW diverges under the quantization while PowerStep remains stable and matches full-precision convergence.


## 5.5 Scaling to large models

Finally, we evaluate PowerStep on large-scale models to verify its scalability. We train DeepSeek-V2-Lite (16B), Qwen3-30B-A3B, Qwen3-32B and Qwen3-235B-A22B, spanning both dense and MoE architectures, and compare full-precision AdamW against PowerStep with int8 quantization. Figure 5 reports training loss trajectories. Across all four models, PowerStep (int8) matches the convergence of AdamW (fp32) without divergence or degradation. We do not observe noticeable wall-clock time or throughput differences between PowerStep and AdamW, since all additional operations (sign, power, and quantization) are elementwise and contribute negligible overhead. Table 1 reports optimizer state memory per NPU. PowerStep reduces the optimizer memory footprint by approximately 8× relative to full-precision AdamW. Table 6 in Appendix D reports final validation loss, confirming that PowerStep with int8 quantization matches full-precision AdamW with negligible difference in validation performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/54b53593fe7cdf7d85b2273f8b02d517e35f4c135a645bad5e42a4ad880c88f0.jpg)



(a) DeepSeek-V2-Lite (16B)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/8ea751830473e98aed3c5770a6e021c33e9d3b6095d48e1ab875725c6e610baa.jpg)



(b) Qwen3-30B-A3B


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/2f62ab5f80230b5cafbd60cfe837e3ab16f8c959f7a621fee7f52e268c20bda6.jpg)



(c) Qwen3-32B


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/88b0b4ee70d6fbc9784da3cad0b543ddff3bc845d3443006cc0c57b28ac3a67b.jpg)



(d) Qwen3-235B-A22B



Figure 5: Training loss on large-scale models. PowerStep with int8 quantization achieves convergence parity with full-precision AdamW.



Table 1: Average optimizer state memory per NPU (MB). All values are reported for training on 256 NPUs (see Table 3 for details). PowerStep with int8 quantization reduces the memory footprint by approximately 8× compared to full-precision AdamW.


<table><tr><td>Model</td><td>AdamW (fp32)</td><td>PowerStep (int8)</td></tr><tr><td>DeepSeek-V2-Lite (16B)</td><td>429.0</td><td>55.3</td></tr><tr><td>Qwen3-30B-A3B</td><td>864.0</td><td>111.4</td></tr><tr><td>Qwen3-32B</td><td>976.5</td><td>125.9</td></tr><tr><td>Qwen3-235B-A22B</td><td>6768.0</td><td>872.4</td></tr></table>

## 6 Conclusion and future work

We introduced PowerStep, a memory-efficient alternative to second-moment adaptive optimizers. By applying the signed power transform to the momentum buffer, PowerStep achieves instantaneous momentum magnitude modulation while converging at the optimal $O ( 1 / \sqrt { T } )$ rate for non-convex stochastic optimization. Extensive experiments on Transformer models up to 235B parameters demonstrate that PowerStep matches Adam’s convergence speed while halving optimizer memory. Low-precision implementations (int8) further reduce the memory footprint by ∼ 8× compared to full precision Adam without sacrificing performance. These results establish PowerStep as a principled, scalable and practical choice for large-scale training. 

A promising direction for future work is integrating PowerStep’s memory-efficient $\ell _ { p }$ adaptivity with structured matrix-valued updates. The recently proposed Muon optimizer [Jordan, 2024], derived from steepest descent under a matrix norm [Bernstein and Newhouse, 2024], accelerates Transformer training by orthogonalizing momentum matrices to regularize their spectrum. Combining PowerStep’s adaptivity and low-precision robustness with Muon’s spectral regularization could yield a new class of optimizers that are simultaneously adaptive, memory-efficient, numerically stable and spectrally accelerated. Beyond this, exploring more aggressive quantization (e.g., int4), evaluating on downstream language tasks (e.g., fine-tuning) and extending PowerStep to vision and multimodal domains are also promising avenues for future investigation. 

## References



Arjevani, Y., Carmon, Y., Duchi, J. C., Foster, D. J., Srebro, N., and Woodworth, B. (2023). Lower bounds for non-convex stochastic optimization. Mathematical Programming. 





Balles, L. and Hennig, P. (2018). Dissecting Adam: The sign, magnitude and variance of stochastic gradients. In International Conference on Machine Learning. 





Beck, A. and Teboulle, M. (2003). Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters. 





Bengio, Y., Simard, P., and Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks. 





Bernstein, J. and Newhouse, L. (2024). Old optimizer, new norm: An anthology. arXiv preprint arXiv:2503.12345. 





Bernstein, J., Wang, Y.-X., Azizzadenesheli, K., and Anandkumar, A. (2018). signSGD: Compressed optimisation for non-convex problems. International Conference on Machine Learning. 





Boyd, S. and Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press. 





Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., Lu, Y., and Le, Q. V. (2024). Symbolic discovery of optimization algorithms. Advances in Neural Information Processing Systems. 





Chen, X., Liu, S., Sun, R., and Hong, M. (2019). On the convergence of a class of Adam-type algorithms for non-convex optimization. International Conference on Learning Representations. 





Choromanska, A., Henaff, M., Mathieu, M., Arous, G. B., and LeCun, Y. (2015). The loss surfaces of multilayer networks. International Conference on Artificial Intelligence and Statistics. 





Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., and Bengio, Y. (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. Advances in Neural Information Processing Systems. 





DeepSeek-AI, Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Deng, C., Ruan, C., Dai, D., et al. (2024). Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434. 





Défossez, A., Bottou, L., Bach, F., and Usunier, N. (2022). A simple convergence proof of Adam and Adagrad. Transactions on Machine Learning Research. 





Dettmers, T., Lewis, M., Shleifer, S., and Zettlemoyer, L. (2022). 8-bit optimizers via block-wise quantization. International Conference on Learning Representations. 





Duchi, J., Hazan, E., and Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal ofMachine Learning Research. 





Gentile, C. (2003). The robustness of the p-norm algorithms. Machine Learning. 





Ghadimi, S. and Lan, G. (2013). Stochastic first- and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization. 





Gokaslan, A., Cohen, V., Pavlick, E., and Tellex, S. (2019). Openwebtext corpus. 





Grove, A. J., Littlestone, N., and Schuurmans, D. (2001). General convergence results for linear discriminant updates. Machine Learning. 





Han, Y., Yang, C., Chen, C., Wang, X., and Sun, R. (2025). Q-adam-mini: Memory-efficient 8-bit quantized optimizer for large language model training. ICML Workshop on Large Language Models and Cognition. 





Hochreiter, S. (1991). Untersuchungen zu dynamischen neuronalen netzen. Diploma, Technische Universität München. 





Jin, R., Liang, Y., and Zou, S. (2026). Why adam can beat sgd: Second-moment normalization yields sharper tails. arXiv preprint arXiv:2603.03099. 





Jordan, K. (2024). Muon: An optimizer for hidden layers in neural networks. Blog Post. 





Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. International Conference on Learning Representations. 





Kunstner, F., Chen, J., Lavington, J. W., and Schmidt, M. (2023). Noise is not the main factor behind the gap between sgd and adam on transformers, but sign descent might be. International Conference on Learning Representations. 





LeCun, Y., Bottou, L., Orr, G. B., and Müller, K.-R. (1998). Efficient backprop. Neural Networks: Tricks ofthe Trade. 





Li, B., Chen, J., and Zhu, J. (2024a). Memory-efficient optimizers with 4-bit states. Advances in Neural Information Processing Systems. 





Li, H., Jadbabaie, A., and Rakhlin, A. (2024b). Convergence of Adam under relaxed assumptions. Advances in Neural Information Processing Systems. 





Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. International Conference on Learning Representations. 





Luo, L., Xiong, Y., Liu, Y., and Sun, X. (2019). Adaptive gradient methods with dynamic bound of learning rate. International Conference on Learning Representations. 





Luo, X., Bai, C. S., Li, B., Drineas, P., Zhang, R., and Bullins, B. (2025). Stacey: Promoting stochastic steepest descent via accelerated ℓ<sub>p</sub>-smooth nonconvex optimization. International Conference on Machine Learning. 





Nemirovsky, A. S. and Yudin, D. B. (1983). Problem Complexity and Method Efficiency in Optimization. John Wiley & Sons. 





Pan, Y. and Li, Y. (2022). Toward understanding why adam converges faster than SGD for transformers. Advances in Neural Information Processing Systems. 





Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. Ussr Computational Mathematics and Mathematical Physics. 





Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI Technical Report. 





Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research. 





Reddi, S. J., Kale, S., and Kumar, S. (2018). On the convergence of Adam and beyond. International Conference on Learning Representations. 





Shazeer, N. and Stern, M. (2018). Adafactor: Adaptive learning rates with sublinear memory cost. International Conference on Machine Learning. 





Tang, X., Li, J., and Zou, D. (2026). A convergence analysis of adaptive optimizers under floatingpoint quantization. International Conference on Learning Representations. 





Tieleman, T. and Hinton, G. (2012). Lecture 6.5—RMSProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning. Lecture slides. 





Tomihari, A. and Sato, I. (2025). Understanding why Adam outperforms SGD: Gradient heterogeneity in transformers. arXiv preprint arXiv:2502.00213. 





Topollai, K. and Choromanska, A. (2026). Understanding quantization of optimizer states in LLM pre-training: Dynamics of state staleness and effectiveness of state resets. arXiv preprint arXiv:2603.16731. 





Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems. 





Wang, B., Fu, J., Zhang, H., Zheng, N., and Chen, W. (2024). Closing the gap between the upper bound and lower bound of Adam’s iteration complexity. Advances in Neural Information Processing Systems. 





Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. (2025). Qwen3 technical report. arXiv preprint arXiv:2505.09388. 





Yuan, Y., Li, M., Liu, J., and Tomlin, C. (2019). On the powerball method: variants of descent methods for accelerated optimization. IEEE Control Systems Letters. 





Zhang, H., Wang, B., and Chen, L. (2025a). Adams: Momentum itself can be a normalizer for llm pretraining and post-training. Conference on Empirical Methods in Natural Language Processing. 





Zhang, Y., Chen, C., Ding, T., Li, Z., Sun, R., and Luo, Z.-Q. (2024). Why transformers need adam: A hessian perspective. Advances in Neural Information Processing Systems. 





Zhang, Y., Chen, C., Li, Z., Ding, T., Wu, C., Kingma, D. D., Ye, Y., Luo, Z.-Q., and Sun, R. (2025b). Adam-mini: Use fewer learning rates to gain more. International Conference on Learning Representations. 





Zhang, Y., Chen, C., Shi, N., Sun, R., and Luo, Z.-Q. (2022). Adam can converge without any modification on update rules. Advances in Neural Information Processing Systems. 





Zhao, R., Morwani, D., Brandfonbrener, D., Vyas, N., and Kakade, S. (2025). Deconstructing what makes a good optimizer for language models. International Conference on Learning Representations. 





Zheng, S. and Kwok, J. T. (2019). Blockwise adaptivity: Faster training and better generalization in deep learning. arXiv preprint arXiv:1905.09899. 





Zhou, B., Liu, J., Sun, W., Chen, R., Tomlin, C., and Yuan, Y. (2021). pbsgd: powered stochastic gradient descent methods for accelerated nonconvex optimization. International Joint Conferences on Artificial Intelligence. 



## A Experimental details

## A.1 Hyperparameters

Table 2 reports the per-model learning rate configurations. Common settings across all experiments: decoupled weight decay $\lambda = 0 . 1$ and gradient norm clipping at 1.0. For MoE models, an auxiliary load-balancing loss with coefficient $\mathrm { 1 \check { \times } 1 0 ^ { - 3 } }$ is applied. A 2000-step linear learning rate warmup from zero to $\eta _ { \mathrm { m a x } }$ followed by cosine decay to $\eta _ { \mathrm { m i n } }$ is applied. For the fairness of comparison, the same learning rate schedule is used for all optimizers within each model. 

We acknowledge that a fully rigorous demonstration would require independent learning rate sweeps for each optimizer at every model scale. Given our evaluation suite of ten models and five optimizers, such sweeps were precluded by computational constraints. However, the structural analogy between the AdamW and PowerStep and the consistent convergence parity across all model scales collectively support the fairness of the comparison. A mismatched learning rate would manifest as scale-dependent degradation; we observe no such trend. We therefore conclude that PowerStep’s ability to match AdamW while halving optimizer memory is not an artifact of learning rate mismatch. A systematic sensitivity analysis is left to future work. We also provide an ablation study of learning rates in Appendix B. 


Table 2: Training hyperparameters


<table><tr><td>Model</td><td>Batch size</td><td>Context length</td><td><eq>\eta_{\text{max}}</eq></td><td><eq>\eta_{\text{min}}</eq></td></tr><tr><td>GPT-2-Small (124M)</td><td>480</td><td>1024</td><td><eq>6 \times 10^{-4}</eq></td><td><eq>6 \times 10^{-5}</eq></td></tr><tr><td>GPT-2-Medium (350M)</td><td>480</td><td>1024</td><td><eq>6 \times 10^{-4}</eq></td><td><eq>6 \times 10^{-5}</eq></td></tr><tr><td>Qwen3-0.6B</td><td>256</td><td>2048</td><td><eq>5 \times 10^{-4}</eq></td><td><eq>5 \times 10^{-5}</eq></td></tr><tr><td>Qwen3-1.7B</td><td>256</td><td>2048</td><td><eq>5 \times 10^{-4}</eq></td><td><eq>5 \times 10^{-5}</eq></td></tr><tr><td>Qwen3-4B</td><td>256</td><td>2048</td><td><eq>5 \times 10^{-4}</eq></td><td><eq>5 \times 10^{-5}</eq></td></tr><tr><td>Qwen3-8B</td><td>256</td><td>2048</td><td><eq>3 \times 10^{-4}</eq></td><td><eq>3 \times 10^{-5}</eq></td></tr><tr><td>DeepSeek-V2-Lite (16B)</td><td>256</td><td>2048</td><td><eq>2 \times 10^{-4}</eq></td><td><eq>2 \times 10^{-5}</eq></td></tr><tr><td>Qwen3-30B-A3B</td><td>256</td><td>2048</td><td><eq>2 \times 10^{-4}</eq></td><td><eq>2 \times 10^{-5}</eq></td></tr><tr><td>Qwen3-32B</td><td>256</td><td>2048</td><td><eq>2 \times 10^{-4}</eq></td><td><eq>2 \times 10^{-5}</eq></td></tr><tr><td>Qwen3-235B-A22B</td><td>256</td><td>2048</td><td><eq>2 \times 10^{-4}</eq></td><td><eq>2 \times 10^{-5}</eq></td></tr></table>

## A.2 Parallelism configuration

Table 3 details the parallelism configuration in training for each model. We denote data parallelism by DP, tensor parallelism by TP, pipeline parallelism by PP and expert parallelism by EP. The total number of NPUs satisfies $\mathrm { \Delta N P U s } = \bar { \mathrm { D P } } \times \bar { \mathrm { T P } } \times \mathrm { P P }$ . For MoE architectures, DP and EP share the same communication group with $\mathbf { E P \le D P }$ , following DeepSeek-AI et al. [2024]. 


Table 3: Parallelism configuration across models


<table><tr><td>Model</td><td>NPUs</td><td>DP</td><td>TP</td><td>PP</td><td>EP</td></tr><tr><td>GPT-2-Small (124M)</td><td>8</td><td>8</td><td>1</td><td>1</td><td>N/A</td></tr><tr><td>GPT-2-Medium (350M)</td><td>8</td><td>8</td><td>1</td><td>1</td><td>N/A</td></tr><tr><td>Qwen3-0.6B</td><td>8</td><td>8</td><td>1</td><td>1</td><td>N/A</td></tr><tr><td>Qwen3-1.7B</td><td>8</td><td>8</td><td>1</td><td>1</td><td>N/A</td></tr><tr><td>Qwen3-4B</td><td>8</td><td>8</td><td>1</td><td>1</td><td>N/A</td></tr><tr><td>Qwen3-8B</td><td>32</td><td>16</td><td>1</td><td>2</td><td>N/A</td></tr><tr><td>DeepSeek-V2-Lite (16B)</td><td>256</td><td>256</td><td>1</td><td>1</td><td>8</td></tr><tr><td>Qwen3-30B-A3B</td><td>256</td><td>64</td><td>1</td><td>4</td><td>4</td></tr><tr><td>Qwen3-32B</td><td>256</td><td>16</td><td>8</td><td>2</td><td>N/A</td></tr><tr><td>Qwen3-235B-A22B</td><td>256</td><td>64</td><td>1</td><td>4</td><td>8</td></tr></table>

## B Ablation on learning rates

A persistent concern in optimizer evaluation is whether reported performance parity reflects genuine algorithmic quality or merely a learning rate mismatch favoring one method. To address this, we conduct a controlled sensitivity analysis on GPT-2-Small (124M), varying $\eta _ { \mathrm { m a x } } \in \{ 1 \times 1 0 ^ { - 4 } , 2 \times$ $1 0 ^ { - 4 } , 4 \times 1 0 ^ { - 4 } , 6 \times 1 0 ^ { - 4 } , 8 \stackrel { \cdot } { \times } 1 0 ^ { - 4 } , 1 \times 1 0 ^ { - 3 } \big \}$ with $\eta _ { \mathrm { m i n } } = 0 . 1 \cdot \eta _ { \mathrm { m a x } } .$ , while keeping all other hyperparameters fixed at the values used in Section 5.2. 

Figure 6 reports the resulting training loss. PowerStep’s sensitivity profile closely mirrors AdamW’s across the full range, with no sign of the systematic divergence or instability that would signal an unfair comparison. Notably, both optimizers converge faster at larger learning rates, indicating that the learning rates used in our main experiments are not biased in favor of either method. These results support the conclusion that PowerStep achieves AdamW-style adaptivity without requiring a second-moment buffer. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/171bd06676a223d2cb61fa1b615e5f4102334f66fd90f6c0f30ab30994e74cb1.jpg)



(a) AdamW


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/03005c5fda358424324500c2fa3a520f06249b5a23c2812cec2f3a9ed02d7764.jpg)



(b) PowerStep



Figure 6: Training loss on GPT-2-Small (124M)


## C Comparison to Stacey

PowerStep can be viewed as a simplified, memory-efficient variant of Stacey- $\cdot ( p , 2 )$ [Luo et al., 2025] that removes the primal-dual auxiliary variables and the ϵ-stabilization term and employs heavy-ball momentum. To assess whether these simplifications incur any performance cost, we conduct a direct comparison between the two optimizers on GPT-2-Small (124M) and GPT-2-Medium (350M). For $\mathrm { S t a c e y } { - } ( p , 2 )$ , we adopt the hyperparameters from [Luo et al., 2025]: $\alpha = 0 . 1 , \beta _ { 1 } = 0 . 9 .$ $\beta _ { 2 } = 0 . 9 9 , \tau = 0 . 0 0 1$ , and $\epsilon = 1 \times 1 0 ^ { - 8 }$ . We set $p = 1 1$ , corresponding to $\beta = 1 / ( p - 1 ) = 0 . 1$ which matches PowerStep’s power exponent for a controlled comparison; lower values of p lead to slower convergence for Stacey. For learning rates, we evaluate $\eta _ { \mathrm { m a x } } \in \{ 6 \times 1 0 ^ { - 4 } , 1 \times 1 0 ^ { - 3 } \}$ with $\eta _ { \mathrm { m i n } } = 0 . 1 \cdot \eta _ { \mathrm { m a x } }$ across both optimizers. All other settings follow Section 5.2. Figure 7 reports the training loss trajectories. PowerStep converges faster in the early stages of training. Both optimizers ultimately reach comparable final loss values. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/b64dff8e6999363240f205a4fc8797360f404962ce248c2418468716374bc3a0.jpg)



(a) GPT-2-Small (124M)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/7b46da74-4d1a-4fd9-bdda-5f8b69f250ca/a1eeca831abdc2f6b40ed7eb8ac8254754847a0e4d9579de75af5cec3cb4ac98.jpg)



(b) GPT-2-Medium (350M)



Figure 7: Training loss comparison between PowerStep and Stacey- $( p , 2 )$


## D Validation results

The validation loss results on small-scale and large-scale models are reported in Tables 4 and 6, respectively. The impact of optimizer state quantization on validation performance is examined separately in Table 5. 

As shown in Table 4, on small-scale dense models ranging from 124M to 8B parameters, PowerStep achieves validation loss comparable to AdamW across all model sizes. While AdamW holds a slight edge on several models, the performance gap is marginal, confirming that PowerStep’s single-buffer design does not sacrifice overall convergence quality. 

Table 5 isolates the effect of int8 quantization on the momentum buffer. On GPT-2-Small and GPT-2-Medium, PowerStep with int8 quantization matches both its full-precision counterpart and full-precision AdamW, with no measurable degradation in validation loss. AdamW under int8 quantization is omitted because it leads to training collapse (see Figure 4 of the main text). These results confirm that PowerStep’s heavy-ball momentum buffer is inherently robust to aggressive compression, in contrast to Adam’s second-moment estimator. 

Table 6 further demonstrates PowerStep’s scalability to large-scale models up to 235B parameters, including both dense and MoE architectures. PowerStep with int8 quantization matches fullprecision AdamW on all four models. On DeepSeek-V2-Lite and Qwen3-235B-A22B, PowerStep (int8) achieves lower validation loss (2.618 and 2.524, respectively), while on Qwen3-30B-A3B and Qwen3-32B the gap is within 0.004 − 0.010. These results establish that PowerStep’s 8-bit training recipe preserves final model quality while reducing optimizer memory by approximately 8×, as quantified in Table 1 of the main text. 


Table 4: Validation loss on small-scale dense models. The best result per model is highlighted in bold. Although PowerStep slightly trails AdamW in some cases, the performance gap remains marginal: at most 0.028 and typically ≤ 0.02 across all models. All optimizer states are in fp32.


<table><tr><td>Model</td><td>AdamW</td><td>AdamS</td><td>SignSGD</td><td>pbSGDM</td><td>PowerStep</td></tr><tr><td>GPT-2-Small (124M)</td><td>2.950</td><td>2.951</td><td>2.942</td><td>3.377</td><td>2.949</td></tr><tr><td>GPT-2-Medium (350M)</td><td>2.711</td><td>2.725</td><td>2.736</td><td>3.110</td><td>2.717</td></tr><tr><td>Qwen3-0.6B</td><td>2.912</td><td>2.912</td><td>2.897</td><td>3.198</td><td>2.923</td></tr><tr><td>Qwen3-1.7B</td><td>2.780</td><td>2.782</td><td>2.794</td><td>3.034</td><td>2.799</td></tr><tr><td>Qwen3-4B</td><td>2.690</td><td>2.672</td><td>2.680</td><td>2.712</td><td>2.712</td></tr><tr><td>Qwen3-8B</td><td>2.660</td><td>4.689</td><td>5.045</td><td>2.853</td><td>2.688</td></tr></table>


Table 5: Validation loss under optimizer state quantization. The best result per model is shown in bold. AdamW with int8 quantization leads to training collapse (Figure 4) and is therefore omitted. PowerStep with int8 matches full-precision AdamW, confirming that aggressive compression incurs no degradation in model quality.


<table><tr><td>Model</td><td>AdamW (fp32)</td><td>PowerStep (fp32)</td><td>PowerStep (int8)</td></tr><tr><td>GPT-2-Small (124M)</td><td>2.950</td><td>2.949</td><td>2.948</td></tr><tr><td>GPT-2-Medium (350M)</td><td>2.711</td><td>2.717</td><td>2.716</td></tr></table>


Table 6: Validation loss on large-scale dense and MoE models. The better result per model is highlighted in bold. PowerStep (int8) matches AdamW (fp32) across all models with negligible differences, demonstrating that the ∼ 8× memory reduction preserves validation performance.


<table><tr><td>Model</td><td>AdamW (fp32)</td><td>PowerStep (int8)</td></tr><tr><td>DeepSeek-V2-Lite (16B)</td><td>2.623</td><td>2.618</td></tr><tr><td>Qwen3-30B-A3B</td><td>2.620</td><td>2.624</td></tr><tr><td>Qwen3-32B</td><td>2.572</td><td>2.582</td></tr><tr><td>Qwen3-235B-A22B</td><td>2.525</td><td>2.524</td></tr></table>

## E Analysis of int8 quantization robustness

PowerStep’s empirical robustness to aggressive int8 quantization (Section 5.4) stands in contrast to AdamW, which diverges under the same compression. We provide an analysis that explains this discrepancy. 

## E.1 Quantization model

We model blockwise dynamic int8 quantization as follows. For a vector $\mathbf { x } \in \mathbb { R } ^ { d }$ partitioned into blocks of size $B ,$ the quantized representation $\mathcal { Q } ( \mathbf { x } )$ satisfies 

$$
\mathcal {Q} (\mathbf {x}) = \mathbf {x} + \boldsymbol {\delta}, \| \boldsymbol {\delta} \| _ {\infty} \leq \frac {\| \mathbf {x} \| _ {\max}}{2 ^ {7}} = \frac {\| \mathbf {x} \| _ {\max}}{1 2 8},\tag{26}
$$

where $\| \mathbf { x } \| _ { \operatorname* { m a x } }$ is the maximum absolute value within the block and the factor $2 ^ { 7 }$ arises from the 7 mantissa bits of int8 (one bit reserved for the sign). The relative error per coordinate is therefore bounded by $1 / 1 2 8 \approx 0 . 7 8 \%$ of the block’s dynamic range. 

## E.2 Error propagation in AdamW

AdamW stores two buffers: the first moment $\mathbf { m } _ { t }$ and the second moment $\mathbf { v } _ { t }$ . Under quantization, the stored quantities are $\tilde { \mathbf { m } } _ { t } = \mathbf { m } _ { t } + \delta _ { t } ^ { m }$ and $\tilde { \mathbf { v } } _ { t } = \mathbf v _ { t } + \delta _ { t } ^ { v }$ . The AdamW update (ignoring weight decay) is 

$$
\pmb {\theta} _ {t} = \pmb {\theta} _ {t - 1} - \eta \cdot \frac {\tilde {\mathbf {m}} _ {t}}{\sqrt {\tilde {\mathbf {v}} _ {t}} + \epsilon},\tag{27}
$$

where $\epsilon = 1 0 ^ { - 8 }$ is a small constant for numerical stability. 

The critical vulnerability lies in the reciprocal square root operation. The first-order Taylor expansion of $f ( x ) = 1 / ( \sqrt { x } + \epsilon )$ around $v _ { t , i }$ is 

$$
f (v _ {t, i} + \delta_ {t, i} ^ {v}) \approx f (v _ {t, i}) + f ^ {\prime} (v _ {t, i}) \delta_ {t, i} ^ {v}, \qquad f ^ {\prime} (x) = - \frac {1}{2 \sqrt {x} (\sqrt {x} + \epsilon) ^ {2}}.\tag{28}
$$

For a coordinate i, substituting $\tilde { v } _ { t , i } = v _ { t , i } + \delta _ { t , i } ^ { v }$ yields 

$$
\frac {1}{\sqrt {\tilde {v} _ {t , i}} + \epsilon} \approx \frac {1}{\sqrt {v _ {t , i}} + \epsilon} - \frac {\delta_ {t , i} ^ {v}}{2 \sqrt {v _ {t , i}} (\sqrt {v _ {t , i}} + \epsilon) ^ {2}}.\tag{29}
$$

The role of ϵ is to prevent division by zero when $v _ { t , i }$ is very small. However, under int8 quantization, this protection becomes insufficient. Because $v _ { t , i }$ accumulates squared gradients, it is typically very small at initialization and for parameters receiving sparse or weak gradients. In full precision, $\epsilon =$ $1 0 ^ { - 8 }$ safely dominates the denominator when $v _ { t , i } \ll \bar { 1 } 0 ^ { - 8 }$ , yielding a stable update of approximately $\tilde { m } _ { t , i } / 1 0 ^ { - 8 }$ . Under int8 quantization, the stored $\tilde { v } _ { t , i }$ carries an error $\delta _ { t , i } ^ { v }$ whose magnitude scales with the block’s dynamic range. When the block contains a single large gradient $( \mathbf { e . g . } , \| \mathbf { v } _ { t } \| _ { \operatorname* { m a x } } \approx 1 )$ , the quantization error $\delta _ { t , i } ^ { v } \approx \bar { 1 } / 1 2 8 \approx 0 . 0 0 7 8$ can vastly exceed the true value for coordinates where $v _ { t , i } \approx 1 0 ^ { - 8 }$ . The error term in Equation (29) then becomes 

$$
\frac {\delta_ {t , i} ^ {v}}{2 \sqrt {v _ {t , i}} (\sqrt {v _ {t , i}} + \epsilon) ^ {2}} \approx \frac {0 . 0 0 7 8}{2 \cdot 1 0 ^ {- 4} \cdot 1 0 ^ {- 8}} \approx 4 \times 1 0 ^ {9},\tag{30}
$$

completely overwhelming the signal. Moreover, unlike the benign ϵ safeguard, this quantization error is not zero-mean and accumulates in the momentum buffer across iterations, leading to rapid divergence. 

## E.3 Error propagation in PowerStep

PowerStep stores only the momentum buffer $\mathbf { m } _ { t }$ in int8. The quantized buffer is $\tilde { \mathbf { m } } _ { t } = \mathbf { m } _ { t } + \boldsymbol { \delta } _ { t }$ The update is 

$$
\boldsymbol {\theta} _ {t} = \boldsymbol {\theta} _ {t - 1} - \eta \cdot \Phi_ {\beta} (\tilde {\mathbf {m}} _ {t}).\tag{31}
$$

The signed power transform $\Phi _ { \beta } ( \mathbf { x } ) = \mathrm { s i g n } ( \mathbf { x } ) \odot | \mathbf { x } | ^ { \beta }$ with $\beta = 0 .$ 1 provides four layers of protection. 

Layer 1: Hölder continuity of $\Phi _ { \beta } ,$ . Lemma 3 establishes that for any $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d }$ 

$$
\| \Phi_ {\beta} (\mathbf {x}) - \Phi_ {\beta} (\mathbf {y}) \| _ {1 + \beta} \leq C _ {\beta} \| \mathbf {x} - \mathbf {y} \| _ {1 + \beta} ^ {\beta},\tag{32}
$$

with $C _ { \beta } = 2 ^ { 1 - \beta } d ^ { \left( 1 - \beta \right) / \left( 1 + \beta \right) }$ . Setting $\mathbf { x } = \tilde { \mathbf { m } } _ { t }$ and $\textbf { y } = \mathbf { m } _ { t }$ and using the norm equivalence $\lVert \delta _ { t } \rVert _ { 1 + \beta } \leq d ^ { \frac { 1 } { 1 + \beta } } \lVert \delta _ { t } \rVert _ { \infty }$ , we obtain 

$$
\| \Phi_ {\beta} (\tilde {\mathbf {m}} _ {t}) - \Phi_ {\beta} (\mathbf {m} _ {t}) \| _ {1 + \beta} \leq C _ {\beta} d ^ {\frac {\beta}{1 + \beta}} \| \pmb {\delta} _ {t} \| _ {\infty} ^ {\beta}.\tag{33}
$$

Since $\beta = 0 . 1$ , the exponent $\beta$ on the quantization error $\| \delta _ { t } \| _ { \infty }$ significantly attenuates its impact. For a block with dynamic range $\| \mathbf { m } _ { t } \| _ { \operatorname* { m a x } } \approx 1$ , the int8 error is $\| \delta _ { t } \| _ { \infty } \approx 1 / \dot { 1 } 2 8 \approx 0 . 0 0 7 8$ . Raising this to the power 0.1 yields $0 . 0 0 7 8 ^ { 0 . 1 } \approx 0 . 6 1$ , making the perturbation comparable in magnitude to the signal. Critically, however, $\Phi _ { \beta }$ is a continuous, bounded nonlinearity: unlike the reciprocal square root, it has no singularity at zero. 

Layer 2: bounded amplification at small signal values. The derivative of $\Phi _ { \beta }$ at coordinate x is 

$$
\frac {d}{d x} \Phi_ {\beta} (x) = \beta | x | ^ {\beta - 1}.\tag{34}
$$

For AdamW, the analogous sensitivity is $\scriptstyle { \frac { 1 } { 2 } } v ^ { - 3 / 2 }$ , which diverges as $v  0$ . For PowerStep with $\beta = 0 . 1$ , the sensitivity is $0 . 1 \cdot | x | ^ { - 0 . 9 }$ , which also grows as $| x | \to 0$ , but the exponent −0.9 is less severe than −1.5 for the reciprocal square root. More importantly, this sensitivity is integrated over the momentum buffer, which is a linear accumulator of gradients and thus does not suffer from the extreme dynamic range compression that affects $\mathbf { v } _ { t }$ 

Layer 3: linear accumulation preserves dynamic range. The momentum buffer $\begin{array} { r l } { \mathbf { m } _ { t } } & { { } = } \end{array}$ $\scriptstyle \sum _ { k = 0 } ^ { t - 1 } \gamma ^ { k } \mathbf { g } _ { t - k }$ is a linear combination of gradient vectors. In contrast, Adam’s second moment $\mathbf { v } _ { t }$ accumulates squared gradients, compressing the dynamic range by squaring. For a typical Transformer, gradient magnitudes span several orders of magnitude. Squaring compresses a factor of $1 0 ^ { 4 }$ into $\mathrm { \bar { 1 0 ^ { 8 } } }$ in the second moment, pushing small values close to machine precision. The linear accumulation in $\mathbf { m } _ { t }$ preserves the original dynamic range, keeping values well above the quantization granularity. 

Layer 4: heavy-ball momentum alleviates update stalling. EMA momentum is defined as 

$$
\mathbf {m} _ {t} = \beta \mathbf {m} _ {t - 1} + (1 - \beta) \mathbf {g} _ {t},\tag{35}
$$

where $\beta \in ( 0 , 1 )$ is the decay coefficient. Topollai and Choromanska [2026] show that EMA-based moments under optimizer state quantization suffer from the state-update stalling problem: because the state $\mathbf { m } _ { t - 1 }$ is in low-precision and the incoming high-precision gradient $\mathbf { g } _ { t }$ is scaled by $1 - \beta _ { : }$ the effective per-step increment can be too small to change the stored value. PowerStep’s heavy-ball momentum [Polyak, 1964] alleviates this stalling: whereas EMA momentum weights the current gradient by only $\mathrm { \bar { 1 } } - \beta \approx 0 . 1$ , heavy-ball momentum adds the full gradient $\mathbf { g } _ { t }$ (weight 1), yielding an order-of-magnitude larger per-step increment. This makes it substantially less likely that the update falls below the quantization step size and stalls. 

## E.4 Summary

PowerStep’s robustness to int8 quantization arises from a confluence of four factors: (i) the absence of a reciprocal square root singularity, (ii) the Hölder continuity of $\Phi _ { \beta } .$ , which attenuates quantization noise with exponent $\beta = 0 . 1$ , (iii) the elimination of the second-moment buffer and use of linear accumulation, which preserves dynamic range, and (iv) heavy-ball momentum, whose full-weight gradient updates make first-moment stalling far less likely than under EMA. As a result, PowerStep’s momentum buffer remains responsive under quantization without requiring the stochastic rounding or periodic resets that are necessary for quantized Adam [Han et al., 2025; Topollai and Choromanska, 2026]. This analysis also motivates the conjecture that PowerStep would remain stable under even more aggressive quantization (e.g., int4), a direction for future investigation. 

## F Complete proofs

We provide complete proofs of all lemmas and Theorem 1 stated in Section 3. For self-containedness, we restate each assumption and result before its proof. 

Assumption 1 (L-Smoothness). f is continuously differentiable and there exists a constant $L > 0$ such that,for all x, $\mathbf { y } \in \mathbb { R } ^ { d } .$ 

$$
\left\| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \right\| _ {2} \leq L \| \mathbf {x} - \mathbf {y} \| _ {2}.\tag{36}
$$

Assumption 2 (Bounded Below). There exists $f ^ { * } \in \mathbb { R }$ such that $f ( \pmb { \theta } ) \geq f ^ { * }$ for all $\pmb \theta \in \mathbb R ^ { d } .$ 

Assumption 3 (Bounded Gradient). There exists a constant $G > 0$ such that $\| \nabla f ( \pmb \theta ) \| _ { 2 } \le G$ for all $\pmb \theta \in \mathbb { R } ^ { d }$ 

Assumption 4 (Unbiased Gradient). At each iteration t, the stochastic gradient g<sub>t</sub> satisfies 

$$
\mathbb {E} \left[ \mathbf {g} _ {t} \mid \boldsymbol {\theta} _ {t - 1} \right] = \nabla f (\boldsymbol {\theta} _ {t - 1}).\tag{37}
$$

Assumption 5 (Bounded Variance). There exists a constant $\sigma > 0$ such that,for all $t \geq 1$ 

$$
\mathbb {E} \left[ \| \mathbf {g} _ {t} - \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {2} ^ {2} | \boldsymbol {\theta} _ {t - 1} \right] \leq \sigma^ {2}.\tag{38}
$$

Lemma 1 (Induced Norm Structure). For any vector m $\in \mathbb { R } ^ { d }$ and $\beta \in ( 0 , 1 ]$ 

$$
\langle \mathbf {m}, \Phi_ {\beta} (\mathbf {m}) \rangle = \| \mathbf {m} \| _ {1 + \beta} ^ {1 + \beta}.\tag{39}
$$

Proof. By definition of the signed power transform $\Phi _ { \beta } ( \mathbf { x } ) = \mathrm { s i g n } ( \mathbf { x } ) \odot | \mathbf { x } | ^ { \beta }$ , we compute the inner product coordinate-wise, 

$$
\langle \mathbf {m}, \Phi_ {\beta} (\mathbf {m}) \rangle = \sum_ {i = 1} ^ {d} m _ {i} \cdot \mathrm{sign} (m _ {i}) | m _ {i} | ^ {\beta}\tag{40}
$$

$$
= \sum_ {i = 1} ^ {d} \left(\operatorname{sign} (m _ {i}) | m _ {i} |\right) \cdot \operatorname{sign} (m _ {i}) | m _ {i} | ^ {\beta}\tag{41}
$$

$$
= \sum_ {i = 1} ^ {d} (\mathrm{sign} (m _ {i})) ^ {2} | m _ {i} | ^ {1 + \beta}\tag{42}
$$

$$
= \sum_ {i = 1} ^ {d} | m _ {i} | ^ {1 + \beta} = \| \mathbf {m} \| _ {1 + \beta} ^ {1 + \beta},\tag{43}
$$

where we used $m _ { i } = \mathrm { s i g n } ( m _ { i } ) | m _ { i } |$ and $( \mathrm { s i g n } ( m _ { i } ) ) ^ { 2 } = 1$ for $m _ { i } \neq 0$ . When $m _ { i } = 0$ , the term contributes zero to the sum regardless. □ 

Lemma 2 (Norm Relationship). For any m $\in \mathbb { R } ^ { d }$ and $\beta \in ( 0 , 1 ]$ 

$$
\| \Phi_ {\beta} (\mathbf {m}) \| _ {2} ^ {2} = \| \mathbf {m} \| _ {2 \beta} ^ {2 \beta} \leq d ^ {1 - \beta} \| \mathbf {m} \| _ {2} ^ {2 \beta}.\tag{44}
$$

Proof. The equality follows directly from the definition, 

$$
\| \Phi_ {\beta} (\mathbf {m}) \| _ {2} ^ {2} = \sum_ {i = 1} ^ {d} \bigl | \mathrm{sign} (m _ {i}) | m _ {i} | ^ {\beta} \bigr | ^ {2} = \sum_ {i = 1} ^ {d} | m _ {i} | ^ {2 \beta} = \| \mathbf {m} \| _ {2 \beta} ^ {2 \beta}.\tag{45}
$$

For the inequality, we consider two cases. If $\mathbf { \partial } \cdot \boldsymbol { \beta } = 1$ , then $| \Phi _ { 1 } ( { \bf m } ) | 2 ^ { 2 } = | { \bf m } | 2 ^ { 2 }$ and the bound holds with equality. If $\beta \in ( 0 , 1 )$ , we apply Hölder’s inequality, which states that for vectors u, $\mathbf { v } \in \mathbb { R } ^ { d }$ and conjugate exponents $p , q \geq 1$ with $1 / p + 1 / q = 1$ 

$$
\sum_ {i = 1} ^ {d} \left| u _ {i} v _ {i} \right| \leq \left(\sum_ {i = 1} ^ {d} \left| u _ {i} \right| ^ {p}\right) ^ {1 / p} \left(\sum_ {i = 1} ^ {d} \left| v _ {i} \right| ^ {q}\right) ^ {1 / q}.\tag{46}
$$

Setting $u _ { i } = | m _ { i } | ^ { 2 \beta } , v _ { i } = 1$ , with exponents $p = 1 / \beta$ and $q = 1 / ( 1 - \beta )$ (which satisfy $1 / p + 1 / q =$ $\beta + ( \bar { 1 } - \beta ) \overset { \cdot } { = } 1 \overset { \cdot } { ) }$ , 

$$
\| \mathbf {m} \| _ {2 \beta} ^ {2 \beta} = \sum_ {i = 1} ^ {d} | m _ {i} | ^ {2 \beta} \cdot 1 \leq \left(\sum_ {i = 1} ^ {d} \bigl (| m _ {i} | ^ {2 \beta} \bigr) ^ {1 / \beta}\right) ^ {\beta} \left(\sum_ {i = 1} ^ {d} 1 ^ {1 / (1 - \beta)}\right) ^ {1 - \beta}\tag{47}
$$

$$
= \left(\sum_ {i = 1} ^ {d} | m _ {i} | ^ {2}\right) ^ {\beta} \cdot d ^ {1 - \beta} = d ^ {1 - \beta} \| \mathbf {m} \| _ {2} ^ {2 \beta}.\tag{48}
$$

Lemma 3 (Hölder Continuity of $\Phi _ { \beta } )$ . For any $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d }$ and $\beta \in ( 0 , 1 ]$ 

$$
\left\| \Phi_ {\beta} (\mathbf {x}) - \Phi_ {\beta} (\mathbf {y}) \right\| _ {1 + \beta} \leq C _ {\beta} \left\| \mathbf {x} - \mathbf {y} \right\| _ {1 + \beta} ^ {\beta},\tag{49}
$$

where $C _ { \beta } = 2 ^ { 1 - \beta } d ^ { ( 1 - \beta ) / ( 1 + \beta ) } \leq 2 d .$ 

Proof. We first establish a coordinate-wise bound. For any $a , b \in \mathbb { R } .$ 

$$
| \operatorname{sign} (a) | a | ^ {\beta} - \operatorname{sign} (b) | b | ^ {\beta} | \leq 2 ^ {1 - \beta} | a - b | ^ {\beta}.\tag{50}
$$

This follows by a sign case analysis. If a and b share the same sign, the left-hand side reduces to $\left| | a | ^ { \beta } - | b | ^ { \beta } \right| \leq \left| | a | - | b | \right| ^ { \beta } \leq | a - b | ^ { \beta }$ , using $| u ^ { \beta } - v ^ { \beta } | \leq | u - v | ^ { \beta }$ for $u , v \geq 0$ (by subadditivity of $f ( x ) = x ^ { \beta }$ , which is concave with $f ( 0 ) = 0 )$ . If a and b have opposite signs, then $| \mathrm { s i g n } ( a ) | a | ^ { \beta } -$ $\operatorname { s i g n } ( b ) | b | ^ { \beta } | = | a | ^ { \beta } + | b | ^ { \beta } \leq 2 ^ { 1 - \beta } \widetilde ( | a | ^ { \prime } + | b | ) ^ { \beta } = 2 ^ { 1 - \beta } | a - b | ^ { \beta }$ , where the inequality $x ^ { \beta } + \dot { y ^ { \beta } } \leq$ $2 ^ { 1 - \beta } ( x + y ) ^ { \beta }$ for $x , y \geq 0$ follows from Jensen’s inequality and concavity of $f ( x ) = x ^ { \beta }$ 

Summing over coordinates and raising to the power $1 + \beta _ { \ast }$ 

$$
\| \Phi_ {\beta} (\mathbf {x}) - \Phi_ {\beta} (\mathbf {y}) \| _ {1 + \beta} ^ {1 + \beta} = \sum_ {i = 1} ^ {d} | \mathrm{sign} (x _ {i}) | x _ {i} | ^ {\beta} - \mathrm{sign} (y _ {i}) | y _ {i} | ^ {\beta} | ^ {1 + \beta}\tag{51}
$$

$$
\leq 2 ^ {(1 - \beta) (1 + \beta)} \sum_ {i = 1} ^ {d} | x _ {i} - y _ {i} | ^ {\beta (1 + \beta)}.\tag{52}
$$

Apply Hölder’s inequality with $p = 1 / \beta$ and $q = 1 / ( 1 - \beta )$ , 

$$
\sum_ {i = 1} ^ {d} \left| x _ {i} - y _ {i} \right| ^ {\beta (1 + \beta)} \cdot 1 \leq \left(\sum_ {i = 1} ^ {d} \left(\left| x _ {i} - y _ {i} \right| ^ {\beta (1 + \beta)}\right) ^ {1 / \beta}\right) ^ {\beta} \left(\sum_ {i = 1} ^ {d} 1 ^ {1 / (1 - \beta)}\right) ^ {1 - \beta}\tag{53}
$$

$$
= \| \mathbf {x} - \mathbf {y} \| _ {1 + \beta} ^ {\beta (1 + \beta)} \cdot d ^ {1 - \beta}.\tag{54}
$$

Therefore, 

$$
\| \Phi_ {\beta} (\mathbf {x}) - \Phi_ {\beta} (\mathbf {y}) \| _ {1 + \beta} ^ {1 + \beta} \leq 2 ^ {1 - \beta^ {2}} d ^ {1 - \beta} \| \mathbf {x} - \mathbf {y} \| _ {1 + \beta} ^ {\beta (1 + \beta)}.\tag{55}
$$

Taking the (1 + β)-th root and noting $2 ^ { ( 1 - \beta ^ { 2 } ) / ( 1 + \beta ) } = 2 ^ { 1 - \beta }$ 

$$
\| \Phi_ {\beta} (\mathbf {x}) - \Phi_ {\beta} (\mathbf {y}) \| _ {1 + \beta} \leq 2 ^ {1 - \beta} d ^ {(1 - \beta) / (1 + \beta)} \| \mathbf {x} - \mathbf {y} \| _ {1 + \beta} ^ {\beta} = C _ {\beta} \| \mathbf {x} - \mathbf {y} \| _ {1 + \beta} ^ {\beta},\tag{56}
$$

with $C _ { \beta } \leq 2 d$ since $2 ^ { 1 - \beta } \leq 2$ and $d ^ { ( 1 - \beta ) / ( 1 + \beta ) } \leq d .$ 

□ 

Lemma 4 (Momentum Bound). Under Assumptions 3–5, for PowerStep with $\gamma \in [ 0 , 1 )$ and any $t \geq 1$ 

$$
\mathbb {E} \left[ \| \mathbf {m} _ {t} \| _ {2} ^ {2} \right] \leq \frac {2 (G ^ {2} + \sigma^ {2})}{(1 - \gamma) ^ {2}}.
$$

(57) 

Consequently, for any $\beta \in ( 0 , 1 ]$ 

$$
\mathbb {E} \left[ \| \Phi_ {\beta} (\mathbf {m} _ {t}) \| _ {2} ^ {2} \right] \leq d ^ {1 - \beta} 2 ^ {\beta} \left(\frac {G ^ {2} + \sigma^ {2}}{(1 - \gamma) ^ {2}}\right) ^ {\beta} =: M _ {\beta}.\tag{58}
$$

Proof. Step 1: Unrolling the momentum. Expanding $\mathbf { m } _ { t } = \gamma \mathbf { m } _ { t - 1 } + \mathbf { g } _ { t }$ with $\mathbf { m } _ { 0 } = \mathbf { 0 }$ gives 

$$
\mathbf {m} _ {t} = \sum_ {k = 0} ^ {t - 1} \gamma^ {k} \mathbf {g} _ {t - k}.\tag{59}
$$

Decompose $\mathbf { g } _ { s } = \nabla f ( \pmb { \theta } _ { s - 1 } ) + \pmb { \xi } _ { s }$ , where $\pmb { \xi } _ { s } = \mathbf { g } _ { s } - \nabla f ( \pmb { \theta } _ { s - 1 } )$ is zero-mean with $\mathbb { E } [ \| \pmb { \xi } _ { s } \| _ { 2 } ^ { 2 } | \pmb { \theta } _ { s - 1 } ] \le$ $\sigma ^ { 2 }$ . Then 

$$
\mathbf {m} _ {t} = \underbrace {\sum_ {k = 0} ^ {t - 1} \gamma^ {k} \nabla f (\boldsymbol {\theta} _ {t - k - 1})} _ {\mathbf {a} _ {t}} + \underbrace {\sum_ {k = 0} ^ {t - 1} \gamma^ {k} \boldsymbol {\xi} _ {t - k}} _ {\mathbf {b} _ {t}}.\tag{60}
$$

Step 2: Bounding the signal term. By the triangle inequality and $\| \nabla f ( \cdot ) \| _ { 2 } \leq G$ 

$$
\| \mathbf {a} _ {t} \| _ {2} \leq \sum_ {k = 0} ^ {t - 1} \gamma^ {k} G \leq \frac {G}{1 - \gamma}.\tag{61}
$$

Step 3: Bounding the noise term. Since {ξ<sub>s</sub>} is a martingale difference sequence, cross terms vanish: $\mathbb { E } [ \langle \pmb { \xi } _ { t - i } , \pmb { \xi } _ { t - j } \rangle ] = 0$ for $i \neq j$ . Hence, 

$$
\mathbb {E} \big [ \| \mathbf {b} _ {t} \| _ {2} ^ {2} \big ] = \sum_ {k = 0} ^ {t - 1} \gamma^ {2 k} \mathbb {E} \big [ \| \boldsymbol {\xi} _ {t - k} \| _ {2} ^ {2} \big ] \leq \sigma^ {2} \sum_ {k = 0} ^ {t - 1} \gamma^ {2 k} \leq \frac {\sigma^ {2}}{1 - \gamma^ {2}} \leq \frac {\sigma^ {2}}{1 - \gamma}.\tag{62}
$$

Step 4: Combining. Using $\| \mathbf { a } _ { t } + \mathbf { b } _ { t } \| _ { 2 } ^ { 2 } \leq 2 \| \mathbf { a } _ { t } \| _ { 2 } ^ { 2 } + 2 \| \mathbf { b } _ { t } \| _ { 2 } ^ { 2 }$ and taking expectations, 

$$
\mathbb {E} \big [ \| \mathbf {m} _ {t} \| _ {2} ^ {2} \big ] \leq 2 \mathbb {E} \big [ \| \mathbf {a} _ {t} \| _ {2} ^ {2} \big ] + 2 \mathbb {E} \big [ \| \mathbf {b} _ {t} \| _ {2} ^ {2} \big ]\tag{63}
$$

$$
\leq 2 \left(\frac {G}{1 - \gamma}\right) ^ {2} + 2 \frac {\sigma^ {2}}{1 - \gamma} \leq \frac {2 (G ^ {2} + \sigma^ {2})}{(1 - \gamma) ^ {2}},\tag{64}
$$

where the last step uses $1 / ( 1 - \gamma ) \leq 1 / ( 1 - \gamma ) ^ { 2 }$ for $\gamma \in [ 0 , 1 )$ . 

Step 5: Bounding the transformed update. From Lemma $2 , \| \Phi _ { \beta } ( \mathbf { m } _ { t } ) \| _ { 2 } ^ { 2 } \leq d ^ { 1 - \beta } \| \mathbf { m } _ { t } \| _ { 2 } ^ { 2 \beta }$ . Taking expectations and applying Jensen’s inequality (since $f ( x ) = x ^ { \beta }$ is concave for $\beta \in ( 0 , 1 ] )$ , 

$$
\mathbb {E} \big [ \| \Phi_ {\beta} (\mathbf {m} _ {t}) \| _ {2} ^ {2} \big ] \leq d ^ {1 - \beta} \mathbb {E} \big [ \| \mathbf {m} _ {t} \| _ {2} ^ {2 \beta} \big ] \leq d ^ {1 - \beta} \big (\mathbb {E} \big [ \| \mathbf {m} _ {t} \| _ {2} ^ {2} \big ] \big) ^ {\beta}
$$

(65) 

$$
\leq d ^ {1 - \beta} \left(\frac {2 (G ^ {2} + \sigma^ {2})}{(1 - \gamma) ^ {2}}\right) ^ {\beta} =: M _ {\beta}.\tag{66}
$$

Lemma 5 (Descent Inequality). Under Assumption 1, the iterates of PowerStep with learning rate η<sub>t</sub> satisfy 

$$
\mathbb {E} [ f (\pmb {\theta} _ {t}) ] \leq \mathbb {E} [ f (\pmb {\theta} _ {t - 1}) ] - \eta_ {t} \mathbb {E} \big [ \langle \nabla f (\pmb {\theta} _ {t - 1}), \Phi_ {\beta} (\mathbf {m} _ {t}) \rangle \big ] + \frac {L \eta_ {t} ^ {2}}{2} \mathbb {E} \big [ \| \Phi_ {\beta} (\mathbf {m} _ {t}) \| _ {2} ^ {2} \big ].\tag{67}
$$

Proof. By L-smoothness (Assumption 1), for all $\pmb { \theta } , \pmb { \phi } \in \mathbb { R } ^ { d }$ 

$$
f (\phi) \leq f (\boldsymbol {\theta}) + \langle \nabla f (\boldsymbol {\theta}), \phi - \boldsymbol {\theta} \rangle + \frac {L}{2} \| \phi - \boldsymbol {\theta} \| _ {2} ^ {2}.\tag{68}
$$

Substituting $\pmb \theta = \pmb \theta _ { t - 1 } , \pmb \phi = \pmb \theta _ { t }$ , and using the update $\pmb { \theta } _ { t } = \pmb { \theta } _ { t - 1 } - \eta _ { t } \Phi _ { \beta } ( \mathbf { m } _ { t } )$ 

$$
f (\pmb {\theta} _ {t}) \leq f (\pmb {\theta} _ {t - 1}) - \eta_ {t} \langle \nabla f (\pmb {\theta} _ {t - 1}), \Phi_ {\beta} (\mathbf {m} _ {t}) \rangle + \frac {L \eta_ {t} ^ {2}}{2} \| \Phi_ {\beta} (\mathbf {m} _ {t}) \| _ {2} ^ {2}.\tag{69}
$$

Taking total expectation yields the claim. 

Lemma 6 (Gradient Alignment). Under Assumptions 1–3, for PowerStep with learning rate $\eta _ { t }$ and $\gamma \in [ 0 , 1 )$ , there exists a constant $C _ { 0 } > 0$ depending on $L , \gamma , G , \sigma , d ,$ , and $\beta$ such thatfor all $t \geq 1$ 

$$
\mathbb {E} \big [ \langle \nabla f (\boldsymbol {\theta} _ {t - 1}), \Phi_ {\beta} (\mathbf {m} _ {t}) \rangle \big ] \geq \mathbb {E} \big [ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {1 + \beta} ^ {1 + \beta} \big ] - C _ {0} (1 + \eta_ {t} ^ {\beta}).\tag{70}
$$

Proof. Let $\bar { \bf g } _ { t } = \nabla f ( \pmb \theta _ { t - 1 } )$ and $\delta _ { t } = \mathbf { m } _ { t } - \bar { \mathbf { g } } _ { t }$ . Unrolling the momentum update, 

$$
\boldsymbol {\delta} _ {t} = \underbrace {\sum_ {k = 0} ^ {t - 1} \gamma^ {k} \boldsymbol {\xi} _ {t - k}} _ {\mathbf {b} _ {t}} + \underbrace {\sum_ {k = 0} ^ {t - 1} \gamma^ {k + 1} (\bar {\mathbf {g}} _ {t - 1 - k} - \bar {\mathbf {g}} _ {t - k})} _ {\mathbf {d} _ {t}},\tag{71}
$$

where $\pmb { \xi } _ { s } = \mathbf { g } _ { s } - \bar { \mathbf { g } } _ { s }$ 

By L-smoothness and Lemma 4, 

$$
\mathbb {E} \left[ \| \mathbf {d} _ {t} \| _ {1 + \beta} \right] \leq \mathbb {E} \left[ \| \mathbf {d} _ {t} \| _ {2} \right] \leq \frac {\gamma L \sqrt {M _ {\beta}}}{1 - \gamma} \eta_ {t}.\tag{72}
$$

By Lemma 1 and the dual norm inequality, 

$$
\langle \bar {\mathbf {g}} _ {t}, \Phi_ {\beta} (\mathbf {m} _ {t}) \rangle \geq \| \bar {\mathbf {g}} _ {t} \| _ {1 + \beta} ^ {1 + \beta} - \| \bar {\mathbf {g}} _ {t} \| _ {(1 + \beta) / \beta} \cdot \| \Phi_ {\beta} (\bar {\mathbf {g}} _ {t} + \pmb {\delta} _ {t}) - \Phi_ {\beta} (\bar {\mathbf {g}} _ {t}) \| _ {1 + \beta}\tag{73}
$$

$$
\geq \| \bar {\mathbf {g}} _ {t} \| _ {1 + \beta} ^ {1 + \beta} - G \cdot C _ {\beta} \| \pmb {\delta} _ {t} \| _ {1 + \beta} ^ {\beta},\tag{74}
$$

since $\| \bar { \mathbf { g } } _ { t } \| _ { ( 1 + \beta ) / \beta } \leq \| \bar { \mathbf { g } } _ { t } \| _ { 2 } \leq G$ and by Lemma 3. 

Using $\delta _ { t } = \mathbf { b } _ { t } + \mathbf { d } _ { t }$ and the subadditivity of $f ( x ) = x ^ { \beta }$ 

$$
\| \pmb {\delta} _ {t} \| _ {1 + \beta} ^ {\beta} \leq \| \mathbf {b} _ {t} \| _ {1 + \beta} ^ {\beta} + \| \mathbf {d} _ {t} \| _ {1 + \beta} ^ {\beta}.\tag{75}
$$

Taking expectations and applying Jensen’s inequality, 

$$
\mathbb {E} \left[ \| \boldsymbol {\delta} _ {t} \| _ {1 + \beta} ^ {\beta} \right] \leq \left(\mathbb {E} \left[ \| \mathbf {b} _ {t} \| _ {1 + \beta} \right]\right) ^ {\beta} + \left(\mathbb {E} \left[ \| \mathbf {d} _ {t} \| _ {1 + \beta} \right]\right) ^ {\beta}.\tag{76}
$$

For the noise term, by Lemma 4, 

$$
\mathbb {E} \big [ \| \mathbf {b} _ {t} \| _ {1 + \beta} \big ] \leq \sqrt {\mathbb {E} \big [ \| \mathbf {b} _ {t} \| _ {2} ^ {2} \big ]} \leq \frac {\sigma}{\sqrt {1 - \gamma}}.\tag{77}
$$

For the drift term, 

$$
\mathbb {E} \left[ \| \mathbf {d} _ {t} \| _ {1 + \beta} \right] \leq \frac {\gamma L \sqrt {M _ {\beta}}}{1 - \gamma} \eta_ {t}.\tag{78}
$$

Substituting, 

$$
\mathbb {E} \left[ \| \boldsymbol {\delta} _ {t} \| _ {1 + \beta} ^ {\beta} \right] \leq \left(\frac {\sigma}{\sqrt {1 - \gamma}}\right) ^ {\beta} + \left(\frac {\gamma L \sqrt {M _ {\beta}}}{1 - \gamma}\right) ^ {\beta} \eta_ {t} ^ {\beta}.\tag{79}
$$

Setting $C _ { 0 } = G C _ { \beta } \operatorname* { m a x } ( ( \sigma / \sqrt { 1 - \gamma } ) ^ { \beta } , ( \gamma L \sqrt { M _ { \beta } } / ( 1 - \gamma ) ) ^ { \beta } )$ completes the proof. 

Theorem 1 (Convergence Rate). Under Assumptions $I { - } 5 ,$ let $\{ \pmb { \theta } _ { t } \} _ { t = 1 } ^ { T }$ be generated by PowerStep with learning rate $\eta _ { t } = \eta / \sqrt { t } f o r$ some $\eta > 0$ and momentum coefficient $\gamma \in [ 0 , 1 )$ . Then for any $\beta \in ( 0 , 1 ]$ 

$$
\min _ {t \in [ T ]} \mathbb {E} \left[ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {2} ^ {2} \right] = O \left(\frac {1}{\sqrt {T}}\right).\tag{80}
$$

Proof. From Lemma 5 with learning rate $\eta _ { t }$ , we have 

$$
\mathbb {E} [ f (\boldsymbol {\theta} _ {t}) ] \leq \mathbb {E} [ f (\boldsymbol {\theta} _ {t - 1}) ] - \eta_ {t} \mathbb {E} \big [ \langle \nabla f (\boldsymbol {\theta} _ {t - 1}), \Phi_ {\beta} (\mathbf {m} _ {t}) \rangle \big ] + \frac {L \eta_ {t} ^ {2}}{2} \mathbb {E} \big [ \| \Phi_ {\beta} (\mathbf {m} _ {t}) \| _ {2} ^ {2} \big ].\tag{81}
$$

Summing over $t = 1 , \dots , T$ and telescoping the left-hand side, 

$$
\mathbb {E} [ f (\boldsymbol {\theta} _ {T}) ] - f (\boldsymbol {\theta} _ {0}) \leq - \sum_ {t = 1} ^ {T} \eta_ {t} \mathbb {E} \big [ \langle \nabla f (\boldsymbol {\theta} _ {t - 1}), \Phi_ {\beta} (\mathbf {m} _ {t}) \rangle \big ] + \frac {L}{2} \sum_ {t = 1} ^ {T} \eta_ {t} ^ {2} \mathbb {E} \big [ \| \Phi_ {\beta} (\mathbf {m} _ {t}) \| _ {2} ^ {2} \big ].\tag{82}
$$

By Assumption 2 (bounded below), $\mathbb { E } [ f ( \pmb { \theta } _ { T } ) ] \ge f ^ { * }$ , so $\begin{array} { r } { \mathbb { E } [ f ( \pmb { \theta } _ { T } ) ] - f ( \pmb { \theta } _ { 0 } ) \ge - \Delta _ { 0 } } \end{array}$ where $\Delta _ { 0 } =$ $f ( \pmb { \theta } _ { 0 } ) - f ^ { * }$ . Rearranging, 

$$
\sum_ {t = 1} ^ {T} \eta_ {t} \mathbb {E} \left[ \langle \nabla f (\boldsymbol {\theta} _ {t - 1}), \Phi_ {\beta} (\mathbf {m} _ {t}) \rangle \right] \leq \Delta_ {0} + \frac {L}{2} \sum_ {t = 1} ^ {T} \eta_ {t} ^ {2} \mathbb {E} \left[ \| \Phi_ {\beta} (\mathbf {m} _ {t}) \| _ {2} ^ {2} \right].\tag{83}
$$

By Lemma 4, $\mathbb { E } [ \| \Phi _ { \beta } ( \mathbf { m } _ { t } ) \| _ { 2 } ^ { 2 } ] \le M _ { \beta }$ for all t. With $\eta _ { t } = \eta / \sqrt { t } .$ 

$$
\sum_ {t = 1} ^ {T} \eta_ {t} ^ {2} = \eta^ {2} \sum_ {t = 1} ^ {T} \frac {1}{t} \leq \eta^ {2} (1 + \log T).\tag{84}
$$

Thus, 

$$
\frac {L}{2} \sum_ {t = 1} ^ {T} \eta_ {t} ^ {2} \mathbb {E} \big [ \| \Phi_ {\beta} (\mathbf {m} _ {t}) \| _ {2} ^ {2} \big ] \leq \frac {L M _ {\beta} \eta^ {2}}{2} (1 + \log T).\tag{85}
$$

From Lemma $^ { 6 , }$ we have 

$$
\mathbb {E} \big [ \langle \nabla f (\boldsymbol {\theta} _ {t - 1}), \Phi_ {\beta} (\mathbf {m} _ {t}) \rangle \big ] \geq \mathbb {E} \big [ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {1 + \beta} ^ {1 + \beta} \big ] - C _ {0} (1 + \eta_ {t} ^ {\beta}).\tag{86}
$$

Substituting into the telescoped inequality, 

$$
\sum_ {t = 1} ^ {T} \eta_ {t} \mathbb {E} \left[ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {1 + \beta} ^ {1 + \beta} \right] \leq \Delta_ {0} + C _ {0} \sum_ {t = 1} ^ {T} \eta_ {t} + C _ {0} \sum_ {t = 1} ^ {T} \eta_ {t} ^ {1 + \beta} + \frac {L M _ {\beta} \eta^ {2}}{2} (1 + \log T).\tag{87}
$$

Now bound the sums involving $\eta _ { t } = \eta / \sqrt { t } .$ 

$$
\sum_ {t = 1} ^ {T} \eta_ {t} = \eta \sum_ {t = 1} ^ {T} \frac {1}{\sqrt {t}} \leq 2 \eta \sqrt {T},\tag{88}
$$

$$
\sum_ {t = 1} ^ {T} \eta_ {t} ^ {1 + \beta} = \eta^ {1 + \beta} \sum_ {t = 1} ^ {T} \frac {1}{t ^ {(1 + \beta) / 2}} \leq \eta^ {1 + \beta} \cdot \frac {2}{1 - \beta} T ^ {(1 - \beta) / 2} \quad (\text { for } \beta <   1).\tag{89}
$$

For $\beta = 1$ , the sum is $\textstyle \eta ^ { 2 } \sum _ { t = 1 } ^ { T } 1 / t \leq \eta ^ { 2 } ( 1 + \log T )$ 

Therefore, 

$$
\sum_ {t = 1} ^ {T} \eta_ {t} \mathbb {E} \left[ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {1 + \beta} ^ {1 + \beta} \right] \leq \Delta_ {0} + 2 C _ {0} \eta \sqrt {T} + \frac {2 C _ {0} \eta^ {1 + \beta}}{1 - \beta} T ^ {(1 - \beta) / 2} + \frac {L M _ {\beta} \eta^ {2}}{2} (1 + \log T) = O (\sqrt {T}).\tag{90}
$$

Since ${ \textstyle \sum _ { t = 1 } ^ { T } } \eta _ { t } = \Theta ( { \sqrt { T } } )$ , the weighted average satisfies 

$$
\frac {\sum_ {t = 1} ^ {T} \eta_ {t} \mathbb {E} \left[ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {1 + \beta} ^ {1 + \beta} \right]}{\sum_ {t = 1} ^ {T} \eta_ {t}} = O \left(\frac {1}{\sqrt {T}}\right).\tag{91}
$$

The minimum over iterates is bounded by this weighted average, yielding 

$$
\min _ {t \in [ T ]} \mathbb {E} \left[ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {1 + \beta} ^ {1 + \beta} \right] = O \left(\frac {1}{\sqrt {T}}\right).\tag{92}
$$

To convert to the $\ell _ { 2 }$ norm, we apply norm equivalence. For any $\mathbf { x } \in \mathbb { R } ^ { d }$ and $0 < p < q , \| \mathbf { x } \| _ { p } \geq$ $d ^ { 1 / p - 1 / q } \lVert \mathbf { x } \rVert _ { q }$ . Setting $p = 1 + \beta$ and $q = 2$ (valid since $1 + \beta \leq 2$ for $\beta \leq 1 )$ , 

$$
\| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {1 + \beta} \geq d ^ {\frac {1}{1 + \beta} - \frac {1}{2}} \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {2} = d ^ {\frac {1 - \beta}{2 (1 + \beta)}} \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {2}.\tag{93}
$$

Raising both sides to power $1 + \beta$ and taking expectations, 

$$
\mathbb {E} \big [ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {1 + \beta} ^ {1 + \beta} \big ] \geq d ^ {\frac {1 - \beta}{2}} \mathbb {E} \big [ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {2} ^ {1 + \beta} \big ].\tag{94}
$$

By Jensen’s inequality, $\begin{array} { r } { \mathbb { E } [ \| \nabla f ( \theta _ { t - 1 } ) \| _ { 2 } ^ { 1 + \beta } ] \ge \left( \mathbb { E } [ \| \nabla f ( \theta _ { t - 1 } ) \| _ { 2 } ^ { 2 } ] \right) ^ { \frac { 1 + \beta } { 2 } } } \end{array}$ . Substituting, 

$$
\min _ {t \in [ T ]} \left(\mathbb {E} \big [ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {2} ^ {2} \big ]\right) ^ {\frac {1 + \beta}{2}} = O \left(\frac {1}{\sqrt {T}}\right).\tag{95}
$$

Raising both sides to power $2 / ( 1 + \beta )$ absorbs the exponent into the hidden constant, yielding 

$$
\min _ {t \in [ T ]} \mathbb {E} \left[ \| \nabla f (\boldsymbol {\theta} _ {t - 1}) \| _ {2} ^ {2} \right] = O \left(\frac {1}{\sqrt {T}}\right),\tag{96}
$$

where the hidden constant includes a factor of $d ^ { ( 1 - \beta ) / ( 1 + \beta ) }$ from the norm equivalence step. This completes the proof. □ 