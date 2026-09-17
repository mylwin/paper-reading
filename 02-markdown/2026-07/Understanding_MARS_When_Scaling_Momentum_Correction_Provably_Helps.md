# Understanding MARS: When Scaling Momentum Correction Provably Helps

Egor Shulgin <sup>1</sup> Tamaz Gadaev <sup>2</sup> Sarit Khirirat <sup>1</sup> Peter Richtarik´ <sup>1</sup> 

## Abstract

MARS (Yuan et al., 2025) has recently emerged as a strong optimizer for large language model (LLM) training by scaling the correction term in momentum-based variance reduction (MVR). However, existing theory does not explain why this modification can improve convergence over the unscaled MVR choice $\gamma = 1$ . In this paper, we provide a theoretical explanation for this phenomenon. We introduce γ-similarity, a refined similarity condition that captures how the scaling coefficient interacts with the stochastic gradientdifference structure. This condition recovers standard similarity at $\gamma = 1$ and smoothness at $\gamma = 0 .$ Using γ-similarity, we derive convergence guarantees for fixed-γ MARS whose complexity depends explicitly on γ and the corresponding γ-similarity constant. The bound reveals why small values of γ can be beneficial: they may reduce the similarity term enough to outweigh the penalty from deviating from MVR. We prove that optimizing γ gives MARS a lower complexity guarantee than MVR. Experiments with MARS-AdamW on GPT-style LLM pretraining corroborate the theory, showing that properly chosen small values of $\gamma$ improve token efficiency over $\gamma = 1$ and AdamW. 

## 1. Introduction

The success of deep learning has driven significant attention toward nonconvex stochastic optimization problems of the form: 

$$
\min _ {x \in \mathbb {R} ^ {d}} f (x) := \mathbb {E} _ {\xi} [ f _ {\xi} (x) ],\tag{1}
$$

where $f _ { \xi } ( x )$ is a possibly nonconvex function, $x \in \mathbb { R } ^ { d }$ represents high-dimensional parameters, and $\xi$ is a random variable from an unknown data distribution D. This formulation is central to the training of deep neural network models. State-of-the-art models, such as GPT-5 (Bubeck et al., 2025), LLaMa-3 (Grattafiori et al., 2024), and DeepSeek-R1 (Guo et al., 2025), comprise billions of parameters, and are trained on massive datasets. For solving such huge-scale tasks, standard stochastic optimizers are Stochastic Gradient Descent (SGD), including its adaptive variants, such as Adam (Kingma, 2015) and AdamW (Loshchilov & Hutter, 2019). These algorithms are widely used, because they construct inexpensive gradient estimates using only a few data points at each iteration, thus making them both memory- and computationally efficient. Furthermore, to minimize smooth functions, SGD achieves an $\mathcal { O } ( 1 / T ^ { 1 / 4 } )$ convergence in the gradient norm (Ghadimi & Lan, 2013). 

## 1.1. Variance Reduction

To improve the convergence of SGD, various techniques have been proposed. One variance-reduction technique form an estimator combining stochastic gradients with periodically computed full gradients. Popular stochastic algorithms using this technique are Stochastic Variance Reduced Gradient (SVRG) algorithms (Johnson & Zhang, 2013; Konecnˇ y´ & Richtarik´ , 2013; Reddi et al., 2016). Despite their theoretical advantages over SGD, SVRG algorithms have achieved limited empirical success in solving huge-scale, nonconvex neural network training tasks. However, in these tasks, computing full gradients is infeasible, and the algorithms using this gradient estimator fail to effectively reduce gradient variance, as shown by Defazio & Bottou (2019). To improve the empirical training efficiency of SVRG, Yin et al. (2025) recently introduced a multiplicative coefficient into the variance-reduced gradient estimator. 

Another prominent variance-reduction technique that does not require full gradients is Polyak momentum. A provably variance-reduced momentum variant is known as Momentum-based Variance Reduction (MVR) (Cutkosky & Orabona, 2019). MVR incorporates stochastic gradient difference as the correction term into the momentum update. 

Specifically, it performs the iterations 

$$
x _ {t + 1} = x _ {t} - \eta g _ {t}, \quad \text { where }\tag{2}
$$

$$
\begin{array}{c} g _ {t} = (1 - \beta) (g _ {t - 1} + [ \nabla f _ {\xi_ {t}} (x _ {t}) - \nabla f _ {\xi_ {t}} (x _ {t - 1}) ]) \\ + \beta \nabla f _ {\xi_ {t}} (x _ {t}). \end{array}\tag{3}
$$

Here, $\eta \geq 0$ is the stepsize, and $\beta \in [ 0 , 1 ]$ is the momentum parameter. Note that MVR reduces to SGD when $\beta = 1$ and SGD with momentum when we omit the correction term $\nabla f _ { \xi _ { t } } ( x _ { t } ) - \nabla f _ { \xi _ { t } } ( x _ { t - 1 } )$ . Furthermore, MVR achieves the $\mathcal { O } ( 1 / T ^ { 1 / 3 } )$ convergence in the gradient norm, which improves upon the $\mathcal { O } ( 1 \bar { / } T ^ { 1 / 4 } )$ rate of SGD and SGD with momentum. Its convergence guarantee nearly matches the lower bounds originally established by Arjevani et al. (2023) and recently tightened by Fradin et al. (2026). 

## 1.2. MARS

Motivated by recent advances in the use of the multiplicative coefficient in SVRG (Yin et al., 2025), Yuan et al. (2025) proposed the Momentum with Adaptive Residual Scaling (MARS) algorithm, which further enhances MVR by explicitly scaling the momentum correction term. Specifically, MARS uses a coefficient $\gamma \geq 0$ to control the strength of the momentum correction term in MVR, thus resulting in the update 

$$
g _ {t} = (1 - \beta) (g _ {t - 1} + \gamma \Delta_ {t}) + \beta \nabla f _ {\xi_ {t}} (x _ {t}),\tag{4}
$$

where $\Delta _ { t } = \nabla f _ { \xi _ { t } } ( x _ { t } ) - \nabla f _ { \xi _ { t } } ( x _ { t - 1 } )$ . A full description of MARS is provided in Algorithm 1. Moreover, MARS encompasses both MVR and SGD with momentum. In particular, MARS reduces to MVR when $\gamma = 1$ , and to SGD with momentum when $\gamma = 0$ 

```txt
Algorithm 1 MARS (γ-MVR)
1: Input: stepsize η > 0; correction scale γ ≥ 0; momentum β ∈ (0, 1]; initial point and gradient estimator x₀, g₀ ∈ ℝᵈ.
2: for t = 0, 1, ..., T - 1 do
3: Compute xₜ₊₁ = xₜ - ηgₜ.
4: Sample ξₜ₊₁.
5: Compute Δₜ₊₁ = ∇fξₜ₊₁(xₜ₊₁) - ∇fξₜ₊₁(xₜ).
6: Update
    gₜ₊₁ = (1 - β)(gₜ + γΔₜ₊₁) + β∇fξₜ₊₁(xₜ₊₁).
7: end for
8: Output: x̂ₜ ~ Unif{x₀, ..., xₜ₋₁}. 
```

## 1.3. Theoretical Limitations of MARS

Yuan et al. (2025) study a variant of MARS (Algorithm 1) incorporating gradient clipping and a Hessian conditioning matrix. Furthermore, they show that introducing the scaling coefficient $\gamma$ leads to gradient estimators with strictly smaller variance than those produced by MVR. However, their theoretical analysis does not distill this favorable property of the MARS estimator into an explicit convergence advantage of MARS over MVR. More precisely, Yuan et al. (2025) establish $\mathcal { O } ( 1 / T ^ { 2 / 3 } )$ rate bounds for $\mathbb { E } \| \nabla f ( x _ { t } ) - g _ { t } \| ^ { 2 }$ and $\mathbb { E } \| x _ { t + 1 } - x _ { t } \| ^ { 2 }$ . Subsequent work by Liu et al. (2025) shows that MARS-M, which adapts MARS to the Muon optimizer for LLM training, achieves an $\mathcal { O } ( 1 / T ^ { 1 / 3 } )$ convergence rate in the gradient norm. Nevertheless, since MVR attains the same $\bar { \mathcal { O } } ( 1 / T ^ { 1 / 3 } )$ rate, these results do not distinguish the theoretical performance of MARS from that of MVR, and therefore do not explain the consistently superior empirical behavior of MARS over MVR. Although Chang et al. (2025) consider MARS-M, their analysis focuses solely on the special case $\gamma = 1$ , corresponding to MVR within the Muon framework, which has been recently studied by Huang et al. (2025); Khirirat et al. (2025); Qian et al. (2025). 

Furthermore, the convergence guarantees for MARS by Yuan et al. (2025) and for MARS-M by Liu et al. (2025) rely on impractical, time-varying parameters. For instance, Theorem B.5 of Yuan et al. (2025) requires the scaling factor $\gamma _ { t } \geq 0$ to depend on inaccessible quantities in practice: 

$$
\gamma_ {t} = \frac {\| d _ {t} \| ^ {2} - G _ {t}}{\beta_ {t} \| d _ {t} \| ^ {2}},
$$

where $G _ { t } ~ = ~ ( 1 - \beta _ { t } ) \mathbb { E } \langle \Delta _ { t } , \nabla f _ { \xi _ { t } } ( x _ { t } ) - \nabla f ( x _ { t } ) \rangle ~ + ~$ $\beta _ { t } \mathbb { E } \langle d _ { t } , g _ { t - 1 } - \nabla f ( x _ { t - 1 } ) \rangle$ , and $d _ { t } = \nabla f ( x _ { t } ) - \nabla f ( x _ { t - 1 } )$ The requirement of computing stochastic and full gradients at each iteration renders the implementation of their theoretical scaling factor $\gamma _ { t }$ impossible. This highlights the need for a new theoretical framework that provides sharp provable guarantees for MARS, and explains its empirical superiority over MVR through improved complexity guarantees. 

A single-gradient variant of MARS. The MARS momentum update in (4) requires evaluating two stochastic gradients per iteration, namely $\nabla f _ { \xi _ { t } } ( x _ { t } )$ and $\nabla f _ { \xi _ { t } } ( x _ { t - 1 } )$ . To reduce the per-iteration computational cost, one may instead form the correction term using $\nabla f _ { \xi _ { t } } ( x _ { t } ) - \nabla f _ { \xi _ { t - 1 } } ( x _ { t - 1 } )$ reusing the stochastic gradient computed at the previous iteration. However, in this paper, we focus on the two-gradient variant of MARS rather than its single-gradient variant. This is because the former achieves the $\mathcal { O } ( 1 / T ^ { 1 / 3 } )$ rate, which improves upon the $\mathcal { O } ( 1 / T ^ { 1 / 4 } )$ rate attained by the latter. 

## 2. Contributions

The goal of this paper is to provide a theoretical explanation of why MARS empirically outperforms MVR without the need to invoke any non-standard assumptions. Our key contributions include: 

• A refined similarity condition: γ-similarity. To resolve this issue, we propose a more refined similarity notion called γ-similarity, which measures, in expectation, how well $\gamma ( \nabla f _ { \xi } ( x ) - \nabla f _ { \xi } ( y ) )$ tracks $\nabla f ( x ) - \nabla f ( y )$ in the squared Euclidean norm. It provides a unified framework that recovers two widely used assumptions for analyzing stochastic optimization methods: (1) standard similarity $( \gamma = 1 )$ and (2) smoothness $( \gamma = 0 )$ as specific cases. Furthermore, under appropriate choices of γ, our γ-similarity constant is proved to be strictly smaller than the standard similarity and smoothness constants. This results in the conclusion that MARS theoretically outperforms MVR. 

• Superior convergence of MARS over MVR for nonconvex functions. In Section 6, by using our γ- similarity condition, we derive a convergence guarantee for MARS under our γ-similarity condition and standard assumptions (smoothness and bounded variance). In contrast to Theorem B.5 of Yuan et al. (2025), which chooses γ based on infeasible-to-compute quantities, our result holds for any fixed $\gamma \in [ 0 , 1 ] .$ , and shows explicitly that MARS with a proper choice of $\gamma \in [ 0 , 1 ]$ provides a strictly lower gradient complexity than standard MVR. 

• Stronger empirical performance of MARS over MVR and AdamW. In Section 7.2, we benchmarked MARS, MVR, and AdamW for training a 124M parameter LLM model. Our experiments confirm the existence of an optimal γ that maximizes token efficiency, outperforming both MVR and AdamW baselines. 

## 3. Related Works

We review prior literature on variance reduction and stochastic momentum algorithms. 

Variance reduction. While SGD is easy to implement and has been extensively studied, its stochastic gradient estimator suffers from high variance. To mitigate this issue, many variance reduction techniques have been proposed. Popular variance-reduction algorithms include SVRG (Johnson & Zhang, 2013), L-SVRG (Kovalev et al., 2020)), S2GD (Konecnˇ y & Richt´ arik´ , 2013), SAG (Roux et al., 2012; Schmidt et al., 2017), SAGA (Defazio et al., 2014a), FINITO (Defazio et al., 2014b), SPIDER (Fang et al., 2018), SARAH (Nguyen et al., 2017), and PAGE (Li et al., 2021). These algorithms construct variance-reduced stochastic gradient estimators by exploiting the difference between the stochastic gradient and its full gradient. Despite many studies demonstrating the theoretical advantages of these variance reduction algorithms, they have limited success in training huge-scale neural network models. A key performance bottleneck is the need to compute the full gradient, which renders variance-reduction algorithms impractical for these tasks. 

Polyak momentum. Inspired by Polyak’s heavy-ball method (Polyak, 1964) for deterministic optimization, Polyak momentum is a widely used technique for improving the convergence of SGD. The convergence behaviors of SGD with momentum have been extensively studied under various settings (Yan et al., 2018; Yu et al., 2019; Gitman et al., 2019; Loizou & Richtarik´ , 2020; Liu et al., 2020; Sebbouh et al., 2021; Wang et al., 2023; Zhang et al., 2025; Oikonomou & Loizou, 2025). In particular, under fixed step-size and momentum parameters, Liu et al. (2020); Sebbouh et al. (2021) showed that SGD with momentum achieve convergence rates comparable to those of SGD when minimizing (strongly) convex functions. However, it often yields superior empirical performance, and is adopted as the default optimizers in open-source software libraries such as PyTorch (Paszke et al., 2019) and JAX (Bradbury et al., 2018). 

Novel momentum. To improve upon Polyak momentum, several novel momentum techniques have been proposed. Notable techniques include Implicit Gradient Transport (IGT) (Arnold et al., 2019), Momentum Variance Reduction (MVR) (Cutkosky & Orabona, 2019), and various secondorder momentum methods (Salehkaleybar et al., 2024; Tran & Cutkosky, 2022). In the context of non-convex stochastic optimization, while IGT achieves an $\mathcal { O } ( 1 / T ^ { 2 / 7 } )$ convergence rate in gradient norm (Cutkosky & Mehta, 2020), both MVR and second-order momentum reach $\mathcal { O } ( 1 / T ^ { 1 / 3 } )$ nearly matching the theoretical lower bounds established by Arjevani et al. (2023) and recently tightened by Fradin et al. (2026). Yuan et al. (2025) introduced MARS to improve the empirical performance of MVR. This approach was further extended by Liu et al. (2025) through the development of MARS-M, which adapts the MARS update for use within the Muon optimizer in LLM training. However, the existing theoretical analysis of MARS-M yields an $\mathcal { O } ( 1 / T ^ { 1 / 3 } )$ convergence rate. This rate is identical to the rates derived for MVR-based Muon variants (Chang et al., 2025; Huang et al., 2025; Khirirat et al., 2025; Qian et al., 2025). Consequently, despite its clear empirical advantages, a theoretical framework that explicitly demonstrates the superior convergence of MARS over MVR remains a significant gap in the literature. 

## 4. Notations and Assumptions

We introduce notations and assumptions used throughout this paper. 

## 4.1. Notations

We denote by $\nabla f _ { \xi }$ the stochastic gradient associated with a random variable ξ, and by $\nabla f$ the full gradient. Expectation is denoted by $\mathbb { E } [ \cdot ]$ , while $\mathbb { E } _ { \xi } [ \cdot ]$ specifies expectation with respect to the randomness of $\xi .$ . Let xˆ<sub>T</sub> ∼ $\mathrm { U n i f } \{ x _ { 0 } , \ldots , x _ { T - 1 } \}$ denote an iterate sampled uniformly at random from the sequence $\{ x _ { 0 } , x _ { 1 } , \dotsc , x _ { T - 1 } \}$ . For vectors $x , y \in \mathbb { R } ^ { d } , \langle x , y \rangle$ denotes their inner product, and $\| x \| : =$ $\sqrt { \langle x , x \rangle }$ the associated Euclidean norm. We use $e _ { i } \in \mathbb { R } ^ { d }$ to denote the ith standard basis vector, Diag $( a _ { 1 } , \ldots , a _ { d } )$ the diagonal matrix with entries $a _ { 1 } , \dots , a _ { d } \in \mathbb { R }$ , and $I _ { d }$ the d×d identity matrix. Finally, for a square matrix $A \in \mathbb { R } ^ { d \times d }$ $\lambda _ { \mathrm { m a x } } ( A )$ denotes its largest eigenvalue. 

## 4.2. Assumptions

To facilitate our analysis, we impose assumptions on objectives and stochastic gradients, which are standard for analyzing stochastic algorithms for non-convex stochastic problems. 

Assumption 1 (Smoothness). The function $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ is bounded from below, i.e., $\begin{array} { r } { f _ { \operatorname* { i n f } } : = \operatorname* { i n f } _ { x \in \mathbb { R } ^ { d } } f ( x ) > - \infty } \end{array}$ and is L-smooth, i.e., 

$$
\| \nabla f (x) - \nabla f (y) \| \leq L \| x - y \|, \quad \forall x, y \in \mathbb {R} ^ {d}.
$$

L-Smoothness modulus of f can be defined as 

$$
L := \sup _ {x \neq y} \frac {\| \nabla f (x) - \nabla f (y) \|}{\| x - y \|} <   \infty .
$$

Assumption 2 (Unbiased and variance-bounded stochastic gradients). The estimator $\nabla f _ { \xi } ( x )$ is unbiased of the gradient $\nabla f ( x )$ and its variance is bounded, i.e., for all $x \in \mathbb { R } ^ { d }$ 

$$
\begin{array}{r c l} \mathbb {E} _ {\xi} [ \nabla f _ {\xi} (x) ] & = & \nabla f (x), \quad \text {and} \\ \mathbb {E} _ {\xi} \| \nabla f _ {\xi} (x) - \nabla f (x) \| ^ {2} & \leq & \sigma^ {2}. \end{array}
$$

## 5. Similarity

We can measure how similar the stochastic gradient $\nabla f _ { \xi } ( x )$ is to the full gradient $\nabla f ( x )$ by the following measure: 

Definition 1 (γ-similarity). For a fixed $\gamma ~ \in ~ \mathbb { R }$ , the $\gamma \mathrm { - }$ similarity is 

$$
\delta_ {\gamma} ^ {2} := \sup _ {x \neq y} \frac {\mathbb {E} _ {\xi} \left[ \| \gamma d _ {\xi} (x , y) - d (x , y) \| ^ {2} \right]}{\| x - y \| ^ {2}},\tag{5}
$$

where $d _ { \xi } ( x , y ) ~ = ~ \nabla f _ { \xi } ( x ) - \nabla f _ { \xi } ( y )$ and $d ( x , y ) \ =$ $\nabla f ( x ) - \nabla f ( y )$ . We call $\delta _ { \gamma }$ the γ-similarity constant whenever the above supremum is finite. Equivalently, for all $x , y \in \mathbb { R } ^ { d }$ 

$$
\mathbb {E} _ {\xi} \| \gamma (\nabla f _ {\xi} (x) - \nabla f _ {\xi} (y)) - (\nabla f (x) - \nabla f (y)) \| ^ {2} \leq \delta_ {\gamma} ^ {2} \| x - y \| ^ {2}.
$$

Definition 1, which we call a γ-similarity constant, provides a unified framework that recovers constants related to several standard assumptions for analyzing stochastic optimization methods. When $\gamma = 0$ , Definition 1 recovers smoothness in the sense that 

$$
\delta_ {0} ^ {2} = \sup _ {x \neq y} \frac {\| d (x , y) \| ^ {2}}{\| x - y \| ^ {2}} \leq L ^ {2}.
$$

If L is chosen as the smallest valid smoothness constant, equality holds. 

Conversely, by setting $\gamma = 1$ , Definition 1 obtains the constant of a standard similarity condition (i.e., Lipschitz continuity of $\nabla f _ { \xi } ( x ) - \nabla f ( x )$ in expectation), since: 

$$
\delta^ {2} := \delta_ {1} ^ {2} = \sup _ {x \neq y} \frac {\mathbb {E} _ {\xi} \left[ \| d _ {\xi} (x , y) - d (x , y) \| ^ {2} \right]}{\| x - y \| ^ {2}}.\tag{6}
$$

The standard similarity condition with $\delta \geq 0$ in (6) is used to demonstrate improved convergence results for stochastic algorithms in both centralized (Tyurin et al., 2022; Chayti & Karimireddy, 2024) and distributed settings (Khaled & Jin, 2023; Karagulyan et al., 2024; Takezawa et al., 2025). This condition is more general than Assumption 2 of Karagulyan et al. (2024), a star similarity condition in Tovmasyan et al. (2026), and an expected similarity condition in Sadiev et al. (2024). If $\mathbb { E } _ { \xi } \left\| \dot { \nabla } f _ { \xi } ( x ) - \nabla f _ { \xi } ( \dot { y } ) \right\| ^ { 2 } \leq \hat { L } ^ { 2 } \left\| x - y \right\| ^ { 2 }$ for some $\hat { L } > 0$ , then it follows that $\delta \leq { \hat { L } }$ 

For finite-sum minimization problems, or equivalently Problem (1) with ξ being sampled uniformly at random from $\{ 1 , 2 , \ldots , n \}$ , the standard similarity condition in (6) reduces to second-order similarity (Mairal, 2015; Khaled & Jin, 2023; Chayti & Karimireddy, 2024; Takezawa et al., 2025; Gasanov & Richtarik, 2024), i.e. 

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \| d _ {i} (x, y) - d (x, y) \| ^ {2} \leq \delta^ {2} \| x - y \| ^ {2},\tag{7}
$$

which is more relaxed than Assumption 3.2. in Condat et al. (2025). 

The next lemma provides the upper-bound of $\delta _ { \gamma } ^ { 2 }$ according to Definition 1, based on the knowledge of $\delta ^ { 2 }$ and $L ^ { 2 }$ 

Lemma 1. Consider Problem (1), and let $\delta _ { \gamma } ^ { 2 }$ be defined in (5). Suppose that Assumptions 1, 2 hold, and let $\delta ^ { 2 } : = \delta _ { 1 } ^ { 2 }$ be the standard similarity modulus in (6). Then: 

• For any $\gamma \in \mathbb { R } ,$ 

$$
\delta_ {\gamma} ^ {2} \leq \gamma^ {2} \delta^ {2} + (\gamma - 1) ^ {2} L ^ {2}.
$$

• The minimizer of the right-hand side is 

$$
\gamma_ {\star} := \arg \min _ {\gamma \in \mathbb {R}} \left\{\gamma^ {2} \delta^ {2} + (\gamma - 1) ^ {2} L ^ {2} \right\} = \frac {L ^ {2}}{\delta^ {2} + L ^ {2}},
$$

which lies in the interval [0, 1]. Furthermore 

$$
\delta_ {\gamma_ {\star}} ^ {2} \leq \frac {L ^ {2} \delta^ {2}}{\delta^ {2} + L ^ {2}}.
$$

Lemma 1 establishes that the γ-similarity constant $\delta _ { \gamma } ^ { 2 }$ is upper-bounded by a combination of the standard similarity constant $\delta ^ { 2 }$ and the smoothness constant $L ^ { 2 }$ . By selecting the optimal $\gamma _ { \star }$ according to this lemma, we show that $\delta _ { \gamma \star } ^ { 2 }$ strictly improves upon both $\delta ^ { 2 }$ and $L ^ { 2 }$ . In the case of quadratic problems, this upper bound becomes tight, and also $\delta _ { \gamma _ { \star } } ^ { 2 }$ can be substantially lower than its standard counterpart $\hat { \delta } ^ { 2 }$ 

Example 1. Consider the problem of minimizing $f ( x ) =$ $\textstyle { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n ^ { - } } f _ { i } ( x )$ with $\begin{array} { r } { f _ { i } ( x ) = \frac { 1 } { 2 } x ^ { T } A _ { i } x f o r x \in \mathbb { R } ^ { d } , d = n } \end{array}$ and $A _ { i } = \hat { L } e _ { i } e _ { i } ^ { T } \in \mathbb { R } ^ { d \times d } f o r \hat { L } > 0$ and i being selected uniformly at random from $\{ 1 , 2 , \ldots , n \}$ . Then, $\delta _ { \gamma } ^ { 2 } = ( \gamma ^ { 2 } n -$ $\begin{array} { r } { 2 \gamma + 1 ) \frac { \hat { L } ^ { 2 } } { n ^ { 2 } } , ~ \delta ^ { 2 } = ( n - 1 ) \frac { \hat { L } ^ { 2 } } { n ^ { 2 } } } \end{array}$ , and the L-smoothness is $\textstyle L ^ { 2 } = { \frac { { \hat { L } } ^ { 2 } } { n ^ { 2 } } }$ . This implies 

$$
\delta_ {\gamma} ^ {2} = \gamma^ {2} \delta^ {2} + (\gamma - 1) ^ {2} L ^ {2}.
$$

Also, γ<sub>⋆</sub> := argmin $\begin{array} { r } { \gamma ^ { 2 } \delta ^ { 2 } + ( \gamma - 1 ) ^ { 2 } L ^ { 2 } = \frac { 1 } { n } } \end{array}$ and $\delta _ { \gamma _ { \star } } ^ { 2 } =$ γ $\textstyle { \frac { n - 1 } { n } } { \frac { { \hat { L } } ^ { 2 } } { n ^ { 2 } } }$ , which yields $\begin{array} { r } { \frac { \delta ^ { 2 } } { \delta _ { \gamma \star } ^ { 2 } } = n . } \end{array}$ 

## 6. Convergence of MARS

In this section, we establish the gradient complexity of MARS. In particular, we rely on the γ-similarity condition to explicitly prove that MARS attains a lower complexity than MVR. 

To this end, we provide the convergence theorem for MARS to minimize nonconvex, smooth functions in the next theorem. 

Theorem 1. Consider MARS (Algorithm 1) for solving Problem (1). Suppose that Assumptions 1, 2 hold, and let $\delta _ { \gamma }$ be the γ-similarity from Definition 1. Initialize $g _ { 0 } =$ $\begin{array} { r } { \frac { 1 } { B _ { \mathrm { i n i t } } } \sum _ { j = 1 } ^ { B _ { \mathrm { i n i t } } } \nabla f _ { \xi _ { j } } ( x _ { 0 } ) } \end{array}$ , and $B _ { \mathrm { i n i t } } = \lceil 1 / \beta \rceil$ . For $\sigma = 0$ set $\beta \stackrel {  } { = } 1$ , and $f o r \sigma > 0 ,$ , set 

$$
\beta = \min \left\{1, \frac {\epsilon^ {2}}{\sigma^ {2}} \right\} a n d \eta = \frac {1}{L + \sqrt {a}},
$$

where $\begin{array} { r } { a = \left( \frac { ( 1 - \beta ) ^ { 3 } | \gamma - 1 | ^ { 2 } } { \beta } L ^ { 2 } + 2 ( 1 - \beta ) ^ { 2 } \delta _ { \gamma } ^ { 2 } \right) \frac { 1 } { \beta } . } \end{array}$ . Then, for $\begin{array} { r } { T \ge \Big \lceil \frac { 2 \Delta } { \eta \epsilon ^ { 2 } } + \frac { \sigma ^ { 2 } } { \epsilon ^ { 2 } } \Big \rceil , f o r \Delta = f ( x _ { 0 } ) - f _ { \mathrm { i n f } } . } \end{array}$ , the output ${ \hat { x } } _ { T } \sim \operatorname { U n i f } \{ x _ { 0 } , \dots , x _ { T - 1 } \}$ satisfies $\mathbb { E } \| \nabla f ( \hat { x } _ { T } ) \| ^ { 2 } \leq 4 \epsilon ^ { 2 }$ Consequently, the number of stochastic gradient evaluations is 

$$
\mathcal {O} \left(\frac {\sigma^ {2}}{\epsilon^ {2}} + \frac {L \Delta}{\epsilon^ {2}} + \frac {\delta_ {\gamma} \Delta \sigma}{\epsilon^ {3}} + \frac {| \gamma - 1 | L \Delta \sigma^ {2}}{\epsilon^ {4}}\right).\tag{8}
$$

Theorem 1 generalizes the result by Yuan et al. (2025), in terms of the valid range for the coefficient γ and gradient complexity bound. First, Theorem 1 holds for any γ-values, whereas Yuan et al. (2025) require specific γ choices that depend on quantities typically unavailable in practice. Second, we provide explicit gradient complexity bounds, while Yuan et al. (2025) can establish only the rate bounds for $\mathbb { E } \| \nabla f ( x _ { t } ) - g _ { t } \| ^ { 2 }$ and $\mathbb { E } \| x _ { t + 1 } - x _ { t } \| ^ { 2 }$ . Third, in contrast to Yuan et al. (2025), our theorem recovers the complexity bounds for MVR and SGD with momentum. Specifically, Theorem 1 obtains the $\mathcal { O } ( 1 / \epsilon ^ { 3 } )$ complexity for MVR analyzed by Cutkosky & Orabona (2019); Fradin et al. (2026) when we let $\gamma = 1$ , and the $\mathcal { O } ( 1 / \epsilon ^ { 4 } )$ complexity for SGD with momentum. 

When $\gamma = 1$ , Theorem 1 recovers the MVR-type upperbound scaling 

$$
\mathcal {O} \left(\frac {\sigma^ {2}}{\epsilon^ {2}} + \frac {L \Delta}{\epsilon^ {2}} + \frac {\delta \Delta \sigma}{\epsilon^ {3}}\right).
$$

This is consistent with the known optimal-rate picture for stochastic nonconvex optimization under similarity-type conditions, including the bounds discussed by Fradin et al. (2026). We emphasize that our main contribution is not a new lower bound for $\gamma = 1$ , but rather the refined $\gamma -$ dependent upper bound showing when the scaled correction can improve the MVR bound expression. 

In addition to the convergence for MARS under constant tuning parameters $\beta , \eta .$ , we also present the convergence for MARS under T -dependent parameters in the next corollary. 

Corollary 1 (A T -dependent parameter choice). Consider MARS (Algorithm 1) for solving Problem (1), where Assumptions 1, 2 hold. Let $\delta _ { \gamma } ^ { 2 }$ be defined by (5), and let the algorithm run for horizon $T \geq 1$ , such that $\begin{array} { r } { \sqrt { \frac { | \gamma - 1 | L \Delta } { \sigma ^ { 2 } T } } + } \end{array}$ $\begin{array} { r } { 2 ^ { - 1 / 3 } \left( \frac { \delta _ { \gamma } \Delta } { \sigma ^ { 2 } T } \right) ^ { 2 / 3 } \leq 1 } \end{array}$ , and choose 

$$
\begin{array}{r} \beta_ {T} = \sqrt {\frac {| \gamma - 1 | L \Delta}{\sigma^ {2} T}} + 2 ^ {- 1 / 3} \left(\frac {\delta_ {\gamma} \Delta}{\sigma^ {2} T}\right) ^ {2 / 3}, \\ \eta = \frac {1}{L + \frac {| \gamma - 1 | L}{\beta_ {T}} + \frac {\sqrt {2} \delta_ {\gamma}}{\sqrt {\beta_ {T}}}}. \end{array}
$$

Then for $\begin{array} { r } { g _ { 0 } = \frac { 1 } { B _ { \mathrm { i n i t } } } \sum _ { j = 1 } ^ { B _ { \mathrm { i n i t } } } \nabla f _ { \xi _ { j } } ( x _ { 0 } ) \mathrm { ~ } w i t h \mathrm { ~ } B _ { \mathrm { i n i t } } = \lceil 1 / \beta _ { T } \rceil , } \end{array}$ 

$$
\begin{array}{r l} \mathbb {E} \big [ \| \nabla f (\hat {x} _ {T}) \| ^ {2} \big ] & \leq \frac {2 L \Delta}{T} + \frac {4 \sigma \sqrt {| \gamma - 1 | L \Delta}}{\sqrt {T}} \\ & + \frac {3 \cdot 2 ^ {2 / 3} (\delta_ {\gamma} \Delta \sigma) ^ {2 / 3}}{T ^ {2 / 3}} + \frac {\sigma^ {2}}{T}, \end{array}\tag{9}
$$

where ${ \hat { x } } _ { T } \sim \operatorname { U n i f } \{ x _ { 0 } , \dots , x _ { T - 1 } \}$ 

Like Theorem 1, Corollary 1 encompasses the convergence rate bound for MVR and SGD with momentum, depending on the coefficient $\gamma .$ On the one hand, when we let $\gamma =$ 1, Corollary 1 obtains the $\mathcal { O } ( T ^ { - 2 / 3 } )$ rate in the squared gradient norm for MVR (up to the additional $\sigma ^ { 2 } / \bar { T }$ term arising from initialization $\Delta )$ . On the other hand, when we let $\gamma = 0$ , Corollary 1 yields the $\mathcal { O } ( T ^ { - 1 / 2 } )$ rate for SGD with momentum. 

## 6.1. Theoretical Advantage of MARS over MVR

Next, we demonstrate how Theorem 1 implies the superior performance of MARS with $\gamma \in [ 0 , 1 ]$ to MVR, in the next corollary. 

Corollary 2. Consider the setting of Theorem 1 with $\gamma \in$ 

[0, 1]. Let 

$$
A = \frac {\Delta \sigma}{\epsilon^ {3}}, \qquad B = \frac {L \Delta \sigma^ {2}}{\epsilon^ {4}}, \qquad D = \delta^ {2} + L ^ {2}.
$$

Define the surrogate non-common part of the MARS bound by $J ( \gamma ) = A \sqrt { \gamma ^ { 2 } \delta ^ { 2 } + ( 1 - \gamma ) ^ { 2 } L ^ { 2 } } + B ( 1 - \gamma ) . I f B < A \delta$ and $\begin{array} { r } { \gamma _ { \star } = \frac { L ^ { 2 } } { D } + \frac { B L \delta } { D \sqrt { A ^ { 2 } D - B ^ { 2 } } } } \end{array}$ , then $\gamma _ { \star } \in [ 0 , 1 ]$ and 

$$
J (\gamma_ {\star}) \leq J (1) = A \delta .
$$

Consequently, after replacing $\delta _ { \gamma }$ by the upper bound from Lemma 1, MARS complexity bound in (8) is no larger than the corresponding displayed MVR bound at $\gamma = 1$ 

Proof. The terms common to the displayed MARS and MVR bounds are $\sigma ^ { 2 } / \epsilon ^ { 2 }$ and $L \Delta / \epsilon ^ { 2 }$ . Hence it suffices to compare the remaining displayed terms. By Lemma 1, for any $\gamma \in [ 0 , 1 ]$ 

$$
\frac {\Delta \sigma}{\epsilon^ {3}} \delta_ {\gamma} + \frac {L \Delta \sigma^ {2}}{\epsilon^ {4}} (1 - \gamma)
$$

is upper-bounded by $A \sqrt { \gamma ^ { 2 } \delta ^ { 2 } + ( 1 - \gamma ) ^ { 2 } L ^ { 2 } } + B ( 1 - \gamma ) =$ $J ( \gamma )$ . Proposition 1 gives $J ( \gamma _ { \star } ) \leq J ( 1 ) = A \delta$ whenever 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/8eaad0b6-8cb4-4ca8-8ebc-ad8b2b31742c/130a2a5f244b63c23a3e942c5f7e69b8488abf93180560e06bcf46d8621165cc.jpg)



Figure 1. Bound-implied MARS vs. MVR speedup. Brighter colors indicate settings where MARS offers larger improvements over the MVR baseline $( \gamma = 1 )$ and dark regions show no benefit. Color encodes $\log _ { 1 0 } ( A \delta / J ( \gamma _ { \star } ) )$ , where $J ( \gamma ) = A \sqrt { \gamma ^ { 2 } \delta ^ { 2 } + ( 1 - \gamma ) ^ { 2 } L ^ { 2 } } +$ $B ( 1 - \gamma )$ with $A = \sigma / \epsilon ^ { 3 } B = L \bar { \sigma } ^ { 2 } / \epsilon ^ { 4 }$ , and $\gamma _ { \star }$ is given in Corollary $2 .$ Each panel sweeps over target accuracy ϵ (horizontal) and gradient heterogeneity δ (vertical). Rows vary smoothness $L \in \{ 0 . 1 , \dot { 1 } , 1 0 \}$ , columns vary noise $\sigma \in \{ \breve { 1 } 0 ^ { - 4 } , 1 0 ^ { - 2 } , 1 \}$ . Dashed curves show optimal $\gamma _ { \star }$ contours.


$B < A \delta$ . Therefore the non-common part of the displayed MARS upper-bound expression at $\gamma _ { \star }$ is no larger than the corresponding non-common part of the displayed MVR expression at $\gamma = 1$ □ 

This corollary shows that, under the stated condition and surrogate upper bound on $\delta _ { \gamma } ,$ , the displayed MARS upperbound expression can be made no larger than the corresponding MVR expression. 

## 6.2. Illustrative Speed-up Gains from MARS over MVR

The only γ-dependent part of the displayed bound in Theorem 1 is controlled by 

$$
A \delta_ {\gamma} + B (1 - \gamma), \qquad A = \frac {\Delta \sigma}{\epsilon^ {3}}, \qquad B = \frac {L \Delta \sigma^ {2}}{\epsilon^ {4}}.
$$

Using Lemma 1, this is upper-bounded by the surrogate 

$$
J (\gamma) = A \sqrt {\gamma^ {2} \delta^ {2} + (1 - \gamma) ^ {2} L ^ {2}} + B (1 - \gamma).
$$

Since the MVR baseline corresponds to $J ( 1 ) = A \delta $ , we visualize the bound-implied speedup by the ratio $A \delta / J ( \gamma _ { \star } )$ 

To illustrate the magnitude of the complexity improvement offered by MARS over MVR, Figure 1 displays the quantity log $_ { \mathrm { 1 0 } } ( A \delta / J ( \gamma _ { \star } ) )$ across various parameter settings. Specifically, we evaluate this speed-up for smoothness constants $L \in \{ 0 . 1 , 1 . 0 , 1 0 . 0 \}$ and noise levels $\sigma \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 2 } , 1 \}$ 

In Figure 1, three patterns emerge. First, speedup grows toward larger ϵ and δ: the upper-right of each panel is bright, reflecting regimes where the $( 1 - \gamma )$ -penalty becomes negligible relative to the baseline cost. The $\gamma _ { \star }$ contours shift toward smaller values in this region, indicating that the surrogate increasingly prefers $\gamma _ { \star } ~ < ~ 1$ . Second, a diagonal transition separates improvement from no-improvement, governed by the ratio $L \sigma / ( \epsilon \delta )$ : improvement emerges when ϵδ is sufficiently large relative to $L \sigma$ . Third, increasing L (top to bottom) or σ (left to right) shifts the improvement region outward. Overall, MARS yields the largest gains in moderate-accuracy, high-heterogeneity regimes, while harder problems (large Lσ) require proportionally larger ϵδ to benefit. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/8eaad0b6-8cb4-4ca8-8ebc-ad8b2b31742c/a5ef791e8d1079bf9d1fb208fad50e094b3dfdfe7e3a40ec5266a573fb8dda5a.jpg)


## 7. Experiments

## 7.1. A CIFAR-10 Probe of the Predicted Correction Scale

We first run a small theorem-aligned probe to connect the analysis to quantities that can be estimated along a training trajectory. This experiment is not intended as a competitive CIFAR-10 benchmark. Its purpose is to test whether the local gradient-difference statistics appearing in γ-similarity prefer a correction scale below the MVR value $\gamma = 1$ 

We train a small CNN on CIFAR-10 using the vanilla twogradient γ-MVR update analyzed in Theorem 1. At a checkpoint t, let 

$$
\begin{array}{c} {d _ {t} = \nabla f (x _ {t}) - \nabla f (x _ {t - 1}),} \\ {d _ {B, t} = \nabla f _ {B} (x _ {t}) - \nabla f _ {B} (x _ {t - 1}),} \end{array}
$$

where B is a mini-batch. Using $M = 5 1 2$ sampled minibatches, we estimate the local predicted correction scale 

$$
\widehat {\gamma} _ {t} ^ {\star} = \frac {\| d _ {t} \| ^ {2}}{\frac {1}{M} \sum_ {m = 1} ^ {M} \| d _ {B _ {m} , t} \| ^ {2}},
$$

which is the empirical analogue of the minimizer suggested by Lemma 1. We also compute the grid minimizer of the corresponding local correction proxy over the tested γ values. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/8eaad0b6-8cb4-4ca8-8ebc-ad8b2b31742c/db0ed8612d40ad008f8ba1baeebceb026b9998012738aacb524ddd7c83457831.jpg)



Figure 2. CIFAR-10 probe of the MARS correction scale. Left: checkpoint-level predicted scale $\begin{array} { r } { \widehat { \gamma } _ { t } ^ { \star } = \| d _ { t } \| ^ { 2 } / ( \frac { 1 } { M } \sum _ { m } \| d _ { B _ { m } , t } \| ^ { 2 } ) } \end{array}$ and the grid minimizer of the local correction proxy. Both remain below the MVR value $\gamma = 1$ and vary during training. Right: fixed-γ sweep in the same vanilla two-gradient $\gamma { \mathrm { - } } \mathbf { M } \mathbf { V } \mathbf { R }$ setup. Final training loss is minimized at $\gamma < 1$ , while the MVR setting $\gamma = 1$ is worse under the same hyperparameters.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/8eaad0b6-8cb4-4ca8-8ebc-ad8b2b31742c/1089e39217f8558da4d00b0ae3159f3d0c02e7054d02256e21b8a34309d669ea.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/8eaad0b6-8cb4-4ca8-8ebc-ad8b2b31742c/3285d73d5f0c47cbdaad5c1c0a6a6737dbcca4c3c012c7902bc196878f7e129b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/8eaad0b6-8cb4-4ca8-8ebc-ad8b2b31742c/427b5608f6624424c7c029858c3bfa4234ecd6a205b9327a36be6b8545dbad7d.jpg)



Figure 3. Validation loss vs. tokens for a 124M γ-sweep. Left: full training run. Center: early training (first ≈ 0.2B tokens). Right: late stage near the Chinchilla-optimal budget (log-scale).


Figure 2 shows that the predicted correction scale stays below 1 throughout training and changes substantially over the trajectory. In this run, $\widehat { \gamma } _ { t } ^ { \star }$ ranges from approximately 0.10 to 0.68, with median about 0.44. The grid minimizer follows the same qualitative pattern and is also consistently below 1. A fixed-γ sweep in the same setup further shows that final training loss is minimized at $\gamma = 0 . 2 5$ , whereas the MVR setting $\gamma = 1$ is worse under the same hyperparameters. These results support the qualitative message of the theory: the useful correction scale depends on local gradient-difference statistics and need not coincide with MVR. 

## 7.2. MARS-AdamW on GPT-style Pretraining

We evaluate the effect of $\gamma$ in MARS-AdamW in an applied setting, since LLM pretraining is a primary modern workload. Concretely, we follow the pretraining protocol and codebase of Semenov et al. (2025) on a 124M Llamastyle model (12L/12H/768) and a Chinchilla-optimal budget (Hoffmann et al., 2022) of approximately 2.10B tokens; the protocol is well-tuned, so we rely on their hyperparameters rather than re-tuning for each γ. Our goal is not exhaustive hyperparameter search, but to observe the behavior of different γ values under a sane and inexpensive setup; we sweep $\gamma \in \{ 0 . 0 1 , 0 . 0 2 5 , 0 . 0 4 , 0 . 1 , 1 \}$ and include AdamW as a baseline. We keep all hyperparameters fixed across γ to isolate the effect of the MARS scaling. 

Results. Figure 3 shows three views of the same training run: the full trajectory (left), an early-stage zoom (center), and a late-stage zoom (right). Overall, we observe three phenomena. First, small γ values can outperform AdamW; the late-stage panel (right) shows several $\gamma < 1$ curves below AdamW near the end of training. Second, different small $\gamma$ values behave differently and there is an optimal choice; the early-stage ordering (center) differs from the latestage ordering (right), which is consistent with the iterates traversing regions with different local properties that change how $\gamma$ trades off the relevant terms in our bound (e.g., the $\delta _ { \gamma }$ -dependent term versus the (1 − γ)-dependent penalty), though the final-loss differences among small $\gamma$ values are relatively small. Third, the classic MVR setting $( \gamma = 1 )$ can be sensitive to hyperparameters and exhibit instability without dedicated tuning, as visible in the full-trajectory panel (left). Additional experimental details are provided in Appendix G. 

## 8. Conclusion

In this paper, we have provided a rigorous theoretical explanation for the superior convergence performance of MARS over MVR algorithms. By introducing the $\gamma -$ similarity measure, we derive convergence guarantees of MARS for minimizing nonconvex functions solely under assumptions commonly used for analyzing stochastic algorithms. Our results prove that MARS with an appropriately tuned $\gamma \in [ 0 , 1 ]$ is explicitly shown to achieve a strictly lower gradient complexity than MVR. This theoretical framework is corroborated by our empirical studies on GPT pretraining, which show that there exists an optimal choice of $\gamma$ maximizing token efficiency of MARS over both MVR and AdamW. 

## Acknowledgements

The research reported in this publication was supported by funding from King Abdullah University of Science and Technology (KAUST): i) KAUST Baseline Research Scheme, ii) Center of Excellence for Generative AI, under award number 5940, iii) SDAIA-KAUST Center of Excellence in Artificial Intelligence and Data Science. 

## Impact Statement

Our paper introduces the γ-similarity condition, a novel tool to better capture the nuances of specific algorithmic modifications. Using this condition, we provide the first theoretical justification for the superior performance of MARS. This will pave the way for novel analysis frameworks for scaled momentum and variance reduction methods. 

Furthermore, our theoretical findings show how the optimal scaled coefficient $\gamma$ improves efficiency in large-scale model training. This will lead to the development of scaled momentum and variance reduction methods, which reduce computational costs and improve resource efficiency in training state-of-the-art learning models. 

## References



Arjevani, Y., Carmon, Y., Duchi, J. C., Foster, D. J., Srebro, N., and Woodworth, B. Lower bounds for non-convex stochastic optimization. Mathematical Programming, 199 (1):165–214, 2023. (Cited on pages 2 and 3) 





Arnold, S., Manzagol, P.-A., Babanezhad Harikandeh, R., Mitliagkas, I., and Le Roux, N. Reducing the variance in online optimization by transporting past gradients. Advances in Neural Information Processing Systems, 32, 2019. (Cited on page 3) 





Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., and Zhang, Q. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax. (Cited on page 3) 





Bubeck, S., Coester, C., Eldan, R., Gowers, T., Lee, Y. T., Lupsasca, A., Sawhney, M., Scherrer, R., Sellke, M., Spears, B. K., et al. Early science acceleration experiments with GPT-5. arXiv preprint arXiv:2511.16072, 2025. (Cited on page 1) 





Chang, D., Liu, Y., and Yuan, G. On the convergence of Muon and beyond. arXiv preprint arXiv:2509.15816, 2025. (Cited on pages 2 and 3) 





Chayti, E. M. and Karimireddy, S. P. Optimization with access to auxiliary information. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. (Cited on page 4) 





Condat, L., Gasanov, E., and Richtarik, P. The stochas-´ tic multi-proximal method for nonsmooth optimization. arXiv preprint arXiv:2505.12409, 2025. (Cited on page 4) 





Cutkosky, A. and Mehta, H. Momentum improves normalized SGD. In International Conference on Machine Learning, pp. 2260–2268. PMLR, 2020. (Cited on page 3) 





Cutkosky, A. and Orabona, F. Momentum-based variance reduction in non-convex SGD. Advances in Neural Information Processing Systems, 32, 2019. (Cited on pages 1, 3, and 5) 





Defazio, A. and Bottou, L. On the ineffectiveness of variance reduced optimization for deep learning. Advances in Neural Information Processing Systems, 32, 2019. (Cited on page 1) 





Defazio, A., Bach, F., and Lacoste-Julien, S. SAGA: A fast incremental gradient method with support for nonstrongly convex composite objectives. Advances in Neural Information Processing Systems, 27, 2014a. (Cited on page 3) 





Defazio, A., Domke, J., et al. Finito: A faster, permutable incremental gradient method for big data problems. In International Conference on Machine Learning, pp. 1125– 1133. PMLR, 2014b. (Cited on page 3) 





Fang, C., Li, C. J., Lin, Z., and Zhang, T. Spider: Nearoptimal non-convex optimization via stochastic pathintegrated differential estimator. Advances in Neural Information Processing Systems, 31, 2018. (Cited on page 3) 





Fradin, A., Sadiev, A., Condat, L., and Richtarik, P. Tight´ lower bounds and optimal algorithms for stochastic nonconvex optimization with heavy-tailed noise. In The 29th International Conference on Artificial Intelligence and Statistics, 2026. (Cited on pages 2, 3, 5, and 20) 





Gasanov, E. and Richtarik, P. Speeding up stochastic proximal optimization in the high hessian dissimilarity setting. arXiv preprint arXiv:2412.13619, 2024. (Cited on page 4) 





Ghadimi, S. and Lan, G. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization, 23(4):2341–2368, 2013. (Cited on page 1) 





Gitman, I., Lang, H., Zhang, P., and Xiao, L. Understanding the role of momentum in stochastic gradient methods. Advances in Neural Information Processing Systems, 32, 2019. (Cited on page 3) 





Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. (Cited on page 1) 





Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. DeepSeek-R1: incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. (Cited on page 1) 





Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., and Sifre, L. Training compute-optimal 





large language models. Advances in Neural Information Processing Systems, 35:30016–30030, 2022. (Cited on page 8) 





Huang, F., Luo, Y., and Chen, S. LiMuon: Light and fast Muon optimizer for large models. arXiv preprint arXiv:2509.14562, 2025. (Cited on pages 2 and 3) 





Johnson, R. and Zhang, T. Accelerating stochastic gradient descent using predictive variance reduction. Advances in Neural Information Processing Systems, 26, 2013. (Cited on pages 1 and 3) 





Karagulyan, A., Shulgin, E., Sadiev, A., and Richtarik, P.´ Spam: Stochastic proximal point method with momentum variance reduction for non-convex cross-device federated learning. arXiv preprint arXiv:2405.20127, 2024. (Cited on page 4) 





Khaled, A. and Jin, C. Faster federated optimization under second-order similarity. In The Eleventh International Conference on Learning Representations, 2023. (Cited on page 4) 





Khirirat, S., Sadiev, A., Demidovich, Y., and Richtarik,´ P. Better LMO-based momentum methods with secondorder information. arXiv preprint arXiv:2512.13227, 2025. (Cited on pages 2 and 3) 





Kingma, D. P. Adam: A method for stochastic optimization. In The Third International Conference on Learning Representations, 2015. (Cited on page 1) 





Konecnˇ y, J. and Richt´ arik, P. Semi-stochastic gradient´ descent methods. arXiv preprint arXiv:1312.1666, 2013. (Cited on pages 1 and 3) 





Kovalev, D., Horvath, S., and Richt´ arik, P. Don’t jump´ through hoops and remove those loops: SVRG and katyusha are better without the outer loop. In Algorithmic Learning Theory, pp. 451–467. PMLR, 2020. (Cited on page 3) 





Li, Z., Bao, H., Zhang, X., and Richtarik, P. PAGE: a´ simple and optimal probabilistic gradient estimator for nonconvex optimization. In International Conference on Machine Learning, pp. 6286–6295. PMLR, 2021. (Cited on page 3) 





Liu, Y., Gao, Y., and Yin, W. An improved analysis of stochastic gradient descent with momentum. Advances in Neural Information Processing Systems, 33:18261– 18271, 2020. (Cited on page 3) 





Liu, Y., Yuan, A., and Gu, Q. MARS-M: when variance reduction meets matrices. arXiv preprint arXiv:2510.21800, 2025. (Cited on pages 2 and 3) 





Loizou, N. and Richtarik, P. Momentum and stochastic mo-´ mentum for stochastic gradient, newton, proximal point and subspace descent methods. Computational Optimization and Applications, 77(3):653–710, 2020. (Cited on page 3) 





Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. (Cited on page 1) 





Mairal, J. Incremental majorization-minimization optimization with application to large-scale machine learning. SIAM Journal on Optimization, 25(2):829–855, 2015. (Cited on page 4) 





Nguyen, L. M., Liu, J., Scheinberg, K., and Taka´c, M.ˇ SARAH: A novel method for machine learning problems using stochastic recursive gradient. In International Conference on Machine Learning, pp. 2613–2621. PMLR, 2017. (Cited on page 3) 





Oikonomou, D. and Loizou, N. Stochastic Polyak step-sizes and momentum: Convergence guarantees and practical performance. In The Thirteenth International Conference on Learning Representations, 2025. (Cited on page 3) 





Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32, 2019. (Cited on page 3) 





Penedo, G., Kydl´ıcek, H., Ben allal, L., Lozhkov, A.,ˇ Mitchell, M., Raffel, C., Von Werra, L., and Wolf, T. The FineWeb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 37, 2024. (Cited on page 27) 





Polyak, B. T. Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 4(5):1–17, 1964. (Cited on page 3) 





Qian, X., Rammal, H., Kovalev, D., and Richtarik, P. Muon is provably faster with momentum variance reduction. arXiv preprint arXiv:2512.16598, 2025. (Cited on pages 2 and 3) 





Reddi, S. J., Hefny, A., Sra, S., Poczos, B., and Smola, A. Stochastic variance reduction for nonconvex optimization. In International Conference on Machine Learning, pp. 314–323. PMLR, 2016. (Cited on page 1) 





Richtarik, P., Sokolov, I., and Fatkhullin, I. EF21: A new,´ simpler, theoretically better, and practically faster error feedback. Advances in Neural Information Processing Systems, 34, 2021. (Cited on page 16) 





Roux, N., Schmidt, M., and Bach, F. A stochastic gradient method with an exponential convergence rate for finite training sets. Advances in Neural Information Processing Systems, 25, 2012. (Cited on page 3) 





Sadiev, A., Condat, L., and Richtarik, P. Stochastic proximal´ point methods for monotone inclusions under expected similarity. arXiv preprint arXiv:2405.14255, 2024. (Cited on page 4) 





Salehkaleybar, S., Khorasani, M., Kiyavash, N., He, N., and Thiran, P. Momentum-based policy gradient with secondorder information. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. (Cited on page 3) 





Schmidt, M., Le Roux, N., and Bach, F. Minimizing finite sums with the stochastic average gradient. Mathematical Programming, 162(1):83–112, 2017. (Cited on page 3) 





Sebbouh, O., Gower, R. M., and Defazio, A. Almost sure convergence rates for stochastic gradient descent and stochastic heavy ball. In Conference on Learning Theory, pp. 3935–3971. PMLR, 2021. (Cited on page 3) 





Semenov, A., Pagliardini, M., and Jaggi, M. Benchmarking optimizers for large language model pretraining. arXiv preprint arXiv:2509.01440, 2025. (Cited on pages 8 and 27) 





Shen, Z., Tao, T., Ma, L., Neiswanger, W., Liu, Z., Wang, H., Tan, B., Hestness, J., Vassilieva, N., Soboleva, D., and Xing, E. SlimPajama-DC: understanding data combinations for LLM training. arXiv preprint arXiv:2309.10818, 2023. URL https://arxiv. org/abs/2309.10818. (Cited on page 27) 





Takezawa, Y., Jiang, X., Rodomanov, A., and Stich, S. U. Exploiting similarity for computation and communication-efficient decentralized optimization. In Forty-second International Conference on Machine Learning, 2025. (Cited on page 4) 





Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E.,` Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. (Cited on page 27) 





Tovmasyan, Z., Malinovsky, G., Condat, L., and Richtarik,´ P. Revisiting stochastic proximal point methods: Generalized smoothness and similarity. Journal of Nonlinear and Variational Analysis, 10:471–505, 2026. (Cited on page 4) 





Tran, H. and Cutkosky, A. Better SGD using second-order momentum. Advances in Neural Information Processing Systems, 35, 2022. (Cited on page 3) 





Tyurin, A., Sun, L., Burlachenko, K., and Richtarik, P.´ Sharper rates and flexible framework for nonconvex SGD with client and data sampling. arXiv preprint arXiv:2206.02275, 2022. (Cited on page 4) 





Wang, X., Johansson, M., and Zhang, T. Generalized Polyak step size for first order optimization with momentum. In International Conference on Machine Learning, pp. 35836–35863. PMLR, 2023. (Cited on page 3) 





Yan, Y., Yang, T., Li, Z., Lin, Q., and Yang, Y. A unified analysis of stochastic momentum methods for deep learning. arXiv preprint arXiv:1808.10396, 2018. (Cited on page 3) 





Yin, Y., Xu, Z., Li, Z., Darrell, T., and Liu, Z. A coefficient makes SVRG effective. In The Thirteenth International Conference on Learning Representations, 2025. (Cited on pages 1 and 2) 





Yu, H., Jin, R., and Yang, S. On the linear speedup analysis of communication efficient momentum SGD for distributed non-convex optimization. In International Conference on Machine Learning, pp. 7184–7193. PMLR, 2019. (Cited on page 3) 





Yuan, H., Liu, Y., Wu, S., Xun, Z., and Gu, Q. MARS: unleashing the power of variance reduction for training large models. In International Conference on Machine Learning, pp. 73553–73587. PMLR, 2025. (Cited on pages 1, 2, 3, and 5) 





Zhang, J., Jin, C., and Gu, Y. Adaptive polyak step-size for momentum accelerated stochastic gradient descent with general convergence guarantee. IEEE Transactions on Signal Processing, 73:462–476, 2025. (Cited on page 3) 



## Contents

1 Introduction 1
    1.1 Variance Reduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
    1.2 MARS . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
    1.3 Theoretical Limitations of MARS 2
2 Contributions 2
3 Related Works 3
4 Notations and Assumptions 4
    4.1 Notations 4
    4.2 Assumptions 4
5 Similarity 4
6 Convergence of MARS 5
    6.1 Theoretical Advantage of MARS over MVR 6
    6.2 Illustrative Speed-up Gains from MARS over MVR 7
7 Experiments 7
    7.1 A CIFAR-10 Probe of the Predicted Correction Scale 7
    7.2 MARS-AdamW on GPT-style Pretraining 8
8 Conclusion 8
A Proof of Lemma 1 13
B Additional Lemma for γ-Similarity 13
    B.1 Proof of Lemma 2 14
    B.2 Proof of Example 1 15
C Descent Lemma for MARS Algorithms 16
D Convergence Results of MARS 19
    D.1 Proof of Theorem 1 19
    D.2 Comparison to Existing MVR Bounds 20
    D.3 A T-horizon Convergence Bound for MARS 21
    D.3.1 Proof of Corollary 1 21
E Theoretical Advantage of MARS over MVR 23 

E.1 Why the quadratic surrogate is needed 23  
E.2 Proposition 1 23  
E.3 Proof of Proposition 1 23  
F Auxiliary CIFAR-10 Probe 24  
G LLM pretraining 27  
H Theory visualization: bound-implied MARS vs. MVR speedup 29 

## A. Proof of Lemma 1

Fix x $\neq y$ and write 

$$
d _ {\xi} = d _ {\xi} (x, y), \qquad d = d (x, y).
$$

By Assumption 2, $, \mathbb { E } _ { \xi } [ d _ { \xi } ] = d$ , and hence $\mathbb { E } _ { \xi } [ d _ { \xi } - d ] = 0 .$ . Therefore, 

$$
\begin{array}{r l} & {\mathbb {E} _ {\xi} \| \gamma d _ {\xi} - d \| ^ {2} = \mathbb {E} _ {\xi} \| \gamma (d _ {\xi} - d) + (\gamma - 1) d \| ^ {2}} \\ & {\qquad = \gamma^ {2} \mathbb {E} _ {\xi} \| d _ {\xi} - d \| ^ {2} + (\gamma - 1) ^ {2} \| d \| ^ {2} + 2 \gamma (\gamma - 1) \langle \mathbb {E} _ {\xi} [ d _ {\xi} - d ], d \rangle} \\ & {\qquad = \gamma^ {2} \mathbb {E} _ {\xi} \| d _ {\xi} - d \| ^ {2} + (\gamma - 1) ^ {2} \| d \| ^ {2}.} \end{array}
$$

Dividing by $\| x - y \| ^ { 2 }$ and using the definitions of δ and $L$ gives 

$$
\frac {\mathbb {E} _ {\xi} \| \gamma d _ {\xi} (x , y) - d (x , y) \| ^ {2}}{\| x - y \| ^ {2}} \leq \gamma^ {2} \delta^ {2} + (\gamma - 1) ^ {2} L ^ {2}.
$$

Taking the supremum over x $\neq y$ proves the first claim. 

The function $q ( \gamma ) = \gamma ^ { 2 } \delta ^ { 2 } + ( \gamma - 1 ) ^ { 2 } L ^ { 2 }$ is a convex quadratic, and $q ^ { \prime } ( \gamma ) = 2 \gamma \delta ^ { 2 } + 2 ( \gamma - 1 ) L ^ { 2 }$ . Solving $q ^ { \prime } ( \gamma ) = 0$ yields 

$$
\gamma_ {\star} = \frac {L ^ {2}}{\delta^ {2} + L ^ {2}} \in [ 0, 1 ].
$$

Substituting this value gives 

$$
q (\gamma_ {\star}) = \frac {L ^ {2} \delta^ {2}}{\delta^ {2} + L ^ {2}}.
$$

Since $\delta _ { \gamma _ { \star } } ^ { 2 } \leq q ( \gamma _ { \star } )$ , the proof is complete. 

## B. Additional Lemma for γ-Similarity

In this section, we include one additional lemma, which characterizes the property of γ-similarity. 

Lemma 2. Under Assumption $^ { 2 , }$ consider $h ( x , y ; \gamma ) = \mathbb { E } _ { \xi } \left\| \gamma d _ { \xi } ( x , y ) - d ( x , y ) \right\| ^ { 2 }$ , where $d _ { \xi } ( x , y ) = \nabla f _ { \xi } ( x ) - \nabla f _ { \xi } ( y )$ and $d ( x , y ) = \nabla f ( x ) - \nabla f ( y )$ . Then, for a fixed $x , y \in \mathbb { R } ^ { d }$ 

$$
h (x, y; \gamma) = \gamma^ {2} \mathbb {E} _ {\xi} \left\| d _ {\xi} (x, y) - d (x, y) \right\| ^ {2} + (\gamma - 1) ^ {2} \left\| d (x, y) \right\| ^ {2}.
$$

In addition, 

1. $h ( x , y ; \gamma )$ is convex with respect to γ. 

2. $h ( x , y ; \gamma )$ is non-increasing for $\gamma < 0 ,$ and is non-decreasing $f o r \gamma > 1$ 

3. $I f \mathbb { E } _ { \xi } \| d _ { \xi } ( x , y ) \| ^ { 2 } > 0 \mathrm { ~ }$ , then the minimizer $\gamma _ { \star } : = \mathrm { a r g m i n } _ { \gamma \in \mathbb { R } } h ( x , y ; \gamma )$ is 

$$
\gamma_ {\star} = \frac {\| d (x , y) \| ^ {2}}{\mathbb {E} _ {\xi} \| d _ {\xi} (x , y) - d (x , y) \| ^ {2} + \| d (x , y) \| ^ {2}} = \frac {\| d (x , y) \| ^ {2}}{\mathbb {E} _ {\xi} \| d _ {\xi} (x , y) \| ^ {2}} \in [ 0, 1 ],
$$

and 

$$
h (x, y; \gamma_ {\star}) = \gamma_ {\star} \mathbb {E} _ {\xi} \| d _ {\xi} (x, y) - d (x, y) \| ^ {2} = (1 - \gamma_ {\star}) \| d (x, y) \| ^ {2}.
$$

$I f \mathbb { E } _ { \xi } \| d _ { \xi } ( x , y ) \| ^ { 2 } = 0 ,$ then $h ( x , y ; \gamma ) = 0 f o r { a l l } \gamma ,$ and every $\gamma \in \mathbb { R }$ is a minimizer. 

## B.1. Proof of Lemma 2

We begin by proving the first statement. From the definition of $h ( x , y ; \gamma )$ , and the Euclidean norm, for any fixed $x , y \in \mathbb { R } ^ { d }$ 

$$
\begin{array}{r l} & h (x, y; \gamma) = \mathbb {E} _ {\xi} \left\| \gamma (d _ {\xi} (x, y) - d (x, y)) + (\gamma - 1) d (x, y) \right\| ^ {2} \\ & \qquad = \gamma^ {2} \mathbb {E} _ {\xi} \left\| d _ {\xi} (x, y) - d (x, y) \right\| ^ {2} + (\gamma - 1) ^ {2} \mathbb {E} _ {\xi} \left\| d (x, y) \right\| ^ {2} + 2 \gamma (\gamma - 1) \mathbb {E} _ {\xi} \langle d _ {\xi} (x, y) - d (x, y), d (x, y) \rangle \\ & \qquad = \gamma^ {2} \mathbb {E} _ {\xi} \left\| d _ {\xi} (x, y) - d (x, y) \right\| ^ {2} + (\gamma - 1) ^ {2} \mathbb {E} _ {\xi} \left\| d (x, y) \right\| ^ {2} + 2 \gamma (\gamma - 0) \langle \mathbb {E} _ {\xi} [ d _ {\xi} (x, y) - d (x, y) ], d (x, y) \rangle . \end{array}
$$

By Assumption 2, i.e. $\mathbb { E } _ { \xi } [ d _ { \xi } ( x , y ) - d ( x , y ) ] = 0$ , and by the fact that $\mathbb { E } _ { \xi } \left\| d ( x , y ) \right\| ^ { 2 } = \left\| d ( x , y ) \right\| ^ { 2 } ,$ 

$$
h (x, y; \gamma) = \gamma^ {2} \mathbb {E} _ {\xi} \left\| d _ {\xi} (x, y) - d (x, y) \right\| ^ {2} + (\gamma - 1) ^ {2} \left\| d (x, y) \right\| ^ {2}.
$$

Next, we prove the first statement. By taking the second derivative with respect to $\gamma ,$ 

$$
\frac {d ^ {2}}{d \gamma^ {2}} h (x, y; \gamma) = 2 \mathbb {E} _ {\xi} \left\| d _ {\xi} (x, y) - d (x, y) \right\| ^ {2} + 2 \left\| d (x, y) \right\| ^ {2}.
$$

Since $\begin{array} { r } { \frac { d ^ { 2 } } { d \gamma ^ { 2 } } h ( x , y ; \gamma ) \ge 0 } \end{array}$ for any $\gamma \in \mathbb { R } , h ( x , y ; \gamma )$ is a convex function. 

Next, we prove the second statement. By taking the first derivative with respect to γ, 

$$
\frac {d}{d \gamma} h (x, y; \gamma) = 2 \gamma \mathbb {E} _ {\xi} \left\| d _ {\xi} (x, y) - d (x, y) \right\| ^ {2} + 2 (\gamma - 1) \left\| d (x, y) \right\| ^ {2}.
$$

If $\gamma < 0 .$ , then $\begin{array} { r } { \frac { d } { d \gamma } h ( x , y ; \gamma ) < 0 } \end{array}$ , thus implying that $h ( x , y ; \gamma )$ is non-increasing. 

If $\gamma > 1$ , then $\begin{array} { r } { \frac { d } { d \gamma } h ( x , y ; \gamma ) > 0 } \end{array}$ , thus meaning that $h ( x , y ; \gamma )$ is non-decreasing. 

Next, we prove the third statement. As $h ( x , y ; \gamma )$ is convex, we can find γ<sub>⋆</sub> := argmin h(x, y; γ) by solving 

$$
\gamma_ {\star} := \underset {\gamma} {\operatorname{argmin}} h (x, y; \gamma)
$$

$$
\frac {d}{d \gamma} h (x, y; \gamma) = 0.
$$

This implies that 

$$
\gamma_ {\star} = \frac {\| d (x , y) \| ^ {2}}{\mathbb {E} _ {\xi} \| d _ {\xi} (x , y) - d (x , y) \| ^ {2} + \| d (x , y) \| ^ {2}} \in [ 0, 1 ],
$$

provided that the denominator is nonzero. Substituting this value into h gives $h ( x , y ; \gamma _ { \star } ) = \gamma _ { \star } \mathbb { E } _ { \xi } \| d _ { \xi } ( x , y ) - d ( x , y ) \| ^ { 2 }$ Furthermore, since $\mathbb { E } _ { \xi } d _ { \xi } ( x , y ) = d ( x , y )$ and $\mathbb { E } _ { \xi } \left\| d _ { \xi } ( x , y ) - d ( x , y ) \right\| ^ { 2 } = \mathbb { E } _ { \xi } \left\| d _ { \xi } ( x , y ) \right\| ^ { 2 } - \left\| d ( x , y ) \right\| ^ { 2 }$ , we can show that 

$$
h (x, y; \gamma) = \gamma^ {2} \mathbb {E} _ {\xi} \left\| d _ {\xi} (x, y) \right\| ^ {2} + (1 - 2 \gamma) \left\| d (x, y) \right\| ^ {2},
$$

and that 

$$
\gamma_ {\star} = \frac {\| d (x , y) \| ^ {2}}{\mathbb {E} _ {\xi} \| d _ {\xi} (x , y) \| ^ {2}},
$$

which yields $h ( x , y ; \gamma _ { \star } ) = ( 1 - \gamma _ { \star } ) \left\| d ( x , y ) \right\| ^ { 2 }$ 

## B.2. Proof of Example 1

Consider the problem of minimizing $\begin{array} { r } { f ( x ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } f _ { i } ( x ) } \end{array}$ with each component function $f _ { i } ( x ) = { \textstyle \frac { 1 } { 2 } } x ^ { T } A _ { i } x \operatorname { f o r } x \in \mathbb { R } ^ { d }$ and $A _ { i } = \hat { L } e _ { i } e _ { i } ^ { T } \in \mathbb { R } ^ { d \times d }$ for $\hat { L } > 0$ and i being selected uniformly at random from $\{ 1 , 2 , \ldots , n \}$ . Then, 

$$
\begin{array}{l} \mathbb {E} _ {i} \left\| \gamma d _ {i} (x, y) - d (x, y) \right\| ^ {2} \\ = \mathbb {E} _ {i} \left\| (\gamma A _ {i} - \bar {A}) (x - y) \right\| ^ {2} \\ = \frac {1}{n} \sum_ {i = 1} ^ {n} \left\| (\gamma A _ {i} - \bar {A}) (x - y) \right\| ^ {2} \\ = (x - y) ^ {T} \left(\frac {1}{n} \sum_ {i = 1} ^ {n} (\gamma A _ {i} - \bar {A}) ^ {T} (\gamma A _ {i} - \bar {A})\right) (x - y) \\ \leq \lambda_ {\max} \left(\frac {1}{n} \sum_ {i = 1} ^ {n} (\gamma A _ {i} - \bar {A}) ^ {T} (\gamma A _ {i} - \bar {A})\right) \| x - y \| ^ {2}, \end{array}
$$

where $\begin{array} { r } { \bar { A } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } A _ { i } = \frac { \hat { L } } { n } I _ { n } } \end{array}$ 

Next, by the definition of $A _ { i }$ and ${ \bar { A } } ,$ , we can show that 

$$
\begin{array}{l} \gamma A _ {i} - \bar {A} \\ = \text {Diag} \left(- \frac {\hat {L}}{n}, \ldots , - \frac {\hat {L}}{n}, \underbrace {\gamma \hat {L} - \frac {\hat {L}}{n}} _ {i ^ {\text {th}} - \text {coordinate}}, - \frac {\hat {L}}{n}, \ldots , - \frac {\hat {L}}{n}\right), \end{array}
$$

and that 

$$
\begin{array}{l} \lambda_ {\max} \left(\frac {1}{n} \sum_ {i = 1} ^ {n} (\gamma A _ {i} - \bar {A}) ^ {T} (\gamma A _ {i} - \bar {A})\right) \\ = \frac {1}{n} \left[ \left(\gamma \hat {L} - \frac {\hat {L}}{n}\right) ^ {2} + (n - 1) \frac {\hat {L} ^ {2}}{n ^ {2}} \right] \\ = (\gamma^ {2} n - 2 \gamma + 1) \frac {\hat {L} ^ {2}}{n ^ {2}}. \end{array}
$$

Therefore, 

$$
\mathbb {E} _ {i} \left\| \gamma d _ {i} (x, y) - d (x, y) \right\| ^ {2} \leq \delta_ {\gamma} ^ {2} \left\| x - y \right\| ^ {2},\tag{10}
$$

where $\begin{array} { r } { \delta _ { \gamma } ^ { 2 } = \left\lceil ( \gamma ^ { 2 } n - 2 \gamma + 1 ) \frac { \hat { L } ^ { 2 } } { n ^ { 2 } } \right\rceil } \end{array}$ . This implies that the γ-similarity parameter (10) is $\begin{array} { r } { \delta _ { \gamma } ^ { 2 } = ( \gamma ^ { 2 } n - 2 \gamma + 1 ) \frac { \hat { L } ^ { 2 } } { n ^ { 2 } } } \end{array}$ , the standard similarity parameter ((10) with $\gamma = 1 )$ is $\begin{array} { r } { \delta ^ { 2 } = ( n - 1 ) \frac { \hat { L } ^ { 2 } } { n ^ { 2 } } } \end{array}$ , and the L-smoothness ((10) with $\gamma = 0 )$ is $\textstyle L ^ { 2 } = { \frac { { \hat { L } } ^ { 2 } } { n ^ { 2 } } }$ In conclusion, $\delta _ { \gamma } ^ { 2 } = \gamma ^ { 2 } \delta ^ { 2 } + ( \gamma - 1 ) ^ { 2 } L ^ { 2 }$ , and 

$$
\gamma_ {\star} := \underset {\gamma} {\operatorname{argmin}}   \gamma^ {2} \delta^ {2} + (\gamma - 1) ^ {2} L ^ {2} = \frac {1}{n},
$$

with its associated optimal value at $\begin{array} { r } { \delta _ { \gamma _ { \star } } ^ { 2 } = \frac { n - 1 } { n } \frac { \hat { L } ^ { 2 } } { n ^ { 2 } } } \end{array}$ . This implies that $\begin{array} { r } { \frac { \delta ^ { 2 } } { \delta _ { \gamma \star } ^ { 2 } } = n } \end{array}$ 

## C. Descent Lemma for MARS Algorithms

We provide the descent inequality for MARS algorithms to solve stochastic optimization problems in (1). 

We begin by introducing one useful lemma for deriving an easy-to-write stepsize bound. 

Lemma 3 (Lemma 5 of Richtarik et al. ´ (2021)). Let $a , b > 0 .$ $\begin{array} { r } { I f 0 < \eta \leq \frac { 1 } { \sqrt { a } + b } , } \end{array}$ , then $a \eta ^ { 2 } + b \eta \leq 1 .$ 

Now, we provide the descent lemma below. 

Lemma 4. Consider MARS (Algorithm 1) for solving Problem (1), where Assumptions 1, 2 hold and let $\delta _ { \gamma }$ be defined by (5). If 

$$
0 <   \eta \leq \frac {1}{\sqrt {a} + L}, \quad a = \left(\frac {(1 - \beta) ^ {3} | \gamma - 1 | ^ {2}}{\beta} L ^ {2} + 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2}\right) \frac {1}{\beta},
$$

then 

$$
\mathbb {E} [ H _ {t + 1} ] \leq \mathbb {E} [ H _ {t} ] - \frac {\eta}{2} \mathbb {E} \| \nabla f (x _ {t}) \| ^ {2} + \beta \eta \sigma^ {2},
$$

where $\begin{array} { r } { H _ { t } = f ( x _ { t } ) - f _ { \mathrm { i n f } } + \frac { \eta } { 2 \beta } \left. e _ { t } \right. ^ { 2 } a n d e _ { t } = g _ { t } - \nabla f ( x _ { t } ) ; } \end{array}$ 

Proof. Let $\mathcal { F } _ { t }$ be the history up to iteration $t ,$ i.e. $\mathcal { F } _ { t } : = \sigma ( x _ { 0 } , g _ { 0 } , \xi _ { 1 } , \xi _ { 2 } , . . . , \xi _ { t } )$ . For $t \geq 1$ , define the conditional expectation 

$$
\mathbb {E} _ {t} [ \cdot ] := \mathbb {E} [ \cdot | \mathcal {F} _ {t - 1} ],
$$

so the only randomness in $\mathbb { E } _ { t } [ \cdot ]$ comes from the fresh sample $\xi _ { t }$ . 

Step 1) Error term. Let $d _ { t } = \nabla f _ { \xi _ { t } } ( x _ { t } ) - \nabla f _ { \xi _ { t } } ( x _ { t - 1 } ) , D _ { t } = \nabla f ( x _ { t } ) - \nabla f ( x _ { t - 1 } ) , \mathrm { a n d } z _ { t } = \nabla f _ { \xi _ { t } } ( x _ { t } ) - \nabla f ( x _ { t } )$ . From the update for $g _ { t }$ 

$$
\begin{array}{r c l} {g _ {t}} & = & {(1 - \beta) (g _ {t - 1} + \gamma d _ {t}) + \beta \nabla f _ {\xi_ {t}} (x _ {t})} \\ & = & {(1 - \beta) (g _ {t - 1} + \gamma d _ {t} - D _ {t}) + (1 - \beta) D _ {t} + \beta \nabla f (x _ {t}) + \beta z _ {t}.} \end{array}
$$

Define the error $e _ { t } : = g _ { t } - \nabla f ( x _ { t } )$ . Then, 

$$
\begin{array}{r c l} e _ {t} & = & (1 - \beta) g _ {t - 1} + (1 - \beta) (\gamma d _ {t} - D _ {t}) + (1 - \beta) D _ {t} - (1 - \beta) \nabla f (x _ {t}) + \beta z _ {t} \\ & = & (1 - \beta) e _ {t - 1} + (1 - \beta) (\gamma d _ {t} - D _ {t}) + \beta z _ {t} \\ & = & (1 - \beta) e _ {t - 1} + w _ {t}, \end{array}
$$

where $w _ { t } : = ( 1 - \beta ) ( \gamma d _ { t } - D _ { t } ) + \beta z _ { t }$ 

Taking conditional expectation $\mathbb { E } _ { t } [ \cdot ]$ yields 

$$
\begin{array}{r c l} \mathbb {E} _ {t} \| e _ {t} \| ^ {2} & = & \mathbb {E} _ {t} \| (1 - \beta) e _ {t - 1} + w _ {t} \| ^ {2} \\ & = & (1 - \beta) ^ {2} \| e _ {t - 1} \| ^ {2} + 2 (1 - \beta) \langle e _ {t - 1}, \mathbb {E} _ {t} [ w _ {t} ] \rangle + \mathbb {E} _ {t} \| w _ {t} \| ^ {2}, \end{array}
$$

since $e _ { t - 1 }$ is $\mathcal { F } _ { t - 1 }$ -measurable. 

Next, from Assumption 2 (unbiasedness), 

$$
\mathbb {E} _ {t} [ w _ {t} ] = (1 - \beta) \mathbb {E} _ {t} [ \gamma d _ {t} - D _ {t} ] + \beta \mathbb {E} _ {t} [ z _ {t} ] = (1 - \beta) (\gamma - 1) D _ {t}.
$$

Therefore, 

$$
\begin{array}{r c l} \mathbb {E} _ {t} \| e _ {t} \| ^ {2} & = & (1 - \beta) ^ {2} \| e _ {t - 1} \| ^ {2} + 2 (1 - \beta) ^ {2} (\gamma - 1) \langle e _ {t - 1}, D _ {t} \rangle + \mathbb {E} _ {t} \| w _ {t} \| ^ {2} \\ & \leq & (1 - \beta) ^ {2} \| e _ {t - 1} \| ^ {2} + 2 (1 - \beta) ^ {2} | \gamma - 1 | \| e _ {t - 1} \| \| D _ {t} \| + \mathbb {E} _ {t} \| w _ {t} \| ^ {2} \\ & \leq & (1 - \beta) ^ {2} (1 + \theta) \| e _ {t - 1} \| ^ {2} + \frac {(1 - \beta) ^ {2} | \gamma - 1 | ^ {2}}{\theta} \| D _ {t} \| ^ {2} + \mathbb {E} _ {t} \| w _ {t} \| ^ {2}, \end{array}
$$

where we reach the first inequality by the Cauchy-Schwarz inequality, and the last inequality by the fact that $2 a b \leq \theta a ^ { 2 } + { \textstyle { \frac { 1 } { \theta } } } b ^ { 2 }$ with $\theta > 0$ 

To complete the bound of $\mathbb { E } _ { t } \Vert e _ { t } \Vert ^ { 2 }$ , we now bound $\mathbb { E } _ { t } \Vert w _ { t } \Vert ^ { 2 } \mathrm { . ~ }$ 

$$
\begin{array}{r c l} \mathbb {E} _ {t} \| w _ {t} \| ^ {2} & = & \mathbb {E} _ {t} \| (1 - \beta) (\gamma d _ {t} - D _ {t}) + \beta z _ {t} \| ^ {2} \\ & \leq & 2 (1 - \beta) ^ {2} \mathbb {E} _ {t} \| \gamma d _ {t} - D _ {t} \| ^ {2} + 2 \beta^ {2} \mathbb {E} _ {t} \| z _ {t} \| ^ {2} \\ & \overset {(5)} {\leq} & 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2} \| x _ {t} - x _ {t - 1} \| ^ {2} + 2 \beta^ {2} \mathbb {E} _ {t} \| z _ {t} \| ^ {2} \\ & \overset {\text {Assumption 2}} {\leq} & 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2} \| x _ {t} - x _ {t - 1} \| ^ {2} + 2 \beta^ {2} \sigma^ {2}. \end{array}
$$

Also, by L-smoothness of f (Assumption 1), 

$$
\| D _ {t} \| \leq L \| x _ {t} - x _ {t - 1} \| \Rightarrow \| D _ {t} \| ^ {2} \leq L ^ {2} \| x _ {t} - x _ {t - 1} \| ^ {2}.
$$

Thus, 

$$
\begin{array}{r c l} \mathbb {E} _ {t} \| e _ {t} \| ^ {2} & \leq & (1 - \beta) ^ {2} (1 + \theta) \| e _ {t - 1} \| ^ {2} \\ & & + \left(\frac {(1 - \beta) ^ {2} | \gamma - 1 | ^ {2}}{\theta} L ^ {2} + 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2}\right) \| x _ {t} - x _ {t - 1} \| ^ {2} + 2 \beta^ {2} \sigma^ {2}. \end{array}
$$

Taking full expectation and using $\mathbb { E } [ \mathbb { E } _ { t } [ \cdot ] ] = \mathbb { E } [ \cdot ]$ gives 

$$
\begin{array}{r c l} \mathbb {E} \| e _ {t} \| ^ {2} & \leq & (1 - \beta) ^ {2} (1 + \theta) \mathbb {E} \| e _ {t - 1} \| ^ {2} \\ & & + \left(\frac {(1 - \beta) ^ {2} | \gamma - 1 | ^ {2}}{\theta} L ^ {2} + 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2}\right) \mathbb {E} \| x _ {t} - x _ {t - 1} \| ^ {2} + 2 \beta^ {2} \sigma^ {2}. \end{array}
$$

Choose $\begin{array} { r } { \theta = \frac { \beta } { 1 - \beta } } \end{array}$ (for $\beta \in ( 0 , 1 ) )$ , so that $( 1 - \beta ) ^ { 2 } ( 1 + \theta ) = 1 - \beta$ and $\begin{array} { r } { \frac { ( 1 - \beta ) ^ { 2 } } { \theta } = \frac { ( 1 - \beta ) ^ { 3 } } { \beta } } \end{array}$ , yielding 

$$
\mathbb {E} \| e _ {t} \| ^ {2} \leq (1 - \beta) \mathbb {E} \| e _ {t - 1} \| ^ {2} + \left(\frac {(1 - \beta) ^ {3} | \gamma - 1 | ^ {2}}{\beta} L ^ {2} + 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2}\right) \mathbb {E} \| x _ {t} - x _ {t - 1} \| ^ {2} + 2 \beta^ {2} \sigma^ {2}.\tag{11}
$$

(When $\beta = 1 , g _ { t } = \nabla f _ { \xi _ { t } } ( x _ { t } )$ and $e _ { t } = z _ { t } , \mathbf { s o } \mathbb { E } \| e _ { t } \| ^ { 2 } \leq \sigma ^ { 2 }$ , and the subsequent steps still go through with $a = 0 . )$ 

Step 2) Descent inequality. By L-smoothness of f, for any $x \in \mathbb { R } ^ { d }$ and $x ^ { + } = x - \eta g$ 

$$
\begin{array}{r c l} f (x ^ {+}) & \leq & f (x) + \langle \nabla f (x), x ^ {+} - x \rangle + \frac {L}{2} \| x ^ {+} - x \| ^ {2} \\ & = & f (x) - \eta \langle \nabla f (x), g \rangle + \frac {L}{2} \| x ^ {+} - x \| ^ {2} \\ & = & f (x) - \frac {\eta}{2} \| \nabla f (x) \| ^ {2} + \frac {\eta}{2} \| g - \nabla f (x) \| ^ {2} + \left(\frac {L}{2} - \frac {1}{2 \eta}\right) \| x ^ {+} - x \| ^ {2}, \end{array}
$$

where the last line uses the identity $\begin{array} { r } { - \eta \langle \nabla f ( x ) , g \rangle = - \frac { \eta } { 2 } \| \nabla f ( x ) \| ^ { 2 } - \frac { \eta } { 2 } \| g \| ^ { 2 } + \frac { \eta } { 2 } \| g - \nabla f ( x ) \| ^ { 2 } \mathrm { a n d } \| x ^ { + } - x \| ^ { 2 } = \eta ^ { 2 } \| g \| ^ { 2 } } \end{array}$ 

Applying this with $x = x _ { t } , x ^ { + } = x _ { t + 1 }$ , and $g = g _ { t }$ gives 

$$
f (x _ {t + 1}) - f _ {\mathrm{inf}} \leq f (x _ {t}) - f _ {\mathrm{inf}} - \frac {\eta}{2} \| \nabla f (x _ {t}) \| ^ {2} + \frac {\eta}{2} \| e _ {t} \| ^ {2} + \left(\frac {L}{2} - \frac {1}{2 \eta}\right) \| x _ {t + 1} - x _ {t} \| ^ {2}.\tag{12}
$$

Define $\begin{array} { r } { H _ { t } : = f ( x _ { t } ) - f _ { \mathrm { i n f } } + A \| e _ { t } \| ^ { 2 } } \end{array}$ with $A > 0$ . Then 

$$
\begin{array}{r c l} \mathbb {E} [ H _ {t + 1} ] & = & \mathbb {E} [ f (x _ {t + 1}) - f _ {\text {inf}} ] + A \mathbb {E} \| e _ {t + 1} \| ^ {2} \\ & \overset {(1 2)} {\leq} & \mathbb {E} [ f (x _ {t}) - f _ {\text {inf}} ] - \frac {\eta}{2} \mathbb {E} \| \nabla f (x _ {t}) \| ^ {2} + \frac {\eta}{2} \mathbb {E} \| e _ {t} \| ^ {2} + \left(\frac {L}{2} - \frac {1}{2 \eta}\right) \mathbb {E} \| x _ {t + 1} - x _ {t} \| ^ {2} + A \mathbb {E} \| e _ {t + 1} \| ^ {2} \\ & \overset {(1 1)} {\leq} & \mathbb {E} [ f (x _ {t}) - f _ {\text {inf}} ] - \frac {\eta}{2} \mathbb {E} \| \nabla f (x _ {t}) \| ^ {2} + \left(\frac {\eta}{2} + A (1 - \beta)\right) \mathbb {E} \| e _ {t} \| ^ {2} + 2 \beta^ {2} A \sigma^ {2} \\ & & + \left(\frac {L}{2} - \frac {1}{2 \eta} + \left(\frac {(1 - \beta) ^ {3} | \gamma - 1 | ^ {2}}{\beta} L ^ {2} + 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2}\right) A\right) \mathbb {E} \| x _ {t + 1} - x _ {t} \| ^ {2}. \end{array}
$$

Choose $\begin{array} { r } { A = \frac { \eta } { 2 \beta } } \end{array}$ , so that $\begin{array} { r } { \frac { \eta } { 2 } + A ( 1 - \beta ) = A } \end{array}$ and $2 \beta ^ { 2 } A \sigma ^ { 2 } = \beta \eta \sigma ^ { 2 }$ . Thus, 

$$
\mathbb {E} \left[ H _ {t + 1} \right] \leq \mathbb {E} \left[ H _ {t} \right] - \frac {\eta}{2} \mathbb {E} \| \nabla f (x _ {t}) \| ^ {2} + \beta \eta \sigma^ {2} + \left(\frac {L}{2} - \frac {1}{2 \eta} + \left(\frac {(1 - \beta) ^ {3} | \gamma - 1 | ^ {2}}{\beta} L ^ {2} + 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2}\right) \frac {\eta}{2 \beta}\right) \mathbb {E} \| x _ {t + 1} - x _ {t} \| ^ {2}.
$$

Step 3) Stepsize choices. It suffices to ensure 

$$
\frac {L}{2} - \frac {1}{2 \eta} + \left(\frac {(1 - \beta) ^ {3} | \gamma - 1 | ^ {2}}{\beta} L ^ {2} + 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2}\right) \frac {\eta}{2 \beta} \leq 0,
$$

which is equivalent to 

$$
\left(\frac {(1 - \beta) ^ {3} | \gamma - 1 | ^ {2}}{\beta} L ^ {2} + 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2}\right) \frac {\eta^ {2}}{\beta} + L \eta \leq 1.
$$

By Lemma 3, this holds whenever $\begin{array} { r } { 0 < \eta \leq \frac { 1 } { \sqrt { a } + L } } \end{array}$ with 

$$
a = \left(\frac {(1 - \beta) ^ {3} | \gamma - 1 | ^ {2}}{\beta} L ^ {2} + 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2}\right) \frac {1}{\beta}.
$$

Under this stepsize, 

$$
\mathbb {E} [ H _ {t + 1} ] \leq \mathbb {E} [ H _ {t} ] - \frac {\eta}{2} \mathbb {E} \| \nabla f (x _ {t}) \| ^ {2} + \beta \eta \sigma^ {2}.
$$

## D. Convergence Results of MARS

## D.1. Proof of Theorem 1

Let $\hat { x } _ { T }$ be chosen uniformly at random from $\{ x _ { 0 } , x _ { 1 } , \dotsc , x _ { T - 1 } \}$ . Then 

$$
\begin{array}{r c l} \mathbb {E} \| \nabla f (\hat {x} _ {T}) \| ^ {2} & = & \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (x _ {t}) \| ^ {2} \\ & \stackrel {\text {Lemma 4}} {\leq} & \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \left(\frac {2}{\eta} \big (\mathbb {E} [ H _ {t} ] - \mathbb {E} [ H _ {t + 1} ] \big) + 2 \beta \sigma^ {2}\right) \\ & = & \frac {2}{\eta T} \sum_ {t = 0} ^ {T - 1} \big (\mathbb {E} [ H _ {t} ] - \mathbb {E} [ H _ {t + 1} ] \big) + 2 \beta \sigma^ {2} \\ & = & \frac {2}{\eta T} \big (\mathbb {E} [ H _ {0} ] - \mathbb {E} [ H _ {T} ] \big) + 2 \beta \sigma^ {2} \\ & \leq & \frac {2}{\eta T} \mathbb {E} [ H _ {0} ] + 2 \beta \sigma^ {2}, \end{array}
$$

since $H _ { T } \geq 0$ 

By definition, 

$$
H _ {0} = f (x _ {0}) - f _ {\mathrm{inf}} + \frac {\eta}{2 \beta} \| g _ {0} - \nabla f (x _ {0}) \| ^ {2}.
$$

If $\begin{array} { r } { g _ { 0 } = \frac { 1 } { B _ { \mathrm { i n i t } } } \sum _ { j = 1 } ^ { B _ { \mathrm { i n i t } } } \nabla f _ { \xi _ { j } } ( x _ { 0 } ) } \end{array}$ , then by Assumption Assumption 2, 

$$
\mathbb {E} \| g _ {0} - \nabla f (x _ {0}) \| ^ {2} \leq \frac {\sigma^ {2}}{B _ {\mathrm{init}}}.
$$

Hence 

$$
\mathbb {E} \| \nabla f (\hat {x} _ {T}) \| ^ {2} \leq \frac {2}{\eta T} \left(f (x _ {0}) - f _ {\mathrm{inf}} + \frac {\eta}{2 \beta} \cdot \frac {\sigma^ {2}}{B _ {\mathrm{init}}}\right) + 2 \beta \sigma^ {2}.
$$

Choose $\begin{array} { r } { B _ { \mathrm { i n i t } } = \left\lceil \frac { 1 } { \beta } \right\rceil } \end{array}$ . Then $B _ { \mathrm { i n i t } } \geq { \frac { 1 } { \beta } }$ and thus $\begin{array} { r } { \frac { 1 } { B _ { \mathrm { i n i t } } } \le \beta _ { \mathrm { : } } } \end{array}$ , so 

$$
\begin{array}{r c l} \mathbb {E} \| \nabla f (\hat {x} _ {T}) \| ^ {2} & \leq & \frac {2}{\eta T} \left(f (x _ {0}) - f _ {\mathrm{inf}} + \frac {\eta}{2} \sigma^ {2}\right) + 2 \beta \sigma^ {2} \\ & = & \frac {2}{\eta T} (f (x _ {0}) - f _ {\mathrm{inf}}) + \frac {\sigma^ {2}}{T} + 2 \beta \sigma^ {2}. \end{array}
$$

Let $\Delta : = f ( x _ { 0 } ) - f _ { \mathrm { i n f } }$ . If we choose 

$$
T = \frac {2 \Delta}{\eta} \cdot \frac {1}{\epsilon^ {2}} + \frac {\sigma^ {2}}{\epsilon^ {2}}, \quad \beta = \min \left(1, \frac {\epsilon^ {2}}{\sigma^ {2}}\right),
$$

then $\begin{array} { r } { \frac { 2 \Delta } { \eta T } \leq \epsilon ^ { 2 } , \frac { \sigma ^ { 2 } } { T } \leq \epsilon ^ { 2 } } \end{array}$ , and $2 \beta \sigma ^ { 2 } \le 2 \epsilon ^ { 2 }$ , giving 

$$
\mathbb {E} \| \nabla f (\hat {x} _ {T}) \| ^ {2} \leq 4 \epsilon^ {2}.
$$

Gradient complexity. Each iteration uses two stochastic gradients (at $x _ { t }$ and $x _ { t - 1 }$ with the same $\xi _ { t } )$ , so the total number of gradient evaluations is 

$$
B _ {\mathrm{init}} + 2 T.
$$

With $\begin{array} { r } { \beta = \operatorname* { m i n } \left( 1 , \frac { \epsilon ^ { 2 } } { \sigma ^ { 2 } } \right) } \end{array}$ we have 

$$
\frac {1}{\beta} = \max \left(1, \frac {\sigma^ {2}}{\epsilon^ {2}}\right),
$$

hence $\begin{array} { r } { B _ { \mathrm { i n i t } } = \left\lceil \frac { 1 } { \beta } \right\rceil = \mathcal { O } \Bigl ( \operatorname* { m a x } \left( 1 , \frac { \sigma ^ { 2 } } { \epsilon ^ { 2 } } \right) \Bigr ) } \end{array}$ , and 

$$
B _ {\mathrm{init}} + 2 T = \mathcal {O} \left(\max \left(1, \frac {\sigma^ {2}}{\epsilon^ {2}}\right) + \frac {\Delta}{\eta \epsilon^ {2}} + \frac {\sigma^ {2}}{\epsilon^ {2}}\right) = \mathcal {O} \left(1 + \frac {\sigma^ {2}}{\epsilon^ {2}} + \frac {\Delta}{\eta \epsilon^ {2}}\right).
$$

Plugging in $\begin{array} { r } { \eta = \frac { 1 } { L + \sqrt { a } } } \end{array}$ . I $\begin{array} { r } { \textrm { f } \eta = \frac { 1 } { L + \sqrt { a } } } \end{array}$ , then $\textstyle { \frac { 1 } { \eta } } = L + { \sqrt { a } }$ and 

$$
\sqrt {a} = \sqrt {\frac {(1 - \beta) ^ {3} | \gamma - 1 | ^ {2}}{\beta^ {2}} L ^ {2} + \frac {2 (1 - \beta) ^ {2}}{\beta} \delta_ {\gamma} ^ {2}} \leq \frac {(1 - \beta) ^ {3 / 2} | \gamma - 1 |}{\beta} L + \frac {\sqrt {2} (1 - \beta)}{\sqrt {\beta}} \delta_ {\gamma},
$$

using ${ \sqrt { u + v } } \leq { \sqrt { u } } + { \sqrt { v } } .$ . Therefore, 

$$
\frac {1}{\eta} \leq L + \frac {(1 - \beta) ^ {3 / 2} | \gamma - 1 |}{\beta} L + \frac {\sqrt {2} (1 - \beta)}{\sqrt {\beta}} \delta_ {\gamma}.
$$

Combining this with $\begin{array} { r } { \beta = \operatorname* { m i n } \left( 1 , \frac { \epsilon ^ { 2 } } { \sigma ^ { 2 } } \right) } \end{array}$ yields the stated bound 

$$
B _ {\text { init }} + 2 T \leq \mathcal {O} \left(\frac {\sigma^ {2}}{\epsilon^ {2}} + \frac {L \Delta}{\epsilon^ {2}} + \frac {\delta_ {\gamma} \Delta \sigma}{\epsilon^ {3}} + \frac {| \gamma - 1 | L \Delta \sigma^ {2}}{\epsilon^ {4}}\right).
$$

## D.2. Comparison to Existing MVR Bounds

The gradient-evaluation complexity in Theorem 1 is stated as 

$$
B _ {\mathrm{init}} + 2 T = \mathcal {O} \left(\frac {\sigma^ {2}}{\epsilon^ {2}} + \frac {L \Delta}{\epsilon^ {2}} + \frac {\delta_ {\gamma} \Delta \sigma}{\epsilon^ {3}} + \frac {| \gamma - 1 | L \Delta \sigma^ {2}}{\epsilon^ {4}}\right).
$$

In contrast, some references state the corresponding MVR bound under standard similarity in the form 

$$
\mathcal {O} \left(\frac {\sigma^ {2}}{\epsilon^ {2}} + \frac {(L + \delta) \Delta}{\epsilon^ {2}} + \frac {\delta \Delta \sigma}{\epsilon^ {3}}\right),
$$

see, e.g., Fradin et al. (2026, Appendix I, Theorem I.1). The apparent discrepancy is only the extra $\delta \Delta / \epsilon ^ { 2 }$ term: it is not a fundamentally different rate, but rather a different way to upper bound the same stepsize-dependent expression. 

Indeed, in our proof we use the stepsize choice $\eta = 1 / ( L + { \sqrt { a } } )$ with 

$$
a = \left(\frac {(1 - \beta) ^ {3} | \gamma - 1 | ^ {2}}{\beta} L ^ {2} + 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2}\right) \frac {1}{\beta},
$$

hence 

$$
\frac {1}{\eta} = L + \sqrt {a} \leq L + \frac {(1 - \beta) ^ {3 / 2} | \gamma - 1 |}{\beta} L + \frac {\sqrt {2} (1 - \beta)}{\sqrt {\beta}} \delta_ {\gamma}.
$$

With $\beta = \operatorname* { m i n } \{ 1 , \epsilon ^ { 2 } / \sigma ^ { 2 } \}$ we have $\beta ^ { - 1 / 2 } = \operatorname* { m a x } \{ 1 , \sigma / \epsilon \}$ , so a uniform (but looser) bound is obtained by using $( 1 - \beta ) \leq 1$ and σ 

$$
\max \left\{1, \frac {\sigma}{\epsilon} \right\} \leq 1 + \frac {\sigma}{\epsilon},
$$

which yields an additional contribution proportional to $\delta _ { \gamma }$ in $1 / \eta$ , and hence an additional $\delta _ { \gamma } \Delta / \epsilon ^ { 2 }$ term in the overall complexity. Concretely, one may equivalently state the following (slightly looser, but stylistically closer) bound: 

$$
B _ {\text { init }} + 2 T = \mathcal {O} \left(\frac {\sigma^ {2}}{\epsilon^ {2}} + \frac {(L + \delta_ {\gamma}) \Delta}{\epsilon^ {2}} + \frac {\delta_ {\gamma} \Delta \sigma}{\epsilon^ {3}} + \frac {| \gamma - 1 | L \Delta \sigma^ {2}}{\epsilon^ {4}}\right).
$$

When $\gamma = 1 , \delta _ { \gamma }$ becomes the standard similarity constant $\delta ,$ and the above reduces (up to constants) to the form reported in Fradin et al. (2026, Appendix I, Theorem I.1). 

Finally, note that in the variance-reduction regime $\epsilon \leq \sigma$ (equivalently $\beta = \epsilon ^ { 2 } / \sigma ^ { 2 } < 1 )$ , the extra $\delta _ { \gamma } \Delta / \epsilon ^ { 2 }$ term is always dominated by $\delta _ { \gamma } \Delta \sigma / \epsilon ^ { 3 }$ because $\sigma / \epsilon \geq 1$ . This is why we omit $\delta _ { \gamma } \Delta / \epsilon ^ { 2 }$ in the main displayed bound. 

## D.3. A T -horizon Convergence Bound for MARS

Theorem 2 (Generic T -horizon bound). Consider MARS (Algorithm 1) for solving (1). Suppose Assumptions 1, 2 hold, and let $\delta _ { \gamma }$ be the γ-similarity from Definition 1. Fix any $\beta \in ( 0 , 1 ]$ and $\gamma \in \mathbb { R } ,$ , and choose a constant stepsize $\eta > 0$ satisfying 

$$
0 <   \eta \leq \frac {1}{L + \sqrt {a}}, \quad a \triangleq \left(\frac {(1 - \beta) ^ {3} | \gamma - 1 | ^ {2}}{\beta} L ^ {2} + 2 (1 - \beta) ^ {2} \delta_ {\gamma} ^ {2}\right) \frac {1}{\beta}.
$$

Let $\hat { x } _ { T }$ be drawn uniformly at random from $\{ x _ { 0 } , \dotsc , x _ { T - 1 } \}$ . Then 

$$
\mathbb {E} \| \nabla f (\hat {x} _ {T}) \| ^ {2} \leq \frac {2}{\eta T} \left(f (x _ {0}) - f _ {\mathrm{inf}} + \frac {\eta}{2 \beta} \mathbb {E} \| g _ {0} - \nabla f (x _ {0}) \| ^ {2}\right) + 2 \beta \sigma^ {2}.\tag{13}
$$

In particular, $\begin{array} { r } { i f g _ { 0 } = \frac { 1 } { B _ { \mathrm { i n i t } } } \sum _ { j = 1 } ^ { B _ { \mathrm { i n i t } } } \nabla f _ { \xi _ { j } } ( x _ { 0 } ) } \end{array}$ with $B _ { \mathrm { i n i t } } \geq 1 / \beta$ , then $\begin{array} { r } { \mathbb { E } \| g _ { 0 } - \nabla f ( x _ { 0 } ) \| ^ { 2 } \leq \sigma ^ { 2 } / B _ { \mathrm { i n i t } } \leq \beta \sigma ^ { 2 } } \end{array}$ , and hence 

$$
\mathbb {E} \| \nabla f (\hat {x} _ {T}) \| ^ {2} \leq \frac {2 \Delta}{\eta T} + \frac {\sigma^ {2}}{T} + 2 \beta \sigma^ {2}, \quad \Delta \triangleq f (x _ {0}) - f _ {\mathrm{inf}}.\tag{14}
$$

Proof. By Lemma 4, for $\begin{array} { r } { H _ { t } = f ( x _ { t } ) - f _ { \mathrm { i n f } } + \frac { \eta } { 2 \beta } \Vert e _ { t } \Vert ^ { 2 } } \end{array}$ we have 

$$
\mathbb {E} [ H _ {t + 1} ] \leq \mathbb {E} [ H _ {t} ] - \frac {\eta}{2} \mathbb {E} \| \nabla f (x _ {t}) \| ^ {2} + \beta \eta \sigma^ {2}.
$$

Summing from t = 0 to T − 1 and using telescoping yields 

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (x _ {t}) \| ^ {2} \leq \frac {2}{\eta T} \bigl (\mathbb {E} [ H _ {0} ] - \mathbb {E} [ H _ {T} ] \bigr) + 2 \beta \sigma^ {2} \leq \frac {2}{\eta T} \mathbb {E} [ H _ {0} ] + 2 \beta \sigma^ {2},
$$

where we used $H _ { T } \geq 0$ . Finally, since $\hat { x } _ { T }$ is uniform over $\{ x _ { 0 } , \dots , x _ { T - 1 } \}$ 

$$
\mathbb {E} \| \nabla f (\hat {x} _ {T}) \| ^ {2} = \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (x _ {t}) \| ^ {2},
$$

which gives (13). The simplified bound (14) follows from $\begin{array} { r } { \mathbb { E } \| g _ { 0 } - \nabla f ( x _ { 0 } ) \| ^ { 2 } \leq \sigma ^ { 2 } / B _ { \mathrm { i n i t } } \leq \beta \sigma ^ { 2 } . } \end{array}$ 

Remark 1 (Why a random output iterate?). Lemma 4 directly gives a bound on the average $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } } \end{array}$ . Returning $\hat { x } _ { T }$ uniformly at random is a standard device to turn this average guarantee into a guarantee for a single iterate without changing the bound: $\begin{array} { r } { \mathbb { E } \| \nabla f ( \hat { x } _ { T } ) \| ^ { 2 } = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } } \end{array}$ 

## D.3.1. PROOF OF COROLLARY 1

Proof. First note that since $( 1 - \beta _ { T } ) \le 1$ , we have the crude upper bound 

$$
a = \left(\frac {(1 - \beta_ {T}) ^ {3} | \gamma - 1 | ^ {2}}{\beta_ {T}} L ^ {2} + 2 (1 - \beta_ {T}) ^ {2} \delta_ {\gamma} ^ {2}\right) \frac {1}{\beta_ {T}} \leq \frac {| \gamma - 1 | ^ {2} L ^ {2}}{\beta_ {T} ^ {2}} + \frac {2 \delta_ {\gamma} ^ {2}}{\beta_ {T}}.
$$

Hence $\begin{array} { r } { \sqrt { a } \le \frac { | \gamma - 1 | L } { \beta _ { T } } + \frac { \sqrt { 2 } \delta _ { \gamma } } { \sqrt { \beta _ { T } } } } \end{array}$ , so $\begin{array} { r } { \eta _ { T } \leq \frac { 1 } { L + \sqrt { a } } } \end{array}$ and Theorem 2 applies. Using (14) gives 

$$
\mathbb {E} \| \nabla f (\hat {x} _ {T}) \| ^ {2} \leq \frac {2 \Delta}{\eta_ {T} T} + \frac {\sigma^ {2}}{T} + 2 \beta_ {T} \sigma^ {2} = \frac {2 \Delta}{T} \left(L + \frac {| \gamma - 1 | L}{\beta_ {T}} + \frac {\sqrt {2} \delta_ {\gamma}}{\sqrt {\beta_ {T}}}\right) + \frac {\sigma^ {2}}{T} + 2 \beta_ {T} \sigma^ {2}.
$$

By the imposed horizon condition, $\beta _ { T } = \beta _ { A } + \beta _ { B } \leq 1$ , where 

$$
\beta_ {A} = \sqrt {\frac {| \gamma - 1 | L \Delta}{\sigma^ {2} T}}, \quad \beta_ {B} = 2 ^ {- 1 / 3} \left(\frac {\delta_ {\gamma} \Delta}{\sigma^ {2} T}\right) ^ {2 / 3}.
$$

Hence $\beta _ { T } \geq \beta _ { A }$ and $\beta _ { T } \geq \beta _ { B }$ . Then 

$$
\frac {2 \Delta | \gamma - 1 | L}{T \beta_ {T}} + 2 \sigma^ {2} \beta_ {A} \leq \frac {2 \Delta | \gamma - 1 | L}{T \beta_ {A}} + 2 \sigma^ {2} \beta_ {A} = \frac {4 \sigma \sqrt {| \gamma - 1 | L \Delta}}{\sqrt {T}},
$$

and similarly 

$$
\frac {2 \sqrt {2} \Delta \delta_ {\gamma}}{T \sqrt {\beta_ {T}}} + 2 \sigma^ {2} \beta_ {B} \leq \frac {2 \sqrt {2} \Delta \delta_ {\gamma}}{T \sqrt {\beta_ {B}}} + 2 \sigma^ {2} \beta_ {B} = \frac {3 \cdot 2 ^ {2 / 3} (\delta_ {\gamma} \Delta \sigma) ^ {2 / 3}}{T ^ {2 / 3}}.
$$

Combining these estimates yields (9). 

## E. Theoretical Advantage of MARS over MVR

E.1. Why the quadratic surrogate is needed 

A cruder consequence of Lemma 1 is 

$$
\delta_ {\gamma} \leq \gamma \delta + (1 - \gamma) L.
$$

Using this bound in the non-common part of the displayed complexity bound gives 

$$
A \delta_ {\gamma} + B (1 - \gamma) \leq A \left[ \gamma \delta + L \left(1 + \frac {\sigma}{\epsilon}\right) (1 - \gamma) \right].
$$

Therefore this crude certificate is no larger than the MVR term $A \delta$ only when 

$$
(1 - \gamma) \left(L \left(1 + \frac {\sigma}{\epsilon}\right) - \delta\right) \leq 0.
$$

Thus, unless $\delta \ge L ( 1 + \sigma / \epsilon )$ , this linear bound certifies no improvement over $\gamma = 1$ . This is why we use the sharper quadratic surrogate 

$$
J (\gamma) = A \sqrt {\gamma^ {2} \delta^ {2} + (1 - \gamma) ^ {2} L ^ {2}} + B (1 - \gamma),
$$

which leads to Proposition 1. 

## E.2. Proposition 1

Proposition 1. Consider the objective function with respect to $\gamma \in [ 0 , 1 ]$ 

$$
J (\gamma) = A \sqrt {\gamma^ {2} \delta^ {2} + (1 - \gamma) ^ {2} L ^ {2}} + B (1 - \gamma),
$$

where $A , B , \delta , L > 0 .$ Let $D : = \delta ^ { 2 } + L ^ { 2 }$ and $\begin{array} { r } { \gamma _ { \star } = \arg \operatorname* { m i n } _ { \gamma \in [ 0 , 1 ] } J ( \gamma ) } \end{array}$ . Then: 

$I f B \geq A \delta ,$ , then the minimum is at the boundary $\gamma _ { \star } = 1 , a n d J ( \gamma _ { \star } ) = A \delta .$ 

$I f B < A \delta$ , then the minimum is interior: 

$$
\gamma_ {\star} = \frac {L ^ {2}}{D} + \frac {B L \delta}{D \sqrt {A ^ {2} D - B ^ {2}}},
$$

and the optimal objective value satisfies 

$$
J (\gamma_ {\star}) = \frac {\delta (L \sqrt {A ^ {2} D - B ^ {2}} + B \delta)}{D} \leq A \delta .
$$

## E.3. Proof of Proposition 1

Let $D = \delta ^ { 2 } + L ^ { 2 }$ . We analyze the convexity and critical points of $J ( \gamma )$ 

1. Convexity. The function $f ( \gamma ) = \sqrt { \gamma ^ { 2 } \delta ^ { 2 } + ( 1 - \gamma ) ^ { 2 } L ^ { 2 } }$ is the Euclidean norm of the affine map $\gamma \mapsto ( \gamma \delta , ( 1 - \gamma ) L )$ Since the composition of a convex function (the norm) and an affine map is convex, $f ( \gamma )$ is convex. The term $B ( 1 - \gamma )$ is linear and thus convex. Therefore, $J ( \gamma )$ is convex on $\gamma \in [ 0 , 1 ]$ 

2. Boundary Solution $( \gamma _ { \star } = 1 )$ . Since $J ( \gamma )$ is convex, the minimum lies at the boundary $\gamma = 1$ if and only if the derivative at $\gamma = 1$ is non-positive $( J ^ { \prime } ( 1 ) \leq 0 )$ . Differentiating $J ( \gamma )$ 

$$
J ^ {\prime} (\gamma) = A \frac {2 \gamma \delta^ {2} - 2 (1 - \gamma) L ^ {2}}{2 \sqrt {\gamma^ {2} \delta^ {2} + (1 - \gamma) ^ {2} L ^ {2}}} - B = A \frac {\gamma \delta^ {2} - (1 - \gamma) L ^ {2}}{\sqrt {\gamma^ {2} \delta^ {2} + (1 - \gamma) ^ {2} L ^ {2}}} - B.
$$

Evaluating at $\gamma = 1$ : 

$$
J ^ {\prime} (1) = A \frac {1 \cdot \delta^ {2} - 0}{\sqrt {\delta^ {2}}} - B = A \delta - B.
$$

Thus, the condition for $\gamma _ { \star } = 1 { \mathrm { ~ i s ~ } } A \delta - B \leq 0 \iff B \geq A \delta .$ . In this case, $J ( 1 ) = A \delta$ 

3. Interior Solution $( \gamma _ { \star } < 1 )$ . Assume $B < A \delta$ . Since $J ^ { \prime } ( 1 ) = A \delta - B > 0 \quad$ , the minimizer cannot be $\gamma = 1$ . Moreover, $J ^ { \prime } ( 0 ) = - A L - B < 0$ , so by convexity the minimizer is the unique point in (0, 1) satisfying $J ^ { \prime } ( \gamma ) = 0$ 

Let $r ( \gamma ) = \sqrt { \gamma ^ { 2 } \delta ^ { 2 } + ( 1 - \gamma ) ^ { 2 } L ^ { 2 } } , D = \delta ^ { 2 } + L ^ { 2 }$ . The stationarity condition is 

$$
A \frac {\gamma \delta^ {2} - (1 - \gamma) L ^ {2}}{r (\gamma)} = B.
$$

Set $s = D \gamma - L ^ { 2 } = \gamma \delta ^ { 2 } - ( 1 - \gamma ) L ^ { 2 }$ . One checks that $D r ( \gamma ) ^ { 2 } = s ^ { 2 } + L ^ { 2 } \delta ^ { 2 }$ . Thus the stationarity condition gives 

$$
A ^ {2} s ^ {2} = B ^ {2} \frac {s ^ {2} + L ^ {2} \delta^ {2}}{D},
$$

or equivalently $( A ^ { 2 } D - B ^ { 2 } ) s ^ { 2 } = B ^ { 2 } L ^ { 2 } \delta ^ { 2 }$ . Since $B > 0$ and the stationary point satisfies $s > 0 ,$ , we obtain $\begin{array} { r } { s = \frac { B L \delta } { \sqrt { A ^ { 2 } D - B ^ { 2 } } } } \end{array}$ Therefore 

$$
\gamma_ {\star} = \frac {L ^ {2} + s}{D} = \frac {L ^ {2}}{D} + \frac {B L \delta}{D \sqrt {A ^ {2} D - B ^ {2}}}.
$$

The condition $B < A \delta$ implies $s < \delta ^ { 2 }$ , and hence $\gamma _ { \star } < 1 ;$ clearly $\gamma _ { \star } > 0$ 

4. Optimal Value and Inequality. Substituting γ<sub>⋆</sub> back into $J ( \gamma )$ yields the expression: 

$$
J (\gamma_ {\star}) = \frac {\delta (L \sqrt {A ^ {2} D - B ^ {2}} + B \delta)}{D}.
$$

To prove $J ( \gamma _ { \star } ) \leq A \delta$ , we rewrite the inequality as $L \sqrt { A ^ { 2 } D - B ^ { 2 } } + B \delta \leq A D$ . We apply the Cauchy-Schwarz inequality to vectors $\mathbf { u } = ( L , \delta )$ and $\mathbf { v } = ( \sqrt { A ^ { 2 } D - B ^ { 2 } } , B )$ 

$$
\mathbf {u} \cdot \mathbf {v} \leq \| \mathbf {u} \| \| \mathbf {v} \| = \sqrt {L ^ {2} + \delta^ {2}} \sqrt {(A ^ {2} D - B ^ {2}) + B ^ {2}} = \sqrt {D} \sqrt {A ^ {2} D} = A D.
$$

Dividing by D and multiplying by δ recovers $J ( \gamma _ { \star } ) \leq A \delta$ 

## F. Auxiliary CIFAR-10 Probe

Setup. We use CIFAR-10 with a small CNN containing approximately $8 . 1 \times 1 0 ^ { 5 }$ parameters. To keep the experiment close to the finite-sum stochastic optimization setting, we use the vanilla two-gradient γ-MVR update from Algorithm 1. We train for 20 epochs with batch size 128, learning rate 0.05, momentum coefficient 0.99 in the implementation, no weight decay, and a single random seed. For the checkpoint-level probe, we use a trajectory trained with $\gamma = 0 . 2 5$ and estimate local quantities every 500 steps using $M = 5 1 2$ sampled mini-batches. 

Local correction-scale estimate. At each checkpoint, we compute the full-gradient difference 

$$
d _ {t} = \nabla f (x _ {t}) - \nabla f (x _ {t - 1})
$$

and mini-batch differences 

$$
d _ {B _ {m}, t} = \nabla f _ {B _ {m}} (x _ {t}) - \nabla f _ {B _ {m}} (x _ {t - 1}), \qquad m = 1, \ldots , M.
$$

We estimate the local correction proxy 

$$
\widehat {h} _ {t} (\gamma) = \frac {1}{M} \sum_ {m = 1} ^ {M} \| \gamma d _ {B _ {m}, t} - d _ {t} \| ^ {2},
$$

and the corresponding local γ-similarity proxy 

$$
\widehat {\delta} _ {\gamma , t} ^ {2} = \frac {\widehat {h} _ {t} (\gamma)}{\| x _ {t} - x _ {t - 1} \| ^ {2}}.
$$

The local quantities 

$$
\widehat {L} _ {t} ^ {2} = \frac {\| d _ {t} \| ^ {2}}{\| x _ {t} - x _ {t - 1} \| ^ {2}}, \qquad \widehat {\delta} _ {t} ^ {2} = \frac {\frac {1}{M} \sum_ {m} \| d _ {B _ {m} , t} - d _ {t} \| ^ {2}}{\| x _ {t} - x _ {t - 1} \| ^ {2}}
$$

give the predicted minimizer 

$$
\widehat {\gamma} _ {t} ^ {\star} = \frac {\widehat {L} _ {t} ^ {2}}{\widehat {L} _ {t} ^ {2} + \widehat {\delta} _ {t} ^ {2}} = \frac {\| d _ {t} \| ^ {2}}{\frac {1}{M} \sum_ {m} \| d _ {B _ {m , t}} \| ^ {2}}.
$$


Table 1. Fixed-γ CIFAR-10 sweep. The best final training loss is attained at $\gamma = 0 . 2 5$ . Test accuracy is reported for completeness, but this setup is not tuned as a CIFAR-10 generalization benchmark.


<table><tr><td>γ</td><td>Final train loss</td><td>Final test acc.</td><td>Best test acc.</td></tr><tr><td>0</td><td>0.1925</td><td>72.90</td><td>75.23</td></tr><tr><td>0.025</td><td>0.1392</td><td>76.17</td><td>76.17</td></tr><tr><td>0.1</td><td>0.1023</td><td>74.25</td><td>74.68</td></tr><tr><td>0.25</td><td>0.0854</td><td>72.10</td><td>73.70</td></tr><tr><td>0.5</td><td>0.0911</td><td>70.37</td><td>71.79</td></tr><tr><td>0.75</td><td>0.1168</td><td>70.60</td><td>71.95</td></tr><tr><td>1</td><td>0.1415</td><td>70.69</td><td>71.72</td></tr></table>

Fixed-γ sweep. As a complementary sanity check, we run a fixed-hyperparameter sweep over 

$$
\gamma \in \{0, 0. 0 2 5, 0. 1, 0. 2 5, 0. 5, 0. 7 5, 1 \}.
$$

The final training loss is minimized at $\gamma = 0 . 2 5$ with final training loss 0.085, while the MVR setting $\gamma = 1$ gives final training loss 0.142 under the same hyperparameters. The best final test accuracy occurs at a different value, $\gamma = 0 . 0 2 5$ , so we do not interpret this sweep as a tuned generalization benchmark. Its purpose is to show that the choice of γ materially affects optimization in a theorem-aligned setting. 


Local correction proxy


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/8eaad0b6-8cb4-4ca8-8ebc-ad8b2b31742c/6c7c6113127877fe85dcdd5d25bb6dd293edba7f31c7edf8d99e8932a2971b23.jpg)



Figure 4. Local correction proxy on CIFAR-10. We plot $\widehat { h } _ { t } ( \gamma ) / \widehat { h } _ { t } ( 1 )$ at representative checkpoints. Values below 1 indicate that the scaled correction has smaller local discrepancy than the MVR correction $\gamma = 1$ . This diagnostic is a checkpoint-level proxy for the numerator of γ-similarity and is not a global estimate of $\delta _ { \gamma }$


## G. LLM pretraining

Motivation. We study how performance varies with the scaling parameter γ in a realistic LLM pretraining workload, since this is a primary modern use case. This experiment complements the theory by empirically characterizing sensitivity to γ under a standard, well-tuned training protocol and by probing whether the preferred γ appears to change across training stages, consistent with a regime-dependent tradeoff. 

Setup. We train the 124M-parameter Llama-style model (Touvron et al., 2023) following the pretraining protocol of Semenov et al. (2025) and use their public training codebase.<sup>1</sup> We run the same training pipeline and hyperparameterization, except that we train on SlimPajama (Shen et al., 2023) instead of FineWeb (Penedo et al., 2024). Since the protocol is already carefully tuned, our goal is not exhaustive hyperparameter search but to observe the behavior of different γ values under a sane and inexpensive setup; accordingly, we keep the remaining hyperparameters fixed rather than re-tuning for each γ. The training configuration is summarized in Table 2. 


Table 2. 124M LLM pretraining configuration (fixed across the sweep). We follow Semenov et al. (2025) and sweep only γ.


<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Model</td><td>Llama-style transformer, ≈ 124M params</td></tr><tr><td>Dataset</td><td>SlimPajama (Shen et al., 2023)</td></tr><tr><td>Sequence length</td><td>512</td></tr><tr><td>Batch</td><td>256 sequences (length 512)</td></tr><tr><td>Token budget</td><td>≈ 2.10B tokens</td></tr><tr><td>γ values</td><td>{1, 0.1, 0.04, 0.025, 0.02, 0.01}</td></tr><tr><td>Learning rates (AdamW; MARS)</td><td><eq>10^{-3}</eq>; <eq>3 \cdot 10^{-3}</eq></td></tr><tr><td>Betas (AdamW; MARS)</td><td>(0.8, 0.999); (0.95, 0.99)</td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td>Warmup steps</td><td>2000</td></tr><tr><td>Scheduler</td><td>cosine</td></tr></table>


Results. We report validation loss as a function of processed tokens in Figure 6.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/8eaad0b6-8cb4-4ca8-8ebc-ad8b2b31742c/09e4f64f13af46745eb5ff1c22c9c6d76bf02df8203db7fe59292c126b082442.jpg)



(a) Full training run.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/8eaad0b6-8cb4-4ca8-8ebc-ad8b2b31742c/eed0e829599aa42e30370ef514ec7acb07396c60f2e90abe776b5b59ef7af39d.jpg)



(b) Early training.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/8eaad0b6-8cb4-4ca8-8ebc-ad8b2b31742c/d7f1a735d0d0778fe4bcd0bea86515e704bb825c4892f690025a6f920e00329e.jpg)



(c) Late stage (log-scale).



Figure 6. Validation loss vs. tokens for 124M pretraining under a γ-sweep. We train a 124M Llama-style model on SlimPajama for ≈ 2.10B tokens (configuration in Table 2). Curves correspond to MARS-AdamW with different γ values and an AdamW baseline.


Figure 6 shows three views of the same training run: Figure 6a is the full trajectory, Figure 6b zooms into the beginning, and Figure 6c zooms into the end. The goal is to compare token-efficiency and final loss across γ values. Overall, the validation-loss trajectory appears sensitive to γ, with several $\gamma < 1$ settings tracking below AdamW for substantial portions of training. 

Figure 6b highlights the early stage: in our runs, smaller γ values tend to reduce validation loss more quickly as a function of tokens processed in this regime. 

Figure 6c highlights the end of training. Here, differences become subtler and the ordering among γ values can shift relative to the early stage $( \mathrm { e . g . , a \gamma } )$ that looks best early need not be best late). This suggests that as training progresses the iterates may traverse regions with different local properties, which can change which γ best balances the relevant terms in our bound (e.g., the δ -dependent term versus the (1 − γ)-dependent penalty), though the differences in final loss among small γ values are relatively small in this run. Finally, Figure 6a shows that the $\gamma = 1$ run exhibits pronounced instability under the fixed hyperparameters, indicating that the out-of-the-box $\gamma = 1$ setting may require re-tuning. 

To summarize the main observations more directly: (i) small γ values can outperform AdamW, which is visible near the end of training in Figure 6c, where several $\gamma < 1$ curves lie below AdamW; (ii) different small $\gamma$ values behave differently and there is an optimal choice, and the fact that the early-stage ordering in Figure 6b differs from the late-stage ordering in Figure 6c is consistent (though not conclusive) with stage-dependent local properties affecting which terms dominate in the bound; and (iii) the classic MVR setting $( \gamma = 1 )$ can be sensitive to hyperparameters and exhibit instability without dedicated tuning, as illustrated by the full trajectory in Figure 6a. 

## Implications.

• Small γ values can outperform AdamW and the classic MVR setting $( \gamma = 1 )$ in this workload, indicating that the choice of γ is practically important. 

• Different small γ values behave differently and there is an optimal choice; the early/late comparison in Figure 6b and Figure 6c suggests (cautiously) that changes in local properties along training can affect which γ is preferred, even though the final-loss differences among small γ values are relatively small here. 

• The classic MVR setting $( \gamma = 1 )$ is unstable under the fixed hyperparameters (Figure 6a), suggesting that $\gamma = 1$ can be hyperparameter-sensitive and may require dedicated tuning. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/8eaad0b6-8cb4-4ca8-8ebc-ad8b2b31742c/4bcf06daccd31105397f0934c3e6831dfdbda9291cb85e803c38f2fa68081f69.jpg)



Figure 7. Bound-implied MARS vs. MVR speedup (full grid). Rows correspond to $L \in \{ 0 . 1 , 1 , 1 0 \}$ , columns correspond to $\sigma ~ \in ~ \{ 1 0 ^ { - 4 } , 1 0 ^ { - 2 } , 1 \}$ . Each panel plots $\log _ { 1 0 } \bar { ( } ( A \delta ) \bar { / } J ( \gamma _ { \star } ) \big )$ on a log–log grid over ϵ (horizontal) and δ (vertical), with $\overset { \cdot } { J } ( \gamma ) =$ $A \sqrt { \gamma ^ { 2 } \delta ^ { 2 } + ( 1 - \gamma ) ^ { 2 } L ^ { 2 } } + B ( 1 - \gamma ) , A = \sigma / \epsilon ^ { 3 } , B = L \sigma ^ { 2 } / \epsilon ^ { 4 }$ (here $\Delta = 1 )$ ). Dashed curves show $\gamma _ { \star }$ contour levels.


## H. Theory visualization: bound-implied MARS vs. MVR speedup

Motivation. The analysis in the main text shows that, in a low-target-accuracy regime, MARS can attain a lower gradient/complexity bound than MVR due to the way γ trades off a $\delta _ { \gamma }$ -dependent term against a $( 1 - \gamma )$ )-dependent penalty. This appendix figure makes that tradeoff concrete by visualizing, across a wide range of problem/accuracy parameters, (i) how much the bound improves when optimizing $\gamma ,$ and (ii) which $\gamma _ { \star }$ the surrogate prefers. 

Setup (surrogate objective and normalization). For fixed $( L , \sigma , \epsilon , \delta )$ we consider the one-dimensional convex objective 

$$
J (\gamma) = A \sqrt {\gamma^ {2} \delta^ {2} + (1 - \gamma) ^ {2} L ^ {2}} + B (1 - \gamma), \quad \gamma \in [ 0, 1 ],
$$

with $\Delta = 1$ and the identifications 

$$
A = \frac {\sigma}{\epsilon^ {3}}, \qquad B = \frac {L \sigma^ {2}}{\epsilon^ {4}}.
$$

The MVR baseline corresponds to $\gamma = 1$ , in which case $J ( 1 ) = A \delta$ . We report the bound-implied speedup of optimizing γ relative to this baseline, 

$$
\log_ {1 0} \text { speedup } = \log_ {1 0} \left(\frac {A \delta}{J (\gamma_ {\star})}\right),
$$

so that $\log _ { 1 0 }$ speedup = 0 means “no improvement over $\gamma = 1 ^ { \mathfrak { s } }$ and larger values indicate stronger improvement in the surrogate. 

What is shown. Figure 7 is a $3 \times 3$ grid of $\log _ { 1 0 }$ speedup heatmaps. Rows correspond to $L \in \{ 0 . 1 , 1 , 1 0 \}$ , columns correspond to $\sigma \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 2 } , 1 \}$ . Each panel is evaluated on a log–log grid over ϵ (horizontal axis) and δ (vertical axis). Dashed curves overlay contour lines of the optimizer $\gamma _ { \star }$ (selected levels shown in the legend); smaller $\gamma _ { \star }$ indicates that the surrogate prefers a more aggressive deviation from the MVR baseline. 

Qualitative trends and interpretation. The grid exhibits three robust qualitative behaviors: 

• Larger ϵ and δ yield larger speedups. Across all panels, the heatmaps brighten toward the upper-right: increasing ϵ and $\delta$ tends to decrease the relative importance of the $( 1 - \gamma )$ -penalty term (through B) compared to the baseline $A \delta .$ making $\gamma < 1$ more beneficial in the surrogate. 

• A diagonal transition is governed by the ratio $L \sigma / ( \epsilon \delta )$ . The boundary between the dark region $( \log _ { 1 0 }$ speedup ≈ $0 ,$ $\gamma _ { \star } \approx 1 )$ and the brighter improvement region aligns approximately along diagonals in $( \epsilon , \delta )$ . This is consistent with the surrogate depending on 

$$
\frac {B}{A \delta} = \frac {L \sigma}{\epsilon \delta},
$$

so that improvement emerges when $\epsilon \delta$ is sufficiently large relative to $L \sigma$ . The $\gamma _ { \star }$ contours track this transition: $\gamma _ { \star }$ drops below 1 as one moves into the improvement regime. 

• Dependence on L and $\sigma { : }$ larger $L$ or $\sigma$ shifts the improvement regime “outward”. Increasing either $L$ (down the rows) or σ (across columns) increases $L \sigma$ and hence increases $B / ( A \delta )$ at fixed $( \epsilon , \delta )$ . In the plots, this manifests as a shift of the improvement region toward larger ϵ and/or larger $\delta ,$ and a corresponding tendency for $\gamma _ { \star }$ <sub>⋆</sub> to remain closer to 1 unless ϵδ is sufficiently large. 

Takeaway. Overall, the figure supports the qualitative message that the surrogate favors smaller $\gamma$ (and predicts larger improvements over MVR) in regimes with larger ϵ and $\delta ,$ while larger $L$ and $\sigma$ make improvement harder to realize unless ϵδ grows commensurately. In the main text we include a compact $1 \times 2$ view at fixed $L = 0 . 1$ and two representative $\sigma$ values; this appendix provides the full sweep. 