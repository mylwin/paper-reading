# Softmax is 1/2-Lipschitz: A tight bound across all $\ell _ { p }$ norms

Pravin Nair 

Department of Electrical Engineering 

pravinnair@ee.iitm.ac.in 

Indian Institute of Technology, Madras 

Reviewed on OpenReview: https: // openreview. net/ forum? id= 6dowaHsa6D 

## Abstract

The softmax function is a basic operator in machine learning and optimization, used in classification, attention mechanisms, reinforcement learning, game theory, and problems involving log-sum-exp terms. Existing robustness guarantees of learning models and convergence analysis of optimization algorithms typically consider the softmax operator to have a Lipschitz constant of 1 with respect to the $\ell _ { 2 }$ norm. In this work, we prove that the softmax function is contractive with the Lipschitz constant 1/2, uniformly across all $\ell _ { p }$ norms with $p \geq 1$ . We also show that the local Lipschitz constant of softmax attains $1 / 2$ for $p = 1$ and $p = \infty$ , and for $p \in ( 1 , \infty )$ , the constant remains strictly below 1/2 and the supremum 1/2 is achieved only in the limit. To our knowledge, this is the first comprehensive norm-uniform analysis of softmax Lipschitz continuity. We demonstrate how the sharper constant directly improves a range of existing theoretical results on robustness and convergence. We further validate the sharpness of the 1/2 Lipschitz constant of the softmax operator through empirical studies on attention-based architectures (ViT, GPT-2, Qwen3-8B) and on stochastic policies in reinforcement learning. 

## 1 Introduction

The softmax function has applications across diverse areas of machine learning and optimization. In classification models, model output logits are normalized using the softmax function to form a probability distribution (Goodfellow et al., 2016). In Transformer architectures, softmax normalizes attention scores to compute weighted combinations of input token features, enabling high-quality feature refinement (Vaswani et al., 2017). These transformer architectures have shown state-of-the-art performance in various applications in natural language processing (Tunstall et al., 2022), computer vision (Khan et al., 2022), reinforcement learning (Li et al., 2023), etc. Since softmax is the gradient of the log-sum-exp function, its properties are crucial for analyzing optimization algorithms (Gao & Pavel, 2017; Nachum et al., 2017; Nesterov, 2005). In reinforcement learning, softmax is used to convert action-value estimates (Q-values) to stochastic policies. This results in a probability distribution over actions that favours actions with higher Q-values and still assigns probabilities to suboptimal actions. In entropy-regularized reinforcement learning, the softmax operator arises naturally as the solution of the policy optimization problem (Sutton & Barto, 2018; Nachum et al., 2017). 

The softmax operator maps any real vector to a probability distribution. Formally, the softmax function $\sigma _ { \lambda } : \mathbb { R } ^ { n } \to \Delta _ { n } ^ { \circ }$ with inverse–temperature parameter $\lambda > 0$ is defined as 

$$
\sigma_ {\lambda} (\boldsymbol {x}) _ {i} = \frac {\exp (\lambda x _ {i})}{\sum_ {j = 1} ^ {n} \exp (\lambda x _ {j})}, \qquad i = 1, \ldots , n,\tag{1}
$$

where $\Delta _ { n } = \{ { \pmb u } \in \mathbb { R } ^ { n } : u _ { i } \geq 0 , \sum _ { i } u _ { i } = 1 \}$ is the probability simplex and $\begin{array} { r } { \Delta _ { n } ^ { \circ } = \{ \pmb { u } \in \mathbb { R } ^ { n } : u _ { i } > 0 , \sum _ { i } u _ { i } = 1 \} } \end{array}$ denotes its interior. Since $\exp ( x _ { i } ) > 0$ for all $x _ { i } \in \mathbb { R } , \sigma _ { \lambda } ( { \pmb x } )$ never attains the boundary of the simplex $\partial \Delta _ { n } = \{ \pmb { u } \in \mathbb { R } ^ { n } : u _ { i } = 0$ for some $i \}$ . The coeficient λ measures the smoothness of the output distribution. A larger λ makes the distribution more peaked on the largest entries, and a smaller λ makes it smoother with more evenly spread probability values. The standard softmax function is when $\lambda = 1$ 

Accurate analysis of softmax’s Lipschitz constant has important applications across machine learning and optimization. It enables accurate robustness analysis of attention mechanisms, where Lipschitz bounds can be used to set hyperparameters that ensure stable training (Qi et al., 2023). In game-theoretic learning, modeling action selection via softmax allows Lipschitz continuity to be directly leveraged for establishing convergence rates toward Nash equilibria (Gao & Pavel, 2017). Many existing generalization results for learning models require computing the Lipschitz constant of the loss function with respect to either network input or network parameters. When softmax appears in the final layer or in attention, an accurate Lipschitz analysis becomes essential for deriving sharp generalization bounds (Asadi & Abbe, 2020). Moreover, when a model is expressed as a composition of layers, standard network analyses bound the global Lipschitz constant by an appropriate composition (often a product) of layer-wise constants Bartlett et al. (2017); Miyato et al. (2018); Virmaux & Scaman (2018); Fazlyab et al. (2019). In such bounds, the softmax Lipschitz constant acts as a multiplicative factor, so tightening it leads directly to sharper network-level guarantees. In classification networks, robustness and domain adaptation depend on how sensitive output probabilities are to input perturbations, which is again controlled by the Lipschitz constant of softmax (Chen et al., 2022). The log sum-exp (LSE) function is a smooth approximation of the maximum operator, and the gradient of LSE is the softmax function. Hence, in optimization problems with LSE-based objective functions, step-size rules for gradient descent and smoothness-based convergence rates can be sharpened using a tight softmax Lipschitz bound. Lipschitz constant of softmax is also used in analysing several other frameworks, including meta learning (Jeon et al., 2024), structural causal models (Le Priol et al., 2021), prior-data fitted networks (Nagler, 2023), and sparse training (Lei et al., 2024). The above-mentioned diverse set of applications highlights tha deriving a tight Lipschitz constant of softmax has a substantial practical impact. 

Despite the theoretical and empirical significance of softmax’s Lipschitz constant, an accurate analysis is missing in the literature. A number of works (Gao & Pavel, 2017; Laha et al., 2018; Asadi & Abbe, 2020; Gouk et al., 2021; Le Priol et al., 2021; Chen et al., 2022; Nagler, 2023; Jeon et al., 2024; Lei et al., 2024) assume the softmax function to be 1-Lipschitz with respect to the $\ell _ { 2 }$ norm. This assumption is often attributed to Proposition 4 in Gao & Pavel (2017). In some works, this is mentioned as a straightforward fact, perhaps because showing the Lipschitz constant to be bounded by 1 is trivial. We also note that Xu et al. (2022) reports a bound of $1 / 4$ for the softmax’s Lipschitz constant. The main motivation for this work is the discrepancy between the commonly assumed Lipschitz constant of the softmax function and the smaller values observed in our empirical studies on transformer architectures. This prompted us to derive a tighter bound for the Lipschitz constant of the softmax function. In this regard, our contributions are as follows: 

• We prove that softmax is $1 / 2 \AA$ -Lipschitz across all $\ell _ { p }$ norms $( p \geq 1 )$ , thereby improving the usually assumed bound of 1. 

• Furthermore, we show the tightness of this bound by proving that the local Lipschitz constant is attained for $p = 1$ and $p = \infty$ as $1 / 2 ,$ , while for $p \in ( 1 , \infty )$ the local Lipschitz constants remain strictly below $1 / 2$ and the supremum is only approached in the limit. 

• We demonstrate that our tight Lipschitz bound improves the existing theoretical results in robustness and convergence. We also validate our derived Lipschitz constant through experiments on large-scale transformer models and reinforcement learning policies. 

## 2 Preliminaries

In this section, we provide the necessary background for our main results. We start with the definition of the $\ell _ { p }$ norm for vectors and the corresponding induced operator norm for matrices. 

Definition $2 . 1 \ ( \ell _ { p } \ \mathrm { n o r m } )$ . For a vector $\pmb { x } = ( x _ { 1 } , x _ { 2 } , \ldots , x _ { n } ) \in \mathbb { R } ^ { n }$ and for $1 \leq p < \infty$ , the $\ell _ { p }$ norm of x is defined as 

$$
\| \boldsymbol {x} \| _ {p} = \left(\sum_ {i = 1} ^ {n} | x _ {i} | ^ {p}\right) ^ {1 / p}.
$$

For $p = \infty$ , the $\ell _ { \infty }$ norm is defined as 

$$
\| \boldsymbol {x} \| _ {\infty} = \max _ {1 \leq i \leq n} | x _ {i} |.
$$

We can also define the operator norm of a matrix $\pmb { A } = ( A _ { i j } ) \in \mathbb { R } ^ { m \times n }$ induced by the $\ell _ { p }$ norm as 

$$
\| \boldsymbol {A} \| _ {p} := \sup _ {\boldsymbol {v} \neq 0} \frac {\| \boldsymbol {A} \boldsymbol {v} \| _ {p}}{\| \boldsymbol {v} \| _ {p}} = \sup _ {\| \boldsymbol {v} \| _ {p} = 1} \| \boldsymbol {A} \boldsymbol {v} \| _ {p}.
$$

Throughout the paper, we overload $\| \cdot \| _ { p }$ to represent both the vector norm and the corresponding induced norm on matrices. Next, we define the Lipschitz property of functions. 

Definition 2.2 (Lipschitz continuity in $\ell _ { p }$ norm). Let $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ and let $\| \cdot \| _ { p }$ denote the $\ell _ { p }$ norm with $1 \leq p \leq \infty$ . The function f is said to be Lipschitz continuous with respect to $\| \cdot \| _ { p }$ if there exists a constant $L _ { p } \geq 0$ such that 

$$
\| f (\boldsymbol {x}) - f (\boldsymbol {y}) \| _ {p} \leq L _ {p} \| \boldsymbol {x} - \boldsymbol {y} \| _ {p}, \quad \forall \boldsymbol {x}, \boldsymbol {y} \in \mathbb {R} ^ {n}.\tag{2}
$$

The smallest such constant $L _ { p }$ is called the Lipschitz constant of f with respect to the $\ell _ { p }$ norm. 

A mapping f is contractive if $L _ { p } < 1$ , and non-expansive if $L _ { p } \leq 1$ . It is firmly non-expansive if $\| f ( { \pmb x } ) -$ $f ( { \pmb y } ) \| _ { p } ^ { 2 } \le \langle f ( { \pmb x } ) - f ( { \pmb y } ) , { \pmb x } - { \pmb y } \rangle$ for all x, y, and co-coercive with constant $\beta > 0 { \mathrm { ~ i f ~ } } \langle f ( { \pmb x } ) - f ( { \pmb y } ) , { \pmb x } - { \pmb y } \rangle \geq$ $\beta \| f ( \dot { \pmb x } ) - f ( \pmb y ) \| _ { p } ^ { 2 }$ 

Note that the Lipschitz constant in Eq. 2 is a global quantity, since the inequality holds for all $\ b { x } , \ b { y } \in \mathbb { R } ^ { n }$ By definition, $L _ { p }$ can be characterized as, 

$$
L _ {p} = \sup _ {\boldsymbol {x}, \boldsymbol {y} \in \mathbb {R} ^ {n}, \boldsymbol {x} \neq \boldsymbol {y}} \frac {\| f (\boldsymbol {x}) - f (\boldsymbol {y}) \| _ {p}}{\| \boldsymbol {x} - \boldsymbol {y} \| _ {p}}.\tag{3}
$$

We next introduce the notion of a local Lipschitz constant (Rockafellar & Wets, 1998) via the Jacobian of $f ,$ which in turn provides another characterization of the global Lipschitz constant. 

Definition 2.3 (Jacobian matrix (Boyd & Vandenberghe, 2004)). Let $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ be a diferentiable function, where ${ \pmb f } ( { \pmb x } ) = \big ( f _ { 1 } ( { \pmb x } ) , f _ { 2 } ( { \pmb x } ) , \dots , f _ { m } ( { \pmb x } ) \big )$ and each $f _ { i } : \mathbb { R } ^ { n }  \mathbb { F }$ is scalar-valued function. The Jacobian matrix of $f$ at $\mathbf { \delta x } .$ denoted $\pmb { J } _ { f } ( \pmb { x } ) \in \mathbb { R } ^ { m \times n }$ , is the matrix of all partial derivatives, 

$$
\boldsymbol {J} _ {f} (\boldsymbol {x}) = \left[ \begin{array}{c c c} \frac {\partial f _ {1}}{\partial x _ {1}} (\boldsymbol {x}) & \dots & \frac {\partial f _ {1}}{\partial x _ {n}} (\boldsymbol {x}) \\ \vdots & \ddots & \vdots \\ \frac {\partial f _ {m}}{\partial x _ {1}} (\boldsymbol {x}) & \dots & \frac {\partial f _ {m}}{\partial x _ {n}} (\boldsymbol {x}) \end{array} \right].
$$

If the function is continuously diferentiable, the local Lipschitz constant of a function at any $\pmb { x } \in \mathbb { R } ^ { n }$ is equal to the $\ell _ { p }$ norm of the Jacobian at that point. 

Definition 2.4 (Local Lipschitz constant (Rockafellar & Wets, 1998)). Let $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ . The local Lipschitz constant of f at a point $\pmb { x } \in \mathbb { R } ^ { n }$ with respect to the $\ell _ { p }$ norm is defined as 

$$
L _ {p} (\boldsymbol {x}) := \limsup _ {\boldsymbol {y} \to \boldsymbol {x},   \boldsymbol {y} \neq \boldsymbol {x}} \frac {\| f (\boldsymbol {y}) - f (\boldsymbol {x}) \| _ {p}}{\| \boldsymbol {y} - \boldsymbol {x} \| _ {p}}.
$$

Intuitively, $L _ { p } ( { \pmb x } )$ characterizes the tightest Lipschitz bound in an arbitrarily small neighborhood of x. If f is diferentiable at x, then 

$$
L _ {p} (\pmb {x}) = \| \pmb {J} _ {f} (\pmb {x}) \| _ {p},
$$

where $\| J _ { f } ( \pmb { x } ) \| _ { p }$ denotes $\ell _ { p }$ -induced operator norm of the Jacobian matrix. 

We now relate the global Lipschitz constant of a function to its local Lipschitz constant, following a result from Hytönen et al. (2016). 

Lemma 1 (Lipschitz constant via the Jacobian). Let $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ be continuously diferentiable. Then, for $1 \leq p \leq \infty$ , the global Lipschitz constant of f with respect to $\ell _ { p }$ norm is given as 

$$
L _ {p} = \sup _ {\boldsymbol {x} \in \mathbb {R} ^ {n}} L _ {p} (\boldsymbol {x}) = \sup _ {\boldsymbol {x} \in \mathbb {R} ^ {n}} \| \boldsymbol {J} _ {f} (\boldsymbol {x}) \| _ {p}.
$$

Thus, the global Lipschitz constant of a function can be expressed in a variational form involving the $\ell _ { p } -$ induced operator norm of its Jacobian matrix. To apply this principle to the softmax operator, we recall its Jacobian formulation from Gao & Pavel (2017), as stated below. 

Lemma 2 (Jacobian of the softmax). Let $\sigma _ { \lambda } : \mathbb { R } ^ { n } \to \Delta _ { n } ^ { \circ }$ be the softmax function as in Definition 1 and let $\pmb { s } = \sigma _ { \lambda } ( \pmb { x } )$ . Then, the Jacobian of $\sigma _ { \lambda }$ at x is 

$$
\boldsymbol {J} _ {\sigma_ {\lambda}} (\boldsymbol {x}) = \lambda \bigl (\operatorname{Diag} (\boldsymbol {s}) - \boldsymbol {s s} ^ {\top} \bigr),
$$

so, in coordinates, $i f s = \{ s _ { 1 } , s _ { 2 } , . . . s _ { n } \}$ , 

$$
\left[ \boldsymbol {J} _ {\sigma_ {\lambda}} (\boldsymbol {x}) \right] _ {i j} = \left\{ \begin{array}{l l} \lambda   s _ {i} (1 - s _ {i}), & i = j, \\ - \lambda   s _ {i} s _ {j}, & i \neq j. \end{array} \right.
$$

Refer to Proposition 2 in Gao & Pavel (2017) for the proof of Lemma 2. In Section 3, we will make use of Lemma 1 and the result in Lemma 2 for the softmax’s Jacobian to establish our main results. 

## 3 Main Results

We begin by deriving an inequality result for $\ell _ { p } .$ -induced operator norms for matrices. 

Proposition 1 (Norm Interpolation). Let $A \in \mathbb { R } ^ { n \times n }$ . Then: 

(a) $\begin{array} { r } { \| \pmb { A } \| _ { 1 } = \operatorname* { m a x } _ { 1 \leq j \leq n } \ \sum _ { i = 1 } ^ { n } | A _ { i j } | . } \end{array}$ 

(b) $\begin{array} { r } { \| \pmb { A } \| _ { \infty } = \operatorname* { m a x } _ { 1 \leq i \leq n } \sum _ { j = 1 } ^ { n } | A _ { i j } | . } \end{array}$ 

(c) For $1 < p < \infty _ { ; }$ 

$$
\| \boldsymbol {A} \| _ {p} \leq \| \boldsymbol {A} \| _ {1} ^ {1 / p} \| \boldsymbol {A} \| _ {\infty} ^ {1 - 1 / p}.
$$

Proposition 1(a) and (b) state the well-known results that the $\ell _ { 1 }$ -induced operator norm of a matrix is equal to its maximum absolute column sum, and the $\ell _ { \infty } .$ -induced operator norm is equal to the maximum absolute row sum (Golub & Van Loan, 2013). Proposition 1(c) establishes an interpolation inequality that upper bounds the $\ell _ { p } .$ -induced operator norm of a matrix in terms of $\| A \| _ { 1 }$ <sub>1</sub> and $\| A \| _ { \infty }$ . While this inequality is a special case of the Riesz–Thorin interpolation theorem (Riesz, 1927; Stein, 1956; Thorin, 1939), we provide a self-contained proof in the Appendix A.1 for clarity. Importantly, Proposition 1(c) forms the key technical tool for this section, as it enables us to derive norm-uniform bounds on the Lipschitz constant of the softmax operator. In particular, we reformulate the optimization problem in Lemma 1 using the Jacobian form of the softmax provided in Lemma 2. 

Lemma 3 (Lipschitz constant of the softmax operator). Let $\sigma _ { \lambda } : \mathbb { R } ^ { n } \to \Delta _ { n } ^ { \circ }$ be the softmax function as in Definition $^ { 1 , }$ and let $\pmb { s } = \sigma _ { \lambda } ( \pmb { x } )$ . Then, for any $1 \leq p \leq \infty$ , the Lipschitz constant of $\sigma _ { \lambda }$ with respect to $\ell _ { p }$ norm is given by 

$$
L _ {p} = \lambda \sup _ {\boldsymbol {s} \in \Delta_ {n} ^ {\circ}} \left\| \mathrm{Diag} (\boldsymbol {s}) - \boldsymbol {s s} ^ {\top} \right\| _ {p}.\tag{4}
$$

The key implication of Lemma 3 is that the global Lipschitz constant reduces to the supremum of the induced matrix norm in $\operatorname { E q }$ . 4 over the interior of the probability simplex. The boundary points of the simplex, $\partial \Delta _ { n } .$ do not influence the formulation of the Lipschitz constant, since all components of the softmax output are strictly positive. 

Next, we combine the interpolation inequality from Proposition 1 with the Jacobian formulation of the softmax operator from Lemma 3 to establish its Lipschitz constant across all $\ell _ { p }$ norms. This result is summarized as the main contribution of our work in Theorem 1. 

Theorem 1 (Lipschitz constant of softmax function). Irrespective of the $\ell _ { p }$ norms $( p \ge 1 )$ , 

(a) $\begin{array} { r } { \| \pmb { J } _ { \sigma _ { 1 } } ( \pmb { x } ) \| _ { p } \ \leq \ \frac { 1 } { 2 } } \end{array}$ 

$$
(b) \| \sigma_ {\lambda} (\boldsymbol {x}) - \sigma_ {\lambda} (\boldsymbol {y}) \| _ {p} \leq \frac {\lambda}{2} \| \boldsymbol {x} - \boldsymbol {y} \| _ {p}
$$

The upper bound on the Jacobian of the softmax operator across $\ell _ { p }$ norms, established in Theorem 1, directly provides the global Lipschitz constant of the softmax function. Refer to Appendix A.2 for a detailed proof. A natural question arises that while many works assume the constant to be 1, we show it is in fact $1 / 2 ;$ but could it be even smaller, for instance $1 / 4$ as claimed by Xu et al. (2022). Proposition 2 resolves this question by proving that $\lambda / 2$ is indeed the tight global Lipschitz constant of the softmax operator for any $\lambda > 0$ 

Proposition 2 (Tightness of the Lipschitz constant for softmax). Consider the optimization problem, 

$$
\sup _ {\boldsymbol {x} \in \mathbb {R} ^ {n}} \| \boldsymbol {J} _ {\sigma_ {1}} (\boldsymbol {x}) \| _ {p} = \sup _ {\boldsymbol {s} \in \Delta_ {n} ^ {\circ}} \| \operatorname{Diag} (\boldsymbol {s}) - \boldsymbol {s s} ^ {\top} \| _ {p},
$$

where $\Delta _ { n } ^ { \circ }$ is the interior of the probability simplex. 

(a) For $p = 1$ or $p = \infty$ , the supremum is $1 / 2$ and is attained at an interior point in $\Delta _ { n } ^ { \circ }$ . In particular, $f o r \pmb { x } = ( \ln ( n - 1 ) , 0 , 0 , \dots , 0 ) \in \mathbb { R } ^ { n }$ , the softmax output $\pmb { s } = \sigma _ { 1 } ( \pmb { x } )$ satisfies $s \in \Delta _ { n } ^ { \circ }$ and 

$$
\| \mathrm{Diag} (\pmb {s}) - \pmb {s s} ^ {\top} \| _ {p} = \frac {1}{2}.
$$

(b) For $p \in ( 1 , \infty )$ , the supremum value is again $1 / 2$ but is not attained in $\Delta _ { n } ^ { \circ }$ for $n > 2$ . Instead, there exists a sequence $\{ s _ { k } \} _ { k \geq 1 } \subset \Delta _ { n } ^ { \circ }$ converging to the boundary of the probability simplex $\partial \Delta _ { n }$ such that 

$$
\lim _ {k \to \infty} \| \mathrm{Diag} (\pmb {s} _ {k}) - \pmb {s} _ {k} \pmb {s} _ {k} ^ {\top} \| _ {p} = \frac {1}{2},
$$

with $\begin{array} { r } { \pmb { s } _ { k } = \sigma _ { 1 } ( \pmb { x } _ { k } ) } \end{array}$ for some $\pmb { x } _ { k } \in \mathbb { R } ^ { n }$ . For $n = 2$ , the supremum $\frac { 1 } { 2 }$ is attained at a point in $\Delta _ { 2 } ^ { \circ }$ 

The proof of Proposition 2 is deferred to Appendix A.3. In particular, part (b) relies on the fact that for $1 <$ $p < \infty$ , the interpolation inequality in Proposition 1 is indeed strict for $\scriptstyle J _ { \sigma _ { 1 } } ( { \pmb x } )$ for all $\mathbf { x } \in \mathbb { R } ^ { n }$ . This is because the supremum in the optimization problem of Proposition 2 can be attained only on the boundary points of the probability simplex. Specifically, the extremal points are the permutations of $( 1 / 2 , 1 / 2 , 0 , \ldots , 0 )$ , which lie in $\partial \Delta _ { n }$ 

Remark. There are multiple works showing that the Lipschitz constant of the softmax operator with respect to the $\ell _ { 2 }$ norm is upper bounded by $1 / 2 .$ . Alghamdi et al. (2022) upper bounds $\| J _ { \sigma _ { 1 } } ( \pmb { x } ) \| _ { 2 }$ by first controlling the Frobenius norm of the softmax Jacobian (using $\| A \| _ { 2 } \leq \| A \| _ { F } )$ and then maximising this bound over the simplex to obtain the same constant $1 / 2$ . Yudin et al. (2025) analyse the Jacobian matrix $M ( s ) =$ dia $\mathbf { \boldsymbol { s } } ( \mathbf { \boldsymbol { s } } ) - \mathbf { \boldsymbol { s } } \mathbf { \boldsymbol { s } } ^ { \intercal }$ on the simplex and upper bound its spectral norm via an eigenvalue argument (reducing the worst case to a distribution supported on two classes), yielding $\| M ( s ) \| _ { 2 } \le 1 / 2$ . Similarly, Newhouse (2025) gives an independent proof by applying Gershgorin’s circle theorem to $\mathrm { d i a g } ( s ) - s s ^ { \top }$ , and also shows tightness in the $\ell _ { 2 }$ setting. In contrast, we prove that the bound $1 / 2$ is tight uniformly across all $\ell _ { p }$ norms for $p \geq 1$ , and give a detailed analysis of attainability on the simplex for diferent $p .$ 

We next provide an example of a pair of points for which the empirically computed Lipschitz ratio approaches $1 / 2$ . This gives empirical evidence for the tightness of our theoretical bound. 

Example 1 (Tightness of the Lipschitz bound). Let $K = 2 0$ and $\varepsilon = 1 0 ^ { - 4 }$ . Consider the vectors 

$$
\boldsymbol {x} = (0, 0, - K, - K, \dots , - K) \in \mathbb {R} ^ {1 0}, \qquad \boldsymbol {y} = \boldsymbol {x} + \varepsilon \boldsymbol {v},
$$

where v is eigenvector corresponding to the maximum eigenvalue of $J _ { \sigma _ { 1 } } ( { \pmb x } )$ . Then the ratio 

$$
\frac {\| \sigma_ {1} (\boldsymbol {y}) - \sigma_ {1} (\boldsymbol {x}) \| _ {p}}{\| \boldsymbol {y} - \boldsymbol {x} \| _ {p}}
$$

evaluates approximately to 0.49999999504472 for all $p > = 1$ . This demonstrates a concrete pair $( { \pmb x } , { \pmb y } )$ where the Lipschitz constant of the softmax function is nearly attained. We can extend the example to any dimension by adding −K as a value to the other added dimensions. 

## 4 Implications of the Improved Softmax Lipschitz Constant

Our result in Theorem 1 establishes that the softmax function is $\lambda / 2 .$ -Lipschitz with respect to all $\ell _ { p }$ norms for $p \geq 1$ . This tighter characterization enables us to revisit existing results where softmax Lipschitz continuity is relevant, leading to sharper constants or simplified identities. Next, we illustrate this in a few representative settings. 

## 4.1 Refinement of (Gao & Pavel, 2017, Cor. 3).

Leveraging our sharper Lipschitz estimate, Corollary 3 in Gao & Pavel (2017) can be strengthened as follows. Corollary 1 (Softmax regularity with improved constants). For any $\lambda > 0$ , the softmax map $\sigma _ { \lambda } : \mathbb { R } ^ { n } \to \Delta _ { n } ^ { \circ }$ satisfies 

$$
\begin{array}{r l} (L i p s c h i t z) & \| \sigma_ {\lambda} (\boldsymbol {x}) - \sigma_ {\lambda} (\boldsymbol {y}) \| _ {p} \leq \frac {\lambda}{2}   \| \boldsymbol {x} - \boldsymbol {y} \| _ {p}, \qquad \forall p \geq 1, \\ (c o - c o e r c i v e) & \langle \sigma_ {\lambda} (\boldsymbol {x}) - \sigma_ {\lambda} (\boldsymbol {y}),   \boldsymbol {x} - \boldsymbol {y} \rangle \geq \frac {2}{\lambda}   \| \sigma_ {\lambda} (\boldsymbol {x}) - \sigma_ {\lambda} (\boldsymbol {y}) \| _ {2} ^ {2}. \end{array}
$$

In particular, 

$\sigma _ { \lambda }$ is nonexpansive and firmly nonexpansive for $\lambda \in ( 0 , 2 ]$ ; 

$\sigma _ { \lambda }$ is contractive $f o r \lambda \in ( 0 , 2 )$ 

The proof of this corollary follows directly from the analysis in Gao & Pavel (2017), with the Lipschitz constant of the softmax operator replaced by the sharper value $\lambda / 2$ 

## 4.2 Lipschitz Analysis of Attention variant

The self-attention module (Vaswani et al., 2017) refines an input feature matrix $\ b { X } \in \mathbb { R } ^ { n \times d }$ by projecting it into queries, keys, and values via learnable matrices $\pmb { W } ^ { Q } \in \mathring { \mathbb { R } } ^ { d \times d _ { k } } , \pmb { W } ^ { K } \in \mathbb { R } ^ { d \times d _ { k } }$ , and $W ^ { V } \in \overline { { \mathbb { R } ^ { d \times d _ { v } } } }$ . The attention operation is then given by 

$$
\mathrm{Att} (\boldsymbol {X}; \boldsymbol {W} ^ {Q}, \boldsymbol {W} ^ {K}, \boldsymbol {W} ^ {V}) = \mathrm{softmax} \Bigl (\frac {\boldsymbol {X} \boldsymbol {W} ^ {Q} \left(\boldsymbol {X} \boldsymbol {W} ^ {K}\right) ^ {\top}}{\sqrt {d _ {k}}} \Bigr) \boldsymbol {X} \boldsymbol {W} ^ {V},\tag{5}
$$

where softmax(·) denotes the row-wise application of the function $\sigma _ { 1 }$ . Although we obtain a tight global Lipschitz constant for the softmax operator $\sigma _ { 1 }$ acting on the rows of the attention score matrix, the self-attention map ${ \pmb X } \mapsto \mathrm { A t t } ( { \pmb X } ; { \pmb W } ^ { Q } , { \pmb W } ^ { K } , { \pmb W } ^ { V } )$ defined on an unbounded input domain is not globally Lipschitz. Kim et al. (Kim et al., 2021) show that, for this map, the Lipschitz constant with respect to any $\ell _ { p }$ norm is infinite. Motivated by this negative result, Qi et al. (2023) propose scaled cosine similarity attention (SCSA) as a Lipschitz-controlled variant of self-attention, which normalizes the projections $X W ^ { Q }$ and $X { \dot { W } } ^ { K }$ to unit norm and applies an inverse temperature coeficient $\tau > 0$ 

$$
\operatorname{SCSA} (\boldsymbol {X}; \boldsymbol {W} ^ {Q}, \boldsymbol {W} ^ {K}, \boldsymbol {W} ^ {V}, \nu , \tau) = \nu \operatorname{softmax} \Bigl (\tau \frac {\boldsymbol {X} \boldsymbol {W} ^ {Q} \left(\boldsymbol {X} \boldsymbol {W} ^ {K}\right) ^ {\top}}{\| \boldsymbol {X} \boldsymbol {W} ^ {Q} \| \| \boldsymbol {X} \boldsymbol {W} ^ {K} \|} \Bigr) \boldsymbol {V}.\tag{6}
$$

SCSA can be viewed as a Lipschitz variant of the standard attention mechanism. Theorem 1 in Qi et al. (2023) establishes an Lipschitz bound for scaled cosine similarity attention (SCSA), but their bound depends on the choice $( n - 1 ) / n$ as a Lipschitz constant for the softmax operator. Using our sharper analysis of the softmax Lipschitz constant, we can replace $( n - 1 ) / n$ by $1 / 2$ in their argument. Thus Lipschitz bound for SCSA can be tightened as shown below. 

Theorem 2 (Refined ℓ -Lipschitz bound for SCSA). Let $L _ { 2 } ( \mathrm { S C S A } )$ denote the global Lipschitz constant of $X \mapsto \operatorname { S C S A } ( X , W ^ { Q } , W ^ { K } , W ^ { V } , \nu , \tau )$ with respect to the $\ell _ { 2 }$ norm. Then, for fixed $( W ^ { Q } , { \bar { W } } ^ { K } , W ^ { V } , \nu , \tau )$ 

$$
L _ {2} (\mathrm{SCSA}) \leq n ^ {2} \nu \tau \varepsilon^ {- 1 / 2} \| \boldsymbol {W} ^ {K} \| _ {2} + n \nu \tau \varepsilon^ {- 1 / 2} \| \boldsymbol {W} ^ {Q} \| _ {2} + 2 n \nu \varepsilon^ {- 1 / 2} \| \boldsymbol {W} ^ {V ^ {\top}} \| _ {2}.
$$

In particular, compared with the Lipschitz bound stated in Theorem 1 of Qi et al. (2023), Theorem 2 improves the estimate by a factor of 2, removing the extra factor 2 that appears in their original bound Such refinements can directly impact theoretical robustness guarantees for attention-based architectures. 

## 4.3 Entropy-regularized zero-sum games via double-softmax fixed point

Consider a two-player zero-sum matrix game with a payof matrix $\pmb { A } \in \mathbb { R } ^ { n \times m }$ . The standard minimax formulation is 

$$
\min _ {\boldsymbol {x} \in \Delta_ {n}} \max _ {\boldsymbol {y} \in \Delta_ {m}} \boldsymbol {x} ^ {\top} \boldsymbol {A} \boldsymbol {y},
$$

where $\Delta _ { n } , \Delta _ { m }$ denote the probability simplices. This formulation characterizes the Nash equilibrium of the game, capturing the row player’s strategy that minimizes the worst-case loss against an adversarial opponent, and thus serves as a foundation for robust decision making, reinforcement learning, and adversarial training. To ensure unique mixed strategies and to improve algorithmic stability, it is common to add entropic regularization terms (Cen et al., 2021). The entropy-regularized problem becomes 

$$
\min _ {\boldsymbol {x} \in \Delta_ {n}} \max _ {\boldsymbol {y} \in \Delta_ {m}} \boldsymbol {x} ^ {\top} \boldsymbol {A} \boldsymbol {y} + \tau \big (H (\boldsymbol {x}) - H (\boldsymbol {y}) \big),
$$

where, $\tau > 0$ is the regularization parameter and $H ( \cdot )$ is the Shannon entropy, defined for any probability vector $\pmb { x } \in \Delta _ { n }$ as $\begin{array} { r } { H ( \pmb { x } ) = - \sum _ { i = 1 } ^ { n } x _ { i } } \end{array}$ log x<sub>i</sub>. 

A well-known algorithm to solve the above entropy regularization problem is the double-softmax fixed-point (DSFP) iteration (McKelvey & Palfrey, 1995), which updates 

$$
\boldsymbol {y} _ {k + 1} \leftarrow (1 - \alpha) \boldsymbol {y} _ {k} + \alpha \sigma_ {1 / \tau} \left(\boldsymbol {A} ^ {\top} \sigma_ {1 / \tau} (- \boldsymbol {A} \boldsymbol {y} _ {k})\right).
$$

Intuitively, each player responds via a softmax update, yielding an equilibrium. Our improved Lipschitz constant for softmax in Theorem 1 provides a tighter condition to choose the regularization parameter τ depending on the payof matrix A, such that the DSFP iteration is contractive and convergent. 

Theorem 3 (Convergence of DSFP under entropy regularization). Let $A \in \mathbb { R } ^ { n \times m }$ be a payof matrix and let $\tau > 0$ denote the entropy regularization parameter. For all $1 \leq p \leq \infty , i f \tau > \| A \| _ { p } / 2$ and $\alpha \in ( 0 , 1 ]$ , then the DSFP iteration is a Banach contraction on $( \Delta _ { m } , \| \cdot \| _ { p } )$ with contraction factor $\| A \| _ { p } ^ { 2 } / ( 4 \tau ^ { 2 } )$ . Hence, the iterates ${ \mathbf { } } _ { \mathbf { } ^ { 3 k } }$ converge linearly in the $\ell _ { p }$ norm to the unique fixed point $y ^ { \star } \in \Delta _ { m }$ . The corresponding strategy of the row player is recovered as 

$$
\boldsymbol {x} ^ {\star} = \sigma_ {1 / \tau} (- \boldsymbol {A} \boldsymbol {y} ^ {\star}),
$$

which is also convergent since $\pmb { x } _ { k } = \sigma _ { 1 / \tau } ( - \pmb { A } \pmb { y } _ { k } )$ at each step. 

In summary, for DSFP iterations to converge linearly under the $\ell _ { p }$ norm, we can derive a condition for the regularization parameter τ to satisfy and thereby guarantee well-defined equilibrium strategies. 

## 5 Empirical Validation of Lipschitz Constant

In this section, we empirically validate our theoretical result that the softmax operator has a global Lipschitz constant of $1 / 2$ , which is norm-uniform and tight. Our experiments involve a broad range of architectures across vision, language, and reinforcement learning, under varying datasets, prompts, and inverse temperature coeficient λ. All experiments were conducted on NVIDIA A40 GPUs. We compute the empirical Lipschitz constant using the definition in equation 3, using pairs of input obtained from each experimental setting. In particular, we compute, 

$$
\text {Empirical} L _ {p} = \max _ {1 \leq i \leq M} \frac {\| \sigma_ {\lambda} (\boldsymbol {x} _ {i}) - \sigma_ {\lambda} (\boldsymbol {x} _ {i} + \delta \boldsymbol {x} _ {i}) \| _ {p}}{\| \delta \boldsymbol {x} _ {i} \| _ {p}},\tag{7}
$$

where M is the number of inputs. For each input sample ${ \mathbf { x } } _ { i } ,$ we apply small random perturbations $\delta \pmb { x } _ { i }$ normalized to have $\| \delta \pmb { x } _ { i } \| _ { p } = \epsilon$ , where ϵ denotes the perturbation magnitude. We report the computed Empirical $L _ { p }$ constant across multiple values of ϵ and $p .$ 

## 5.1 Vision Models

We empirically evaluate the Lipschitz constant of the softmax operator, which appears in the attention mechanism of vision models. In particular, we consider Vision Transformer (ViT) models (Dosovitskiy et al., 2021) in three variants, Base (86M parameters), Large (307M parameters), and Huge (632M parameters). For each model, we select images from both the CIFAR-100 and ImageNet datasets, pass them through the transformer, and extract the pre-softmax attention scores $Q K ^ { \top } / \sqrt { d _ { k } }$ , which is the input to the softmax in $\operatorname { E q . }$ 5. We then apply perturbations to these scores and compute the empirical Lipschitz constant row-wise according to Definition 7. M is set to be 100 images. 

![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig1.jpg)



(a) ViT-B (CIFAR-100)


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig2.jpg)



(b) ViT-L (CIFAR-100)


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig3.jpg)



(c) ViT-H (CIFAR-100)


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig4.jpg)



(d) ViT-B (ImageNet)


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig5.jpg)



(e) ViT-L (ImageNet)


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig6.jpg)



(f) ViT-H (ImageNet)


Figure 1: Empirical $L _ { p }$ of the softmax operator over attention scores from three Vision Transformer (ViT) variants on CIFAR-100 (top row) and ImageNet (bottom row), across varying perturbation magnitudes ϵ for multiple $\ell _ { p }$ norms. In all cases, the empirical values remain below the derived bound of $1 / 2$ 

![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig7.jpg)



(a) CIFAR dataset


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig8.jpg)



(b) Imagenet dataset



Figure 2: Empirical $L _ { p }$ of the softmax operator for classification logits of ResNET50 model on (a) CIFAR 100 and (b) ImageNet dataset, across varying perturbation magnitudes ϵ for multiple $\ell _ { p }$ norms. In all cases, the empirical values remain well below the derived bound of $1 / 2$


The results are presented as plots of empirical Lipschitz values versus the perturbation norm in Fig. 1 for CIFAR-100 (top row) and ImageNet (bottom row) datasets, with the theoretical baseline of 0.5 indicated. We report results across multiple $\ell _ { p }$ norms. In each case, the plotted value corresponds to the maximum empirical Lipschitz constant over all attention heads, layers, tokens, and images. In particular, the total number of tested vectors is significant, which is equal to $N \times H \times M \times 1 2$ , where N is the number of tokens and H the number of heads. The plots consistently show that the empirical Lipschitz constant remains below our theoretical value of $1 / 2 ,$ regardless of the choice of norm or ViT variant. 

We further evaluate the Lipschitz constant of the softmax operator on the classification logits of ResNET50 model (He et al., 2016), using images from the CIFAR-100 dataset in Fig. 2a and ImageNet dataset in Fig. 2b. In this experiment, we use $M = 2 5 { , } 0 0 0$ images and introduce input perturbations to the logits and compute the empirical $L _ { p }$ constant, for diferent p and perturbation magnitude ϵ. Consistent with our findings for attention scores, the empirical $L _ { p }$ remains strictly below the theoretical bound of $1 / 2$ 

## 5.2 Language Models

We next evaluate the Lipschitz constant of the softmax operator on attention scores of large language models. Specifically, we sample 100 prompts each from the HellaSwag (Zellers et al., 2019) and the PIQA (Bisk et al., 2020) datasets, process them through GPT-2 (Radford et al., 2019) (518M parameters) and Qwen3 (Bai et al., 2023) (8B parameters), extract the pre-softmax attention scores, apply perturbations, and compute the empirical Lipschitz constant as defined in Definition 7. 

![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig9.jpg)


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig10.jpg)



(a) GPT-2 (Hellaswag dataset)


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig11.jpg)



(c) GPT-2 (PIQA dataset)



(b) QWEN-3 (Hellaswag dataset)


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig12.jpg)



(d) QWEN-3 (PIQA dataset)



Figure 3: Empirical $L _ { p }$ of the softmax operator over attention scores from two Large Language models, GPT-2 and QWEN-3, on Hellswag dataset (top row) and PIQA dataset (bottom row), across varying perturbation magnitudes ϵ for multiple $\ell _ { p }$ norms. Across all configurations, the empirical values remain below the theoretical bound of $1 / 2 ,$ , with several instances approaching this limit, thereby confirming the tightness of the derived bound.


The scale of this evaluation is substantial here as well. The total number of tested vectors equals $H \times$ $1 0 0 \times 1 2$ per token per prompt, where the number of tokens varies with prompt length. Consistent with our experiments on vision models, the results in Fig. 3 for the Hellaswag dataset (top row) and the $\mathrm { P I Q A }$ dataset (bottom row) demonstrate that the empirical Lipschitz constant remains strictly below the theoretical value of $1 / 2$ across all norms. Notably, for the PIQA dataset, we observe empirical constants of 0.4995 for the Qwen3 model and 0.4999 for GPT-2, providing empirical evidence for the tightness of our theoretically derived Lipschitz bound. 

## 5.3 Reinforcement Learning Policies

We next investigate the Lipschitz behavior of softmax within stochastic policies in reinforcement learning. In policy-gradient methods (Sutton & Barto, 2018), a stochastic policy is typically parameterized as 

$$
\pi_ {\lambda} (a \mid s) = \frac {\exp (\lambda Q (s , a))}{\sum_ {a ^ {\prime} \in \mathcal {A}} \exp (\lambda Q (s , a ^ {\prime}))},
$$

where $Q ( s , a )$ are the action logits for state $s \in { \mathcal { S } } , A$ is the action space, and $\lambda > 0$ is the inverse temperature coeficient. We evaluate this mapping $Q ( s , \cdot ) \mapsto \pi _ { \lambda } ( \cdot \mid s )$ in the LunarLander environment (discrete action space of size $| { \mathcal { A } } | = 4 )$ and the CartPole environment $( | A | = 2 )$ from the OpenAI Gym benchmark (Brockman et al., 2016). We implement a PPO (Proximal Policy Optimization) agent (Schulman et al., 2017) using the stable_baselines3 library, initialized randomly, and executed to collect a large set of visited states. For each state, we perturb the corresponding action logits $Q ( s , \cdot )$ with random perturbations, and compute the empirical Lipschitz constant as in Definition 7. The reported value is obtained by averaging over multiple states $( M = 2 5 0 0 0 )$ and perturbation trials for diferent coeficients λ. The results in Fig. 4 for Cartpole and Lunarlander environments confirm that the empirical constants scale proportionally to $\lambda / 2 ,$ , in agreement with our theoretical prediction. Note that for the CartPole environment, where the softmax input dimension is $n = 2$ , we obtain an empirical constant of 1.9996 for $p = 8$ and $\lambda = 4 . 0$ , which is very close to the derived bound. 

![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig13.jpg)


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig14.jpg)


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig15.jpg)


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig16.jpg)



(a) $\lambda = 4 . 0$


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig17.jpg)



(b) $\lambda = 2 . 0$


![image](images/Softmax_is_1_2_Lipschitz_A_tight_bound_across_all_lp_norms/fig18.jpg)



(c) $\lambda = 0 . 5$



Figure 4: Empirical Lipschitz sensitivity of the softmax policy in RL environments, Cartpole (top row) and Lunarlander (bottom row), across varying perturbation magnitudes $\epsilon ,$ under varying coeficients λ for diferent p-norms. The empirical values scale with λ and remain below the theoretical bound of $\lambda / 2$ thereby empirically confirming the derived bound.


In summary, across all models, datasets, and settings considered, the empirical Lipschitz constant of softmax never exceeded $1 / 2 ,$ , thereby providing strong empirical validation of our theoretical bound. Furthermore, in several cases, the empirical estimates approached $1 / 2$ illustrating that the derived bound is indeed tight in practice. 

## Conclusion

In this work, we derived that the softmax function admits a Lipschitz constant of $1 / 2$ uniformly across all $\ell _ { p }$ norms. This result clarifies a commonly adopted assumption in the machine learning literature, where the Lipschitz constant of the softmax operator has often been considered as 1 with respect to the $\ell _ { 2 }$ norm. We further prove the tightness of our derived bound and demonstrate its utility in strengthening existing theoretical analyses. We also estimate the empirical Lipschitz constants for softmax transformations applied to attention score matrices from transformer-based architectures such as ViT, GPT-2, and Qwen3, classification logits from ResNet-50, and stochastic policies in reinforcement learning agents. In all cases, the empirical estimates remain strictly below the theoretical bound of $1 / 2 .$ thereby guaranteeing the tightness and generality of our results. 

## Broader Impact Statement

This work is primarily theoretical. By itself, this result does not introduce new datasets, architectures, or applications, but it provides sharper analytical tools for reasoning about the stability and robustness of existing models. On the positive side, such tools can help practitioners design systems with better controlled sensitivity to perturbations, complementing existing Lipschitz-based approaches used to improve robustness and training stability in deep networks (Cisse et al., 2017)(Miyato et al., 2018). At the same time, any advance that makes robust and stable deployment easier can be dual-use - it may also strengthen models used in societally sensitive settings, including surveillance and biometric monitoring (Almeida et al., 2022). A further risk is overstatement or misinterpretation - guarantees for a single component can be misconstrued as end-to-end assurances of safety, fairness, or reliability, which could encourage premature use in high-stake contexts (AI, 2023). 

## References



NIST AI. Artificial intelligence risk management framework (ai rmf 1.0). URL: https://nvlpubs. nist. gov/nistpubs/ai/nist. ai, pp. 100–1, 2023. 





Wael Alghamdi, Hsiang Hsu, Haewon Jeong, Hao Wang, Peter Michalak, Shahab Asoodeh, and Flavio Calmon. Beyond adult and compas: Fair multi-class prediction via information projection. Proc. Advances in Neural Information Processing Systems, 35:38747–38760, 2022. 





Denise Almeida, Konstantin Shmarko, and Elizabeth Lomas. The ethics of facial recognition technologies, surveillance, and accountability in an age of artificial intelligence: a comparative analysis of us, eu, and uk regulatory frameworks. AI and Ethics, 2(3):377–387, 2022. 





Amir R Asadi and Emmanuel Abbe. Chaining meets chain rule: Multilevel entropic regularization and training of neural networks. Journal of Machine Learning Research, 21(139):1–32, 2020. 





Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 





Peter L. Bartlett, Dylan J. Foster, and Matus Telgarsky. Spectrally-normalized margin bounds for neural networks. Proc. Advances in Neural Information Processing Systems, 30, 2017. 





Yonatan Bisk, Rowan Zellers, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. Proc. AAAI Conference on Artificial Intelligence (AAAI), 34(05):7432–7439, 2020. 





Stephen P Boyd and Lieven Vandenberghe. Convex optimization. Cambridge University Press, 2004. 





Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016. 





Shicong Cen, Yuting Wei, and Yuejie Chi. Fast policy extragradient methods for competitive games with entropy regularization. Proc. Advances in Neural Information Processing Systems, 34:27952–27964, 2021. 





Lin Chen, Huaian Chen, Zhixiang Wei, Xin Jin, Xiao Tan, Yi Jin, and Enhong Chen. Reusing the taskspecific classifier as a discriminator: Discriminator-free adversarial domain adaptation. Proc. IEEE/CVF Conference on Computer Vision and Pattern rRecognition, pp. 7181–7190, 2022. 





Moustapha Cisse, Piotr Bojanowski, Edouard Grave, Yann Dauphin, and Nicolas Usunier. Parseval networks: Improving robustness to adversarial examples. Proce. International Conference on Machine Learning, 70: 854–863, 2017. 





Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. Proc. International Conference on Learning Representations, 2021. 





Mahyar Fazlyab, Manfred Morari, and George J. Pappas. Eficient and accurate estimation of lipschitz constants for deep neural networks. Proc. Advances in Neural Information Processing Systems, 32, 2019. 





Bolin Gao and Lacra Pavel. On the properties of the softmax function with application in game theory and reinforcement learning. arXiv preprint arXiv:1704.00805, 2017. 





Gene H Golub and Charles F Van Loan. Matrix computations. JHU press, 2013. 





Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning, volume 1. MIT Press, 2016. 





Henry Gouk, Eibe Frank, Bernhard Pfahringer, and Michael J Cree. Regularisation of neural networks by enforcing lipschitz continuity. Machine Learning, 110(2):393–416, 2021. 





Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. Proc. IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016. 





Tuomas Hytönen, Jan Van Neerven, Mark Veraar, and Lutz Weis. Analysis in Banach spaces, volume 12. Springer, 2016. 





Hong Jun Jeon, Jason D. Lee, Qi Lei, and Benjamin Van Roy. An information-theoretic analysis of in-context learning. Proc. International Conference on Machine Learning, 2024. 





Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. Transformers in vision: A survey. ACM computing surveys (CSUR), 54(10s):1–41, 2022. 





Hyunjik Kim, George Papamakarios, and Andriy Mnih. The lipschitz constant of self-attention. Proc. International Conference on Machine Learning, 139:5562–5571, 2021. 





Anirban Laha, Saneem Ahmed Chemmengath, Priyanka Agrawal, Mitesh Khapra, Karthik Sankaranarayanan, and Harish G Ramaswamy. On controllable sparse alternatives to softmax. Proc. Advances in Neural Information Processing Systems, 31, 2018. 





Rémi Le Priol, Reza Babanezhad, Yoshua Bengio, and Simon Lacoste-Julien. An analysis of the adaptation speed of causal models. Proc. International Conference on Artificial Intelligence and Statistics, pp. 775– 783, 2021. 





Bowen Lei, Dongkuan Xu, Ruqi Zhang, and Bani Mallick. Embracing unknown step by step: Towards reliable sparse training in real world. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. 





Wenzhe Li, Hao Luo, Zichuan Lin, Chongjie Zhang, Zongqing Lu, and Deheng Ye. A survey on transformers in reinforcement learning. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. 





Richard D. McKelvey and Thomas R. Palfrey. Quantal response equilibria for normal form games. Games and Economic Behavior, 10(1):6–38, 1995. 





Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. Proc. International Conference on Learning Representations, 2018. 





Ofir Nachum, Mohammad Norouzi, Kelvin Xu, and Dale Schuurmans. Bridging the gap between value and policy based reinforcement learning. Proc. Advances in neural information processing systems, 30, 2017. 





Thomas Nagler. Statistical foundations of prior-data fitted networks. Proc. International Conference on Machine Learning, pp. 25660–25676, 2023. 





Yu Nesterov. Smooth minimization of non-smooth functions. Mathematical programming, 103(1):127–152, 2005. 





Laker Newhouse. Softmax is <sup>1</sup>-lipschitz (in a norm that may not matter). https://www.lakernewhouse. com/assets/writing/softmax-is-0-5-lipschitz.pdf, 2025. Unpublished note. 





Xianbiao Qi, Jianan Wang, Yihao Chen, Yukai Shi, and Lei Zhang. Lipsformer: Introducing lipschitz continuity to vision transformers. Proc. International Conference on Learning Representations, 2023. 





Alec Radford, Jefrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI Blog, 1(8):9, 2019. 





Marcel Riesz. Sur les maxima des formes bilinéaires et sur les fonctionnelles linéaires. Acta Mathematica, 49 (3):465–497, 1927. 





R. Tyrrell Rockafellar and Roger J-B Wets. Variational Analysis. Springer, 1998. 





John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 





Elias M Stein. Interpolation of linear operators. Transactions of the American Mathematical Society, 83(2): 482–492, 1956. 





Richard Sutton and Andrew Barto. Reinforcement Learning: An Introduction. MIT Press, 2018. 





GO Thorin. An extension of a convexity theorem due to m. riesz, kungl. fysiografiska sällskapets i lund förhandlingar, no. 8, 14 (1938), 166–170; medd. Lunds Univ. Mat. Sem, 4:1–5, 1939. 





Lewis Tunstall, Leandro Von Werra, and Thomas Wolf. Natural language processing with transformers. " O’Reilly Media, Inc.", 2022. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Proc. Advances in neural information processing systems, 30, 2017. 





Aladin Virmaux and Kevin Scaman. Lipschitz regularity of deep neural networks: Analysis and eficient estimation. Proc. Advances in Neural Information Processing Systems, 31, 2018. 





Xiaojun Xu, Linyi Li, Yu Cheng, Subhabrata Mukherjee, Ahmed Hassan Awadallah, and Bo Li. Certifiably robust transformers with 1-lipschitz self-attention. 2022. 





Nikolay Yudin, Alexander Gaponov, Sergei Kudriashov, and Maxim Rakhuba. Pay attention to attention distribution: A new local lipschitz bound for transformers. arXiv preprint arXiv:2507.07814, 2025. 





Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? Proc. Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, 2019. 



## A Appendix

## A.1 Proof of Proposition 1

(a) and (b) follow from standard results in matrix analysis (Golub & Van Loan, 2013). 

Proof of Proposition $\pmb { 1 } ( c )$ . Let $A \in \mathbb { R } ^ { n \times n }$ . For $p = 1$ and $p = \infty$ , the identity trivially satisfies, and hence fix $1 < p < \infty$ . For any $\pmb { x } \in \mathbb { R } ^ { n }$ 

$$
\left\| \boldsymbol {A} \boldsymbol {x} \right\| _ {p} ^ {p} = \sum_ {i = 1} ^ {n} \left| (\boldsymbol {A} \boldsymbol {x}) _ {i} \right| ^ {p},
$$

where $( \boldsymbol { A } \boldsymbol { x } ) _ { i }$ denotes $i ^ { t h }$ element in Ax. Then, for each $i ,$ the triangle inequality gives 

$$
\left| (\boldsymbol {A} \boldsymbol {x}) _ {i} \right| = \left| \sum_ {j = 1} ^ {n} A _ {i j} x _ {j} \right| \leq \sum_ {j = 1} ^ {n} \left| A _ {i j} \right| \left| x _ {j} \right|
$$

By Holder’s inequality, we obtain for each $i ,$ 

$$
\sum_ {j = 1} ^ {n} \left| A _ {i j} \right| \left| x _ {j} \right| \leq \left(\sum_ {j = 1} ^ {n} \left| A _ {i j} \right|\right) ^ {1 - 1 / p} \left(\sum_ {j = 1} ^ {n} \left| A _ {i j} \right| \left| x _ {j} \right| ^ {p}\right) ^ {1 / p}.
$$

Thus, we can bound $\| A x \| _ { p } ^ { p }$ as follows, 

$$
\begin{array}{l} \| \boldsymbol {A x} \| _ {p} ^ {p} \leq \sum_ {i = 1} ^ {n} \Big (\sum_ {j = 1} ^ {n} | A _ {i j} | \Big) ^ {p - 1} \Big (\sum_ {j = 1} ^ {n} | A _ {i j} |   | x _ {j} | ^ {p} \Big) \\ \quad \leq \Big (\max _ {1 \leq i \leq n} \sum_ {j = 1} ^ {n} | A _ {i j} | \Big) ^ {p - 1} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {n} | A _ {i j} |   | x _ {j} | ^ {p} \\ \quad = \| \boldsymbol {A} \| _ {\infty} ^ {p - 1} \sum_ {j = 1} ^ {n} | x _ {j} | ^ {p} \Big (\sum_ {i = 1} ^ {n} | A _ {i j} | \Big) \\ \quad \leq \| \boldsymbol {A} \| _ {\infty} ^ {p - 1} \Big (\max _ {1 \leq j \leq n} \sum_ {i = 1} ^ {n} | A _ {i j} | \Big) \sum_ {j = 1} ^ {n} | x _ {j} | ^ {p} \\ \quad = \| \boldsymbol {A} \| _ {\infty} ^ {p - 1}   \| \boldsymbol {A} \| _ {1}   \| x \| _ {p} ^ {p}. \end{array}
$$

The second and fourth inequalities are obtained by taking the max term out of the summation<sup>1</sup>, i.e, 

$$
\sum_ {j = 1} ^ {n} \left| A _ {i j} \right| \leq \max _ {1 \leq i \leq n} \sum_ {j = 1} ^ {n} \left| A _ {i j} \right| \text {   for   all   } i, \quad \sum_ {i = 1} ^ {n} \left| A _ {i j} \right| \leq \max _ {1 \leq j \leq n} \sum_ {i = 1} ^ {n} \left| A _ {i j} \right| \text {   for   all   } j\tag{8}
$$

The third and fifth equalities follow from (a) and (b). Taking the $p \mathrm { - }$ -th root and maximizing over $\| \pmb { x } \| _ { p } = 1$ , we get the inequality 

$$
\| \pmb {A} \| _ {p} \leq \| \pmb {A} \| _ {\infty} ^ {1 - 1 / p} \| \pmb {A} \| _ {1} ^ {1 / p}.
$$

## A.2 Proof of Theorem 1

Proof of Theorem $\pmb { 1 } ( a )$ . Note that $J _ { \sigma _ { 1 } } ( x )$ is symmetric for all $\pmb { x } \in \mathbb { R } ^ { n }$ . Hence, from Proposition $1 ( \mathrm { a } )$ and (b), we get that for all $\pmb { x } \in \mathbb { R } ^ { n }$ 

$$
\left\| \boldsymbol {J} _ {\sigma_ {1}} (\boldsymbol {x}) \right\| _ {1} = \left\| \boldsymbol {J} _ {\sigma_ {1}} (\boldsymbol {x}) \right\| _ {\infty} = \max _ {1 \leq i \leq n} \sum_ {j = 1} ^ {n} \left| \left[ \boldsymbol {J} _ {\sigma_ {1}} (x) \right] _ {i j} \right|
$$

From Lemma 2, the above optimization problem becomes, 

$$
\begin{array}{l} \max _ {1 \leq i \leq n} s _ {i} (1 - s _ {i}) + \sum_ {j \neq i} s _ {i} s _ {j} \\ = \max _ {1 \leq i \leq n} s _ {i} (1 - s _ {i}) - s _ {i} ^ {2} + \sum_ {j = 1} ^ {n} s _ {i} s _ {j} \\ = \max _ {1 \leq i \leq n} 2 s _ {i} (1 - s _ {i}), \end{array}
$$

where $\pmb { \mathscr { s } } = \sigma _ { 1 } ( \pmb { x } ) = \left( s _ { 1 } , s _ { 2 } , \ldots , s _ { n } \right)$ . The optimal value for the above problem is upper bounded by $1 / 2$ , and $1 / 2$ is attained for all $\pmb { x } \in \mathbb { R } ^ { n }$ such that the corresponding output $\pmb { s } = \sigma _ { 1 } ( \pmb { x } )$ belongs to $\Delta _ { n } ^ { 0 }$ and $s _ { i } = 1 / 2$ for any $i \in \{ 1 , 2 , \ldots , n \}$ . Thus, we get the following bounds, 

$$
\| \boldsymbol {J} _ {\sigma_ {1}} (\boldsymbol {x}) \| _ {1} \leq \frac {1}{2} \quad \text { and } \quad \| \boldsymbol {J} _ {\sigma_ {1}} (\boldsymbol {x}) \| _ {\infty} \leq \frac {1}{2}
$$

By Proposition 1(c), we get 

$$
\left\| \boldsymbol {J} _ {\sigma_ {1}} (\boldsymbol {x}) \right\| _ {p} \leq \left(\frac {1}{2}\right) ^ {\frac {1}{p}} \left(\frac {1}{2}\right) ^ {1 - \frac {1}{p}} = \frac {1}{2}, \quad \text { for   all } 1 \leq p \leq \infty .
$$

Proof of Theorem $\mathbf { \delta } _ { 1 ( b ) }$ . From Theorem 1(a), we get, 

$$
\sup _ {x \in \mathbb {R} ^ {n}} \| \boldsymbol {J} _ {\sigma_ {1}} (\boldsymbol {x}) \| _ {p} \leq \frac {1}{2}, \quad \text { for   all } 1 \leq p \leq \infty .
$$

From Lemma 3(b), for any $\lambda > 0$ , we have, 

$$
\sup _ {x \in \mathbb {R} ^ {n}} \| \boldsymbol {J} _ {\sigma_ {\lambda}} (\boldsymbol {x}) \| _ {p} \leq \frac {\lambda}{2} \quad \text { for   all } 1 <   p <   \infty .
$$

and hence using Lipschitz characterization in Lemma 1, 

$$
\left\| \sigma_ {\lambda} (\boldsymbol {x}) - \sigma_ {\lambda} (\boldsymbol {y}) \right\| _ {p} \leq \frac {\lambda}{2} \| \boldsymbol {x} - \boldsymbol {y} \| _ {p} \quad \text { for   all } \boldsymbol {x}, \boldsymbol {y} \in \mathbb {R} ^ {n} \text { and   for   all } 1 \leq p \leq \infty .
$$

## A.3 Proof of Proposition 2

Define $M ( s ) \in \mathbb { R } ^ { n \times n }$ as $\mathrm { d i a g } ( s ) - s s ^ { T }$ . From Lemma 3, for all $1 \leq p \leq \infty$ 

$$
\sup _ {\boldsymbol {x} \in \mathbb {R} ^ {n}} \| \boldsymbol {J} _ {\sigma_ {1}} (\boldsymbol {x}) \| _ {p} = \sup _ {\boldsymbol {s} \in \Delta_ {n} ^ {\circ}} \| \boldsymbol {M} (\boldsymbol {s}) \| _ {p}
$$

Proof of Proposition $\mathcal { Z } ( a )$ . As discussed in Theorem 1(a), for $p = 1$ and $p = \infty$ , for all $s \in \Delta _ { n } ^ { \circ }$ 

$$
\| \boldsymbol {M} (\boldsymbol {s}) \| _ {1} = \| \boldsymbol {M} (\boldsymbol {s}) \| _ {\infty} = \max _ {1 \leq i \leq n} 2 s _ {i} (1 - s _ {i}) \leq 1 / 2
$$

and $1 / 2$ is attained if there exists $s \in \Delta _ { n } ^ { \circ }$ such that $s _ { i } = 1 / 2$ for some i and $s _ { j } > 0$ for $j \neq i .$ 

To prove that $1 / 2$ is indeed attained, we need to show an example where $1 / 2$ is attained at a point s in the interior of the probability simplex $\Delta _ { n } ^ { \circ }$ 

Let $\pmb { x } \in \mathbb { R } ^ { n }$ such that $x _ { i } = \ln ( n - 1 )$ and $x _ { j } = 0$ for all $j \neq i , s = \sigma _ { 1 } ( \pmb { x } )$ is such that $s _ { i } = 1 / 2$ and $s _ { j } > 0$ for all $j \neq i$ and hence $s \in \Delta _ { n } ^ { \circ }$ . Thus, 

$$
\max _ {\boldsymbol {x} \in \mathbb {R} ^ {n}} \| \boldsymbol {J} _ {\sigma_ {1}} (\boldsymbol {x}) \| _ {1} = \max _ {\boldsymbol {x} \in \mathbb {R} ^ {n}} \| \boldsymbol {J} _ {\sigma_ {1}} (\boldsymbol {x}) \| _ {\infty} = 1 / 2
$$

Proof of Proposition ${ \mathit { 2 } } ( b )$ . We first prove for the case $n > 2 .$ . It sufices to show that for arbitrary $1 < p < \infty , \| M ( s ) \| _ { p } \neq 1 / 2$ for any $s \in \Delta _ { n } ^ { \circ }$ and there exists $s \in \partial \Delta _ { n }$ such that $\Vert M ( s ) \Vert _ { p } = 1 / 2$ for $n > 2$ to prove that $1 / 2$ is indeed approached in limit. We need the following definition of the support of a vector and the corresponding lemma to prove this. 

Definition A.1 (Support of a vector). For a vector $s \in \mathbb { R } ^ { n }$ , the support of s is defined as 

$$
\operatorname{supp} (\boldsymbol {s}) := \left\{i \in \{1, \dots , n \}: s _ {i} \neq 0 \right\}.
$$

Lemma 4 (Support condition for attaining $1 / 2 )$ . Let $M ( s ) \in \mathbb { R } ^ { n \times n }$ be defined by 

$$
\boldsymbol {M} (\boldsymbol {s}) := \operatorname{Diag} (\boldsymbol {s}) - \boldsymbol {s s} ^ {\top},
$$

for $s \in \Delta _ { n }$ . Suppose $1 < p < \infty$ and $n > 2$ . Then, $\| M ( s ) \| _ { p } = { \begin{array} { l } { { \frac { 1 } { 2 } } } \end{array} } i f$ and only $i f s$ is a permutation of $( 1 / 2 , 1 / 2 , 0 , \ldots , 0 )$ 

The proof of the lemma is given towards the end of this section. From Lemma 4, there exists no $s \in \Delta _ { n }$ with $\operatorname { s u p p } ( s ) > 2$ with $\Vert M ( s ) \Vert _ { p } = 1 / 2$ . Hence, there exists no $s \in \Delta _ { n } ^ { \circ }$ with $\Vert M ( s ) \Vert _ { p } = 1 / 2$ since $\operatorname { s u p p } ( \pmb { s } ) = n$ for all $s \in \Delta _ { n } ^ { \circ }$ . Also, there exists s with $\operatorname { s u p p } ( s ) = 2$ such that $\| M ( s ) \| _ { p } = 1 / 2$ 

Let $\hat { s } \in \Delta _ { n }$ with $\operatorname { s u p p } ( { \hat { s } } ) = 2$ be such that $\Vert M ( \hat { s } ) \Vert _ { p } = 1 / 2$ . By definition, $s \in \partial \Delta _ { n }$ (boundary of the simplex). Hence, we can obtain a sequence of $s _ { k } \in \Delta _ { n } ^ { \circ }$ (interior of the simplex) such that $s _ { k } \ \to \ \hat { s }$ as $k  \infty$ (Rockafellar & Wets, 1998). Also since $M ( s )$ is a continuous function, $M ( s _ { k } ) \to 1 / 2$ as $k  \infty ,$ For completeness, we show convergence by constructing example sequence in Example 2. 

Finally, when $n = 2$ , the point $s = ( 1 / 2 , 1 / 2 )$ lies in $\Delta _ { 2 } ^ { \circ }$ and attains $\| M ( s ) \| _ { p } = 1 / 2$ 

Example 2 (Sequence $( \boldsymbol { s } _ { k } ) _ { k \geq 1 } \ \subset \ \Delta _ { n } ^ { \circ }$ such that $M ( s _ { k } )  1 / 2$ as $k  \infty )$ . Consider $s \in \partial \Delta _ { r }$ with $\hat { s } _ { 1 } = \hat { s } _ { 2 } = 1 / 2$ and all other entries zero. By Lemma $\downarrow , \parallel M ( \hat { s } ) \parallel _ { p } = 1 / 2$ To approximate sˆ from the interior of the probability simplex, fix $\epsilon \in ( 0 , 1 / 2 )$ and choose $\delta \in ( 0 , 1 / 2 )$ satisfying $2 \delta ( 1 - \delta ) = \epsilon .$ Define 

$$
s _ {1} = s _ {2} = \frac {1}{2} - \delta , \qquad s _ {j} = \frac {2 \delta}{n - 2}, j \geq 3,
$$

so that $s \in \Delta _ { n } ^ { \circ }$ . Consider the vector 

$$
\boldsymbol {v} = \left(2 ^ {- 1 / p}, - 2 ^ {- 1 / p}, 0, \dots , 0\right) \in \mathbb {R} ^ {n} \quad w i t h \quad \| \boldsymbol {v} \| _ {p} = 1,
$$

A direct computation shows, 

$$
[ \boldsymbol {M} (\boldsymbol {s}) \boldsymbol {v} ] _ {1} = - [ \boldsymbol {M} (\boldsymbol {s}) \boldsymbol {v} ] _ {2} = \frac {2}{2 ^ {1 / p}} \left(\frac {1}{2} - \delta\right) ^ {2} a n d [ \boldsymbol {M} (\boldsymbol {s}) \boldsymbol {v} ] _ {j} = 0 f o r a l l j \notin \{1, 2 \}.
$$

Hence, we get, 

$$
\| \boldsymbol {M} (\boldsymbol {s}) \boldsymbol {v} \| _ {p} = \frac {1}{2} - 2 \delta (1 - \delta) = \frac {1}{2} - \epsilon .
$$

Thus for every $\epsilon \in ( 0 , 1 / 2 )$ , there exists $s \in \Delta _ { n } ^ { \circ }$ with $1 / 2 - \epsilon \le \| M ( s ) \| _ { p } < 1 / 2$ , proving that $1 / 2$ is the tight upper bound. 

Proof of Lemma 4. Let s be a permutation of $( 1 / 2 , 1 / 2 , 0 , \dots , 0 )$ i.e $s _ { i } = s _ { j } = 1 / 2$ for some distinct $i , j \in$ $\{ 1 , 2 , \ldots , n \}$ . M(s) has values $M ( \pmb { s } ) _ { i i } = M ( \pmb { s } ) _ { j j } = 1 / 4 , M ( \pmb { s } ) _ { i j } = M ( \pmb { s } ) _ { j i } = - 1 / 4$ , and zeros elsewhere. To show that $\Vert M ( s ) \Vert _ { p } = 1 / 2$ , it sufices to construct $v \in \mathbb { R } ^ { n }$ such that $\| \pmb { v } \| _ { p } = 1$ and $\| M ( s ) \pmb { v } \| _ { p } = 1 / 2$ Consider v to be a similar permutation of 

$$
\left(2 ^ {- 1 / p}, - 2 ^ {- 1 / p}, 0, \dots , 0\right) \in \mathbb {R} ^ {n}.
$$

i.e. $v _ { i } = 2 ^ { - 1 / p }$ and $v _ { j } = - 2 ^ { - 1 / p }$ . A direct calculation gives $\| M ( s ) \pmb { v } \| _ { p } = 1 / 2$ 

Now we need to prove that $\mathrm { i f } \parallel M ( s ) \parallel _ { p } = 1 / 2$ , s is a permutation of $( 1 / 2 , 1 / 2 , 0 , \ldots , 0 )$ . By Proposition $1 ( \mathrm { c } )$ the interpolation inequality gives 

$$
\| \boldsymbol {M} (\boldsymbol {s}) \| _ {p} \leq \| \boldsymbol {M} (\boldsymbol {s}) \| _ {1} ^ {1 / p} \| \boldsymbol {M} (\boldsymbol {s}) \| _ {\infty} ^ {1 - 1 / p}.
$$

From Proposition $2 ( \mathrm { a } )$ , we know $\| M ( s ) \| _ { 1 } = \| M ( s ) \| _ { \infty } \leq 1 / 2$ . Thus if $\Vert M ( s ) \Vert _ { p } = 1 / 2$ , equality must hold throughout. This means for all $s \in \Delta _ { n }$ such that $\Vert M ( s ) \Vert _ { p } = 1 / 2$ , we have $\| M ( s ) \| _ { 1 } = \| M ( s ) \| _ { \infty } = 1 / 2$ 

We divide into cases according to the support size of s. 

Case (i): $\operatorname { s u p p } ( s ) = 1$ . Suppose $s _ { i } = 1$ for some i, and $s _ { j } = 0$ for all $j \neq i$ . Then $M ( s ) = 0$ . Hence $\| M ( s ) \| _ { p } = 0$ for all p. Thus, the value $1 / 2$ cannot be attained for any s. 

Case $( \mathbf { i i } ) \colon \mathrm { s u p p } ( s ) = 2$ . As discussed in Theorem 1(a), for $p = 1$ and $p = \infty ,$ for all $s \in \Delta _ { n }$ 

$$
\| \boldsymbol {M} (\boldsymbol {s}) \| _ {1} = \| \boldsymbol {M} (\boldsymbol {s}) \| _ {\infty} = \max _ {1 \leq i \leq n} 2 s _ {i} (1 - s _ {i}) \leq 1 / 2
$$

and $1 / 2$ is attained only if there exists $s \in \Delta _ { n }$ such that $s _ { i } = 1 / 2$ for some i. Thus, if $\operatorname { s u p p } ( s ) = 2$ , then s needs to be permutations of $( 1 / 2 , 1 / 2 , 0 , \ldots , 0 )$ for $\| M ( s ) \| _ { 1 } = \| M ( s ) \| _ { \infty } = 1 / 2$ and we have already shown $\Vert M ( s ) \Vert _ { p } = 1 / 2$ for any permutation of $( 1 / 2 , 1 / 2 , 0 , \dots 0 )$ 

Thus, if $\| M ( s ) \| _ { p } = 1 / 2$ with $\operatorname { s u p p } ( s ) = 2$ , then $\| M ( \pmb { s } ) \| _ { 1 } = \| M ( \pmb { s } ) \| _ { \infty } = 1 / 2$ which needs s to be a permutation of $( 1 / 2 , 1 / 2 , 0 , \dots 0 )$ 

Case (iii): supp(s) > 2. Suppose, $\hat { \pmb { s } } \in \Delta _ { n }$ with supp(sˆ) > 2 satisfies $\Vert M ( \hat { s } ) \Vert _ { p } = 1 / 2$ . By Proposition $1 ( \mathrm { c } )$ 

$$
\| \boldsymbol {M} (\hat {\boldsymbol {s}}) \| _ {p} \leq \| \boldsymbol {M} (\hat {\boldsymbol {s}}) \| _ {1} ^ {1 / p} \| \boldsymbol {M} (\hat {\boldsymbol {s}}) \| _ {\infty} ^ {1 - 1 / p}.
$$

Since $\| M ( \hat { \pmb { s } } ) \| _ { 1 } = \| M ( \hat { \pmb { s } } ) \| _ { \infty } = 1 / 2$ holds, equality must hold in the interpolation inequality in Proposition 1. Thus, it sufices to rule out equality. 

We just need to show that one of the inequalities used in the proof of Proposition 1(c) is strict. We show that the inequality in Eq. 8 is strict i.e. there exists $1 \leq i \leq n ,$ , such that, 

$$
\sum_ {j = 1} ^ {n} | \boldsymbol {M} (\boldsymbol {s}) _ {i j} | <   \max _ {1 \leq k \leq n} \sum_ {j = 1} ^ {n} | \boldsymbol {M} (\boldsymbol {s}) _ {k j} |
$$

Since $\| M ( \pmb { \hat { s } } ) \| _ { 1 } = \| M ( \pmb { \hat { s } } ) \| _ { \infty } = 1 / 2 , \hat { s } _ { i _ { 1 } } = 1 / 2$ for some $1 \leq i _ { 1 } \leq n$ and $\hat { s } _ { j } < 1 / 2$ for all $j \neq i _ { 1 }$ . Since supp( ${ \hat { s } } ) > 2$ , there exists an index $i _ { 2 } \in \{ 1 , 2 , \dots , n \}$ and $i _ { 2 } \neq i _ { 1 }$ , where $\hat { s } _ { i _ { 2 } } < 1 / 2 .$ . As discussed in Theorem 1, for any row index $i , \textstyle \sum _ { j = 1 } ^ { n } | M ( { \hat { s } } ) _ { i j } | = 2 { \hat { s } } _ { i } ( 1 - { \hat { s } } _ { i } )$ . Hence, 

$$
\max _ {1 \leq i \leq n} \sum_ {j = 1} ^ {n} | M (\boldsymbol {s}) _ {i j} | = \sum_ {j = 1} ^ {n} | M (\boldsymbol {s}) _ {i _ {1} j} | = 1 / 2,
$$

and for row index i<sub>2</sub>, 

$$
\sum_ {j = 1} ^ {n} | \boldsymbol {M} (\boldsymbol {s}) _ {i _ {2} j} | <   1 / 2 = \max _ {1 \leq k \leq n} \sum_ {j = 1} ^ {n} | \boldsymbol {M} (\boldsymbol {s}) _ {k j} |.
$$

This shows that the inequality in Eq. 8 is strict. Hence, the interpolation inequality in Proposition 1(c) is strict for the matrix $M ( \hat { \pmb s } )$ . This contradicts our assumption that $\Vert M ( \hat { s } ) \Vert _ { p } = 1 / 2$ 

Combining all cases, we conclude that $\Vert M ( s ) \Vert _ { p } = 1 / 2$ if and only if s is a permutation of $( 1 / 2 , 1 / 2 , 0 \dots 0 )$ 

## A.4 Proof of Theorem 3

Let the DSFP map be defined as 

$$
T (\boldsymbol {y}) = \sigma_ {1 / \tau} \left(\boldsymbol {A} ^ {\top} \sigma_ {1 / \tau} (- \boldsymbol {A} \boldsymbol {y})\right), \quad \boldsymbol {y} \in \Delta_ {m},
$$

where $\sigma _ { 1 / \tau }$ denotes the softmax operator with temperature $\tau .$ . By Theorem 1, the softmax operator is globally Lipschitz with constant $\scriptstyle { \frac { 1 } { 2 \tau } }$ , uniformly across all $\ell _ { p }$ norms with $1 \leq p \leq \infty$ . Applying the chain rule of Lipschitz constants, we obtain 

$$
\| T (\boldsymbol {y} _ {1}) - T (\boldsymbol {y} _ {2}) \| _ {p} \leq \frac {1}{2 \tau} \| \boldsymbol {A} ^ {\top} \| _ {p} \frac {1}{2 \tau} \| \boldsymbol {A} \| _ {p} \| \boldsymbol {y} _ {1} - \boldsymbol {y} _ {2} \| _ {p}.
$$

Since $\| A ^ { \top } \| _ { p } = \| A \| _ { p }$ , this yields 

$$
\| T (\boldsymbol {y} _ {1}) - T (\boldsymbol {y} _ {2}) \| _ {p} \leq \frac {\| \boldsymbol {A} \| _ {p} ^ {2}}{4 \tau^ {2}} \| \boldsymbol {y} _ {1} - \boldsymbol {y} _ {2} \| _ {p}.
$$

Hence, the DSFP map has the Lipschitz constant, 

$$
c = \frac {\| \boldsymbol {A} \| _ {p} ^ {2}}{4 \tau^ {2}}.
$$

If $\tau > \| A \| _ { p } / 2$ , then $c < 1$ and $T$ is a Banach contraction on the complete metric space $( \Delta _ { m } , \| \cdot \| _ { p } )$ . By the Banach fixed-point theorem, T then admits a unique fixed point $y ^ { \star } \in \Delta _ { m }$ , and for any initialization $\pmb { y } _ { 0 } \in \Delta _ { m }$ , the iteration 

$$
\boldsymbol {y} _ {k + 1} \leftarrow (1 - \alpha) \boldsymbol {y} _ {k} + \alpha T (\boldsymbol {y} _ {k}), \quad \alpha \in (0, 1 ],
$$

converges linearly in the $\ell _ { p }$ norm to $\boldsymbol { y } ^ { \star }$ . At each step, the row player’s strategy is updated as 

$$
\boldsymbol {x} _ {k} = \sigma_ {1 / \tau} (- \boldsymbol {A} \boldsymbol {y} _ {k}).
$$

Since $y _ { k } \to y ^ { \star }$ and the softmax map is continuous, it follows that $\pmb { x } _ { k }  \pmb { x } ^ { \star } : = \sigma _ { 1 / \tau } ( - \pmb { A } \pmb { y } ^ { \star } )$ . Thus, both players’ equilibrium strategies are recovered in the limit. 