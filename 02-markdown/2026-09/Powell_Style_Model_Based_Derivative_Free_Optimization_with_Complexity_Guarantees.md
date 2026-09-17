# Powell-Style Model-Based Derivative-Free Optimization with Complexity Guarantees

A. Chaudhry<sup>†</sup>, K. Scheinberg<sup>‡</sup>, and Scholar Sun<sup>‡</sup>

## Abstract

We propose variants of model-based trust region derivative free algorithms that are closest to methods initially proposed and implemented by Powell in [22, 21]. These methods rely on low degree polynomial interpolation and carefully maintain geometry of the interpolation sets. We are able to derive complexity bounds for these methods that make them theoretically competitive to other derivative free methods. Applying these methods in randomly generated subspaces recovers what we believe to be nearly tight complexity. This paper builds on recent results in [11] where complexity of a much simplified version of Powell’s methods was derived. Here we extend the analysis to fully incorporate Powell’s geometry handling approach, and conduct extensive numerical comparison of the model-based trust region methods connecting practical and theoretical performance. We also extend the analysis of subspace model-based trust region methods initially developed in [11] to the case of noisy function evaluations.

**MSC Classification:** 90C30, 90C56.

## 1 Introduction

In this paper, we focus on the complexity of model-based derivative free algorithms that aim to solve unconstrained nonlinear optimization problems of the form

$$
\min _ {x \in \mathbb {R} ^ {n}} \phi (x)
$$

where $\phi : \mathbb { R } ^ { n }  \mathbb { R }$ is a smooth objective function that may not be convex. The key premise of these methods is to economize on function values, possibly at the expense of additional linear algebra, since in most applications the function evaluations cost dominates all else. The complexity will be measured in terms of the number of function evaluations needed to achieve an ϵ-stationary point, that is a point x for which $\| \nabla \phi ( x ) \| \leq \epsilon .$

The main goal of this paper is to narrow the gap between theory and practice in model based derivative free optimization (DFO). Derivative free optimization, also known under names zeroth-order and gradient-free optimization, is an area of optimization concerned with developing optimization methods based purely on function value computation without applying any direct differentiation. The area has experienced significant growth both in theory and in a variety of applications in the past several decades. There is a rich literature of DFO starting from around mid 90s which is rapidly growing with new interest spurred by new applications in engineering, machine learning and artificial intelligence. Aside from an increasing number of papers, there are two books [13] and [1], and a survey on the topic [19]. Examples of applications can be found in [1] and [25].

Model-based DFO methods approach the optimization problem by constructing and maintaining (usually local) models of the objective function from function value samples. A particular class of such methods, modelbased trust region methods pioneered in the 90s by M.J.D. Powell [22, 21], proved to be very effective in practice for many applications [20]. These methods use polynomial interpolation models and maintain sample sets using carefully engineered techniques, supported by mathematical properties of Lagrange interpolation polynomials. The complex structure of the algorithms, especially as presented in Powell’s papers, made them difficult to implement, let alone to analyze rigorously. Some underlying theory of asymptotic convergence was developed in [13] and later complexity bounds were provided for those algorithms in [15], however, methods analyzed there are much simplified and include elements that significantly depart from Powell’s ideas. More specifically, Powell’s methods only use one or two function evaluations per iteration, attempting to select the sample points in an optimal way, to improve the objective value while also maintaining or improving geometry of sample points. In other words, they carefully balance exploration and exploitation. On the other hand, the algorithms which have enjoyed complexity bounds so far require occasional complete model reconstruction, which requires far more function evaluations and essentially abandon Powell’s careful geometry correcting approach.

For the past several decades there was a general lack of understanding of how Powell’s methods work and most importantly if they enjoy favorable complexity bounds. Recently, in a considerable implementation effort Powell’s software has been reincarnated in modern platforms by Z. Zhang and his colleagues [24, 27]. Their software packages have seen considerable success and are now being widely used by practitioners. This raises the interest in providing solid theory for these methods. In [11] a geometry correcting method inspired by Powell’s algorithms was proposed and its complexity bounds derived. There were several important questions addressed in that paper.

- It was shown that a model-based TR method which performs only one or two function evaluations per iteration has good complexity guarantees in the case of general polynomials.

- In the case of linear polynomial interpolation the worst case complexity bound is $\mathcal { O } ( n ^ { 2 } \epsilon ^ { - 2 } )$ which matches those of other DFO methods, such as direct search or methods based on finite difference gradient approximation.

- It was shown that model-based trust region methods (Powell’s or other) can be applied within a random subspace algorithm which results in a further improvement in the worst case complexity bound to $\mathcal { O } ( n \epsilon ^ { - 2 } )$ .

In this paper we extend the results of [11] in several important ways.

- The method in [11] still departs from Powell’s method in its geometry handling approach. We will explain the details of this when we describe the algorithm but the key difference is that the method in [11] discards some potentially useful sample points, which Powell’s methods do not. Including these points complicates the theory substantially, but we are able to provide such theory here.

- We derive the theory based on linear interpolation since this is sufficient for the first-order convergence analysis, however, with an easy modification we allow for quadratic models without any additional complexity cost.

- We extend the analysis to functions with deterministic noise, which is essential when addressing real DFO applications. Deterministic noise implies a lower bound on the best reachable optimality criterion and we derive such a lower bound.

- We extend the analysis of the random subspace DFO trust region method to the case of deterministic noise. As we will show, there is an additional complication which requires algorithmic modifications in this case (as opposed to the full-space case or noise-free subspace case).

- Finally, we provide a careful implementation of our ideas and demonstrate that our theoretically supported algorithm can match the performance of other trust region DFO methods, including those in [24, 27].

### 1.1 Preliminaries

The following are the standard assumptions on $\phi ( x )$ for our setting.

**Assumption 1.1** (Lower bound on $\phi$). The function $\phi$ is bounded below by a scalar $\phi^\star$ on $\mathbb{R}^n$.

**Assumption 1.2** (Lipschitz continuous gradient). The function $\phi$ is continuously differentiable, and its gradient is $L$-Lipschitz continuous on $\mathbb{R}^n$, meaning

$$
\|\nabla \phi(y) - \nabla \phi(x)\| \leq L\|y - x\|
$$

for all $x, y \in \mathbb{R}^n$.

Throughout the paper, we assume that we do not have access to $\nabla \phi ( x )$ or its approximation using any form of differentiation. Instead we have access to an inexact zeroth-order oracle $f ( x ) \approx \phi ( x )$ . Specifically, for all x we assume

$$
| f (x) - \phi (x) | \leq \epsilon_ {f}
$$

for some $\epsilon _ { f } > 0$

Our main objective is to derive algorithms with the following type of guarantees. For a given $\epsilon > 0$ , let $\mathcal { C } _ { \epsilon }$ denote the total number of calls to oracle $f ( x )$ performed by the given algorithm that guarantees reaching a point x for which $\| \nabla \phi ( x _ { \epsilon } ) \| \le \epsilon .$ Then we seek the following guarantees: $\forall \epsilon > \psi ( \epsilon _ { f } ) \ { \mathcal { C } } _ { \epsilon } \leq \Psi ( n , \epsilon )$ , if the algorithm is deterministic, or $\mathbb { E } [ \mathcal { C } _ { \epsilon } ] \le \Psi ( n , \epsilon )$ , if the algorithm is stochastic. Here $\psi$ is some function of the oracle noise that gives the best achievable optimality criterion and Ψ is the complexity bound. We will make use of $\mathcal { O } ( )$ notation to suppress dependence on constants in upper bounds, Ω() notation to do the same in the case of lower bounds and Θ() to be used with an equality up to a constant factor. For all methods considered here as for all methods of similar type $\Psi ( ) = \mathcal { O } ( \epsilon ^ { - 2 } )$ . Thus the main focus of this work is in the dependence of $\Psi ( n , \epsilon )$ on n and the dependence of $\psi ( )$ on n and $\epsilon _ { f }$ . This reflects practical concerns with respect to derivative-free optimization, which in contrast to the usual derivative-based optimization has complexity dependence on n and is known not to scale well for large dimensional problems.

The paper is organized as follows: In Section 2 we present the analysis of the basic model-based trust region framework and present some key results based on what is known as fully-linear models that drive complexity analysis. In Section 3 we establish how these fully linear models arise in polynomial interpolation and quantify their properties in terms of Lagrange polynomials associated with the sample sets. In Section 4 we propose our main algorithm that uses Lagrange polynomials as a tool for model maintenance and analyze its complexity. Section 5 focuses on the subspace trust region method with inexact zeroth-order oracle. Finally, in Section 6 we describe implementation enhancements and present our computational comparisons.

## 2 Basic Algorithm and Elements of Complexity Analysis

We first present and analyze the trust region framework with the focus on its key elements.

As in most trust region algorithms, at every iteration $k \in \{ 0 , 1 , \ldots \}$ , we construct a quadratic model to approximate $\phi ( x )$ near the iterate $x _ { k }$

$$
m _ {k} (x _ {k} + s) = \phi (x _ {k}) + g _ {k} (x _ {k}) ^ {T} s + \frac {1}{2} s ^ {T} H _ {k} (x _ {k}) s.
$$

The model is then minimized (approximately) over the trust region $B ( x _ { k } , \Delta _ { k } ) \textrm { - a }$ Euclidean ball around $x _ { k }$ of a radius $\Delta _ { k }$ . In the paper we use the abbreviation $g _ { k } : = g ( x _ { k } ) = \nabla m _ { k } ( x _ { k } )$ and $H _ { k } : = H ( x _ { k } ) = \nabla ^ { 2 } m _ { k } ( x _ { k } ) . ^ { 1 }$

The following definition, also widely used in the literature [13], helps us identify the requirements on models $m _ { k }$ that are critical for convergence.

**Definition 2.1** (Fully-linear model). Given a ball around point x of radius $\Delta , B ( x , \Delta )$ , we say that model $m ( x + s )$ is a $\kappa _ { e f } , \kappa _ { e g } - f u l l y$ linear model $o f \phi ( x + s )$ on $B ( x , \Delta ) \ i f$

$$
\| \nabla m (x) - \nabla \phi (x) \| \leq \kappa_ {e g} \Delta
$$

and

$$
| m (x + s) - \phi (x + s) | \leq \kappa_ {e f} \Delta^ {2}
$$

for all $\| s \| \leq \Delta$

We now state the algorithmic framework where at each iteration updates are made based on progress and also on whether or not the model is known to be fully linear. We do not need to specify $\kappa _ { e g }$ and $\kappa _ { e f }$ constants in the algorithm but we assume they exist and are fixed throughout the iterations. In the framework below we do not even discuss how to verify if a model is $\kappa _ { e f } , \kappa _ { e g } .$ -fully-linear model, we simply assume such mechanism is given as an input. In the later section we will address this aspect in detail and incorporate it into the algorithm.

```text
Algorithm 1: Trust region method based on fully-linear models
Inputs: A zeroth-order oracle \( f(x) \approx \phi(x) \), a starting point \( x_0 \), TR radius \( \Delta_0 \), and hyperparameters \( \eta_1 \in (0,1) \), \( \eta_2 > 0 \), and \( \gamma \in (0,1) \). Mechanism for establishing if a model is \( \kappa_{ef}, \kappa_{eg} \)-fully-linear for some fixed \( \kappa_{ef}, \kappa_{eg} \).
for \( k = 0, 1, 2, \cdots \) do
1 Compute model \( m_k \) and a trial step \( x_k + s_k \) where \( s_k \approx \arg\min_s\{m_k(x_k + s) : s \in B(0, \Delta_k)\} \).
2 Compute the ratio \( \rho_k \) as
\( \rho_k = \frac{f(x_k) - f(x_k + s_k)}{m_k(x_k) - m_k(x_k + s_k)} \).
3 Update the iterate and the TR radius as
\( (x_{k+1}, \Delta_{k+1}) \leftarrow \begin{cases}(x_k + s_k, \gamma^{-1}\Delta_k) & \text{if } \rho_k \geq \eta_1 \text{ and } \|g_k\| \geq \eta_2\Delta_k, \\(x_k, \Delta_k) & \text{else, if model is not FL in } B(x_k, \Delta_k). \\(x_k, \gamma\Delta_k) & \text{otherwise.} \end{cases} \)
4 Perform some model improvement steps if appropriate.
```

We will make the following standard assumption on the models $m _ { k }$ and their minimization.

**Assumption 2.2**

1. The trust region subproblem is solved sufficiently accurately in each iteration k so that $x _ { k } + s _ { k }$ provides at least a fraction of Cauchy decrease, i.e. for some constant $0 < \kappa _ { f c d } < 1$

$$
m _ {k} (x _ {k}) - m _ {k} (x _ {k} + s _ {k}) \geq \frac {\kappa_ {f c d}}{2} \| g _ {k} \| \min \bigg \{\frac {\| g _ {k} \|}{\| H _ {k} \|}, \Delta_ {k} \bigg \}.
$$

2. There exists a constant $\kappa _ { b h m } > 0$ such that, for all $x _ { k }$ generated by Algorithm 1, the spectral norm of the Hessian of the model is bounded as

$$
\| H _ {k} \| \leq \kappa_ {b h m}.
$$

Condition (2.2) is commonly used in the literature and is satisfied by the Cauchy point with $\kappa _ { \mathrm { f c d } } = 1$ . See [12, Section 6.3.2] for more details.

Algorithm 1 is a variant of a standard trust region method, well studied in the literature [12]. There are two key differences that appear in trust region methods specifically in the DFO context. The first one is the condition $\| g _ { k } \| \ge \eta _ { 2 } \Delta _ { k }$ . This condition is not used in classical TR methods where $\nabla m _ { k } ( x _ { k } ) = g _ { k } = \nabla \phi ( x _ { k } )$ and was first introduced in [3] for the case of a TR method based on random models. The reason for this condition is tied to the fact that the trust region radii have a dual function in this setting - controlling the step of the algorithm and also controlling the model accuracy. For the same reasons, most prior deterministic DFO literature relies on a much less practical and cumbersome "criticality step". The condition $\| g _ { k } \| \ge \eta _ { 2 } \Delta _ { k }$ ensures that the trust region radius and thus the gradient accuracy stays on track as the norm of the gradient reduces which removes the need for a separate criticality step. The second difference for the derivative-free model-based TR (versus the classical TR) method is the necessity to ensure that the model is fully-linear in the trust region when the trust region radius is reduced. One can simply assume that this is guaranteed, for example, by using finite difference gradient approximation to build the model at each iteration. This gives an easy and convenient way to analyze the algorithm, but this does not result in a practical method. In various prior works [22, 13, 11] efficient methods have been proposed to recognize whether a model is fully-linear and if not, to make an improving step. The purpose of this paper is to improve on these methods both in terms of theory and practice.

Step 3 of Algorithm 1 identifies three types of iterations: successful iteration, where the trial step is accepted and the trust region radius is increased, unsuccessful iteration, where the step is rejected and the trust region radius is decreased and what we will call model improving iteration where the step is rejected but the trust region radius is not decreased because the model is not fully-linear.

In this section we will derive the bound on the total number of successful and unsuccessful iterations and in the following section we will provide a mechanism for model improvement and will bound the number of the model improving iterations. For a given $\epsilon > 0$ , let $K _ { \epsilon }$ be the first iteration of Algorithms 1 for which $\| \nabla \phi ( x _ { k } ) \| \le \epsilon$ . We

define the index sets

$$
\begin{array}{c} \mathcal {S} _ {\epsilon} := \{k \in \{0, \ldots , K _ {\epsilon} - 1 \}: \text {iteration k is successful} \}. \\ \mathcal {M} _ {\epsilon} := \{k \in \{0, \ldots , K _ {\epsilon} - 1 \}: \text {iteration k is model improving} \}. \\ \mathcal {U} _ {\epsilon} := \{k \in \{0, \ldots , K _ {\epsilon} - 1 \}: \text {iteration k is unsuccessful} \}. \end{array}
$$

We next present two lemmas that are key in the analysis of any trust region algorithm and specifically Algorithm 1. The first lemma establishes that once $\Delta _ { k }$ is sufficiently small compared to the gradient norm, a successful step is ensured.

**Lemma 2.3** (small $\Delta _ { k }$ implies successful step). Under Assumption 2.2, if m<sub>k</sub> is κ<sub>ef</sub>, κ<sub>eg</sub>-fully-linear and

$$
\sqrt {\frac {2 \epsilon_ {f}}{C _ {0}}} \leq \Delta_ {k} \leq C _ {1} \| \nabla \phi (x _ {k}) \|, w h e r e C _ {1} = \left(\max \left\{\eta_ {2}, \kappa_ {b h m}, \frac {2 \kappa_ {e f} + C _ {0}}{(1 - \eta_ {1}) \kappa_ {f c d}} \right\} + \kappa_ {e g}\right) ^ {- 1}
$$

and $C _ { 0 }$ is an arbitrary constant which we pick to equal max $\{ \eta _ { 2 } , \kappa _ { e f } \}$ for future convenience, then $\rho ~ \geq ~ \eta _ { 1 }$ ， $\| g _ { k } \| \ge \eta _ { 2 } \Delta _ { k }$ , thus iteration k is successful and $x _ { k + 1 } = x _ { k } + s _ { k }$

The proof is a simple modification of those in the stochastic trust region literature $[ 6 , 7 ] ^ { 2 }$

**Proof.** By the assumption that $m _ { k }$ is fully-linear, we have

$$
\| \nabla \phi (x _ {k}) \| \leq \| g _ {k} \| + \kappa_ {e g} \Delta_ {k}.
$$

From (2.4) we conclude that

$$
\begin{array}{c} (\max \{\kappa_ {b h m}, \eta_ {2} \} + \kappa_ {e g}) \Delta_ {k} \leq \| \nabla \phi (x _ {k}) \| \leq \| g _ {k} \| + \kappa_ {e g} \Delta_ {k} \\ \max \{\kappa_ {b h m}, \eta_ {2} \} \Delta_ {k} \leq \| g _ {k} \|. \end{array}
$$

This implies that the first condition of the successful step, namely $\| g _ { k } \| \ge \eta _ { 2 } \Delta _ { k }$ , is satisfied. From (2.2) we have $m _ { k } ( x _ { k } ) - m _ { k } ( x _ { k } + s _ { k } ) \geq \kappa _ { f c d } \| g _ { k } \| \Delta _ { k } / 2$ . Thus, using the assumption that $m _ { k }$ is fully-linear again,

$$
\begin{array}{l} \rho_ {k} = \frac {m (x _ {k}) - m (x _ {k} + s _ {k}) + (f (x _ {k}) - m (x _ {k})) - (f (x _ {k} + s _ {k}) - m (x _ {k} + s _ {k}))}{m (x _ {k}) - m (x _ {k} + s _ {k})} \\ = \frac {m (x _ {k}) - m (x _ {k} + s _ {k}) + (\phi (x _ {k}) - m (x _ {k})) - (\phi (x _ {k} + s _ {k}) - m (x _ {k} + s _ {k}))}{m (x _ {k}) - m (x _ {k} + s _ {k})} \\ + \frac {f (x _ {k}) - \phi (x _ {k}) + (f (x _ {k} + s _ {k}) - \phi (x _ {k} + s _ {k}))}{m (x _ {k}) - m (x _ {k} + s _ {k})} \\ \geq 1 - \frac {\kappa_ {e f} \Delta_ {k} ^ {2}}{m (x _ {k}) - m (x _ {k} + s _ {k})} - \frac {2 \epsilon_ {f}}{m (x _ {k}) - m (x _ {k} + s _ {k})} \\ \geq 1 - \frac {\kappa_ {e f} \Delta_ {k} ^ {2}}{\kappa_ {f c d} \| g _ {k} \| \Delta_ {k} / 2} - \frac {2 \epsilon_ {f}}{\kappa_ {f c d} \| g _ {k} \| \Delta_ {k} / 2} \geq 1 - \frac {(2 \kappa_ {e f} + 2 \epsilon_ {f} / \Delta_ {k} ^ {2}) \Delta_ {k}}{\kappa_ {f c d} (\| \nabla \phi (x _ {k}) \| - \kappa_ {e g} \Delta_ {k})} \\ \geq 1 - \frac {(2 \kappa_ {e f} + C _ {0}) \Delta_ {k}}{\kappa_ {f c d} (\| \nabla \phi (x _ {k}) \| - \kappa_ {e g} \Delta_ {k})} \geq \eta_ {1}, \end{array}
$$

where the last inequality follows from (2.4) since $\begin{array} { r } { \| \nabla \phi ( x _ { k } ) \| \ge \big ( \frac { 2 \kappa _ { e f } } { ( 1 - \eta _ { 1 } ) \kappa _ { f c d } } + \kappa _ { e g } \big ) \Delta _ { k } } \end{array}$

**Lemma 2.4** (successful iteration implies function reduction). Let Assumptions 1.2 and 2.2 hold. If the iteration k is successful, then

$$
\phi (x _ {k}) - \phi (x _ {k + 1}) \geq C _ {2} \Delta_ {k} ^ {2} - 2 \epsilon_ {f}, w h e r e C _ {2} = \frac {\eta_ {1} \eta_ {2} \kappa_ {f c d}}{2} \min \{\frac {\eta_ {2}}{\kappa_ {b h m}}, 1 \};
$$

otherwise, we have $x _ { k + 1 } = x _ { k }$ and $\phi ( x _ { k } ) - \phi ( x _ { k + 1 } ) = 0$

This proof can be found in [11]. Thus, under the additional assumption that $\begin{array} { r } { \Delta _ { k } \ge \sqrt { \frac { 2 \epsilon _ { f } } { \tau C _ { 2 } } } } \end{array}$ for some $\tau \in ( 0 , 1 )$ (2.5) can be further stated as

$$
\phi (x _ {k}) - \phi (x _ {k + 1}) \geq (1 - \tau) C _ {2} \Delta_ {k} ^ {2}, \mathrm{where} C _ {2} = \frac {\eta_ {1} \eta_ {2} \kappa_ {f c d}}{2} \min \{\frac {\eta_ {2}}{\kappa_ {b h m}}, 1 \}.
$$

Lemmas 2.3 and 2.4 give us a bound on the total number of successful and unsuccessful iterations, as long as it can be ensured that $\Delta _ { k }$ remains not smaller than $\sqrt { \frac { 2 \epsilon _ { f } } { \operatorname* { m i n } \{ \tau C _ { 2 } , C _ { 0 } \} } }$ . Henceforth, for simplicity we use $\begin{array} { r } { \tau = \frac { 1 } { 2 } } \end{array}$ also since we chose $C _ { 0 } \geq \eta _ { 2 }$ and $\eta _ { 2 } \geq C _ { 2 }$ the bound on $\Delta _ { k }$ reduces to $\sqrt { \frac { 4 \epsilon _ { f } } { C _ { 2 } } }$ . Due to Lemma 2.3 and the update mechanism for $\Delta _ { k }$ the bound is ensured as long as $\| \nabla \phi ( x _ { k } ) \| \ge \epsilon$ for ϵ sufficiently large.

**Lemma 2.5** (Lower bound on $\Delta _ { k } )$ ). For any $\epsilon > \sqrt { \frac { 4 \epsilon _ { f } } { \gamma ^ { 2 } C _ { 2 } C _ { 1 } ^ { 2 } } }$ , assuming that $\Delta _ { 0 } ~ > ~ \gamma C _ { 1 } \epsilon$ , for all $k \in$ $\{ 0 , \ldots , K _ { \epsilon } - 1 \} ~ \Delta _ { k } \geq \gamma C _ { 1 } \epsilon$

**Proof.** According to Lemma 2.3, any iteration $k \in \{ 0 , \ldots , K _ { \epsilon } - 1 \}$ must be successful when $\Delta _ { k } \le C _ { 1 } \epsilon$ and $m _ { k }$ is $\kappa _ { e f } , \kappa _ { e g } \mathrm { - f u l l y }$ linear. Thus no iteration can be unsuccessful when $\Delta _ { k } \leq C _ { 1 } \epsilon .$ . Thus, given $\Delta _ { 0 } > \gamma C _ { 1 } \epsilon$ , and by the mechanism of Algorithm 1 we must have $\Delta _ { k } \geq \gamma C _ { 1 } \epsilon$ for all $k \in \{ 0 , \ldots , K _ { \epsilon } - 1 \}$ □

The following bound holds under the result of Lemma 2.5.

**Lemma 2.6** (Bound of successful iterations). For any $\epsilon > \sqrt { \frac { 4 \epsilon _ { f } } { \gamma ^ { 2 } C _ { 2 } C _ { 1 } ^ { 2 } } }$ , assuming $\Delta _ { 0 } > \gamma C _ { 1 } \epsilon _ { : }$ , we have

$$
| \mathcal {S} _ {\epsilon} | \leq \frac {2 (\phi (x _ {0}) - \phi^ {\star})}{C _ {2} (\gamma C _ {1} \epsilon) ^ {2}}
$$

**Proof.** Using Lemma 2.4 with $\begin{array} { r } { \tau = \frac { 1 } { 2 } } \end{array}$ and from $\Delta _ { k } \geq \gamma C _ { 1 } \epsilon$ for all $k \in \{ 0 , \ldots , K _ { \epsilon } - 1 \}$ we have

$$
\phi (x _ {0}) - \phi^ {\star} \geq \sum_ {k = 0} ^ {K _ {\epsilon} - 1} \phi (x _ {k}) - \phi (x _ {k + 1}) \geq \frac {1}{2} \sum_ {k \in \mathcal {S} _ {\epsilon}} C _ {2} \Delta_ {k} ^ {2} > \frac {1}{2} | \mathcal {S} _ {\epsilon} | C _ {2} (\gamma C _ {1} \epsilon) ^ {2}
$$

which gives the result of the lemma.

**Lemma 2.7**. For any $\epsilon > \sqrt { \frac { 4 \epsilon _ { f } } { \gamma ^ { 2 } C _ { 2 } C _ { 1 } ^ { 2 } } }$ , assuming that the initial trust-region radius $\Delta _ { 0 } > \gamma C _ { 1 } \epsilon$ ，

$$
\left| \mathcal {U} _ {\epsilon} \right| \leq \left| \mathcal {S} _ {\epsilon} \right| + \lceil \left(\log_ {\gamma} \frac {C _ {1} \epsilon}{\Delta_ {0}}\right) \rceil .
$$

**Proof.** We observe that

$$
\Delta_ {K _ {\epsilon}} = \gamma^ {- | \mathcal {S} _ {\epsilon} |} \gamma^ {| \mathcal {U} _ {\epsilon} |} \Delta_ {0} \geq C _ {1} \epsilon ,
$$

The last inequality follows from the fact that $\Delta _ { K _ { \epsilon } - 1 } \geq \gamma C _ { 1 } \epsilon$ and the $K _ { \epsilon } - 1 \mathrm { - t h }$ iteration must be successful. Thus the number of unsuccessful iterations can be bounded using the number of successful ones rearranging the terms and taking the logarithm.

**Theorem 2.8**. Let Assumptions 1.2 and 2.2 hold. For any $\epsilon > \sqrt { \frac { 4 \epsilon _ { f } } { \gamma ^ { 2 } C _ { 2 } C _ { 1 } ^ { 2 } } }$ , assuming that the initial trust-region radius $\Delta _ { 0 } > \gamma C _ { 1 } \epsilon ,$ , where $C _ { 1 }$ is defined as $\begin{array} { r } { \bigg ( \operatorname* { m a x } \left\{ \eta _ { 2 } , \ \kappa _ { b h m } , \ \frac { 2 \kappa _ { e f } + \operatorname* { m a x } \{ \eta _ { 2 } , \kappa _ { e f } \} } { ( 1 - \eta _ { 1 } ) \kappa _ { f c d } } \right\} + \kappa _ { e g } \bigg ) ^ { - 1 } } \end{array}$ , then we have the bound

$$
| \mathcal {S} _ {\epsilon} | + | \mathcal {U} _ {\epsilon} | \leq \frac {4 (\phi (x _ {0}) - \phi^ {\star})}{C _ {2} (\gamma C _ {1} \epsilon) ^ {2}} + \lceil \left(\log_ {\gamma} \frac {C _ {1} \epsilon}{\Delta_ {0}}\right) \rceil .
$$

Constants $C _ { 1 }$ and $C _ { 2 }$ have a direct effect on the complexity and we will be using them (and their variations) throughout the paper. Let us pause here to understand their different components. Specifically, constants γ, η<sub>1</sub> and $\kappa _ { f c d }$ are usually chosen to be fixed in the algorithm and be close to 1 (say 0.9).

The key remaining constants are $\eta _ { 2 }$ which is chosen in the algorithm and is used as specified, and $\kappa _ { e f } , ~ \kappa _ { e g }$ and $\kappa _ { b h m }$ which are all attributes of the constructed models and are not specified by the algorithm but are rather upper bounded by theory.

In what follows we will impose an upper bound on $\kappa _ { b h m }$ which will be dimension independent. Ideally $\kappa _ { b h m }$ should scale similarly to L, since the later is the bound on the norm of the true Hessian and the former is the bound on the model Hessian. Thus it is convenient to think of $\kappa _ { b h m }$ as $\sim \mathcal { O } ( L )$ , although it also can be zero if linear models are used but also can be large if allowed.

What remains is to derive concrete bounds on $\kappa _ { e g } , \kappa _ { e f }$ with explicit dependence on dimension. These would depend on the manner in which the models are constructed.

The following standard lemma shows that a bound $\kappa _ { e g }$ (together with $\kappa _ { b h m } )$ implies a bound on $\kappa _ { e f }$

**Lemma 2.9** (Fully linear models). Under Assumptions 1.2 and 2.2 if

$$
\| \nabla m (x) - \nabla \phi (x) \| \leq \kappa_ {e g} \Delta
$$

then $m _ { k } ( x _ { k } + s )$ is a $\kappa _ { e f } , \kappa _ { e g } - f u l l y$ linear model of $\phi ( x + s )$ on $B ( x , \Delta )$ with $\begin{array} { r } { \kappa _ { e f } = \kappa _ { e g } + \frac { L + \kappa _ { b h m } } { 2 } } \end{array}$

Let us consider a concrete way of building fully-linear models and the complexity implications. From analysis of the finite difference gradient approximation error (see e.g.[4]), if one forms a gradient estimate via

$$
g (x) = \sum_ {i = 1} ^ {n} \frac {f (x + \delta u _ {i}) - f (x)}{\delta} u _ {i}
$$

then one can derive a gradient approximation bound of

$$
\| g (x) - \nabla \phi (x) \| \leq \frac {\sqrt {n} L \delta}{2} + \frac {2 \sqrt {n} \epsilon_ {f}}{\delta}.
$$

Taking $\delta = \Delta _ { k } .$ at each iteration $k ,$ results in a $\kappa _ { e f } , \kappa _ { e g }$ -fully linear model in $B ( x _ { k } , \Delta _ { k } )$ with $\kappa _ { e g } = \sqrt { n } L$ and $\begin{array} { r } { \kappa _ { e f } = \frac { L + \kappa _ { b h m } } { 2 } + \sqrt { n } L } \end{array}$ , as long as $\Delta _ { k } \ge 2 \sqrt { \frac { \epsilon _ { f } } { L } }$ . To ensure this we add the lower bound $\gamma C _ { 1 } \epsilon \geq 2 \sqrt { \frac { \epsilon _ { f } } { L } }$ which translates to $\begin{array} { r } { \epsilon \geq \sqrt { \frac { 4 \epsilon _ { f } } { \gamma ^ { 2 } L C _ { 1 } ^ { 2 } } } } \end{array}$

The total bound on the number of function evaluations, i.e., oracle complexity, easily follows from Theorem 2.8 and from the fact that (2.9) gives a fully linear model at the cost of n + 1 oracle calls. Thus there are no model improvement iterations and the final bound on the oracle complexity is derived via the bound on $( n + 1 ) ( | S _ { \epsilon } | + | \mathcal { U } _ { \epsilon } | )$ As noted in [11], if $\eta _ { 2 }$ is taken to be a constant independent of dimension, using this finite difference scheme, the total worse-case oracle complexity to achieve $\| \nabla \phi ( x _ { k } ) \| \leq \epsilon$ for any $\epsilon > \sqrt { \frac { 4 \epsilon _ { f } } { \gamma ^ { 2 } \operatorname* { m i n } \{ C _ { 2 } , L \} C _ { 1 } ^ { 2 } } }$ is bounded by

$$
\mathcal {C} _ {\epsilon} \leq \mathcal {O} (n ^ {2} \epsilon^ {- 2}).
$$

In the following corollary, we show that this complexity is not optimal and can, in fact, be improved by taking $\eta _ { 2 }$ to grow with the dimension of the problem by improving the bound on $| S _ { \epsilon } | + | U _ { \epsilon } |$ when $\kappa _ { e f }$ and $\kappa _ { e g }$ scale as $\mathcal { O } ( \sqrt { n } )$

**Corollary 2.10**. Under the same assumptions as Theorem ${ \it 2 . 8 , }$ assuming that $g _ { k } ( x _ { k } )$ is computed via (2.9) with $\delta = \Delta _ { k }$ at each iteration, choosing $\eta _ { 2 } = \sqrt { n }$ , and assuming $\kappa _ { b h m } \leq \mathcal { O } ( \sqrt { n } )$ , then for any $\epsilon > \sqrt { \frac { 4 \epsilon _ { f } } { \gamma ^ { 2 } \operatorname* { m i n } \{ C _ { 2 } , L \} C _ { 1 } ^ { 2 } } }$

$$
\left| \mathcal {S} _ {\epsilon} \right| + \left| \mathcal {U} _ {\epsilon} \right| \leq \mathcal {O} \left(\frac {\sqrt {n}}{\epsilon^ {2}}\right)
$$

and the total worst-case oracle complexity is bounded by

$$
\mathcal {C} _ {\epsilon} \leq \mathcal {O} \left(\frac {n ^ {3 / 2}}{\epsilon^ {2}}\right)
$$

**Proof.** By the error bound from [4], we have $\begin{array} { r l r } { \kappa _ { e f } , \kappa _ { e g } } & { { } = } & { O ( \sqrt { n } ) } \end{array}$ Recall $\begin{array} { r l } { C _ { 1 } ^ { - 1 } } & { { } = } \end{array}$ max $\begin{array} { r } { \left\{ \eta _ { 2 } , \kappa _ { b h m } , \frac { 2 \kappa _ { e f } + \operatorname* { m a x } \{ \eta _ { 2 } , \kappa _ { e f } \} } { ( 1 - \eta _ { 1 } ) \kappa _ { f c d } } \right\} + \kappa _ { e g } } \end{array}$ and $\begin{array} { r l r } { C _ { 2 } } & { { } = } & { \frac { \eta _ { 1 } \eta _ { 2 } \kappa _ { f c d } } { 2 } \operatorname* { m i n } \left\{ \frac { \eta _ { 2 } } { \kappa _ { b h m } } , 1 \right\} } \end{array}$ . Thus, under the assumptions of this corollary, we have $C _ { 1 } ^ { - 1 } = \Theta ( { \sqrt { n } } )$ and $C _ { 2 } = \Theta ( \sqrt { n } )$ . Thus, we can bound

$$
\begin{array}{l} | \mathcal {S} _ {\epsilon} | + | \mathcal {U} _ {\epsilon} | \leq \frac {2 (f (x _ {0}) - f ^ {\star})}{C _ {2} (\gamma C _ {1} \epsilon) ^ {2}} + \log_ {\gamma} \frac {C _ {1} \epsilon}{\Delta_ {0}} \\ = \mathcal {O} \left(\frac {\sqrt {n} (\phi (x _ {0}) - \phi^ {\star})}{\epsilon^ {2}} + \log \left(\frac {\sqrt {n} \Delta_ {0}}{\epsilon}\right)\right) = \mathcal {O} \left(\frac {\sqrt {n}}{\epsilon^ {2}}\right). \end{array}
$$

The total oracle complexity follows as discussed before.

We note here that since we assume that L does not scale with n, while $C _ { 1 } ^ { - 1 } = \Theta ( { \sqrt { n } } )$ , then whether we choose $\eta _ { 2 }$ to be constant or to equal $\sqrt { n }$ the lower bound on ϵ is $\Omega \left( \sqrt { n \epsilon _ { f } } \right)$

In the next section we show how fully-linear models can be constructed via polynomial interpolation, using less rigid sample sets than used in (2.9) and yet we are able to derive a competitive bound on $\kappa _ { e g }$ (and thus $\kappa _ { e f } )$

## 3 Lagrange Polynomials and Fully-Linear Models

Before introducing the method we wish to analyze in this paper we need to discuss an important tool utilized by these algorithms: Lagrange polynomials. The concepts and the definitions below can be found in [13].

Lagrange polynomials and associated concepts will be defined with respect to a space of polynomials $\mathcal { P }$ of dimension $p .$ Typically $\mathcal { P }$ is either the set of linear or quadratic polynomials, but it also can be a set of quadratic polynomials with a pre-defined Hessian sparsity pattern.

**Definition 3.1**. Lagrange Polynomials

Given a space of polynomials P of dimension p and a set of points $\mathcal { Y } = \{ y _ { 1 } , . . . , y _ { p } \} \subset \mathbb { R } ^ { n }$ , a set of p polynomials $\ell _ { j } ( s )$ in $\mathcal { P } f o r j = 1 , \dotsc , p ,$ , is called a basis of Lagrange polynomials associated with Y, if

$$
\ell_ {j} (y _ {i}) = \delta_ {i j} = \left\{ \begin{array}{l l} 1 & i f \quad i = j, \\ 0 & i f \quad i \neq j. \end{array} \right.
$$

If the basis of Lagrange polynomials exists for the given Y then Y is said to be poised.

**Definition 3.2**. Λ–poisedness Given a space of polynomials P of dimension $p , \Lambda > 0$ , and a set $B \subset \mathbb { R } ^ { n }$ A poised set $\mathcal { V } = \{ y _ { 1 } , . . . , y _ { p } \}$ is said to be Λ–poised in B $i f y \subset B$ and for the basis of Lagrange polynomials associated with Y, it holds that

$$
\Lambda \geq \max _ {j = 1, \dots , p} \max _ {s \in \mathcal {B}} | \ell_ {j} (s) |.
$$

We now show how Λ–poisedness of the interpolation set can ensure that related interpolation model is fully linear and derive corresponding constants $\kappa _ { e f }$ and $\kappa _ { e g }$ . Throughout this section we apply Assumptions 1.2 and 2.2. By Lemma 2.9, all we need is to ensure (2.8). For this we specify a way to construct the model m(x).

Let ${ \mathcal { V } } _ { k }$ be a set of points and P be a space of polynomials. Assuming that $y _ { k } = \{ y _ { 1 } , . . . , y _ { p } \}$ is poised in $B ( 0 , \Delta )$ we define $g _ { k }$ and $H _ { k }$ to satisfy $\lVert H \rVert \leq \kappa _ { b h m }$ and

$$
g _ {k} ^ {\top} y + \frac {1}{2} y ^ {\top} H _ {k} y = \phi (x _ {k} + y) - \phi (x _ {k}), \quad \forall y \in \mathcal {Y} _ {k}.
$$

The model $m _ { k } ( x )$ is then defined as

$$
m _ {k} (x _ {k} + s) = \phi (x _ {k}) + g _ {k} ^ {T} s + \frac {1}{2} s ^ {T} H _ {k} s.
$$

We now show an important result that establishes a bound on $\kappa _ { e g }$ when $p = n$ and ${ \mathcal { V } } _ { k } = \{ y _ { 1 } , . . . , y _ { n } \}$ is Λ-poised for homogeneous linear interpolation. This result is an extension of a similar result in [11] which allows $m _ { k }$ to include a quadratic term.

**Theorem 3.3**. Let ${ \mathcal { Y } } = \{ y _ { 1 } , \ldots , y _ { n } \}$ be such that Y is Λ–poised in $B ( 0 , \Delta )$ . Let g, H satisfy $\lVert H \rVert \leq \kappa _ { b h m }$ and $g ^ { \top } y + { \textstyle \frac { 1 } { 2 } } y ^ { \top } H y = \phi ( x + y ) - \ddot { \phi } ( x )$ , for each $y \in \mathcal { V }$ . Then

$$
\| \nabla \phi (x) - g \| \leq \frac {1}{2} \left(L + \kappa_ {b h m}\right) \Delta \sqrt {n} \sqrt {n (\Lambda^ {2} - 1) + 2}.
$$

In particular, $i f \Lambda = 1 + O ( 1 / n )$ , then we will have $\| \nabla \phi ( x ) - g \| = O ( \sqrt { n } ) \Delta$

**Proof.** Let $\bar { \phi } ( s ) = \phi ( x + s ) - \phi ( x )$ , then $\bar { \phi } ( 0 ) = 0$ and $\nabla \phi ( x ) = \nabla \bar { \phi } ( 0 )$ . Let Y be the matrix with ith column equal to $y _ { i }$ , let $D$ be the diagonal matrix such that $D _ { i i } = \| y _ { i } \|$ , and let ${ \bar { \phi } } ( Y )$ be the vector with ith entry equal to $\bar { \phi } ( y _ { i } )$ . The interpolation condition is

$$
g ^ {T} y _ {i} = \phi (x + y _ {i}) - \phi (x) - \frac {1}{2} y _ {i} ^ {T} H y _ {i}, i = 1, \ldots , n
$$

thus we have $g ~ = ~ Y ^ { - T } \bar { \phi } ( Y ) - Y ^ { - T } h ( Y )$ , where $h ( Y )$ is a vector with components $\frac { 1 } { 2 } y _ { i } ^ { T } H y _ { i }$ Then we have $\lVert \nabla \phi ( x ) - g \rVert ~ \leq ~ \lVert \nabla \phi ( x ) - Y ^ { - T } \bar { \phi } ( Y ) \rVert ~ + ~ \lVert Y ^ { - T } \dot { h ( Y ) } \rVert$ . By the proof of Theorem 4.3 in [11], we have $\| Y ^ { - T } D \| \le \sqrt { n ( \Lambda ^ { 2 } - 1 ) + 2 }$ and

$$
\| \nabla \phi (x) - Y ^ {- T} \bar {\phi} (Y) \| \leq \frac {1}{2} \sqrt {n} L \Delta \sqrt {n (\Lambda^ {2} - 1) + 2}
$$

and we need only bound $\| Y ^ { - T } h ( Y ) \|$ . We have $\| Y ^ { - T } h ( Y ) \| \le \| Y ^ { - T } D \| \| D ^ { - 1 } h ( Y ) \|$ . Observe that the ith element of $D ^ { - 1 } \bar { h ( Y ) }$ is $\frac { 1 } { 2 } \tilde { y _ { i } ^ { T } } H y _ { i } / \| y _ { i } \|$ and is therefore bounded by $\frac { 1 } { 2 } \kappa _ { b h m } \Delta$ . Now we have $\| D ^ { - 1 } h ( Y ) \| ~ \leq$ $\begin{array} { r } { \sqrt { n } \| D ^ { - 1 } h ( Y ) \| _ { \infty } \leq \frac { 1 } { 2 } \sqrt { n } \kappa _ { b h m } \Delta } \end{array}$ . Thus $\begin{array} { r } { \| Y ^ { - T } h ( Y ) \| \le \frac { 1 } { 2 } \sqrt { n } \kappa _ { b h m } \Delta \sqrt { n ( \bar { \Lambda ^ { 2 } } - 1 ) + 2 } } \end{array}$ , and in total

$$
\| \nabla \phi (x) - g \| \leq \frac {1}{2} \left(L + \kappa_ {b h m}\right) \Delta \sqrt {n} \sqrt {n (\Lambda^ {2} - 1) + 2}.
$$

This theorem allows us to conclude that our model is fully linear when our interpolation set is Λ-poised.

**Corollary 3.4**. $I f m _ { k }$ is defined as in (3.2) and $i f \mathcal { D } _ { k }$ is Λ-poised, then $m _ { k }$ is $\kappa _ { e f } , \kappa _ { e g } - f u l l y - l i n e a r \ w i t h$

$$
\begin{array}{l} \kappa_ {e g} = \frac {1}{2} (L + \kappa_ {b h m}) \sqrt {n} \sqrt {n (\Lambda^ {2} - 1) + 2} \\ \kappa_ {e f} = \kappa_ {e g} + \frac {L + \kappa_ {b h m}}{2}. \end{array}
$$

**Proof.** This follows from the definition of $m _ { k }$ in (3.2), Theorem 3.3 and Lemma 2.9.

We note that constructing the model to satisfy (3.1) is not possible unless $\phi ( x + y ) - \phi ( x )$ can be computed exactly. In the case of inexact oracles we instead compute g and H from

$$
g _ {k} ^ {\top} y + \frac {1}{2} y ^ {\top} H _ {k} y = f (x _ {k} + y) - f (x _ {l}) \quad \forall y \in \mathcal {Y} _ {k}.
$$

The previous theorem is modified as follows.

**Theorem 3.5**. Let ${ \mathcal { Y } } = \{ y _ { 1 } , \ldots , y _ { n } \}$ be such that Y is Λ–poised in $B ( 0 , \Delta )$ . Let $g , H$ be computed to satisfy (3.3) and $\lVert H \rVert \leq \kappa _ { b h m }$ . Then

$$
\| \nabla \phi (x) - g \| \leq \sqrt {n (\Lambda^ {2} - 1) + 2} \left(\frac {1}{2} \left(L + \kappa_ {b h m}\right) \sqrt {n} \Delta + \sqrt {n} \frac {2 \epsilon_ {f} \Lambda}{\Delta}\right).
$$

**Proof.** Define $\bar { \phi } , ~ D , \ \bar { \phi } ( Y ) , \ h ( Y )$ as in the previous proof. Diverging from that proof, the interpolation condition changes to

$$
g ^ {T} y _ {i} = \phi (x + y _ {i}) - \phi (x) - \frac {1}{2} y _ {i} ^ {T} H y _ {i} + (f (x + y _ {i}) - \phi (x + y _ {i})) - (f (x) - \phi (x)), \quad i = 1, \ldots , n
$$

thus we have $g = Y ^ { - T } \bar { \phi } ( Y ) - Y ^ { - T } h ( Y ) + Y ^ { - T } E$ , where E is a vector with components $( f ( x + y _ { i } ) - \phi ( x + y _ { i } ) ) -$ $( f ( x ) - \phi ( x ) )$ . Then we can bound the error

$$
\begin{array}{r l} & {\| \nabla \phi (x) - g \| = \| Y ^ {- T} D (D ^ {- 1} Y ^ {T} \nabla \bar {\phi} (0) - D ^ {- 1} \bar {\phi} (Y) - D ^ {- 1} E) \| + \| Y ^ {- T} h (Y) \|} \\ & {\qquad \leq \sqrt {n} \| Y ^ {- T} D \| \| D ^ {- 1} Y ^ {T} \nabla \bar {\phi} (0) - D ^ {- 1} \bar {\phi} (Y) - D ^ {- 1} E \| _ {\infty} + \| Y ^ {- T} h (Y) \|} \\ & {\qquad \leq \sqrt {n} \| Y ^ {- T} D \| \left(\| D ^ {- 1} Y ^ {T} \nabla \bar {\phi} (0) - D ^ {- 1} \bar {\phi} (Y) \| _ {\infty} + \| D ^ {- 1} E \| _ {\infty}\right) + \| Y ^ {- T} h (Y) \|} \\ & {\qquad \leq \sqrt {n} \| Y ^ {- T} D \| \left(\| D ^ {- 1} Y ^ {T} \nabla \bar {\phi} (0) - D ^ {- 1} \bar {\phi} (Y) \| _ {\infty} + \| D ^ {- 1} \| _ {\infty} \| E \| _ {\infty}\right) + \| Y ^ {- T} h (Y) \|.} \end{array}
$$

We can bound $\| Y ^ { - T } D \| , \| D ^ { - 1 } Y ^ { T } \nabla \bar { \phi } ( 0 ) - D ^ { - 1 } \bar { \phi } ( Y ) \| _ { \infty } , \| Y ^ { - T } h ( Y ) \| _ { }$ identically as in the previous proof. To bound $\| E \| _ { \infty }$ observe that the condition that $| f ( x + y ) - \phi ( x + y ) | \leq \epsilon _ { f }$ for $y \in \mathcal { y } \cup \{ 0 \}$ implies $\| E \| _ { \infty } \leq 2 \epsilon _ { f }$ . To bound $\| D ^ { - 1 } \| _ { \infty }$ recall that $D _ { i i } = \| y _ { i } \|$ . By the properties of Lagrange polynomials we have $\ell _ { i } ( y _ { i } ) = 1$ . By linearity, we have $\begin{array} { r } { \ell _ { i } ( \Delta \frac { y _ { i } } { \| y _ { i } \| } ) = \frac { \Delta } { \| y _ { i } \| } } \end{array}$ . By the poisedness condition we have $\left| \ell _ { i } ( \Delta \frac { y _ { i } } { \| y _ { i } \| } ) \right| \leq \Lambda$ . Combining these bounds we have ${ \frac { \Delta } { \| y _ { i } \| } } \leq \Lambda$ and thus $\begin{array} { r } { \frac { 1 } { \| y _ { i } \| } \le \frac { \Lambda } { \Delta } } \end{array}$ . Thus $\begin{array} { r } { \| D ^ { - 1 } \| _ { \infty } \leq \frac { \Lambda } { \Delta } } \end{array}$ . The result follows. □

This theorem allows us to conclude that our model is fully linear when our interpolation set is Λ-poised and when $\Delta$ is sufficiently large.

**Corollary 3.6**. If m<sub>k</sub> is defined as in (3.2), $i f \mathcal { D } _ { k }$ is Λ-poised, and if $\begin{array} { r } { \Delta _ { k } \ge \sqrt { \frac { 4 \epsilon _ { f } \Lambda } { L + \kappa _ { b h m } } } } \end{array}$ , then $m _ { k }$ is $\kappa _ { e f } , \kappa _ { e g } -$ fully-linear with

$$
\begin{array}{l} \kappa_ {e g} = (L + \kappa_ {b h m}) \sqrt {n} \sqrt {n (\Lambda^ {2} - 1) + 2} \\ \kappa_ {e f} = \kappa_ {e g} + \frac {L + \kappa_ {b h m}}{2}. \end{array}
$$

**Proof.** This follows from the definition of $m _ { k }$ in (3.2), Theorem 3.5 and Lemma 2.9, since the bound on $\Delta _ { k }$ implies that

$$
\sqrt {n (\Lambda^ {2} - 1) + 2} \left(\frac {1}{2} (L + \kappa_ {b h m}) \sqrt {n} \Delta_ {k} + \sqrt {n} \frac {2 \epsilon_ {f} \Lambda}{\Delta_ {k}}\right) \leq 2 \frac {1}{2} (L + \kappa_ {b h m}) \Delta_ {k} \sqrt {n} \sqrt {n (\Lambda^ {2} - 1) + 2}.
$$

In the next section we propose a concrete algorithm for constructing Λ-poised sample sets and deriving the bound on the number of iterations required to do so.

## 4 Model Improving Iterations Based on Lagrange Polynomials

In [11] a specific algorithm based on the framework of Algorithm 1 was proposed and its complexity analyzed. This algorithm maintains a set Y of n sample points and a set of linear Lagrange polynomials. At each model improving iteration the algorithm performs, what we call, a geometry correcting step by either replacing an interpolation point outside the trust region, if such a point exists, or replacing a point whose corresponding Lagrange polynomial violates the Λ-poisedness condition. If no such improvement is possible, then the set is Λ-poised and thus the model is fully-linear, by Theorems 3.3 and 3.5, hence the iteration is unsuccessful and the trust region radius is reduced. It is shown in [11] that the number of geometry correcting iterations between any two other iterations is at most 3n. Each such iteration performs at most two function evaluations, while successful and unsuccessful iterations perform at most one. Thus using Theorem 2.8 one can derive the bound on $| \mathcal { M } _ { \epsilon } |$ and consequently the bound on the total complexity of the algorithm.

The method in [11] fails to include several important practical elements. Firstly, the models that are being constructed by any successful model-based DFO method are quadratic and are usually based on quadratic interpolation. Secondly, since the function value is computed at the trial step, even if the step is not accepted as the new iterate, the step provides a new sample point which may improve the interpolation model. In fact it is likely to do so, since the reason the step is rejected, to begin with, is due to disagreement of the model and the function at the trial step. Thus adding this trial step to the new model provides new information. The improvement guaranteed by replacing some interpolation point by the trial step can be quantified by the value of the corresponding Lagrange polynomial at the trial step. A principled algorithm relying on this property, which is referred to as self-correcting, has been developed in [26] and shown to converge to a stationary point in the limit. This algorithm uses only such self-correcting steps for model improvement and does not perform geometry correction steps. It also includes the criticality step which is not practical but helps the analysis. No complexity bounds have been developed in [26]. Powell utilized both the self-correcting and geometry correcting steps in his algorithms [21, 23]. In what follows we present an algorithm that allows models to be constructed using quadratic interpolation and utilizes the self-correcting and geometry correcting steps. On the other hand it only maintains a set of linear Lagrange polynomials and ensures only a subset of interpolation points to be Λ-poised for linear interpolation. Combined with the results of the previous section this allows the algorithm to ensure eventual construction of fully-linear models. After we present the algorithm we state and prove the bound on the number of consecutive model improving iterations.

Algorithm 2 is the description of our proposed method that on the one hand tries to include most of the practical elements of Powell’s methods and on the other hand nearly matches the complexity of the simplified method in [11].

At each iteration the method maintains two sets of points - set Y of n points whose geometry is monitored and maintained by means of the associated linear Lagrange polynomials and another set Z of $p \leq ( n - 1 ) n / 2$ points whose geometry is only monitored and maintained in terms of distance to the trust region center.

We construct the model by solving the following constrained least squares problem.

(4.1)

$$
\min _ {g, H} \sum_ {z \in \mathcal {Z} _ {k}} \left(f (x _ {k}) + g ^ {\top} z + \frac {1}{2} z ^ {T} H z - f (x _ {k} + z)\right) ^ {2}
$$

subject to: ∥H∥ ≤ K

$$
g ^ {\top} y + \frac {1}{2} y ^ {\top} H y = f (x _ {k} + y) - f (x _ {k}), \quad \forall y \in \mathcal {Y} _ {k}.
$$

Any solution to this problem will satisfy $\lVert H \rVert \leq \kappa _ { b h m }$ with $\kappa _ { b h m } \leq K$ . If a quadratic model exists whose Hessian satisfies $\lVert H \rVert \leq \kappa _ { b h m }$ with $\kappa _ { b h m } \leq K$ and that interpolates all points in Y and Z, then such model will be an optimal solution to this problem.<sup>3</sup> When $\begin{array} { r } { p < \frac { n ( n - 1 ) } { 2 } } \end{array}$ then the problem may have multiple optimal solutions. In the case when this happens we can select the solution with the smallest Frobenius norm (see [13]), as long as it satisfies $\| H \| \leq K$ . Alternatively, following Powell’s ideas from [21] we can select the solution for which H is the closest in Frobenius norm to the Hessian from the previous iteration. We will discuss both of these techniques in the computational section.

```text
Algorithm 2: Geometry-correcting algorithm

Inputs: A zeroth-order oracle $f(x) \approx \phi(x)$ , $p = \frac{(n-1)n}{2}$ , $\Delta_0$ , $x_0$ , $\gamma \in (0,1)$ $\eta_1 > 0$ , $\eta_2 > 0$ , $\Lambda > 1$ , $\Lambda_{sc} \geq 1$ .

Initialization An initial set $\mathcal{Y}_0$ such that $|\mathcal{Y}_0| = n$ , an initial set $\mathcal{Z}_0$ such that $|\mathcal{Z}_0| \leq p$ and the function values $f(x_0)$ , $f(x_0 + y_i)$ , $y_i \in \mathcal{Y}_0$ , $f(x_0 + z_i)$ , $z_i \in \mathcal{Z}_0$ . A set of Lagrange Polynomials $\{\ell_i(x), i = 1, \ldots, n\}$ in $\mathcal{P}$ for the set $\mathcal{Y}_0$ .

for $k = 0, 1, 2, \ldots$ do

1 Build a quadratic model $m_k(x_k + s)$ as in (3.2) using $f(x_k)$ and $f(x_k + y_i)$ , $y_i \in \mathcal{Y}_k$ , $f(x_0 + z_i)$ , $z_i \in \mathcal{Z}_k$ .

2 Compute a trial step $s_k$ and ratio $\rho_k$ as in Algorithm 1.

3 Successful iteration: $\rho_k \geq \eta_1$ and $\|g_k\| \geq \eta_2\Delta_k$ .

Set $x_{k+1} = x_k + s_k$ , $\Delta_{k+1} = \gamma^{-1}\Delta_k$ . $j_k^* = \arg \max_{j=1,\dots,n} \|y_j - s_k\|$ , $i_k^* = \arg \max_{i=1,\dots,p} \|z_i - s_k\|$ .

If $\|y_{j_k^*} - s_k\| > \|z_{i_k^*} - s_k\|^a$ and $|\ell_{j_k^*}(s_k)| > 0$ then update set $\mathcal{Y}_{k+1} = (\mathcal{Y}_k \setminus \{y_{j_k^*}\} \cup \{0\}) - s_k$ .

Recompute Lagrange Polynomials for $\mathcal{Y}_{k+1}$ . Otherwise update set $\mathcal{Z}_{k+1} = (\mathcal{Z}_k \setminus \{z_{i_k^*}\} \cup \{0\}) - s_k$ if $|\mathcal{Z}_k| = p$ ,

or $\mathcal{Z}_{k+1} = \mathcal{Z}_k \cup \{0\} - s_k$ if $|\mathcal{Z}_k| < p$ .

4 Model improving or unsuccessful iteration: $\rho_k < \eta_1$ or $\|g_k\| < \eta_2\Delta_k$ . Set $x_{k+1} = x_k$ , $I_{imp} = 0$ and perform all the applicable steps below.

(i) Geometry correction by replacing a far point in $\mathcal{Y}$ : Let $j_k^* = \arg \max_{j=1,\dots,n} \|y_j\|$ .

If $\|y_{j_k^*}\| > \Delta_k \Rightarrow s^* = y_{j_k^*}$ $I_{imp} = 1$ . If $|\ell_{j_k^*}(s_k)| > 0$ , then $s_k^* = s_k$ , otherwise $s_k^* = \arg \max_{s \in B(0, \Delta_k)} |\ell_{j_k^*}(s)|$ , compute $f(x_k + s_k^*)$ . Update $\mathcal{Y}_{k+1} = \mathcal{Y}_k \setminus \{y_{j_k^*}\} \cup \{s_k^*\}$ and Lagrange polynomials.

(ii) Self-correction by replacing a point in $\mathcal{Y}$ with a large Lagrange Polynomial value at the trial step: If $I_{imp} = 0$ , $j_k^* = \arg \max_{j=1,\dots,n} |\ell_j(s_k)|$ .

If $|\ell_{j_k^*}(s_k)| > \Lambda_{sc}$ , then $s^* = y_{j_k^*}$ , update $\mathcal{Y}_{k+\frac{1}{2}} = \mathcal{Y}_k \setminus \{y_{j_k^*}\} \cup \{s_k\}$ . Update the set of Lagrange Polynomials for $\mathcal{Y}_{k+\frac{1}{2}}$ . Otherwise $s^* = s_k$ , $\mathcal{Y}_{k+\frac{1}{2}} = \mathcal{Y}_k$ .

(iii) Geometry correction of $\mathcal{Y}$ by replacing a "bad" point: If $I_{imp} = 0$ compute $(j_k^*, s_k^*) = \arg \max_{j=1,\dots,p, s \in B(0, \Delta_k)} |\ell_j(s)|$ .

If $|\ell_{j_k^*}(s_k^*)| > \Lambda$ , compute $f(x_k + s_k^*)$ , $\mathcal{Y}_{k+1} = \mathcal{Y}_{k+\frac{1}{2}} \setminus \{y_{j_k^*}\} \cup \{s_k^*\}$ . Update the set of Lagrange Polynomials for $\mathcal{Y}_{k+1}$ and set $I_{imp} = 1$ . Otherwise $\mathcal{Y}_{k+1} = \mathcal{Y}_{k+\frac{1}{2}}$ .

(iv) Attempt to improve Z using available points. For each defined $s \in \{s^*, s_k^*\}$ repeat:

- Improvement to Z by adding a point: If $|\mathcal{Z}_k| < p$ , $\mathcal{Z}_{k+1} = \mathcal{Z}_k \cup \{s\}$ .

- Improvement to Z to by replacing a far point: Else, let $i_k^* = \arg \max_{i=1,\dots,n} \|z_i\|$ .

If $||z_{i_k^*}\| > ||s|| \Rightarrow \mathcal{Z}_{k+1} = \mathcal{Z}_k \setminus \{z_{i_k^*}\} \cup \{s\}$ .

- Keep Z: Otherwise $Z_{k+1} = Z_k$ .

Model improving iteration: If $I_{imp} = 1$ , $\Delta_{k+1} = \Delta_k$ .

Unsuccessful iteration: If $I_{imp} = 0$ , $\Delta_{k+1} = \gamma\Delta_k$ .
```

> **Footnote a:** If $\mathcal{Z}_k \neq \emptyset$, then
> $\left\|y_{j_k^*} - s_k\right\| > 0$; the procedure is described in [13].

In cases (ii) and (iii) of Step 4 point $y _ { j _ { k } ^ { * } }$ in the current $\mathcal { V }$ set is replaced by a new point, let’s call it s˜. The update for deriving the new Lagrange Polynomial set $\ell ^ { + }$ can be carried out via the following formulae:

(4.4)

$$
\begin{array}{l} \ell_ {j _ {k} ^ {*}} ^ {+} (x) = \frac {\ell_ {j _ {k} ^ {*}} (x)}{\ell_ {j _ {k} ^ {*}} (\tilde {s})} \\ \ell_ {i} ^ {+} (x) = \ell_ {i} (x) - \ell_ {i} (\tilde {s}) \ell_ {j _ {k} ^ {*}} ^ {+} (x) \quad i \neq j _ {k} ^ {*}. \end{array}
$$

Note that since we assume that $\mathcal { V } _ { 0 }$ is poised then so are all consequent ${ \mathcal { V } } _ { k }$ sets by construction.

We now provide results that allow us to bound the number of model improving iterations $\mathcal { M } _ { \epsilon }$

**Theorem 4.1**. Let $\mathcal { P }$ be the set of linear polynomials (with dimension $\textit { p } = \textit { n } { }$ Then the number of oracle calls in any sequence of consecutive model improving iterations; i.e. such that $k \in \mathcal { M } .$ <sub>ϵ</sub> is at most 4n log $n + 8 n + 2 n | \log \log ( \Lambda ) |$ .

We break the proof of this theorem into the following two lemmas, the first of which derives a bound on the number of consecutive model improving iterations required to obtain a desired Λ-poised set starting from a $\Lambda _ { 0 ^ { - } }$ poised set for an arbitrary $\Lambda _ { 0 }$ . The second lemma shows that after the first 2n consecutive model improving iterations $\Lambda _ { \mathrm { 0 ^ { - } F } }$ oised set is obtained with a particular bound on $\Lambda _ { 0 }$

**Lemma 4.2**. Let P be the set of linear polynomials (with dimension $p = n ) . ~ H { \mathcal { Y } } _ { k }$ is $\Lambda _ { 0 } .$ -poised in $B ( 0 , \Delta )$ , then there will be at most ⌈n log $n + n | \log \log \Lambda _ { 0 } | + n | \log \log ( \Lambda ) | \rceil$ additional consecutive model improving iterations.

**Proof.** Assume for simplicity and w.l.o.g that $\Delta = 1$ . Having a $\Lambda _ { 0 } .$ -poised set implies by Hadamard’s inequality that we have

$$
| \det (Y ^ {- T}) | \leq \prod_ {i} ^ {n} \| (Y ^ {- T}) _ {i} \| \leq \Lambda_ {0} ^ {n}.
$$

since $\begin{array} { r } { \operatorname* { m a x } _ { x \in B ( 0 , 1 ) } | \ell _ { i } ( x ) | = \| ( Y ^ { - T } ) _ { i } \| } \end{array}$ . Thus $| \operatorname* { d e t } ( Y ) | \geq \Lambda _ { 0 } ^ { - n }$ . We have by Cramer’s rule that replacing $y _ { i }$ with s results in a matrix $Y ^ { + }$ which satisfies

$$
| \det (Y ^ {+}) | = | \det (Y) | | \ell_ {i} (s) |.
$$

Combining this with the fact that $| \operatorname* { d e t } ( Y ) | \leq 1$ , we have that max $\begin{array} { r } { \dot { \mathbf { \eta } } _ { i \in [ n ] } \operatorname* { m a x } _ { x \in B ( 0 , 1 ) } | \ell _ { i } ( x ) | \leq | \operatorname* { d e t } ( Y ) | ^ { - 1 } } \end{array}$ . Thus Y is Λ-poised $\mathrm { i f } \ | \operatorname* { d e t } ( Y ) | \geq \Lambda ^ { - 1 }$ . All that remains is to show that $| \operatorname* { d e t } ( { \dot { Y } } ) |$ must increase quickly.

After “Geometry correction of Y by replacing a "bad" point” resulting in a matrix $Y ^ { + }$ , we have

$$
| \det (Y ^ {+}) | = | \det (Y) | \left(\max _ {i \in [ n ]} \max _ {x \in B (0, 1)} | \ell_ {i} (x) |\right) = | \det (Y) | \max _ {i \in [ n ]} \| (Y ^ {- T}) _ {i} \|.
$$

Observe by the AM-GM inequality

$$
\max _ {i \in [ n ]} \| (Y ^ {- T}) _ {i} \| ^ {2} \geq \frac {1}{n} \sum_ {i} \| (Y ^ {- T}) _ {i} \| ^ {2} = \frac {1}{n} \| Y ^ {- T} \| _ {F} ^ {2} = \frac {1}{n} \sum_ {i} \sigma_ {i} (Y ^ {- T}) ^ {2} \geq \sqrt [ n ]{\sigma_ {i} (Y ^ {- T}) ^ {2}} = | \det (Y) | ^ {- \frac {2}{n}}.
$$

Thus we have $\begin{array} { r } { \operatorname* { m a x } _ { i \in [ n ] } \operatorname* { m a x } _ { x \in B ( 0 , 1 ) } | \ell _ { i } ( x ) | = \operatorname* { m a x } _ { i \in [ n ] } \| ( Y ^ { - T } ) _ { i } \| \geq | \operatorname* { d e t } ( Y ) | ^ { - \frac { 1 } { n } } } \end{array}$ and hence

$$
| \det (Y ^ {+}) | \geq | \det (Y) | | \det (Y) | ^ {- \frac {1}{n}} = | \det (Y) | ^ {\left(1 - \frac {1}{n}\right)}.
$$

Taking the logarithm of both sides, we have

$$
\log | \det (Y ^ {+}) | \geq \left(1 - \frac {1}{n}\right) \log | \det (Y) |.
$$

Thus, $- \log | \operatorname* { d e t } ( Y ) |$ shrinks exponentially.

If in a single iteration we first do “Geometry correction by replacing a point in Y with a Lagrange Polynomial value” to get $Y ^ { + }$ , and then do “Geometry correction of Y by replacing a "bad" point” to get $Y ^ { + + }$ we have

$$
\log | \det (Y ^ {+ +}) | \geq \left(1 - \frac {1}{n}\right) \log | \det (Y ^ {+}) | \geq \left(1 - \frac {1}{n}\right) \log (| \det (Y) | | \ell_ {i} (s _ {k} ^ {*}) |) \geq \left(1 - \frac {1}{n}\right) \log | \det (Y) |.
$$

Thus, in either case $, - \log | \operatorname* { d e t } ( Y ) |$ shrinks exponentially.

Recall that we have $| \operatorname* { d e t } ( Y ) | \geq \Lambda _ { 0 } ^ { - n }$ and thus log | det $( Y ) | \geq - n \log ( \Lambda _ { 0 } )$ . Also, we will have achieved Λ- poisedness if | det $| Y  | \geq \Lambda ^ { - 1 }$ or equivalently log | det $( Y ) | \geq - \log ( \Lambda )$ . Thus, the number of additional iterations is bounded by

$$
\left\lceil \frac {\log \left(\frac {- n \log (\Lambda_ {0})}{- \log (\Lambda)}\right)}{\log \left(1 - \frac {1}{n}\right)} \right\rceil \leq \left\lceil n \log \left(\frac {n \log (\Lambda_ {0})}{\log (\Lambda)}\right)\right\rceil \leq \lceil n \log n + n | \log \log \Lambda_ {0} | + n | \log \log (\Lambda) | \rceil .
$$

**Lemma 4.3**. Let P be the set of linear polynomials (with dimension $p = n )$ . Then $\mathcal { V } _ { \mathcal { k } } ~ i s ~ 1 4 ^ { n }$ -poised after at most 2n consecutive model improving iterations.

**Proof.** Because the first step of a model improving iteration replaces any point in ${ \mathcal { V } } _ { k }$ that is outside $B ( 0 , \Delta _ { k } )$ then after at most n such iterations (and n oracle calls), ${ \mathcal { V } } _ { k }$ contains n points all of which have norm at most $\Delta _ { k }$ . For simplicity, we assume $\Delta _ { k } = 1$ . We show that in at most n additional iterations, the set $\mathcal { \ V } _ { k }$ becomes 14<sup>n</sup>-poised.

For a subspace $S ,$ we say that a set of points $\mathcal { V }$ is Λ-poised in $S$ if for the Lagrange polynomials $\ell _ { i } ,$ we have $\begin{array} { r } { \operatorname* { m a x } _ { x \in B ( 0 , 1 ) \cap S } | \ell _ { i } ( x ) | \le \Lambda } \end{array}$ . Note that if $y _ { 1 } \in \mathcal { D } _ { k }$ is a unit vector, then Y is 1-poised in $S = \operatorname { s p a n } ( \{ y _ { 1 } \} )$ . Below, we prove two claims regarding poisedness in a subspace.

First, we claim that if Y is Λ-poised in S, then after step (ii): “Self-correction by replacing a point in $\mathcal { V }$ with a large Lagrange Polynomial value”, we will have that the new set of points is 2Λ-poised in $S .$

Let $s _ { k }$ be the point we are adding and let $j _ { k } ^ { \ast } = \arg \operatorname* { m a x } _ { j = 1 , \dots , n } \left| \ell _ { j } ( s _ { k } ) \right|$ . For the replacement to take place, we must have $| \ell _ { j _ { k } ^ { * } } ( s _ { k } ) | \geq 1$ . Let $\ell _ { i } ^ { + }$ denote the Lagrange polynomials after replacement. Then from (4.4) we have

$$
\max _ {x \in B (0, 1) \cap S} | \ell_ {j _ {k} ^ {*}} ^ {+} (x) | = \max _ {x \in B (0, 1) \cap S} \frac {| \ell_ {j _ {k} ^ {*}} (x) |}{| \ell_ {j _ {k} ^ {*}} (s _ {k}) |} \leq \frac {\Lambda}{1} \leq 2 \Lambda .
$$

For $i \neq j _ { k } ^ { * }$ , we have

$$
\begin{array}{l} \max _ {x \in B (0, 1) \cap S} | \ell_ {i} ^ {+} (x) | = \max _ {x \in B (0, 1) \cap S} \left| \ell_ {i} (x) - \frac {\ell_ {i} (s _ {k}) \ell_ {j _ {k} ^ {*}} (x)}{\ell_ {j _ {k} ^ {*}} (s _ {k})} \right| \\ \leq \max _ {x \in B (0, 1) \cap S} | \ell_ {i} (x) | + \left| \frac {\ell_ {i} (s _ {k})}{\ell_ {j _ {k} ^ {*}} (s _ {k})} \right| \left(\max _ {x \in B (0, 1) \cap S} | \ell_ {j _ {k} ^ {*}} (x) |\right) \leq 2 \Lambda . \end{array}
$$

The last inequality is because $| \ell _ { i } ( s _ { k } ) | \leq | \ell _ { j _ { k } ^ { * } } ( s _ { k } ) |$ . This completes the proof of this first claim.

Second, we claim that if $\mathcal { V }$ is Λ-poised in a proper subspace S, then after (iii): “Geometry correction of $\mathcal { V }$ by replacing a "bad" point”, we will have that the new set of points is 7Λ-poised in $S ^ { + }$ , where $S ^ { + }$ is a subspace of one dimension greater than $S .$

Let $( j _ { k } ^ { * } , s _ { k } ^ { * } ) \ = \ \arg \operatorname* { m a x } _ { j = 1 , \ldots , p , s \in B ( 0 , \Delta _ { k } ) } | \ell _ { j } ( s ) |$ . We will replace $y _ { j _ { k } ^ { * } }$ with $s _ { k } ^ { * } ;$ let $\ell _ { i } ^ { + }$ denote the Lagrange polynomials after replacement and apply (4.4).

If Y is 2Λ-poised in the whole space, then we have

$$
\max _ {x \in B (0, 1)} | \ell_ {j _ {k} ^ {*}} ^ {+} (x) | = \max _ {x \in B (0, 1)} \frac {| \ell_ {j _ {k} ^ {*}} (x) |}{| \ell_ {j _ {k} ^ {*}} (s _ {k} ^ {*}) |} = 1 \leq 7 \Lambda .
$$

Furthermore, for $i \neq j _ { k } ^ { * }$ , we have

$$
\begin{array}{l} \max _ {x \in B (0, 1)} | \ell_ {i} ^ {+} (x) | = \max _ {x \in B (0, 1)} \left| \ell_ {i} (x) - \frac {\ell_ {i} (s _ {k} ^ {*}) \ell_ {j _ {k} ^ {*}} (x)}{\ell_ {j _ {k} ^ {*}} (s _ {k} ^ {*})} \right| \\ \leq \max _ {x \in B (0, 1)} | \ell_ {i} (x) | + \left| \frac {\ell_ {i} (s _ {k} ^ {*})}{\ell_ {j _ {k} ^ {*}} (s _ {k} ^ {*})} \right| \left(\max _ {x \in B (0, 1)} | \ell_ {j _ {k} ^ {*}} (x) |\right) \leq 4 \Lambda \leq 7 \Lambda . \end{array}
$$

Thus, we have that the points are 7Λ poised in the whole space and we can take $S ^ { + } \ \mathrm { t o }$ be any subspace of dimension one greater that S. Thus, the claim is shown if Y is 2Λ-poised in the whole space. For the remainder of the proof of the claim we may assume $\mathcal { V }$ is not 2Λ-poised in the whole space.

Let $P _ { S }$ and $P _ { S } ^ { \perp }$ denote the projections onto S and its orthogonal complement, respectively. By linearity, we have

$$
2 \Lambda <   \| \ell (s _ {k} ^ {*}) \| _ {\infty} = \| \ell (P _ {S} ^ {\perp} s _ {k} ^ {*}) + \ell (P _ {S} s _ {k} ^ {*}) \| _ {\infty} \leq \| \ell (P _ {S} ^ {\perp} s _ {k} ^ {*}) \| _ {\infty} + \| \ell (P _ {S} s _ {k} ^ {*}) \| _ {\infty} \leq \| \ell (P _ {S} ^ {\perp} s _ {k} ^ {*}) \| _ {\infty} + \Lambda .
$$

Thus, $\| \ell ( P _ { S } ^ { \perp } s _ { k } ^ { * } ) \| _ { \infty } > \Lambda$ . By homogeneity and optimality we have,

$$
\frac {\| \ell (P _ {S} ^ {\perp} s _ {k} ^ {*}) \| _ {\infty}}{\| P _ {S} ^ {\perp} s _ {k} ^ {*} \|} = \left\| \ell \left(\frac {P _ {S} ^ {\perp} s _ {k} ^ {*}}{\| P _ {S} ^ {\perp} s _ {k} ^ {*} \|}\right) \right\| _ {\infty} \leq \| \ell (s _ {k} ^ {*}) \| _ {\infty} \leq \| \ell (P _ {S} ^ {\perp} s _ {k} ^ {*}) \| _ {\infty} + \Lambda .
$$

Thus we have

$$
\| P _ {S} ^ {\perp} s _ {k} ^ {*} \| \geq \frac {\| \ell (P _ {S} ^ {\perp} s _ {k} ^ {*}) \| _ {\infty}}{\| \ell (P _ {S} ^ {\perp} s _ {k} ^ {*}) \| _ {\infty} + \Lambda} \geq \frac {1}{2}
$$

since $\| \ell ( P _ { S } ^ { \perp } s _ { k } ^ { * } ) \| _ { \infty } > \Lambda$

By an argument identical to that used in the proof of the first claim, we have $\begin{array} { r } { \operatorname* { m a x } _ { x \in B ( 0 , 1 ) \cap S } \| \ell ^ { + } ( x ) \| _ { \infty } \leq 2 \Lambda } \end{array}$ Next we wish to bound $\left\| \ell ^ { + } \left( \frac { P _ { S } ^ { \perp } s _ { k } ^ { * } } { \| P _ { S } ^ { \perp } s _ { k } ^ { * } \| } \right) \right\| _ { \infty }$ . Noting that $\ell ^ { + } ( s _ { k } ^ { * } ) = e _ { j _ { k } ^ { * } }$ , we have

$$
\| \ell^ {+} (P _ {S} ^ {\perp} s _ {k} ^ {*}) \| _ {\infty} = \| \ell^ {+} (s _ {k} ^ {*}) - \ell^ {+} (P _ {S} s _ {k} ^ {*}) \| _ {\infty} \leq \| \ell^ {+} (s _ {k} ^ {*}) \| _ {\infty} + \| \ell^ {+} (P _ {S} s _ {k} ^ {*}) \| _ {\infty} = 1 + 2 \Lambda \leq 3 \Lambda .
$$

Then we have

$$
\left\| \ell^ {+} \left(\frac {P _ {S} ^ {\perp} s _ {k} ^ {*}}{\| P _ {S} ^ {\perp} s _ {k} ^ {*} \|}\right) \right\| _ {\infty} = \frac {\| \ell^ {+} (P _ {S} ^ {\perp} s _ {k} ^ {*}) \| _ {\infty}}{\| P _ {S} ^ {\perp} s _ {k} ^ {*} \|} \leq 6 \Lambda .
$$

Let us define $S ^ { + } = S$ ⊕ span $( \{ P _ { S } ^ { \bot } s _ { k } ^ { * } \} )$ . We can represent any vector x in $B ( 0 , 1 ) \cap S ^ { + }$ , as $\begin{array} { r } { \alpha x ^ { \prime } + \beta \frac { P _ { S } ^ { \bot } s _ { k } ^ { * } } { \| P _ { S } ^ { \bot } s _ { k } ^ { * } \| } } \end{array}$ where $x ^ { \prime } \in B ( 0 , 1 ) \cap S$ and $\alpha ^ { 2 } + \beta ^ { 2 } \leq 1$ Then we have

$$
\left\| \ell^ {+} \left(\alpha x ^ {\prime} + \beta \frac {P _ {S} ^ {\perp} s _ {k} ^ {*}}{\| P _ {S} ^ {\perp} s _ {k} ^ {*} \|}\right) \right\| _ {\infty} \leq \alpha \| \ell^ {+} (x ^ {\prime}) \| _ {\infty} + \beta \left\| \ell^ {+} \left(\frac {P _ {S} ^ {\perp} s _ {k} ^ {*}}{\| P _ {S} ^ {\perp} s _ {k} ^ {*} \|}\right) \right\| _ {\infty} \leq | \alpha | 2 \Lambda + | \beta | 6 \Lambda \leq \left\| \left[ \begin{array}{c} 2 \\ 6 \end{array} \right] \right\| \Lambda \leq 7 \Lambda .
$$

This completes the proof of the second claim.

For $t \leq n$ , we will show by induction that after at most t iterations after removing far points, $\mathcal { \mathrm { y } } _ { k }$ is $1 4 ^ { t }$ -poised in a subspace of dimension at least t.

For the base case, when “Geometry correction of ${ \mathcal { V } } _ { k }$ by replacing a "bad" point” is performed for the first time, it is done by maximizing a linear Lagrange polynomial over a ball. Since the maximum of a linear function over the unit ball is always attained on the boundary, the new point added to $\mathcal { \ V } _ { k }$ is on the boundary, thus ${ \mathcal { V } } _ { k }$ contains at least one unit vector. Recall that if $y _ { 1 } \in \mathcal { D } _ { k }$ is a unit vector, then $\mathcal { V }$ is 1-poised in $S = \operatorname { s p a n } ( \{ y _ { 1 } \} )$ . Thus, ${ \mathcal { V } } _ { k }$ is 1-poised in a one-dimensional subspace, establishing the base case.

Now let us assume that after the tth iteration after removing far points, ${ \mathcal { V } } _ { k }$ is 14<sup>t</sup>-poised in a subspace, $S ,$ of dimension at least $t ,$ and let us show that after an additional iteration, ${ \mathcal { V } } _ { k }$ is $1 4 ^ { t + 1 }$ -poised in a subspace of dimension at least $t + 1$ . In the tth-iteration, the algorithm either performs (ii) “Self-correction by replacing a point in ${ \mathcal { V } } _ { k }$ with a large Lagrange Polynomial value” and (iii) “Geometry correction of $\mathcal { \scriptsize { y } } _ { k }$ by replacing a "bad" point”, or just the latter. If the algorithm performs both corrections, then after the first correction we have that the points are $2 \cdot 1 4 ^ { t _ { - } }$ poised in S, by the first claim. Then after the second correction, by the second claim, the points are $7 \cdot 2 \cdot 1 4 ^ { t } = 1 4 ^ { t + 1 }$ -poised in $S ^ { + }$ which is of dimension at least one greater than S. Alternatively, if the algorithm only performs the latter correction, then by the second claim, the points are $7 \cdot 1 4 ^ { t } \leq 1 4 ^ { t + 1 }$ -poised in $S ^ { + }$ which is of dimension at least one greater than S. Thus, in either case, the inductive argument is complete. Thus, after at most n iterations after removing far points, ${ \mathcal { V } } _ { k }$ is 14<sup>n</sup>-poised in a subspace of dimension at least n (i.e. the whole space). Thus, after at most 2n total iterations (including removal of far away points), ${ \mathcal { V } } _ { k }$ is 14<sup>n</sup>-poised. □

Proof of Theorem $4 . 1 .$ We can combine Lemmas 4.3 and 4.2 by letting $\Lambda _ { 0 } = 1 4 ^ { n }$ . Thus the total number of iterations can be bounded as follows:

$$
\begin{array}{r l} & 2 n + \lceil n \log n + n | \log \log 1 4 ^ {n} | + n | \log \log (\Lambda) | \rceil \\ & = 2 n + \lceil 2 n \log n + n | \log \log 1 4 | + n | \log \log (\Lambda) | \rceil \\ & \leq 2 n \log n + 4 n + n | \log \log (\Lambda) |. \end{array}
$$

Since each iteration uses at most 2 oracle calls, the result follows.

With these results, we may now present a total complexity theorem for Algorithm 2. Recall that Theorem 2.8 provides a bound for $| S _ { \epsilon } | + | U _ { \epsilon } |$ . As we have just demonstrated the number of consecutive model improving iterations cannot be larger than 2n log $n + 4 n + n | \log \log ( \Lambda ) |$ until set ${ \mathcal { V } } _ { k }$ is Λ-poised in $B ( x _ { k } , \Delta _ { k } )$ . Let $\textstyle \Lambda = 1 + { \frac { 1 } { n } } ,$ then the number of consecutive iterations can be bounded by O(n log n). By Corollary 3.6 we have, for any $\begin{array} { r } { \Delta _ { k } \ge \sqrt { \frac { 4 \epsilon _ { f } \Lambda } { L + \kappa _ { b h m } } } , } \end{array}$

$$
\begin{array}{r l} & {\kappa_ {e g} = (L + \kappa_ {b h m}) \sqrt {n} \sqrt {n (\Lambda^ {2} - 1) + 2} = \Theta (\sqrt {n})} \\ & {\kappa_ {e f} = \kappa_ {e g} + \frac {L + \kappa_ {b h m}}{2} = \Theta (\sqrt {n}).} \end{array}
$$

Thus we have the following corollary.

**Corollary 4.4**. Under the same assumptions as Theorem 2.8, for any $\epsilon > \sqrt { \frac { 4 \epsilon _ { f } } { \gamma ^ { 2 } \operatorname* { m i n } \{ C _ { 2 } , L + \kappa _ { b h m } \} C _ { 1 } ^ { 2 } } }$ (i.e $\epsilon \geq \Omega ( \sqrt { n \epsilon _ { f } } ) )$ the total oracle complexity of Algorithm 2 is bounded as

$$
\mathcal {C} _ {\epsilon} \leq \mathcal {O} \left(\frac {n ^ {3 / 2} \log n}{\epsilon^ {2}}\right)
$$

$i f \eta _ { 2 } = \sqrt { n }$ and as

$$
\mathcal {C} _ {\epsilon} \leq \mathcal {O} \left(\frac {n ^ {2} \log n}{\epsilon^ {2}}\right)
$$

$i f \eta _ { 2 }$ is constant.

## 5 Model-Based Trust Region Methods in Subspaces

We now consider a trust region method where a model $m ( x )$ is built and optimized in a random low-dimensional subspace of $\mathbb { R } ^ { n }$ . The idea of using random subspace embeddings within derivative-free methods has gained a lot of popularity in the literature lately. It was shown in [16] that applying direct search methods in a low-dimensional subspace reduces the oracle complexity from $\mathcal { O } ( n ^ { \dot { 2 } } \epsilon ^ { - 2 } )$ to $\bar { \mathcal { O } } ( n \bar { \epsilon } ^ { - 2 } )$ (with dependence on the subspace dimensions suppressed). A random subspace version of a model-based TR method was first studied in [8] with the use of Johnson-Lindenstrauss (JL) subspace embeddings which achieved complexity $\mathcal { O } ( n ^ { 2 } \epsilon ^ { - 2 } )$ . Later in [14] this approach was combined with a stochastic model-based trust region method. Recently it was shown in [11] that a subspace model-based trust region method achieves an improved $\mathcal { O } ( n \epsilon ^ { - 2 } )$ complexity when the model is based on a random projection rather than a JL embedding. The key difference lies in the scaling of the projected gradient that is being estimated. A recent note [9] confirms that by rescaling the JL transformation in [8] the rate $\mathcal { O } ( n \epsilon ^ { - 2 } )$ can be achieved.

The works [8, 11] do not consider noisy function values. In [14] the noise in the function values is stochastic and is assumed to be reducible to any desired accuracy, dictated by the trust region radius, which is allowed to shrink to an arbitrarily small value. Here we extend the analysis of a subspace trust region method from [11] to accommodate fixed (deterministic) noise in the function oracle. The fundamental difficulty of doing so is that the algorithmic framework with random subspaces does not by itself guarantee a lower bound on the trust region radius, which is necessary for the analysis of the noisy function oracles, as we have seen in the sections above. As a consequence we need to introduce two algorithmic modifications - a relaxed step acceptance criterion and an enforced lower bound on the trust region radius.

We begin by introducing the subspace embedding of our problem. Given a matrix $Q \in \mathbb { R } ^ { n \times q }$ , with $q \leq n$ and orthonormal columns, $Q Q ^ { T } \nabla \phi ( x )$ is an orthogonal projection of $\nabla \phi ( x )$ onto a subspace spanned by the columns of $Q$ (we will call it a subspace induced by $Q )$ . We also define a reduction of $\phi ( x )$ to the subspace, given by Q around x: $\hat { \phi } ( v ) = \phi ( x + Q v ) , v \in \mathbb { R } ^ { q }$ , which implies $Q \nabla \hat { \phi } ( 0 ) = Q Q ^ { T } \nabla \phi ( x )$ . Similarly we define ${ \hat { m } } ( v ) = m ( x + Q v )$ ， $v \in \mathbb { R } ^ { q }$ , which implies $Q \nabla \hat { m } ( 0 ) = Q Q ^ { T } \nabla m ( x )$

We now present a modified trust-region algorithm that constructs models and computes steps in the subspace. $\mathrm { A t }$ each iteration $k \in \{ 0 , 1 , \ldots \}$ the algorithm chooses $Q _ { k } \in \mathbb { R } ^ { n \times q }$ with orthonormal columns. The model $m _ { k }$ is defined as

$$
m _ {k} (x _ {k} + Q _ {k} v) = \phi (x _ {k}) + g _ {k} ^ {T} Q _ {k} v + \frac {1}{2} v ^ {T} Q _ {k} ^ {T} H _ {k} Q _ {k} v.
$$

For any vector $v , g _ { k } ^ { T } Q _ { k } v = g _ { k } ^ { T } Q _ { k } Q _ { k } ^ { T } Q _ { k } v .$ thus without loss of generality, we will assume that $Q _ { k } Q _ { k } ^ { T } g _ { k } = g _ { k }$ , in other words, $g _ { k }$ lies in the subspace induced by $Q _ { k }$ . We define the trust region in the subspace induced by $Q _ { k }$ as $B _ { Q _ { k } } ( x _ { k } , \Delta _ { k } ) = \{ z : z = x _ { k } + Q _ { k } v , \ \| v \| \le \Delta _ { k } \}$

We will also need the definition of a fully linear model with respect to the subspace. We use the following definition which is the same as in [11].

**Definition 5.1** (Fully-linear model in a subspace). Given a matrix with orthonormal columns $Q \in \mathbb { R } ^ { n \times q }$ ， let $B _ { Q } ( x , \Delta ) = \{ z : z = x + Q v , \ \| v \| \leq \Delta \}$ . Let

$$
m (x + Q v) = \phi (x) + g ^ {T} Q v + \frac {1}{2} v ^ {T} Q ^ {T} H Q v
$$

and $\hat { m } ( v ) = m ( x + Q v ) , v \in \mathbb R ^ { q }$ . We say that model $m ( x + Q v )$ is a $\kappa _ { e f } , \kappa _ { e g } - f u l l y$ linear model of $\phi ( x + Q v ) \quad$ on $B _ { Q } ( x , \Delta )$ if

$$
\| \nabla \hat {m} (0) - \nabla \hat {\phi} (0) \| \leq \kappa_ {e g} \Delta
$$

$$
| \hat {m} (v) - \hat {\phi} (v) | \leq \kappa_ {e f} \Delta^ {2}
$$

$$
\| v \| \leq \Delta .
$$

Here too (5.3) implies (5.4) with a specific value of $\kappa _ { e f }$

**Lemma 5.2** (Lemma 6.6 from [11]). Under Assumptions 1.2 and 2.2, if (5.3) holds then m is $\kappa _ { e f } , \kappa _ { e g } - f u l l y$ linear on $B _ { Q } ( x , \Delta )$ with

$$
\kappa_ {e f} = \kappa_ {e g} + \frac {L _ {Q} + \kappa_ {b h m}}{2}
$$

where $L _ { Q }$ is the Lipschitz constant of $Q Q ^ { T } \nabla \phi ( x )$

```text
Algorithm 3: Trust region method based on fully-linear models in subspace

Inputs: Inexact zeroth order oracle  \( |f(x)-\phi(x)|\leq\epsilon_{f} \) , minimum radius  \( \Delta_{min} \) , initial  \( x_{0},\Delta_{0}\geq\Delta_{min} \) , initial matrix  \( Q_{0}\inR^{n\times q} \)  with orthonormal columns and  \( \eta_{1}\in(0,1),\eta_{2}>0 \) , and  \( \gamma\in(0,1) \) .

for  \( k=0,1,2,\cdots \)  do

1 For the current  \( Q_{k} \)  compute model  \( m_{k} \)  as in (5.1).

2 Compute a trial step  \( x_{k}+s_{k} \)  where  \( s_{k}=Q_{k}v_{k} \)  with  \( v_{k}\approx\arg\min_{v}\{m_{k}(x_{k}+Q_{k}v):\|v\|\leq\Delta_{k}\} \) .

3 Compute the ratio  \( \rho_{k} \)  as

 \( \rho_{k}=\frac{f(x_{k})-f(x_{k}+s_{k})+2\epsilon_{f}}{m_{k}(x_{k})-m_{k}(x_{k}+s_{k})} \) .

4 Update the iterate and the TR radius as

 \( (x_{k+1},\Delta_{k+1})\leftarrow\left\{\begin{aligned}(x_{k}+s_{k},\gamma^{-1}\Delta_{k})&if\rho_{k}\geq\eta_{1}\text{ and }\|g_{k}\|\geq\eta_{2}\Delta_{k},\\(x_{k},\Delta_{k})&else, if the model is not fully-linear in B_{Q_{k}}(x_{k},\Delta_{k}).\\(x_{k},\max\{\gamma\Delta_{k},\Delta_{min}\})&otherwise.\end{aligned}\right. \)

5 Update the subspace

 \( Q_{k+1}\inR^{n\times q}\leftarrow\left\{\begin{aligned}&random if\rho_{k}\geq\eta_{1}\text{ and }\|g_{k}\|\geq\eta_{2}\Delta_{k}\text{ or if a model is fully-linear}\\Q_{k}&otherwise.\end{aligned}\right. \)

6 Perform some model improvement steps.
```

We will assume, as before, that Assumption 2.2 holds. The two key modifications of Algorithm 3 compared to Algorithm 1 are the additional term $2 \epsilon _ { f }$ in the definition of $\rho$ and the imposed lower bound on the trust region radius $\Delta _ { m i n }$ . Both of these modifications are needed because the noisy zeroth-order oracle makes it essential for $\Delta _ { k }$ to remain sufficiently positive to ensure fully linear models. Such a lower bound on $\Delta _ { k }$ is ensured in the deterministic framework of Algorithm 1 for sufficiently large $\epsilon .$ But this is not the case when random subspaces are used because unsuccessful iterations can occur even if $\Delta _ { k }$ is small, due to the subspace not being chosen well. We note that since $\epsilon _ { f }$ is an upper bound on the error in the zeroth order oracle, any upper estimate of it can be used in the algorithm. Of course, unnecessarily large values will have an adverse effect on the resulting best achievable accuracy ϵ. Similarly, we will see that the choice of $\epsilon _ { f }$ dictates the best choice for $\Delta _ { m i n }$

We now introduce a definition from [11] of a measure of how well the subspace induced by Q aligns with the current gradient.

**Definition 5.3** (Well aligned subspace). The subspace spanned by columns of $Q$ is $\kappa _ { g }$ -well aligned with $\nabla \phi ( x )$ for a given x $i f$

$$
\| Q Q ^ {T} \nabla \phi (x) - \nabla \phi (x) \| \leq \kappa_ {g} \| \nabla \phi (x) \|
$$

for some $\kappa _ { g } \in [ 0 , 1 )$

While condition (5.5) involves the gradient $\| \nabla \phi ( x ) \|$ it ultimately reduces to the properties of the subspace. Essentially, it requires that the gradient is not too close to being orthogonal to the subspace induced by $Q .$ . Similar conditions and terminology have been used in [8, 11].

We also recall the following related lemma (recalling that $g _ { k } = Q _ { k } Q _ { k } ^ { T } g _ { k } )$

**Lemma 5.4** (Lemma 6.3 from [11]). On iteration $k , Q _ { k }$ is $\kappa _ { g } - w e l l$ aligned with $\nabla \phi ( x _ { k } )$ if and only if

$$
\| Q _ {k} Q _ {k} ^ {T} \nabla \phi (x _ {k}) \| ^ {2} \geq (1 - \kappa_ {g} ^ {2}) \| \nabla \phi (x _ {k}) \| ^ {2}.
$$

Also, $i f m ( x _ { k } + s )$ is $\kappa _ { e f } , \kappa _ { e g } - f u l l y$ linear model of $\phi ( x _ { k } + s )$ on $B _ { Q _ { k } } ( x _ { k } , \Delta _ { k } )$ then

$$
\| g _ {k} - Q _ {k} Q _ {k} ^ {T} \nabla \phi (x _ {k}) \| \leq \kappa_ {e g} \Delta_ {k}
$$

We can now show the following lemma which is analogous to Lemma 6.4 from [11].

**Lemma 5.5** (sufficient condition for a successful step). Under Assumptions 1.2 and 2.2, if $Q _ { k }$ is $\kappa _ { g } - w e l l$ aligned with $\begin{array} { r } { \nabla \phi ( x _ { k } ) , m _ { k } ( x _ { k } + s ) i s a \kappa _ { e f } , \kappa _ { e g } - f u l l y } \end{array}$ linear model of $\phi ( x _ { k } + s )$ on $B _ { Q _ { k } } ( x _ { k } , \Delta _ { k } )$ and $i f$

$$
\Delta_ {k} \leq \sqrt {1 - \kappa_ {g} ^ {2}} \tilde {C} _ {1} \| \nabla \phi (x _ {k}) \|
$$

where

$$
\tilde {C} _ {1} = (\max \left\{\eta_ {2}, \kappa_ {b h m}, \frac {2 \kappa_ {e f}}{(1 - \eta_ {1}) \kappa_ {f c d}} \right\} + \kappa_ {e g}) ^ {- 1}
$$

then $\rho _ { k } \ge \eta _ { 1 } , \| g _ { k } \| \ge \eta _ { 2 } \Delta _ { k }$ , and $x _ { k + 1 } = x _ { k } + s _ { k }$ , i.e. the iteration k is successful.

**Proof.** Due to Lemma 5.4, specifically (5.7) by triangle inequality,

$$
\| Q _ {k} Q _ {k} ^ {T} \nabla \phi (x _ {k}) \| \leq \| g _ {k} \| + \kappa_ {e g} \Delta_ {k}
$$

and also due to Lemma 5.4

$$
\Delta_ {k} \leq \sqrt {1 - \kappa_ {g} ^ {2}} \tilde {C} _ {1} \| \nabla \phi (x _ {k}) \| \leq \tilde {C} _ {1} \| Q _ {k} Q _ {k} ^ {T} \nabla \phi (x _ {k}) \|
$$

By (5.9) we have

$$
(\max \{\kappa_ {b h m}, \eta_ {2}, \frac {2 \kappa_ {e f}}{(1 - \eta_ {1}) \kappa_ {f c d}} \} + \kappa_ {e g}) \Delta_ {k} \leq (\| g _ {k} \| + \kappa_ {e g} \Delta_ {k})
$$

which implies

$$
\max \{\kappa_ {b h m}, \eta_ {2} \} \Delta_ {k} \leq \| g _ {k} \|.
$$

This establishes that $\| g _ { k } \| \ge \eta _ { 2 } \Delta _ { k }$ and also $m _ { k } ( x _ { k } ) - m _ { k } ( x _ { k } + s _ { k } ) \geq \kappa _ { f c d } \| g _ { k } \| \Delta _ { k } / 2$ by Assumption 2.2. Then, using the fact that $| f ( x ) - \phi ( x ) | \leq \epsilon _ { f }$ and the fully linear assumption on $m _ { k }$ in the subspace induced by $Q _ { k }$ and recalling that $s _ { k } = Q _ { k } v _ { k }$ we have

$$
\begin{array}{l} \rho_ {k} = \frac {m _ {k} (x _ {k}) - m _ {k} (x _ {k} + s _ {k}) + (f (x _ {k}) - m _ {k} (x _ {k})) - (f (x _ {k} + s _ {k}) - m _ {k} (x _ {k} + s _ {k})) + 2 \epsilon_ {f}}{m _ {k} (x _ {k}) - m (x _ {k} + s _ {k})} \\ \geq \frac {m _ {k} (x _ {k}) - m _ {k} (x _ {k} + s _ {k}) + (\phi (x _ {k}) - m _ {k} (x _ {k})) - (\phi (x _ {k} + s _ {k}) - m _ {k} (x _ {k} + s _ {k}))}{m _ {k} (x _ {k}) - m _ {k} (x _ {k} + s _ {k})} \\ \geq 1 - \frac {\kappa_ {e f} \Delta_ {k} ^ {2}}{m _ {k} (x _ {k}) - m _ {k} (x _ {k} + s _ {k})} \geq 1 - \frac {\kappa_ {e f} \Delta_ {k} ^ {2}}{\kappa_ {f c d} \| g _ {k} \| \Delta_ {k} / 2} \\ \geq 1 - \frac {2 \kappa_ {e f} \Delta_ {k}}{\kappa_ {f c d} (\| Q _ {k} Q _ {k} ^ {T} \nabla \phi (x _ {k}) \| - \kappa_ {e g} \Delta_ {k})} \geq \eta_ {1}, \end{array}
$$

where the last step is true because $\begin{array} { r } { \| Q _ { k } Q _ { k } ^ { T } \nabla \phi ( x _ { k } ) \| \ge \big ( \frac { 2 \kappa _ { e f } } { ( 1 - \eta _ { 1 } ) \kappa _ { f c d } } + \kappa _ { e g } \big ) \Delta _ { k } } \end{array}$ follows from (5.9).

Now we show a lower bound on progress made in successful iterations similar to Lemma 4.3 of $[ 7 ]$

**Lemma 5.6** (Progress made in a successful iteration). Under Assumptions 1.2 and 2.2, in Algorithm 3, if $\rho _ { k } \ge \eta _ { 1 }$ and $\| g _ { k } \| \ge \eta _ { 2 } \Delta _ { k }$ , then

$$
\phi (x _ {k}) - \phi (x _ {k + 1}) \geq C _ {2} \Delta_ {k} ^ {2} - 4 \epsilon_ {f},
$$

where $\begin{array} { r } { C _ { 2 } = \frac { 1 } { 2 } \eta _ { 1 } \eta _ { 2 } \kappa _ { f c d } } \end{array}$ min $\left\{ \frac { \eta _ { 2 } } { \kappa _ { b h m } } , 1 \right\}$

**Proof.** Since $\rho _ { k } \ge \eta _ { 1 }$ , we have

$$
\eta_ {1} \leq \frac {f (x _ {k}) - f (x _ {k} + s _ {k}) + 2 \epsilon_ {f}}{m _ {k} (x _ {k}) - m _ {k} (x _ {k} + s _ {k})} \leq \frac {\phi (x _ {k}) - \phi (x _ {k} + s _ {k}) + 4 \epsilon_ {f}}{m _ {k} (x _ {k}) - m _ {k} (x _ {k} + s _ {k})},
$$

which we can rearrange as $\phi ( x _ { k } ) - \phi ( x _ { k } + s _ { k } ) \geq \eta _ { 1 } \big ( m _ { k } ( x _ { k } ) - m _ { k } ( x _ { k } + s _ { k } ) \big ) - 4 \epsilon _ { f }$ . Since $\| g _ { k } \| \ge \eta _ { 2 } \Delta _ { k }$ , then by (2.2), we have

$$
\eta_ {1} (m _ {k} (x _ {k}) - m _ {k} (x _ {k} + s _ {k})) \geq \frac {\eta_ {1} \kappa_ {f c d}}{2} \| g _ {k} \| \min \left\{\frac {\| g _ {k} \|}{\kappa_ {b h m}}, \Delta_ {k} \right\} \geq C _ {2} \Delta_ {k} ^ {2}.
$$

From these results one could bound the number of needed iterations if one had a guarantee of having wellaligned subspaces, however we wish to cover the random case where the subspace is well-aligned only with some probability. Accordingly, we study the random subspace case next.

### 5.1 Complexity Analysis Under Random Subspace Selection

Observe that the matrix $Q _ { k }$ , and the trust region radius, $\Delta _ { k }$ do not change on model improving iterations. We define $Q _ { t }$ to be the tth random matrix $Q _ { k }$ and similarly $\Delta _ { t }$ and $x _ { t }$ are the trust region radius and iterate which correspond to $Q _ { t }$ . In other words, index t counts iterations that follow either a successful or an unsuccessful iteration. In what follows we will essentially derive a bound for $| S _ { \epsilon } | + | U _ { \epsilon } |$ by examining what happens to the objective function and trust region radius after these iterations (since we know that after model improving iterations neither the TR radius nor the function value change).

We now define some relevant stochastic processes:

$I _ { t } = \mathbb { 1 } \{ Q _ { t }$ is $\kappa _ { g } .$ -well aligned with $\nabla \phi ( x _ { t } ) \}$ ,

$A _ { t } = \mathbb { 1 } \{ Q _ { t }$ leads to a successful iteration i.e., $\Delta _ { t + 1 } = \gamma ^ { - 1 } \Delta _ { t } \}$ ,

$$
B _ {t} = \mathbb {1} \{\Delta_ {t} > \hat {C} _ {1} \| \nabla \phi (x _ {t}) \| \},
$$

where $\hat { C } _ { 1 } = \sqrt { 1 - \kappa _ { g } ^ { 2 } } \tilde { C } _ { 1 }$ for ${ \tilde { C } } _ { 1 }$ as defined in Lemma 5.5.

We will say that matrix $Q _ { t }$ is $" \mathrm { t r u e } "$ if $I _ { t } ~ = ~ 1$ . Let $T _ { \epsilon }$ be the first t such that the iterate $x _ { t }$ produced by Algorithm 3 achieves $\lVert \nabla \phi ( x _ { t } ) \rVert ~ \leq ~ \epsilon .$ Let $\mathcal { F } _ { t - 1 }$ denote the σ-algebra generated by the first t matrices, $\mathcal { F } _ { t - 1 } = \sigma ( Q _ { 0 } , Q _ { 1 } , \dots Q _ { t - 1 } )$ . We note that the random variables $x _ { t }$ and $\Delta _ { t }$ are measurable with respect to $\mathcal { F } _ { t - 1 }$ . We define $m _ { t } , \ s _ { t } , \ \rho _ { t }$ to be the last model, proposed step, and ratio, respectively, which correspond to the matrix $Q _ { t }$ . The random variables $m _ { t } , \ s _ { t }$ and $\rho _ { t }$ are measurable with respect to $\mathcal { F } _ { t }$ . The random variable $T _ { \epsilon } = \operatorname* { m i n } \{ t : \ \| \nabla \phi ( x _ { t } ) \| \le \epsilon \}$ is a stopping time adapted to the filtration $\{ \mathcal { F } _ { t - 1 } \}$

**Assumption 5.7**. There exists a $\theta \in \left( \frac { 1 } { 2 } , 1 \right]$ such that

$$
\mathbb {P} \{I _ {t} = 1 | \mathcal {F} _ {t - 1} \} \geq \theta .
$$

By Lemma 6.7 of [11], if we take $Q _ { t }$ such that the subspace it induces is uniformly distributed, we can take $\theta \ge \frac { 2 4 3 } { 4 4 3 } > 1 / 2$ and $\kappa _ { g } = \textstyle { \sqrt { 1 - { \frac { q } { 1 0 n } } } }$

Note that $\sigma ( B _ { t } ) \subset \mathcal { F } _ { t - 1 }$ and $\sigma ( A _ { t } ) \subset \mathcal { F } _ { t } .$ , that is the random variable $B _ { t }$ is fully determined by matrices $Q _ { 0 } , \ldots Q _ { t - 1 }$ produced by the algorithm, while $A _ { t }$ is fully determined by the matrices $Q _ { 0 } , \ldots Q _ { t }$ . The stochastic process described here has essentially the same dynamics as the process analyzed in [10] and [11] enabling us to reuse the results. The only differences are the presence of lower bound $\Delta _ { m i n }$ and the possible increase of $\phi ( \boldsymbol { x } _ { k } )$ on some successful iterations. The lower bound does not alter the main properties of the dynamics of $\Delta _ { k }$ , since by fixing ϵ to be sufficiently large with respect to $\Delta _ { m i n }$ we ensure that $\Delta _ { \operatorname* { m i n } } < \hat { C } _ { 1 } \| \nabla \phi ( x _ { t } ) \|$ for $t = 0 , 1 , \dots T _ { \epsilon } - 1$ Thus we retain the key property which follows from Lemma 5.5:

$$
A _ {t} \geq I _ {t} (1 - B _ {t}),
$$

in other words, if matrix $Q _ { t }$ is true and the trust region radius is sufficiently small, then the iteration is successful.

The increase of objective function on certain iterations is due to the relaxed definition of $\rho _ { k }$ and the error in the zeroth order oracle. Such situations have been previously analyzed for line (step) search in [5, 18] and trust-region method in [7]. The analysis here is simpler but the key idea is that the increase is bounded by $4 \epsilon _ { f }$ and occurs on iterations whose number is not too large compared to the number of iterations where function decreases. By ensuring that the decrease is sufficiently large to compensate for the increase, the results are derived. Below we present the analysis.

To bound the total number of successful and unsuccessful iterations we first bound the number of matrices that lead to successful iterations with large $\Delta .$ . For that let $\bar { B } _ { t } = \mathbb { 1 } \{ \Delta _ { t } \geq \gamma \hat { C } _ { 1 } \epsilon \}$ (note that $B _ { t } = 1 \Rightarrow \bar { B } _ { 1 } = 1 )$ . Then from the dynamics of Algorithm 3 we have the bound similar to [10] (also used in [11]).

**Lemma 5.8**. Suppose $\begin{array} { r } { \epsilon > \sqrt { \frac { 8 \epsilon _ { f } } { C _ { 2 } \gamma ^ { 2 } \hat { C } _ { 1 } ^ { 2 } } } } \end{array}$ For any $l \in \{ 0 , \ldots , T _ { \epsilon } - 1 \}$ and for all realizations of Algorithm 3, we have

$$
\sum_ {t = 0} ^ {l} \bar {B} _ {t} I _ {t} A _ {t} \leq \sum_ {t = 0} ^ {l} \bar {B} _ {t} A _ {t} \leq \frac {\phi (x _ {0}) - \phi^ {\star} + 4 \epsilon_ {f} (\sum_ {t = 0} ^ {l} (1 - \bar {B} _ {t}) A _ {t})}{\frac {1}{2} C _ {2} (\gamma (\hat {C} _ {1} \epsilon)) ^ {2}},
$$

**Proof.** Since $\begin{array} { r } { \epsilon > \sqrt { \frac { 8 \epsilon _ { f } } { C _ { 2 } \gamma ^ { 2 } \hat { C } _ { 1 } ^ { 2 } } } , \Delta _ { t } \geq \gamma \hat { C } _ { 1 } \epsilon } \end{array}$ implies that $\Delta _ { t } \geq \sqrt { \frac { 8 \epsilon _ { f } } { C _ { 2 } } }$ which in turn implies that for large successful iterations (corresponding to $\begin{array} { r } { \bar { B } _ { t } A _ { t } ) , \ \phi ( x _ { t } ) - \phi ( x _ { t + 1 } ) \geq \frac { 1 } { 2 } C _ { 2 } \Delta _ { t } ^ { 2 } } \end{array}$ by Lemma 5.6. For small successful iterations (corresponding to $\left( 1 - \bar { B } _ { t } \right) A _ { t } )$ , by the same lemma, we have $\phi ( x _ { t } ) - \phi ( x _ { t + 1 } ) \geq - 4 \epsilon _ { f }$ . The result follows. □

A useful lemma that easily follows from the dynamics is as follows.

**Lemma 5.9**. Suppose $\Delta _ { 0 } \geq \hat { C } _ { 1 } \epsilon$ and $\Delta _ { \mathrm { m i n } } \leq \gamma \hat { C } _ { 1 } \epsilon$ . For any $l \in \{ 0 , \ldots , T _ { \epsilon } - 1 \}$ and for all realizations of Algorithm 3, we have

$$
\sum_ {t = 0} ^ {l} B _ {t} (1 - A _ {t}) \leq \sum_ {t = 0} ^ {l} \bar {B} _ {t} A _ {t} + \log_ {\gamma} \left(\frac {\hat {C} _ {1} \epsilon}{\Delta_ {0}}\right).
$$

The following result is shown in [10] under Assumption 5.7,

$$
\mathbb {E} \left(\sum_ {t = 0} ^ {T _ {\epsilon} - 1} \bar {B} _ {t} (1 - I _ {t})\right) \leq \frac {1 - \theta}{\theta} \mathbb {E} \left(\sum_ {t = 0} ^ {T _ {\epsilon} - 1} \bar {B} _ {t} I _ {t}\right),
$$

from which the following lemma is derived.

**Lemma 5.10**. Let Assumption 5.7 hold. Under the condition that $\theta > 1 / 2 , \Delta _ { 0 } \geq \hat { C } _ { 1 } \epsilon$ , and $\Delta _ { \mathrm { m i n } } \leq \gamma \hat { C } _ { 1 } \epsilon$ , we have

$$
\mathbb {E} \left(\sum_ {t = 0} ^ {T _ {\epsilon} - 1} B _ {t}\right) \leq \frac {1}{2 \theta - 1} \left(\sum_ {t = 0} ^ {T _ {\epsilon} - 1} \bar {B} _ {t} A _ {t} + \log_ {\gamma} \left(\frac {\hat {C} _ {1} \epsilon}{\Delta_ {0}}\right)\right).
$$

Finally the following lemma is shown in [10] for the stochastic processes $I _ { t } , A _ { t }$ and $B _ { t }$ since $A _ { t } \geq I _ { t } ( 1 - B _ { t } )$ and by the dynamics of $\Delta _ { t }$

**Lemma 5.11**. Let Assumption 5.7 hold.

$$
\mathbb {E} \left(\sum_ {t = 0} ^ {T _ {\epsilon} - 1} \left(1 - B _ {t}\right)\right) \leq \frac {1}{2 \theta} \mathbb {E} \left[ T _ {\epsilon} \right].
$$

Putting these lemmas together we obtain the final expected complexity result.

**Theorem 5.12**. Let Assumption 1.2, Assumption 2.2 and Assumption 5.7 hold. Then for any $\epsilon \mathrm { ~ > ~ }$ $\sqrt { \frac { 1 6 \epsilon _ { f } } { ( 2 \theta - 1 ) ^ { 2 } C _ { 2 } \hat { C } _ { 1 } ^ { 2 } \gamma ^ { 2 } } }$ , assuming an initial trust-region radius $\Delta _ { 0 } ~ \geq ~ \hat { C } _ { 1 } \epsilon$ , and $\Delta _ { \mathrm { m i n } } ~ \le ~ \gamma \hat { C } _ { 1 } \epsilon$ , let $T _ { \epsilon }$ be the random stopping time for the event $\{ \| \nabla \phi ( x _ { t } ) \| \leq \epsilon \}$ . We have the bound

$$
\mathbb {E} \left[ T _ {\epsilon} \right] \leq \frac {4 \theta}{(2 \theta - 1) ^ {2}} \left(\frac {\phi (x _ {0}) - \phi^ {\star}}{\frac {1}{2} C _ {2} (\gamma \hat {C} _ {1} \epsilon) ^ {2}} + \log_ {\gamma} \left(\frac {\hat {C} _ {1} \epsilon}{\Delta_ {0}}\right)\right)
$$

where $\hat { C } _ { 1 } = \sqrt { 1 - \kappa _ { g } ^ { 2 } } \tilde { C } _ { 1 } ~ f o r ~ \tilde { C } _ { 1 }$ as in (5.8) and $C _ { 2 }$ as in Lemma 5.6.

**Proof.** Observe that $\begin{array} { r } { \sum _ { t = 0 } ^ { T _ { \epsilon } - 1 } ( 1 - \bar { B } _ { t } ) A _ { t } \le \sum _ { t = 0 } ^ { T _ { \epsilon } - 1 } ( 1 - B _ { t } ) } \end{array}$ . Thus, from Lemmas 5.8 and 5.10, we have

$$
\mathbb {E} \left(\sum_ {t = 0} ^ {T _ {\epsilon} - 1} B _ {t}\right) \leq \frac {1}{2 \theta - 1} \left(\frac {\phi (x _ {0}) - \phi^ {\star}}{\frac {1}{2} C _ {2} (\gamma \hat {C} _ {1} \epsilon) ^ {2}} + \log_ {\gamma} \left(\frac {\hat {C} _ {1} \epsilon}{\Delta_ {0}}\right)\right) + C _ {3} \left(\sum_ {t = 0} ^ {T _ {\epsilon} - 1} (1 - B _ {t})\right),
$$

where $\begin{array} { r } { C _ { 3 } = \frac { 4 \epsilon _ { f } } { ( 2 \theta - 1 ) \frac { 1 } { 2 } C _ { 2 } ( \gamma \hat { C } _ { 1 } \epsilon ) ^ { 2 } } } \end{array}$ . It follows that

$$
\mathbb {E} \left[ T _ {\epsilon} \right] \leq \frac {1}{2 \theta - 1} \left(\frac {\phi (x _ {0}) - \phi^ {\star}}{\frac {1}{2} C _ {2} (\gamma \hat {C} _ {1} \epsilon) ^ {2}} + \log_ {\gamma} \left(\frac {\hat {C} _ {1} \epsilon}{\Delta_ {0}}\right)\right) + (1 + C _ {3}) \mathbb {E} \left(\sum_ {t = 0} ^ {T _ {\epsilon} - 1} (1 - B _ {t})\right).
$$

Combining with Lemma 5.11, we obtain

$$
\mathbb {E} \left[ T _ {\epsilon} \right] \leq \frac {1}{2 \theta - 1} \left(\frac {\phi (x _ {0}) - \phi^ {\star}}{\frac {1}{2} C _ {2} (\gamma (\hat {C} _ {1} \epsilon) ^ {2}} + \log_ {\gamma} \left(\frac {\hat {C} _ {1} \epsilon}{\Delta_ {0}}\right)\right) + (1 + C _ {3}) \frac {1}{2 \theta} \mathbb {E} \left[ T _ {\epsilon} \right].
$$

The lower bound on ϵ implies that $\begin{array} { r } { C _ { 3 } \leq \theta - \frac { 1 } { 2 } } \end{array}$ and so $\begin{array} { r } { \frac { 1 + C _ { 3 } } { 2 \theta } \leq \frac { 2 \theta + 1 } { 4 \theta } } \end{array}$ . Thus we have

$$
\frac {2 \theta - 1}{4 \theta} \mathbb {E} \left[ T _ {\epsilon} \right] \leq \frac {1}{2 \theta - 1} \left(\frac {\phi (x _ {0}) - \phi^ {\star}}{\frac {1}{2} C _ {2} (\gamma \hat {C} _ {1} \epsilon) ^ {2}} + \log_ {\gamma} \left(\frac {\hat {C} _ {1} \epsilon}{\Delta_ {0}}\right)\right).
$$

In order to use this theorem for effective complexity bounds, we must specify how to form models in a subspace. In the following subsections we discuss the two different approaches we used in the full space case - finite differences and interpolation based Λ-poised sets. The key difference now is in the lower bound on $\Delta _ { k }$ imposed by $\Delta _ { m i n }$ rather than occurring automatically.

### 5.2 Building Models in a Subspace

One can form a gradient estimate via a subspace version of (2.9). This can take the following form given in [11]:

$$
\hat {g} (0) = \sum_ {i = 1} ^ {q} \frac {f (x + \delta Q u _ {i}) - f (x)}{\delta} u _ {i}
$$

where $u _ { i }$ is the ith column of an orthogonal $q \times q$ matrix. Let us define $g ( x ) = Q { \hat { g } } ( 0 )$

By similar analysis as in $[ 4 ]$ , we can derive the bound

$$
\left\| \nabla \hat {m} (0) - \nabla \hat {\phi} (0) \right\| \leq \frac {\sqrt {q} L \delta}{2} + \frac {2 \sqrt {q} \epsilon_ {f}}{\delta}.
$$

Choosing $\delta = \Delta _ { k }$ , we have that $m _ { k }$ is $\kappa _ { e f } , \kappa _ { e g } .$ -fully linear model for

$$
\kappa_ {e g} = \frac {\sqrt {q} L}{2} + \frac {2 \sqrt {q} \epsilon_ {f}}{\Delta_ {\mathrm{min}} ^ {2}}, \quad \kappa_ {e f} = \kappa_ {e g} + \frac {L _ {Q} + \kappa_ {b h m}}{2}.
$$

In this case there are no model improving iterations. Thus all iterations are either successful or unsuccessful and each iteration requires either q or $q + 1$ function evaluations. With these specifics we can give a final complexity bound for Algorithm 3. For simplicity of the presentation we will give the final bounds in terms of the key components, such as $n , \epsilon , \epsilon _ { f } , L$ and $\Delta _ { m i n }$

**Theorem 5.13**. Let Assumption 1.2 and 2.2 hold. When randomizing, take $Q _ { t }$ such that the subspace it induces is uniformly distributed with $q \geq 3$ . For all $k = 0 , 1 , \ldots K _ { \epsilon } - 1$ , define $m _ { k } ( x _ { k } + s )$ with $g _ { k }$ as in (5.10). Let the parameters $\eta _ { 1 } , \eta _ { 2 } , \kappa _ { b h m } , \gamma$ be constants and assume $L \geq 1$ . Then for $\epsilon > \Omega ( \sqrt { n } ( L + \textstyle { \frac { \epsilon _ { f } } { \Delta _ { \operatorname* { m i n } } ^ { 2 } } } ) ( \sqrt { \epsilon _ { f } } + \Delta _ { \operatorname* { m i n } } ) )$ 2 assuming an initial trust-region radius $\Delta _ { 0 } \geq \Omega ( \sqrt { \epsilon _ { f } } + \Delta _ { \operatorname* { m i n } } )$ , let $K _ { \epsilon }$ be the random stopping time for the event $\{ \| \nabla \phi ( x _ { k } ) \| \leq \epsilon \}$ . We have the bound

$$
\mathbb {E} \left[ K _ {\epsilon} \right] \leq \mathcal {O} \left(\left(\frac {n}{\epsilon^ {2}}\right) \left(L + \frac {\epsilon_ {f}}{\Delta_ {\min} ^ {2}}\right) ^ {2}\right)
$$

where the $\it { \dot { b } i g - O ^ { \prime } { } ^ { \prime } }$ notation suppresses constant factors and an additive logarithmic term.

**Proof.** Since we use (5.10) in every iteration, we have that k and t are equivalent. From the definition of ${ \tilde { C } } _ { 1 }$ and the bound on $\kappa _ { e f }$ , we have $\begin{array} { r } { \tilde { C } _ { 1 } ^ { - 1 } = \Theta ( \kappa _ { e g } ) = \Theta ( \sqrt { q } ( L + \frac { \epsilon _ { f } } { \Delta _ { \operatorname* { m i n } } ^ { 2 } } ) ) } \end{array}$ . By Lemma 6.7 of [11], we have $\sqrt { 1 - \kappa _ { g } ^ { 2 } } = \Theta ( \sqrt { \frac { q } { n } } )$ Thus we have $\begin{array} { r } { \hat { C } _ { 1 } ^ { - 1 } = \Theta ( \sqrt { n } ( L + \frac { \epsilon _ { f } } { \Delta _ { \operatorname* { m i n } } ^ { 2 } } ) ) } \end{array}$ . Then the condition that $\epsilon > \sqrt { \frac { 1 6 \epsilon _ { f } } { ( 2 \theta - 1 ) ^ { 2 } C _ { 2 } \hat { C } _ { 1 } ^ { 2 } \gamma ^ { 2 } } }$ from Theorem 5.12 becomes that $\begin{array} { r } { \epsilon > \Omega ( \sqrt { n } ( L + \frac { \epsilon _ { f } } { \Delta _ { \mathrm { m i n } } ^ { 2 } } ) \sqrt { \epsilon _ { f } } ) } \end{array}$ . The condition that $\Delta _ { \mathrm { m i n } } \leq \gamma \hat { C } _ { 1 } \epsilon$ , becomes $\epsilon > \Omega ( \sqrt { n } ( L + \frac { \epsilon _ { f } } { \Delta _ { \operatorname* { m i n } } ^ { 2 } } ) \Delta _ { \operatorname* { m i n } } )$ . Combining these two bounds results in the condition $\epsilon > \Omega ( \sqrt { n } ( L + \textstyle { \frac { \epsilon _ { f } } { \Delta _ { \operatorname* { m i n } } ^ { 2 } } } ) ( \sqrt { \epsilon _ { f } } + \Delta _ { \operatorname* { m i n } } ) )$ . Finally the condition that $\Delta _ { 0 } \geq \hat { C } _ { 1 } \epsilon$ becomes $\Delta _ { 0 } \ge \mathcal { O } ( \sqrt { \epsilon _ { f } } + \Delta _ { \operatorname* { m i n } } )$ . The expected iteration bound then follows directly from Theorem 5.12.

Here we note that the lower bound on $\begin{array} { r } { \epsilon , \Omega ( \sqrt { n } ( L + \frac { \epsilon _ { f } } { \Delta _ { - } ^ { 2 } . . . } ) ( \sqrt { \epsilon _ { f } } + \Delta _ { \mathrm { m i n } } ) ) } \end{array}$ , can be approximately optimized by min taking $\Delta _ { \operatorname* { m i n } } = \Theta ( \sqrt { \epsilon _ { f } } )$ . The lower bound then becomes $\epsilon \geq \Omega ( \sqrt { n \epsilon _ { f } } )$ with a rate of

$$
\mathbb {E} \left[ K _ {\epsilon} \right] \leq \mathcal {O} \left(\frac {n}{\epsilon^ {2}}\right).
$$

We also note that each iteration requires only $\mathcal O ( q )$ function evaluations. Thus the total expected complexity rate is

$$
\mathbb {E} \left[ \mathcal {C} _ {\epsilon} \right] \leq \mathcal {O} \left(\frac {n q}{\epsilon^ {2}}\right).
$$

### 5.3 Geometry-Correcting Algorithm in Subspaces

We now describe a geometry-correcting version of Algorithm 3. This algorithm performs model improving steps of Algorithm 2 until either successful step is achieved or a fully linear model in the subspace is formed. At that point it terminates the work in that subspace and regenerates a new subspace as well as restarts the models using the initial sample sets $\mathcal { V } _ { 0 } , \mathcal { Z } _ { 0 }$ . This initialization choice is somewhat arbitrary and can be replaced by different initial sets. Each time, however, this requires computation of new function values for all points in the “initial” sample set. Our computational results show that this is quite expensive, if we use $\mathcal { V } _ { 0 }$ and $\mathcal { Z } _ { 0 }$ that contain $q$ points in each. We can delay resampling the random subspace until several successful or unsuccessful steps have been encountered and extend the theory to such strategies. However our computational results so far do not support an advantage of this approach. We can also choose ${ \mathcal { V } } _ { 0 }$ to contain only 1 point and $\mathcal { Z } _ { 0 }$ to be empty by modifying model improvement step and Lagrange polynomial computation to allow for incomplete sets. This modification is simple from the theory point of view but whether it can be practically competitive is yet unclear. Thus we retain the simplest approach for our analysis.

For this algorithm we can show the following complexity rate.

**Corollary 5.14**. Under the same assumptions as Theorem 5.13, letting $\textstyle \Lambda = 1 + { \frac { 1 } { q } }$ and $\Delta _ { \mathrm { m i n } } = \sqrt { \epsilon _ { f } }$ , and $| \mathcal { Z } _ { 0 } | = q$ , for any $\epsilon > \Omega ( \sqrt { n \epsilon _ { f } } )$ , the expected total oracle complexity of Algorithm $\it 4$ is bounded as

$$
O \left(\frac {n q \log q}{\epsilon^ {2}}\right).
$$

```txt
Algorithm 4: Geometry-correcting algorithm in subspace

Inputs: Inexact zeroth order oracle \( |f(x) - \phi(x)| \leq \epsilon_f \), minimum radius \( \Delta_{\min} \), initial \( x_0, \Delta_0 \geq \Delta_{\min} \), initial matrix \( Q_0 \in \mathbb{R}^{n \times q} \) with orthonormal columns and \( \eta_1 \in (0,1) \), \( \eta_2 > 0, \gamma \in (0,1) \), \( \Lambda > 1, \Lambda_{sc} \geq 1 \).

Initialization Initial sets \( \mathcal{Y}_0, \mathcal{Z}_0 \subset \mathbb{R}^q \) and the function values \( f(x_0), f(x_0 + Q_0y_i), y_i \in \mathcal{Y}_0, f(x_0 + Q_0z_i), z_i \in \mathcal{Z}_0 \). A set of Lagrange Polynomials \( \{\ell_i(v), i = 1, \dots, q\} \) in \( \mathcal{P} \) for the set \( \mathcal{Y}_0 \).

for \( k = 0, 1, 2, \cdots \) do

1 For the current \( Q_k \), build a quadratic model \( \hat{m}_k(v) = m_k(x_k + Q_kv) \) as in (3.2) using \( f(x_k) \) and \( f(x_k + Q_ky_i), y_i \in \mathcal{Y}_k, f(x_k + Q_kz_i), z_i \in \mathcal{Z}_k \).

2 Compute a trial step \( x_k + s_k \) where \( s_k = Q_kv_k \) with \( v_k \approx \arg\min_v\{m_k(x_k + Q_kv) : \|v\| \leq \Delta_k\} \).

3 Compute the ratio \( \rho_k \) as

\( \rho_k = \frac{f(x_k) - f(x_k + s_k) + 2\epsilon_f}{m_k(x_k) - m_k(x_k + s_k)} \).

4 Update the iterate and the TR radius as

\( (x_{k+1}, \Delta_{k+1}) \leftarrow \begin{cases}(x_k + s_k, \gamma^{-1}\Delta_k) & \text{if } \rho_k \geq \eta_1 \text{ and } \|g_k\| \geq \eta_2\Delta_k, \\(x_k, \Delta_k) & \text{else, if } \mathcal{Y}_k \text{ is not } \Lambda\text{-poised,} \\(x_k, \max\{\gamma\Delta_k, \Delta_{min}\}) & \text{otherwise.}\end{cases} \)

5 Update the subspace

\( Q_{k+1} \in \mathbb{R}^{n\times q} \leftarrow \begin{cases}random & \text{if } \rho \geq \eta_1 \text{ and } \|g_k\| \geq \eta_2\Delta_k \text{ or if } \mathcal{Y}_k \text{ is } \Lambda\text{-poised} \\Q_k & \text{otherwise.}\end{cases} \)

6 Update the interpolation sets

\( Y_{k+1}, Z_{k+1} \leftarrow \begin{cases}Y_0, Z_0 & \text{if } \rho \geq \eta_1 \text{ and } \|g_k\| \geq \eta_2\Delta_k \text{ or if } Y_k \text{ is } \Lambda\text{-poised} \\Update & as in Step 4 of Algorithm 2 & otherwise.\end{cases} \)
```

**Proof.** $\mathrm { B y }$ Theorem 3.5, since we chose $\textstyle \Lambda = 1 + { \frac { 1 } { q } }$ , we have the error bound

$$
\left\| \nabla \hat {m} _ {k} (0) - \nabla \hat {\phi} (0) \right\| \leq \mathcal {O} \left(\sqrt {q} \left(L \Delta_ {k} + \frac {\epsilon_ {f}}{\Delta_ {k}}\right)\right).
$$

Thus for iterations when ${ \mathcal { V } } _ { k }$ is Λ-poised, we have that $m _ { k }$ is $\kappa _ { e f } , \kappa _ { e g } \mathrm { - f u l l y }$ linear model for

$$
\kappa_ {e g}, \kappa_ {e f} = \mathcal {O} \left(\sqrt {q} \left(L + \frac {\epsilon_ {f}}{\Delta_ {\mathrm{min}} ^ {2}}\right)\right).
$$

From this, Theorem 5.12, and the arguments from the proof of Theorem 5.13, we can bound the number of non-geometry-correcting iterations (i.e. iterations where $Q _ { k }$ is resampled) by $\textstyle { \mathcal { O } } ( { \frac { n } { \epsilon ^ { 2 } } } )$ . For these iterations, we use only one oracle call to evaluate $f ( x _ { k } + s _ { k } )$ , however for the iteration which immediately follows a successful or an unsuccessful iteration, we use $\mathcal O ( q )$ oracle calls to evaluate $f ( x _ { k + 1 } + Q _ { k + 1 } y _ { i } )$ and $f ( x _ { k + 1 } + Q _ { k + 1 } z _ { i } )$ . For all other iterations, as in Algorithm 2, we use at most 2 oracle calls. Finally, by Theorem 4.1, the maximum number of consecutive geometry correcting iterations is $\mathcal { O } ( q \log q )$ . The result follows. □

## 6 Numerical Implementations and Results

In this section we propose an implementation of Algorithm 2, which incorporates all its elements such as the self-correcting and geometry-correcting steps but in addition includes several practical features. Some of these features are borrowed from Powell’s algorithms and some are new. As we will discuss, all these additional features improve practical performance but make the analysis more cumbersome. However, ultimately the order of the worst-case complexity of the algorithm is preserved.

The practical implementation is given in Algorithm 5. It utilizes the two interpolation sets Y and Z to manage linear Lagrange polynomials while fitting quadratic models, as proposed in Algorithm 2. We compare our proposed algorithm to NEWUOA [21], which is arguably the most scalable of Powell’s algorithms and which maintains geometry of the full interpolation set by the use of quadratic Lagrange polynomials, and with DFOTR [2] which is a surprisingly efficient method that does not maintain any Lagrange polynomials and only updates the sample set based on the distance of the points to the TR center. In that respect DFOTR also maintains two separate sets, in that it does not reduce the trust region on steps that are not successful and when there are fewer than $n + 1$ points in the appropriate vicinity of the trust region center.

We also test a variation of Algorithm 2, to which current theory does not extend and which uses the quadratic Lagrange polynomials combined with a NEWUOA-like self-correcting rule in place of the rule described in Algorithm 2. As this method seems to provide improvement in high-dimensional setting, it gives motivation for the theory from Sections 3 and 4 to be extended to quadratic Lagrange polynomials in future work. All solvers will use a novel, adaptive fitting scheme. We test these algorithms on a collection of unconstrained problems from the CUTEst test set [17] and investigate results in low dimension, high dimension, and in randomized subspaces.

The following Section 6.1 describes the design choices of our proposed algorithms and how it still satisfies the theory. In Section 6.2 we discuss the testing methodology and numerical results.

### 6.1 Practical Implementations

Algorithm 5 (GC-YZ-LIN) is a practical implementation of the geometry correction framework which maintains linear Lagrange polynomials for the set Y while interpolating a quadratic model using $\mathcal { V } \cup \mathcal { Z } \cup \{ x _ { k } \}$ . This method can be seen as the middle ground between DFOTR and NEWUOA, where the former makes only minimal effort to ensure good sample set geometry while the latter uses a significant amount of effort.

We now describe the changes implemented in Algorithm 5 as opposed to Algorithm 2. Algorithm 5 makes use of a resolution floor and small step gate, both of which are ideas borrowed from Powell’s methods.

The resolution floor $\sigma _ { k }$ is an adaptive lower bound on $\Delta _ { k }$ which only gets decreased once no progress can be made for that resolution. That is, if $\Delta _ { k } = \sigma _ { k }$ , the geometry is good, and the iteration is still unsuccessful we reset $\sigma _ { k + 1 } = \theta \sigma _ { k }$ for some $0 < \theta < 1$ . We then allow $\Delta _ { k }$ to shrink even on model improving iterations as long as $\Delta _ { k }$ is larger than the floor $\sigma _ { k }$ . This is motivated by the observation that in higher dimensions it can be expensive to always ensure good geometry before shrinking, hence an alternative is to only ensure this good geometry at intervals throughout the trajectory of the algorithm.

A small step gate and small model gradient gate are used by checking the conditions $\| s _ { k } \| \ge c _ { g } \sigma _ { k }$ and $\| g _ { k } \| \ge \eta _ { 2 } \Delta _ { k }$ respectively. The small step gate adds an additional condition under which an iteration is deemed successful, which we note improves the termination speed of the algorithm. In the small model gradient gate, before evaluating a proposed step, we check if the norm of the model gradient is small relative to the trust region radius. That is if we know the step is going to be rejected, we save a function evaluation by skipping the evaluation of the trial step and attempt a geometry correcting step and possibly shrink the trust region radius. The same principle applies for the small step gate.

With these two changes, the complexity analysis in Section 2 follows through with small modifications, provided that the following simple assumption holds.

**Assumption 6.1**. On every iteration k, we have

$$
\left\| s _ {k} \right\| \geq \kappa_ {\mathrm{step}} \min \left\{\Delta_ {k}, \frac {\left\| g _ {k} \right\|}{\kappa_ {b h m}} \right\}.
$$

Note that if the trust region subproblem is solved exactly, the assumption is satisfied with $\kappa _ { \mathrm { s t e p } } = 1$ , while for the Cauchy step we have $\begin{array} { r } { \kappa _ { \mathrm { s t e p } } = \frac { \kappa _ { f c d } } { 3 } } \end{array}$ . Then, for Lemma 2.3 to go through, we require that $c _ { g } \leq \kappa _ { \mathrm { s t e p } }$ and Assumption 6.1 to hold. It would then follow that

$$
\kappa_ {b h m} \Delta_ {k} \leq \max \{\kappa_ {b h m}, \eta_ {2} \} \Delta_ {k} \leq \| g _ {k} \| \implies \min \left\{\Delta_ {k}, \frac {\| g _ {k} \|}{\kappa_ {b h m}} \right\} = \Delta_ {k},
$$

hence $\| s _ { k } \| \ge \kappa _ { \mathrm { s t e p } } \Delta _ { k } \ge c _ { g } \sigma _ { k }$ , and a small $\Delta _ { k }$ still implies a successful iteration, even with a small step gate. The rest of the complexity argument for the full-dimensional method follows very closely to the analysis in Section 2, by applying key results with $\sigma _ { k }$ instead of $\Delta _ { k }$ and deriving a bound on the number of unsuccessful iterations until $\Delta _ { k }$ reaches the floor $\sigma _ { k }$ for each round of $\sigma _ { k }$ reductions. Hence the proposed method will obey the theory, at the cost of some logarithmic factors.

The model fitting procedure fits a quadratic model, which we call the “hedge” model, using the following somewhat elaborate sequence of steps. On each iteration we compute the minimum Frobenius norm (MFN) model and the minimum change Frobenius norm (MCFN) model. Both of these models are computed by solving

$$
\min _ {g, H} \| H _ {p r e v} - H \| _ {F} \quad \text {s.t.} \quad v ^ {\top} g + \frac {1}{2} v ^ {\top} H v = f (x _ {k} + v) - f (x _ {k}), v \in \mathcal {Y} \cup \mathcal {Z}.
$$

where the MFN model sets $H _ { p r e v } = 0$ and the MCFN model sets $H _ { p r e v }$ to the previous iteration’s MCFN Hessian. For both models, an exponentially weighted moving average of a relative error score is maintained and is used to determine which model is to be used for the current iteration. Only one model is “active” at a time and if the error for the alternative model becomes lower than the active model, we switch which model is used to compute the trial step. This change comes from the observation that sometimes it may be beneficial to remember the current local curvature, while at other times it may be beneficial to fit a fresh model as previous model Hessians can become stale. The proposed scoring method is a way to adaptively decide which model is best suited for the current iteration. We remark that a similar idea was proposed in Powell’s original paper, where consecutive poor steps would trigger an MFN model to be fit instead of MCFN.

After this modeling procedure, to ensure a bounded model Hessian, we check if the fitted Hessian exceeds the bound, i.e. if $\| H \| > K$ . If so, we fit $g _ { k }$ and $H _ { k }$ by solving the regression problem (4.1). We note that there are other ways to bound $\| H _ { k } \|$ , such as via clipping. Since several of the objective functions in our data set naturally have extreme curvature, to prevent the truncation of useful curvature information, we set K to be a very large number. As a result, the condition $\| H \| > K$ will rarely trigger, which will yield either one of the MFN or MCFN models throughout an overwhelming majority of the iterations. We note that setting K very large implies that for some problems $\kappa _ { b h m }$ becomes very large. However, it appears to happen only when L is similarly large, thus having large $\kappa _ { b h m }$ or reducing it has no bearing on the order of the theoretical complexity bound.

We also develop a version of Algorithm 5, which we call GC-YZ-V, where the self-correction step computes and updates quadratic Lagrange polynomials over the set $\mathcal { V } \cup \mathcal { Z } .$ , similar to NEWUOA, while still maintaining the $\mathcal { V } \mathcal { Z }$ separation. Specifically, for the self-correcting step, we compute the score of a point in the sample set to be:

$$
\mathrm{score} (y _ {i}) = \max \left(1, \frac {\| y _ {i} \| ^ {2}}{\max (0 . 1 \Delta_ {k} , \sigma_ {k}) ^ {2}}\right) ^ {3} \cdot \ell_ {i} (s _ {k}) ^ {2},
$$

and replace the point that achieved the maximum score by $s _ { k }$ . Intuitively, this attempts to simultaneously remove points that are deemed far away while attempting to improve geometry. After such self-correction, the geometry correction is carried out in the same manner as in GC-YZ-LIN. Since the pseudocode closely mirrors Algorithm 5 and is not currently accompanied by any theoretical guarantees, we omit it and present the results as a proof of concept. We see that this algorithm somewhat outperforms both GC-YZ-LIN and NEWUOA and believe that modifications to only the self-correction step can be ultimately covered by extending our theory.

The random subspace variation with geometry correction has a straightforward implementation, in that there is little deviation from the theoretical framework described in Algorithm 4. We call this algorithm GCsub. The only deviation from the framework is the addition of a successful and unsuccessful iteration “patience” parameter. This value dictates the required number of successful or unsuccessful iterations before a subspace is redrawn. Intuitively, if we are making significant progress on a particular subspace, it may be beneficial to keep this subspace instead of redrawing. In our theoretical framework Algorithm 4 this parameter is set to 1. While one can extend the theory to any fixed value of this parameter, it would complicate the notation of Section 5. On the other hand our computational results suggest that 1 is the best value for this parameter at least in the current setting.

```text
Algorithm 5: Geometry-correcting algorithm with Y and Z separation (GC-YZ-LIN)

Inputs: A zeroth-order oracle $f(x) \approx \phi(x)$ , $\Delta_{0}$ , $x_{0}$ , $\gamma \in (0,1)$ , $\eta_{1} > 0$ , $\eta_{2} > 0$ , $\Lambda > 1$ , $\Lambda_{sc} \geq 1$ , $\theta \in (0,\gamma)$ , $\sigma_{end} \in (0,\Delta_{0}]$ , $c_{g} \in (0,1]$ , model Hessian bound K, averaging weight $\beta \in (0,1]$ .

Initialization: An initial set $Y_{0}$ such that $|Y_{0}| = n$ , an initial set $Z_{0}$ such that $|Z_{0}| \leq n(n+1)/2$ and the function values $f(x_{0})$ , $f(x_{0} + y_{i})$ , $y_{i} \in Y_{0}$ , $f(x_{0} + z_{i})$ , $z_{i} \in Z_{0}$ . Set the resolution floor $\sigma_{0} = \Delta_{0}$ . A set of Lagrange polynomials $\{\ell_{i}(x), i = 1, \ldots, n\}$ in P for the set $Y_{0}$ , model errors $e_{C}, e_{F} \leftarrow undefined$ , $H_{prev} \leftarrow 0$ , set active model label $\alpha \leftarrow F$ .

for k = 0, 1, 2, … do

1 Model building: Construct MFN model $g_{F}$ , $H_{F}$ . Construct MCFN model $g_{C}$ , $H_{C}$ . Set $H_{prev} = H_{C}$ .

2 If $e_{C}$ or $e_{F}$ is undefined, then $\alpha \leftarrow F$ and skip model switching step in Line 3.

3 Model switching: If $\alpha = F$ and $e_{C} < 0.8e_{F}$ then $\alpha \leftarrow C$ , otherwise if $\alpha = C$ and $e_{F} < 0.8e_{C}$ then $\alpha \leftarrow F$ .

4 If $\alpha = F$ then $g_{k}$ , $H_{k} \leftarrow g_{F}$ , $H_{F}$ . If $\alpha = C$ then $g_{k}$ , $H_{k} \leftarrow g_{C}$ , $H_{C}$ .

5 Ensuring bounded model Hessians: If $\|H_{k}\| > K$ set $g_{k}$ , $H_{k}$ by solving (4.1).

6 Construct the set of Lagrange polynomials $\{\ell_{i}(x), i = 1, \ldots, n\}$ in P for the set $Y_{k}$ and find max value

( $j_{k}^{*}, s_{k}^{*}$ ) = arg max _{j=1,...,n,s∈B(0,Δk)} | $\ell_{j}(s)|$ .

7 Small model gradient gate: If $\|g_{k}\| < \eta_{2}\Delta_{k}$ perform the following checks:
If $|\ell_{j_{k}^{*}}(s_{k}^{*})| > \Lambda$ then perform “Geometry correction of Y by replacing a “bad” point”.

If $|\ell_{j_{k}^{*}}(s_{k}^{*})| \leq \Lambda$ and $\Delta_{k} \leq \sigma_{k}$ perform the resolution update: $\sigma_{k+1} = \max\{\theta\sigma_{k},   \sigma_{end}\},   \Delta_{k+1} = \max\{\frac{1}{2}\Delta_{k},   \sigma_{k+1}\}$ .

If $|\ell_{j_{k}^{*}}(s_{k}^{*})| \leq \Lambda$ and $\Delta_{k} > \sigma_{k}$ , then $\Delta_{k+1} \leftarrow \max\{\gamma\Delta_{k},   \sigma_{k}\}$ .

Go to 1 and $k \leftarrow k + 1$ .

8 Compute a trial step $s_{k}$ as in Algorithm 1.

9 Small step gate: If $\|s_{k}\| < c_{g}\sigma_{k}$ , do not evaluate $f(x_{k} + s_{k})$ . Perform the same checks as small model gradient gate.

10 Evaluate $f(x_{k} + s_{k})$ and compute the ratio $\rho_{k}$ as in Algorithm 1.

11 Model scoring: for $μ \in {F,C}$ do

p_μ ← (g_μ)^T s_k + ½s_k^T H_μs_k,   f_c ← f(x_k + s_k) - f(x_k)

ε_μ ← |p_μ - f_c| / max{|f_c|, |p_μ|}

If e_μ is undefined, then e_μ = ε_μ, otherwise e_μ = βe_μ + (1 - β)e_μ.

12 Successful iteration: ρ_k ≥ η₁. Perform the successful update as in Algorithm 2.

13 Unsuccessful or Model Improving iteration: ρ_k < η₁. Perform all applicable steps (i), ..., (iv) from Algorithm 2.

Model improving iteration: If I imp = 1, set Δk+1 = max{γΔk, σk};

Unsuccessful iteration above the floor: If I imp = 0 and Δ_k > σ_k, set Δk+1 = max{γΔk, σ_k};

Unsuccessful iteration at floor: If I imp = 0 and Δ_k ≤ σ_k, perform the resolution update from Line 7.

```

### 6.2 Testing and Results

We now turn to the results. We state upfront that our numerical comparison yields the following observations.

1. The new adaptive Hessian fitting procedure improves performance for all solvers.

2. Methods with geometry correction, i.e. GC-YZ-LIN/V and NEWUOA, outperform DFOTR in higher dimensions.

3. Using the Y-Z separation, GC-YZ-LIN is able to largely match NEWUOA, with trade-offs between early-game and late-game performance. Transplanting NEWUOA’s self-correcting rule into the Y-Z framework, we obtain a variant that outperforms NEWUOA; no theoretical analysis has been developed for it yet.

4. A practical implementation of a randomized subspace algorithm that is competitive with full space solvers remains difficult.

Here we give the details of our testing methodology. Table 6.1 outlines the parameters used for our proposed solver and DFOTR. We will use PRIMA’s [27] implementation of NEWUOA, which has fixed trust region parameters by default, so we omit restating it in the table. All problems are initialized in the same way, where the initial interpolation set $\mathcal { V } _ { 0 }$ is set to $\{ \Delta _ { 0 } u _ { i } \}$ where $u _ { i } , i = 1 , \ldots , n$ are randomly rotated coordinate vectors, and radius $\Delta _ { 0 } = 0 . 5 . ~ \mathcal { Z } _ { 0 }$ is then set to $\{ - \Delta _ { 0 } u _ { i } \}$ . To allow the algorithm to run until an exhausted budget, we set the termination criteria to be $\Delta _ { \mathrm { e n d } } = 1 0 ^ { - 1 2 }$ for DFOTR and $\rho _ { \mathrm { e n d } } , \sigma _ { \mathrm { e n d } } = 1 0 ^ { - 1 2 }$ for NEWUOA and GC-YZ-LIN/V respectively. All trust region subproblems are solved exactly. We compare performances using data profiles at tolerances $\tau \in \{ 1 0 ^ { - 2 } , 1 0 ^ { - 4 } , 1 0 ^ { - 6 } \}$ and report the area under curve score of each solver in the legend. We average the curves obtained from the different randomly rotated initial sets.

We remark on the choice of three parameters in GC-YZ-LIN that also show up in the complexity analysis. Firstly, the model Hessian bound K is set to be extremely large, namely $1 0 ^ { 1 0 0 }$ and essentially inactive. This is largely a byproduct of testing solver performance on problems from CUTEst with naturally extreme curvature. For example, problems such as “SSBRYND” and “SCURLY10” have true Hessian norms greater than $1 0 ^ { 2 0 }$ while “POWERSUM” can have a true Hessian norm even exceeding $1 0 ^ { 1 0 0 }$ . Thus setting a universal, fixed, and small K will handicap the model’s ability to capture the function’s true curvature. Additionally, we set $\eta _ { 2 }$ to be small, namely $5 \times 1 0 ^ { - 9 }$ . While ensuring the norm of the model gradient is not too small relative to the radius is necessary for the theory, a large $\eta _ { 2 }$ tethering the trust region radius to the model gradient can be detrimental to problems with small local Lipschitz constants, for example, the Rosenbrock function outside of the “banana valley”. If there is still progress to be made, iterates in this region will be forced to have a small trust region radius and may stunt progress. Moreover, in practice it appears beneficial to have a relaxed choice for Λ. According to our theory, Λ-poisedness guarantees a certain amount of function decrease on successful iterations each time. However, ensuring a small Λ in practice comes at the cost of more function evaluations for geometry correction. We observe that the best performance comes when $\Lambda = 1 0 0 0$ , which suggests that although the geometry may not be perfect, the algorithm is still able to progress, and geometry correction should only take place when the geometry is exceedingly bad. We also make the distinction between Λ and $\Lambda _ { s c } ,$ where the latter is a threshold on how large a Lagrange polynomial value at a step would need to be to initiate a self-correcting step. In practice we set is to 2, while the analysis applies as long as $\Lambda _ { s c } \geq 1$

These seemingly detrimental parameter choices for $\eta _ { 2 }$ and Λ in terms of the complexity bounds can be explained by the fact that our analysis addresses the worst-case. Setting $\eta _ { 2 } = \sqrt { n }$ and $\textstyle \Lambda = \Theta ( 1 + { \frac { 1 } { n } } )$ essentially forces more model improving iterations before a successful step is allowed or trust region radius is reduced, to ensure best models and progress according to the worst case $\kappa _ { e g }$ constant. However, in practice the error between the model and the function at the trial step can be much smaller than the worst case $\kappa _ { e g }$ bound suggests, so too many model improvement steps may be wasteful. It would be interesting to explore adaptive choices for these constants in future work.

Low-dimensional tests. We begin our testing on 174 low-dimensional test problems with dimension between $2 \leq n \leq 5$ . Since the dimension is low, we set the maximum allowable number of points in the sample set to be $( n + 1 ) ( n + 2 ) / 2$ points. Aside from the first few iterations, the sample set of GC-YZ-LIN/V and NEWUOA will eventually become full, resulting in an interpolation system that is fully determined, so the two models coincide and the switching is inactive. In contrast, DFOTR often removes several sample points at once, so the hedge model can be active throughout the trajectory. Figure 6.1 shows the performance of the solvers on this test set. All solvers seem to perform comparably well.

Medium-/High-dimensional tests. We also benchmark on test sets in dimension 30, 100, and 200, with 95, 101, and 92 problems respectively in each set. For these tests, we will use a maximum of 2n + 1 sample points in the total interpolation set. Firstly, Figure 6.2 compares the solvers using their default fitting routine against using the hedge model. We see that adopting this model strictly improves performance across the board. In what follows, to ensure fair testing, all solvers are benchmarked using the hedge model.

Figures 6.3, 6.4, and 6.5 show the data profiles for the 30-, 100- and 200- dimensional test sets respectively.

> **Table 6.1:** Solver settings used throughout testing.

| Setting | GC-YZ-LIN/V | GC-sub | DFOTR |
| --- | ---: | ---: | ---: |
| Acceptance ratio $\eta_1$ | 0.01 | 0.1 | 0.05 |
| Model gradient threshold $\eta_2$ | $5 \times 10^{-9}$ | $5 \times 10^{-9}$ | — |
| Very successful iteration threshold | — | — | 0.5 |
| Expansion factor $\gamma_{\mathrm{inc}}$ | 1.3 | 1.3 | 1.6 |
| Contraction factor $\gamma_{\mathrm{dec}}$ | 0.8 | 0.8 | 0.8 |
| Resolution shrink $\theta$ | 0.1 | — | — |
| Poisedness threshold $\Lambda$ | 1000 | 1000 | — |
| Self-correcting threshold $\Lambda_{\mathrm{sc}}$ | 2 | 2 | — |
| Hedge: EWMA decay $\beta$ | 0.8 | — | — |
| Model Hessian bound $K$ | $10^{100}$ | — | — |
| Small step gate threshold $c_g$ | 0.5 | — | — |
| Subspace dimension $q$ | — | 5 | — |
| Interpolation set capacity | — | 2q | — |

![Figure 6.1](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/bbcf8205-f525-44f1-b045-6cce953a6f01/be4b14033448f48d87e811b4ed01c8106e0e8d3fe71d3a65a77dc7da216a6ef9.jpg)

> Figure 6.1: Data profiles for the low-dimensional test suite: $2 \leq n \leq 5 .$

![Figure 6.2](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/bbcf8205-f525-44f1-b045-6cce953a6f01/5fbcf7e93237168be387b9c018330c7159b2d6339c39148a1a9d2a13f97d43f5.jpg)

> Figure 6.2: Data profiles comparing the hedge model with each solver’s default fitting procedure.

![Figure 6.3](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/bbcf8205-f525-44f1-b045-6cce953a6f01/494c33de4172e7ab97ccea7136998969b84a34e6a52ef90c5b86de05d560928a.jpg)

> Figure 6.3: Data profiles for the test suite $n = 30$.

![Figure 6.4](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/bbcf8205-f525-44f1-b045-6cce953a6f01/b18a78623a649dbca31596d800bea64c9885e60ff82d3f9604cd1388ae1817b2.jpg)

> Figure 6.4: Data profiles for the test suite $n = 100$.

The first most apparent observation is the deterioration of DFOTR as dimension increases. This is largely due to the absence of a geometry management procedure. As dimension increases, the 2n + 1 points become increasingly sparse and are more likely to be nearly degenerate. Thus, DFOTR performs increasingly aggressive shrinking to eventually force a geometry reset, which leads to radius collapse and premature termination, as seen in the significant flattening of the DFOTR data profile curves.

Another apparent observation is the almost strict improvement of GC-YZ-V over NEWUOA and GC-YZ-LIN. GC-YZ-V is a combination of several algorithmic features from both NEWUOA and GC-YZ-LIN. This result suggests that the use of NEWUOA-style self-correction may be more efficient than first ensuring all points are close, then checking the self-correction property. While the latter is more principled and easier to analyze theoretically, it may be slower in practice. This also suggests it may be worthwhile to maintain a separation of sets Y-Z, even when quadratic Lagrange polynomials are maintained for the entire sample set Y ∪ Z. Again, we note that this variant is not covered by current theory and is a line of future work.

Finally, we observe that GC-YZ-LIN is closely competitive with NEWUOA across dimensions and across tolerances. $\mathrm { A t } ~ \tau = 1 0 ^ { - 2 }$ , the curves are nearly identical. At $\tau = { 1 0 ^ { - 4 } , 1 0 ^ { - 6 } }$ , a pattern emerges where NEWUOA is quicker to solve more problems early in the run, however GC-YZ-LIN eventually catches up and overtakes NEWUOA later in the run. In general, the performance of the two methods is comparable, which is surprising given that GC-YZ-LIN only manages linear geometry for a subset of the sample set compared to NEWUOA’s full quadratic geometry.

Results in subspaces. Figure 6.6 compares the performance of the random subspace method against GC-YZ-LIN. The subspace method sets q = 5, patience parameters equal to 1, and draws a fresh coordinate stencil on every subspace redraw. A maximum of 2q + 1 points is used in the sample. We remark that varying choices of patience parameter, subspace dimension, re-draw strategy, and maximum number of allowed sample points were tested, and we opt to present the best performing combination found in dimension 30 and 100. It is evident that the subspace method does not perform as well as a full space solver and its performance deteriorates as tolerances get smaller. While theory suggests that as dimension increases, the competitiveness of subspace methods should as well, this is not something that is immediately evident. A plausible explanation for the discrepancy between theory and practice is as follows. The random subspace method needs at least $\mathcal O ( q )$ points for each subspace redraw and the bound on the number of subspace redraws is not loose. Thus we arrive at the worst case complexity of order q without fail. However, in the full space solvers, we rarely need more than a few consecutive geometry correcting steps to achieve good geometry, hence in practice we are typically far from actually attaining the worst-case behavior described in the complexity bound. Another explanation is that subspace methods ultimately lose the advantage of quadratic approximation in the whole space, so may only be competitive where full space quadratic approximation is prohibitive or useless. A future implementation which initiates any sample set with as few as two sample points may improve performance of subspace methods. In addition reusing sample points from other subspaces can be beneficial. In conclusion, our contribution in terms of subspace TR method in this paper is mainly theoretical while practical approaches require further investigation. We would like to point the reader to the extensive empirical study carried out in [8].

![Figure 6.5](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/bbcf8205-f525-44f1-b045-6cce953a6f01/54f1598cd40ef08f6676e4fb64cea370f7cf179d556d27276abaf5cfc036e434.jpg)

> Figure 6.5: Data profiles for the test suite $n = 200$.

![Figure 6.6](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/bbcf8205-f525-44f1-b045-6cce953a6f01/e2cfeabcc71fde484c2a44fe6193fb5932bf9919f39e3f84110a322ae1e1de73.jpg)

> Figure 6.6: Data profiles for the random subspace implementation compared against GC-YZ-LIN for $n = 30, 100$.

## Acknowledgments

This work was partially supported by ONR award N00014-22-1-215 and the Gary C. Butler Family Foundation.

## References

[1] Charles Audet and Warren L. Hare. Derivative-Free and Blackbox Optimization. Springer Series in Operations Research and Financial Engineering. Springer International Publishing, Cham, 2017.

[2] Afonso S Bandeira, Katya Scheinberg, and Luís N Vicente. Computation of sparse low degree interpolating polynomials and their application to derivative-free optimization. Mathematical Programming, 134(1):223– 257, 2012.

[3] Afonso S Bandeira, Katya Scheinberg, and Luís N Vicente. Convergence of trust-region methods based on probabilistic models. SIAM Journal on Optimization, 24(3):1238–1264, 2014.

[4] Albert S Berahas, Liyuan Cao, Krzysztof Choromanski, and Katya Scheinberg. A theoretical and empirical comparison of gradient approximations in derivative-free optimization. Foundations of Computational Mathematics, pages 1–54, 2021.

[5] Albert S Berahas, Liyuan Cao, and Katya Scheinberg. Global convergence rate analysis of a generic line search algorithm with noise. SIAM Journal on Optimization, 31(2):1489–1518, 2021.

[6] Jose Blanchet, Coralia Cartis, Matt Menickelly, and Katya Scheinberg. Convergence rate analysis of a stochastic trust-region method via supermartingales. INFORMS Journal on Optimization, 1(2):92–119, 2019.

[7] Liyuan Cao, Albert S. Berahas, and Katya Scheinberg. First- and second-order high probability complexity bounds for trust-region methods with noisy oracles. Mathematical Programming, 207(1-2):573–624, 2023. Also available as a preprint at arXiv, May 2022.

[8] Coralia Cartis and Lindon Roberts. Scalable subspace methods for derivative-free nonlinear least-squares optimization. Mathematical Programming, 199(1):461–524, May 2023.

[9] Coralia Cartis and Lindon Roberts. A note on the complexity of random subspace model-based methods for derivative-free optimization, 2026.

[10] Coralia Cartis and Katya Scheinberg. Global convergence rate analysis of unconstrained optimization methods based on probabilistic models. Mathematical Programming, 169(2):337–375, 2018.

[11] Abraar Chaudhry and Katya Scheinberg. On complexity of model-based derivative-free methods. In Proceedings of the International Congress of Mathematicians (ICM) 2026, volume 7, pages 208–228, Philadelphia, PA, USA, 2026. SIAM.

[12] Andrew R Conn, Nicholas IM Gould, and Philippe L Toint. Trust region methods. SIAM, 2000.

[13] Andrew R Conn, Katya Scheinberg, and Luís N Vicente. Introduction to derivative-free optimization. SIAM, 2009.

[14] Kwassi Joseph Dzahini and Stefan M. Wild. Stochastic trust-region algorithm in random subspaces with convergence and expected complexity analyses. SIAM Journal on Optimization, 34(3):2743–2774, 2024.

[15] R. Garmanjani, D. Júdice, and L. N. Vicente. Trust-region methods without using derivatives: Worst case complexity and the non-smooth case. SIAM Journal on Optimization, 26(4):1987–2011, 2016.

[16] Serge Gratton, Clément W Royer, Luís N Vicente, and Zaikun Zhang. Complexity and global rates of trust-region methods based on probabilistic models. IMA Journal of Numerical Analysis, 38(3):1579–1597, 2018.

[17] Serge Gratton and Philippe L. Toint. S2mpj and cutest optimization problems for matlab, python and julia, 2024.

[18] Billy Jin, Katya Scheinberg, and Miaolan Xie. High probability complexity bounds for line search based on stochastic oracles. Advances in Neural Information Processing Systems, 34:9193–9203, 2021.

[19] Jefrey Larson, Matt Menickelly, and Stefan M Wild. Derivative-free optimization methods. Acta Numerica, 28:287–404, 2019.

[20] Jorge J Moré and Stefan M Wild. Benchmarking derivative-free optimization algorithms. SIAM Journal on Optimization, 20(1):172–191, 2009.

[21] Michael J D Powell. The NEWUOA software for unconstrained optimization without derivatives. Technical Report DAMTP 2004/NA08, Department of Applied Mathematics and Theoretical Physics, University of Cambridge, 2004.

[22] Michael JD Powell. UOBYQA: unconstrained optimization by quadratic approximation. Mathematical Programming, 92(3):555–582, 2002.

[23] MJD Powell. On the lagrange functions of quadratic models that are defined by interpolation. Optimization Methods and Software, 16(1-4):289–309, 2001.

[24] T. M. Ragonneau and Z. Zhang. PDFO: a cross-platform package for Powell’s derivative-free optimization solvers. Mathematical Programming Computation, 16(4):535–559, 2024.

[25] Luis Miguel Rios and Nikolaos V. Sahinidis. Derivative-free optimization: a review of algorithms and comparison of software implementations. Journal of Global Optimization, 56(3):1247–1293, 2013.

[26] K. Scheinberg and Ph. L. Toint. Self-correcting geometry in model-based algorithms for derivative-free unconstrained optimization. SIAM Journal on Optimization, 20(6):3512–3532, 2010.

[27] Z. Zhang. PRIMA: Reference Implementation for Powell’s Methods with Modernization and Amelioration. available at http://www.libprima.net, DOI: 10.5281/zenodo.8052654, 2023.
