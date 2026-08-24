# Self-Tuning Stochastic Optimization with Curvature-Aware Gradient Filtering

Ricky T. Q. Chen<sup>∗</sup> <sup>†</sup> Dami Choi<sup>∗</sup> <sup>†</sup> Lukas Balles<sup>∗</sup> <sup>‡</sup> David Duvenaud<sup>†</sup> Philipp Hennig<sup>‡</sup> 

## Abstract

Standard first-order stochastic optimization algorithms base their updates solely on the average mini-batch gradient, and it has been shown that tracking additional quantities such as the curvature can help de-sensitize common hyperparameters. Based on this intuition, we explore the use of exact per-sample Hessian-vector products and gradients to construct optimizers that are self-tuning and hyperparameter-free. Based on a dynamics model of the gradient, we derive a process which leads to a curvature-corrected, noise-adaptive online gradient estimate. The smoothness of our updates makes it more amenable to simple step size selection schemes, which we also base off of our estimates quantities. We prove that our model-based procedure converges in the noisy quadratic setting. Though we do not see similar gains in deep learning tasks, we can match the performance of well-tuned optimizers and ultimately, this is an interesting step for constructing self-tuning optimizers. 

## 1 Introduction

Stochastic gradient-based optimization is plagued by the presence of numerous hyperparameters. While these can often be set to rule-of-thumb constants or manually-designed schedules, it is also common belief that a more information regarding the optimization landscape can help present alternative strategies such that manual tuning has less of an impact on the end result. For instance, the use of curvature information in the form of Hessian matrices or Fisher information can be used to de-sensitize or completely remove step size parameter (Ypma, 1995; Amari, 1998; Martens, 2014), and the momentum coefficient can be set to reduce the local gradient variance (Arnold et al., 2019b). 

Based on these intuitions, we investigate the use of efficient curvature and variance estimates during training to construct a self-tuning optimization framework. Under a Bayesian paradigm, we treat the true gradient as the unobserved state of a dynamical system and seek to automatically infer the true gradient conditioned on the history of parameter updates and stochastic gradient observations. 

Our method is enabled by evaluations of exact persample gradients and Hessian-vector products. With recent improvements in automatic differentiation tooling (e.g., Bradbury et al., 2018; Agarwal and Ganichev, 2019; Dangel et al., 2020), this matches the asymptotic time cost of minibatch gradient and Hessian-vector product evaluations. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-27/e5a1e9f0-0ad2-40f9-bc9e-43d13f521b11/24227d451289c0475254f2f9864d2de4bb6c94893aeb1508c30204194a266acd.jpg)



Figure 1: Stochastic gradient eventually goes into diffusion and does not converge. Our filtered gradients offer smooth convergence and complements adaptive step sizes.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-27/e5a1e9f0-0ad2-40f9-bc9e-43d13f521b11/6d42661a90472a5bc1f3a178b994d19db5fb5895c77f13695206d4a747bfb814.jpg)



Figure 2: Graphical model of the hidden Markov dynamics model. The main idea of our algorithm is that the dynamics parameters can be cheaply estimated on each minibatch, and smoothed across time using exact Kalman filter inference. These dynamics parameters are the gradient variance Σ, the directional curvature Bδ and its variance Q. We stabilize Σ with an exponential moving average, which is effectively another, more elementary form of Kalman filtering.


While our framework contains the good properties of both curvature-based updates and variance reduction—which are attested in toy and synthetic scenarios—we do not observe significant improvements empirically in optimizing deep neural networks. Notably, our approach can be viewed as an explicit form of the implicit gradient transport of Arnold et al. (2019b), yet it does not achieve the same acceleration empirically observed in practice. While we do not fully understand this behavior, we analyze the estimated quantites along the training trajectory and hypothesize that our method has a higher tendency of going down high-variance high-curvature regions whereas standard stochastic gradient descent is repelled from such regions due to gradient variance. This potentially serves as a downside of our method in the deep learning setting. Regardless, the use of efficient variance estimation and the interpretation of gradient estimation within a Bayesian filtering framework are useful constructs in the development of self-tuning stochastic optimization. 

## 2 Bayesian Filtering for Stochastic Gradients

We consider stochastic optimization problems of the general form 

$$
\arg \min _ {\theta \in \mathbb {R} ^ {d}} f (\theta), \quad f (\theta) = \mathbb {E} _ {\xi} \left[ \tilde {f} (\theta , \xi) \right]\tag{1}
$$

where we only have access to samples $\xi .$ . Stochastic gradient descent—the prototypical algorithm for this setting—iteratively updates $\theta _ { t + 1 } = \theta _ { t } - \alpha _ { t } g _ { t }$ , where 

$$
g _ {t} = \frac {1}{n} \sum_ {i = 1} ^ {n} \nabla_ {\theta} \tilde {f} (\theta_ {t}, \xi_ {t} ^ {(i)}), \quad \xi_ {t} ^ {(1)}, \ldots , \xi_ {t} ^ {(n)} \stackrel {\mathrm{iid}} {\sim} p (\xi),\tag{2}
$$

and $\alpha _ { t }$ is a scalar step size. We may use notational shorthands like $f _ { t } = f ( \theta _ { t } ) , \nabla f _ { t } = \nabla f ( \theta _ { t } )$ 

SGD is hampered by the effects of gradient noise. It famously needs a decreasing step size schedule to converge; used with a constant step size, it goes into diffusion in a region around the optimum (see, $e . g .$ ., Bottou et al., 2018). Gradient noise also makes stochastic optimization algorithms difficult to tune. In particular, unreliable directions are not amenable to step size adaptation. 

To stabilize update directions, we build a framework for estimating the true gradient $\nabla f$ based on Kalman filtering. This can also be viewed as a variance reduction method, but does not require the typical finite-sum structure assumption of $e . g .$ . Schmidt et al. (2017); Johnson and Zhang (2013). 

## 2.1 Dynamical System Model

We treat the true gradient $\nabla f _ { t }$ as the latent state of a dynamical system. This dynamical system is comprised of an observation model $p ( g _ { t } \mid f _ { t } )$ and a dynamics model $p ( \nabla f _ { t } \mid \dot { \nabla } f _ { t - 1 } , \delta _ { t - 1 } )$ where $\delta _ { t - 1 } = \theta _ { t } - \theta _ { t - 1 }$ is the update direction. We will later choose $\delta _ { t }$ to be depend on our variance-reduced gradient estimates, but the gradient inference framework itself is agnostic to the choice of $\delta _ { t }$ 

The observation model $p ( g _ { t } \mid \nabla f _ { t } )$ describes how the gradient observations relate to the state of the dynamical system. In our case, it is relatively straight-forward, since $g _ { t }$ is simply an unbiased stochastic estimate of $\nabla f _ { t }$ , but the exact distribution remains to be specified. We make the assumption that $g _ { t }$ follows a Gaussian distribution, 

$$
g _ {t} \mid \nabla f _ {t} \sim \mathcal {N} (\nabla f _ {t}, \Sigma_ {t}),\tag{3}
$$

with covariance $\Sigma _ { t }$ . Since $g _ { t }$ is the mean of iid terms (Eq. 2), this assumption is supported by the central limit theorem when sufficiently large batch sizes are used. 

The dynamics model $p ( \nabla f _ { t } \mid \nabla f _ { t - 1 } )$ describes how the gradient evolves between iterations. We base our dynamics model on a first order Taylor expansion of the gradient function centered at $\theta _ { \mathbf { t } } , \nabla f ( \theta _ { t - 1 } ) \approx \nabla f ( \theta _ { t } ) - \nabla ^ { 2 } f ( \theta _ { t } ) \delta _ { t - 1 }$ . We propose to approximate the gradient dynamics by computing a stochastic estimate of the Hessian-vector product, $B _ { t } \delta _ { t - 1 }$ , where $\mathbb { E } [ B _ { t } ] \stackrel { \cdot } { = } \nabla ^ { 2 } f ( \theta _ { t } )$ Again, we make a Gaussian noise assumption. This implies the dynamics model 

$$
\nabla f _ {t} \mid \nabla f _ {t - 1} \sim \mathcal {N} (\nabla f _ {t - 1} + B _ {t} \delta_ {t - 1}, Q _ {t}).\tag{4}
$$

where $Q _ { t }$ is the covariance of $B _ { t } \delta _ { t - 1 }$ , taking into account the stochasticity in $B _ { t }$ . 

A key insight is that the parameters $B _ { t } \delta _ { t - 1 } , Q _ { t } , \Sigma _ { t }$ of the model can all be “observed” directly using automatic differentiation of the loss on each minibatch of samples. We use the Hessian at $\theta _ { t }$ so that the Hessian-vector product can be simultaneously computed with $g _ { t }$ with just one extra call to automatic differentiation (or “backward pass”) in each iteration (note this does not require constructing the full matrix $B _ { t } )$ . The variances $Q _ { t }$ and $\Sigma _ { t }$ can also be empirically estimated with some memory overhead by using auto-vectorized automatic differentiation routines. We discuss implementation details later in Section 4. 

## 2.2 Filtering Framework for Gradient Inference

As Equations (3) and (4) define a linear-Gaussian dynamical system, exact inference on the true gradient conditioned on the history of gradient observations $p ( \nabla f _ { t } | g _ { 1 : t } , \delta _ { 1 : t - 1 } )$ takes the form of the well-known Kalman filtering equations (Kalman, 1960) (review in Särkkä, 2013): We define parameters $m _ { t } ^ { - } , m _ { t } , P _ { t } ^ { - }$ and $P _ { t }$ such that 

$$
\begin{array}{c} \nabla f _ {t} \mid g _ {1: t - 1}, \delta_ {1: t - 1} \sim \mathcal {N} (m _ {t} ^ {-}, P _ {t} ^ {-}) \\ \nabla f _ {t} \mid g _ {1: t}, \delta_ {1: t - 1} \sim \mathcal {N} (m _ {t}, P _ {t}). \end{array}\tag{5}
$$

Starting from a prior belief $\nabla f _ { 0 } \sim { \mathcal { N } } ( m _ { 0 } , P _ { 0 } )$ , these parameters are updated iteratively: 

$$
m _ {t} ^ {-} = m _ {t - 1} + B _ {t} \delta_ {t - 1}, \qquad P _ {t} ^ {-} = P _ {t - 1} + Q _ {t - 1}\tag{6}
$$

(7) 

$$
m _ {t} = (I - K _ {t}) m _ {t} ^ {-} + K _ {t} g _ {t}, \qquad P _ {t} = (I - K _ {t}) P _ {t} ^ {-} (I - K _ {t}) ^ {T} + K _ {t} \Sigma_ {t} K _ {t} ^ {T}\tag{8}
$$

Equation (6) is referred to as the prediction step as it computes mean and covariance of the predictive distribution $p ( \nabla f _ { t } | g _ { 1 : t - 1 } )$ . In our setting, it predicts the gradient $\nabla f _ { t }$ based on our estimate of the previous gradient $( m _ { t - 1 } )$ and the Hessian-vector product approximating the change in gradient from the step $\theta _ { t } = \theta _ { t - 1 } + \delta _ { t - 1 }$ . Equation (8) is the correction step. Here, the local stochastic gradient evaluation $g _ { t }$ is used to correct the prediction. Importantly, the Kalman gain (7) determines the blend between the prediction and the observations according to the uncertainty in each. 

The resulting algorithm gives an online estimation of the true gradients as the parameters $\theta _ { t }$ are updated. We refer to this framework as MEKA, loosely based on model-based Kalman-adjusted gradient estimation. During optimization, we may use the posterior mean $m _ { t }$ as a variance-reduced gradient estimator and take steps in the direction of $\delta _ { t } = - \alpha _ { t } m _ { t }$ 

We note two key insights enabling MEKA: First, all parameters of the filter are not set ad hoc, but are directly evaluated or estimated using automatic differentiation. Secondly, the dynamics model makes explicit use of the Hessian to predict gradients. This is a first-order update. In contrast to second-order methods, like quasi-Newton methods, MEKA does not try to estimate the Hessian from gradients, but instead leverages a (noisy) projection with the actual Hessian to improve gradient estimates. This is both cheaper and more robust than second-order methods, because it does not involve solving a linear system. 

## 2.3 ADAM-style Update Directions

While MEKA produces variance-reduced gradient estimates, it does not help with ill-conditioned optimization problems, a case where full batch gradient descent can perform poorly. To alleviate this, we may instead take update directions motivated by the ADAGRAD (Duchi et al., 2011) line of optimizers. We follow ADAM (Kingma and Ba, 2014) which proposes dividing the first moment of the gradient element-wise by the square root of the second moment, to arrive at 

$$
\delta_ {t} = - \alpha_ {t} \frac {m _ {t}}{\sqrt {m _ {t} + \mathrm{diag} (P _ {t})} + \varepsilon}.\tag{9}
$$

where $\varepsilon$ is taken for numerical stability and simply set to $1 0 ^ { - 8 }$ . Whereas ADAM makes use of two exponential moving averages to estimate the first and second moments of $g _ { t }$ , we have estimates automatically inferred through the filtering framework. We refer to this variant as ADAMEKA. 

## 3 Uncertainty-informed Step Size Selection

We can adopt a similar Bayesian filtering framework for probabilistic step size adaptation. Our step size adaptation will be a simple enhancement to the quadratic rule, but takes into account uncertainty in the stochastic regime and is much more robust to stochastic observations. The standard quadratic rule if the objective $f$ can be computed exactly is α<sub>quadratic</sub> $\begin{array} { r } { : = \frac { - \delta _ { t } ^ { T } \nabla f _ { t } } { \delta _ { t } ^ { T } \nabla ^ { 2 } f _ { t - 1 } \delta _ { t } } } \end{array}$ , which is based on minimizing a local quadratic approximation $\begin{array} { r } { f ( \theta _ { t } + \alpha _ { t } \delta _ { t } ) - f _ { t } \approx \alpha \delta _ { t } ^ { T } \nabla f _ { t } + \frac { \alpha ^ { 2 } } { 2 } \delta _ { t } ^ { T } \nabla ^ { 2 } f _ { t - 1 } \delta _ { t } . } \end{array}$ 

However, since we only have access to stochastic estimates of $\nabla f$ and $\nabla ^ { 2 } f ,$ naïvely taking this step size with high variance samples results in unpredictable behavior and can cause divergence during optimization. To compensate for the stochasticity and inaccuracy of a quadratic approximation, adaptive step size approaches often include a “damping” term $( e . g .$ . Martens (2010))—where a constant is added to the denominator—and an additional scaling factor on $\alpha _ { t }$ , both of which aim to avoid large steps but introduces more hyperparameters. 

As an alternative, we propose a scheme that uses the variance of the estimates to adapt the step size, only taking steps into regions where we are confident about minimizing the objective function. Once again leveraging the availability of $Q _ { t }$ and $\Sigma _ { t } .$ , our approach allows automatic trade-off between minimizing a local quadratic approximation and the uncertainty over large step sizes, foregoing manual tuning methods such as damping. 

We adopt a similar linear-Gaussian dynamics model for tracking the true objective $f _ { t } ,$ , with the same assumptions as in Section 2. Due to its similarity with Section 2, we delegate the derivations to Appendix B. We again define the posterior distribution, 

$$
f _ {t} \mid y _ {1: t}, \delta_ {1: t - 1} \sim \mathcal {N} (u _ {t}, s _ {t}).\tag{10}
$$

where $u _ { t }$ and $s _ { t }$ are inferred using the Kalman update equations. Finally, setting $f _ { t + 1 } = f ( \theta _ { t } + \alpha _ { t } \delta _ { t } )$ for some direction $\delta _ { t } .$ , we have a predictive model of the change in function value as 

$$
f _ {t + 1} - f _ {t} \mid y _ {1: t}, g _ {1: t}, \delta_ {1: t} \sim \mathcal {N} \bigg (\alpha_ {t} \delta_ {t} ^ {T} m _ {t} + \frac {\alpha_ {t} ^ {2}}{2} \delta_ {t} ^ {T} B _ {t} \delta_ {t}, 2 s _ {t} + \alpha_ {t} ^ {2} \delta_ {t} ^ {T} P _ {t} \delta_ {t} + \frac {\alpha_ {t} ^ {4}}{4} \delta_ {t} ^ {T} Q _ {t} \delta_ {t} \bigg)\tag{11}
$$

Contrasting this with the simple quadratic approximation, the main difference is now we take into account the uncertainty in $f _ { t } , \bar { \nabla } f _ { t }$ , and $\nabla ^ { 2 } f _ { t }$ . Each term makes different contributions to the variance as $\alpha _ { t }$ increases, corresponding to different trade-offs between staying near where we are more certain about the function value and exploring regions we believe have a lower function value. Explicitly specifying this trade-off gives an acquisition function. These decision rules are typically used in the context of Bayesian optimization (Shahriari et al., 2016), but we adopt their use for step size selection. 

## 3.1 Acquisition Functions for Step Size Selection

Computing the optimal step size in the context of a long but finite sequence of optimization steps is intractable in general, but many reasonable heuristics have been developed. These heuristics usually balance immediate progress against information gathering likely to be useful for later steps. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-27/e5a1e9f0-0ad2-40f9-bc9e-43d13f521b11/f091de8d189185c2a8fc01c52deb436b17680a88668191b54d11a8587e53e3d5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-27/e5a1e9f0-0ad2-40f9-bc9e-43d13f521b11/fb11efbc23905d2539cfea1e91ed4cfdd3e5cd98972aeca4cf1fa8bf1e924958.jpg)



(a) Positive curvature



(b) Negative curvature



Figure 3: Illustration of different acquisition functions for selecting a step size $\alpha ,$ based on the mean and variance of our local quadratic estimate of the loss surface.


One natural and hyperparameter-free heuristic is maximizing the probability of improvement (PI) (Kushner, 1964), 

$$
\alpha_ {\mathrm{PI}} := \underset {\alpha} {\arg \max} \mathbb {P} \left(f _ {t + 1} - f _ {t} \leq 0 \mid y _ {1: t}, g _ {1: t}\right)\tag{12}
$$

which is simply the cumulative distribution function of (11) evaluated at zero. 

Figure 3 visualizes the different step sizes chosen by maximizing different acquisition functions. The heuristic of choosing the minimum of the quadratic approximation can be a poor decision when the uncertainty rises quickly. The optimum for PI interpolates between zero and the quadratic minimum in such a way that avoids regions of high uncertainty. Expected improvement (Jones et al., 1998) is another popular acquisition function; however, in tests we found it to not be as robust as PI and often results in step sizes that require additional scaling. 

Maximizing probability of improvement is equivalent to the following optimization problem 

$$
\alpha_ {\mathrm{PI}} = \underset {\alpha} {\arg \min} \frac {- \alpha \delta_ {t} ^ {T} m _ {t} + \frac {\alpha^ {2}}{2} \delta_ {t} ^ {T} B _ {t} \delta_ {t}}{\sqrt {2 s _ {t} + \alpha^ {2} \delta_ {t} ^ {T} P _ {t} \delta_ {t} + \frac {\alpha^ {4}}{4} \delta_ {t} ^ {T} Q _ {t} \delta_ {t}}}.\tag{13}
$$

We numerically solve for $\alpha _ { \mathrm { { P I } } }$ using Newton’s method, which itself is a very small overhead since we only optimize in one variable with fixed constants: no further evaluations of $f$ are required. For optimization problems where negative curvature is a significant concern, we include a third-order correction term that ensures finite and positive step sizes (details in Appendix B.2). 

## 4 A Practical Implementation

While the above derivations have principled motivations and are free of hyperparameters, a practical implementation of MEKA is not entirely straightforward. Below we discuss some technical aspects, simplifications and design choices that increase stability in practice, as well as recent software advances that simplify the computation of quantities of interest. 

Computing Per-Example Quantities for Estimating Variance Recent extensions for automatic differentiation in the machine learning software stack (Bradbury et al., 2018; Agarwal and Ganichev, 2019; Dangel et al., 2020) implement an automatic vectorization map function. Vectorizing over minibatch elements allows efficient computation of gradients and Hessian-vector products of neural network parameters with respect to each data sample independently. These advances allow efficient computation of the empirical variances of gradients and Hessian-vector products, and enable our filtering-based approach to gradient estimation. 

Stabilizing Filter Estimates Instead of working with the full covariance matrices $\Sigma _ { t }$ and $Q _ { t }$ , we approximate them as scalar objects $\sigma _ { t } I$ and $q _ { t } I ,$ with $\sigma _ { t } , q _ { t } \in \mathbb { R } .$ <sub>+</sub> by averaging over all dimensions. We have experimented with diagonal matrices, but found that the scalar form increases stability, generally performing better on our benchmarks. Furthermore, we use an exponential moving average for smoothing the estimated gradient variance $\sigma _ { t }$ as well as the adaptive step sizes $\alpha _ { t }$ . The coefficients of these exponential moving average are kept at 0.999 in our experiments and seem to be quite insensitive, with values in {0.9, 0.99, 0.999} all performing near identically (see Appendix F). 

## 5 Related Work

Designing algorithms that can self-tune its own parameters is a central theme in optimization (Eiben and Smit, 2011; Yang et al., 2013); we focus on the stochastic setting, building on and merging ideas from several research directions. The Bayesian filtering framework itself has previously been applied to stochastic optimization. To the best of our knowledge, the idea goes back to Bittner and Pronzato (2004) who used a filtering approach to devise an automatic stopping criterion for stochastic gradient methods. Patel (2016) proposed filtering-based optimization methods for large-scale linear regression problems. Vuckovic (2018) and Mahsereci (2018) used Kalman filters on general stochastic optimization problems with the goal of reducing the variance of gradient estimates. In contrast to our work, none of these existing approaches leverage evaluations of Hessian-vector products to give curvature-informed dynamics for the gradient. 

In terms of online variance reduction, Gower et al. (2017) have discussed the use of Hessian-vector products to correct the gradient estimate; however, they propose methods that approximate the Hessian whereas we compute exact Hessian-vector products by automatic differentiation. Arnold et al. (2019a) recently proposed an implicit gradient transport formula analogous to our dynamics model, but they require a rather strong assumption that the Hessian is the same for all samples and parameter values. In contrast, we focus on explicitly transporting via the full Hessian. This allows us to stay within the filtering framework and automatically infer the gain parameter, whereas the implicit formulation of Arnold et al. (2019a) requires the use of a manually-tuned averaging schedule. 

Step size selection under noisy observations is a difficult problem and has been tackled from multiple viewpoints. Methods include meta-learning approaches (Almeida et al., 1999; Schraudolph, 1999; Plagianakos et al., 2001; Yu et al., 2006; Baydin et al., 2017) or by assuming the interpolation regime (Vaswani et al., 2019; Berrada et al., 2019). Rolinek and Martius (2018) proposed extending a linear approximation to adapt step sizes but introduces multiple hyperparameters to adjust for the presence of noise, whereas we extend a quadratic approximation and automatically infer parameters based on noise estimates. Taking into account observation noise, Mahsereci and Hennig (2017) proposed a probabilistic line search that is done by fitting a Gaussian process to the optimization landscape. However, inference in Gaussian processes is more costly than our filtering approach. 

## 6 Convergence in the Noisy Quadratic Setting

As a motivating example, consider a simple toy problem, where 

$$
f (\theta , \xi) = \frac {1}{2} (\theta - \xi) ^ {T} H (\theta - \xi),\tag{14}
$$

i.e., a mixture of quadratic functions with identical Hessian but varying location determined by the $\vec { \bf \Phi } ^ { 6 } \mathrm { d a t a } ^ { 3 } \ \xi .$ The full gradient is $\nabla f ( \theta ) ~ = ~ H ( \theta - \mathbf { \bar { \mathbb { E } } } [ \xi ] )$ and perexample gradients evaluate to $\nabla f ( { \boldsymbol { \theta } } , { \boldsymbol { \xi } } ) = \mathbf { \bar { \cal H } } ( { \boldsymbol { \theta } } - { \boldsymbol { \xi } } ) =$ $\nabla f ( { \dot { \theta } } ) - \mathbf { \bar { \theta } } H ( \xi - \mathbb { E } [ \xi ] )$ . Hence, we have additive gradient noise with covariance $\begin{array} { r } { \Sigma = H \mathbf { C o v } | \xi | H ^ { T } } \end{array}$ independent of θ. Moreover, since the Hessian $\nabla ^ { 2 } { \bar { f } } ( \theta , \xi ) = { \bar { \cal H } }$ is independent of $\xi ,$ we have that $B _ { t } \delta _ { t - 1 } \equiv \nabla f _ { t } - \nabla f _ { t - 1 }$ . The covariance $Q _ { t }$ is zero and the filter equations simplify to 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-27/e5a1e9f0-0ad2-40f9-bc9e-43d13f521b11/351eb281202ca01548f463195be341455e014fe14853c2db986721422a392787.jpg)


$$
\begin{array}{l} K _ {t} = P _ {t - 1} (P _ {t - 1} + \Sigma) ^ {- 1}, \\ m _ {t} = (I - K _ {t}) (m _ {t - 1} + B _ {t} \delta_ {t - 1}) + K _ {t} g _ {t}, \\ P _ {t} = (I - K _ {t}) P _ {t - 1}, \end{array}\tag{15}
$$


Figure 4: Filtered gradients converge with a fixed step size in the noisy quadratic regime, whereas SGD results in diffusion for the same step size.


initialized with $m _ { 0 } = g _ { 0 } , P _ { 0 } = \Sigma$ . The filter covariance $P _ { t }$ contracts in every step and in fact, shrinks at a rate of $O ( 1 / t )$ , meaning that the filter will narrow in on the exact gradient. We show that this enables $O ( 1 / t )$ convergence with a constant step size. 

Proposition 1. Assume a problem of the form (14) with $\mu I \preceq H \preceq L I .$ . If we update $\theta _ { t + 1 } = \theta _ { t } - \alpha m _ { t }$ with $\alpha \leq 1 / L$ and $m _ { t }$ obtained via Eq. (15), then $\mathbb { E } [ f ( \theta _ { t } ) - f _ { * } ] \in O \left( 1 / t \right)$ 

Figure 4 shows experimental results for such a noisy quadratic problem of dimension $d = 2 0$ with a randomly-generated Hessian (with condition number $> 1 0 0 0 )$ and $\xi \sim \mathcal { N } ( 0 , I )$ . Using SGD with a high learning rate simply results in diffusion, and setting the learning rate smaller results in slow convergence. Gradient descent (GD) converges nicely with the high learning rate, and using adaptive steps sizes leads to a better convergence rate. The filtered gradients from MEKA converge almost as well as gradient descent, and adaptive step sizes provide an improvement. On the other hand, SGD produces unreliable gradient directions and does not work well with adaptive step sizes. We note that the stochastic gradient has a full covariance matrix and does not match our modeling assumptions, as our model uses a diagonal covariance for efficiency. Even so, the training loss of MEKA follows that of gradient descent very closely after just a few iterations. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-27/e5a1e9f0-0ad2-40f9-bc9e-43d13f521b11/f634422e2ebf7ca490c8f1d8eae62228cfc9b301716d185d10a716edfd1f4e5b.jpg)



Figure 6: Adaptive step sizes based on probability of improvement work best without any additional scaling factor c for modifying the update rule: $\theta _ { t + 1 } = \theta _ { t } + c \alpha _ { t } \delta _ { t }$


## 7 Classification Experiments

Next we test and diagnose our approach on classification benchmarks, MNIST and CIFAR-10. We use JAX’s (Bradbury et al., 2018) vectorized map functionality for efficient per-example gradients and Hessian-vector products. For MNIST, we test using a multi-layer perceptron (MLP); for CIFAR-10, a convolutional neural network (CNN) and a residual network (ResNet-32) (He et al., 2016a,b). One key distinction is we replace the batch normalization layers with group normalization (Wu and He, 2018) as batch-dependent transformations conflict with our assumption that the gradient samples are independent. We note that the empirical per-iteration cost of MEKA is 1.0–1.6× that of SGD due to the computation of Hessian-vector products. Full experiment details are provided in Appendix E. A detailed comparison to tuned baseline optimizers is presented in Appendix D.1. 

Online Variance Reduction We test whether the filtering procedure is correctly aligning the gradient estimate with the true gradient. For this, we use CIFAR-10 with a CNN and no data augmentation, so that the true full-batch gradient over the entire dataset can be computed. Figure 5 shows the $L _ { 2 }$ norm difference between the gradient estimators and the full-batch gradient $\nabla f _ { t }$ . MEKA’s estimated gradients are closer to the true around by around a factor of 5 compared to the minibatch gradient sample. 

Adaptive Step Sizes are Appropriately Scaled Without uncertainty quantification, the quadratic minimum step size scheme tends to result in step sizes too large. As such, one may include a scaling factor such that the update is modified as $\theta _ { t + 1 } = \theta _ { t } - c \alpha _ { t } \delta _ { t }$ . In contrast, we find that the adaptive step sizes based on probability of improvement (PI) are already correctly scaled in the sense that a c different from 1.0 will generally result in worse performance. Figure 6 shows a comparison of different values for c for the quadratic and PI (12) adaptive schemes. We plot expected improvement in Appendix D, which performs poorly and requires non-unit scaling factors. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-27/e5a1e9f0-0ad2-40f9-bc9e-43d13f521b11/fe5d9fbe83e8edcc9ed028412c64e35fc187927ef231145125e5bb6478e1198d.jpg)



Figure 5: MEKA’s estimated gradients are closer to the true full-batch gradient in $L _ { 2 }$ norm than stochastically observed gradients by a factor of around 5.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-27/e5a1e9f0-0ad2-40f9-bc9e-43d13f521b11/d762469f3adf46d4ffc55bbb9f17ad4036706c4fa6a01236539bd3cbc48e5f59.jpg)



Figure 7: The performance of MEKA with adaptive step sizes on ResNet-32 can be explained by quantities captured during optimization. MEKA reaches high-curvature high-variance local minima, as soon as adaptive step sizes are used.


## 7.1 Adaptive Step Sizes Dives into High-curvature, High-variance Regions

A core aspect of our filtering approach is the ability to estimate quantities of interest during optimization. We now use these to help understand the loss landscape and training dynamics of ResNet-32 on CIFAR-10. We find that a cause for slow convergence of MEKA with adaptive step sizes is due to an abundance of minima that are usually too high variance for standard SGD. 

Figure 7 shows estimates of the normalized curvature along the descent direction $\frac { \delta ^ { T } B _ { t } \delta } { \delta ^ { T } \delta }$ as well as the per-sample gradient variance, averaged over parameters. To understand the loss landscape along the trajectory of optimization, we use multiple runs of MEKA with the same initialization. Each run takes a different fixed number of constant-size steps before switching to the adaptive step size scheme. 

It is clear that immediately after switching to adaptive step sizes, MEKA falls into an increasingly high curvature region and remains there. The gradient variance also remains high. As our optimization procedure can handle relatively high variance and curvature, it proceeds to optimize within this sharp but potentally non-local minimum. On the other hand, it may be an advantage of fixed-step-size SGD that it skips over both high-variance and high-curvature minima. 

This failing of adaptive step sizes during the initial phase of training may be related to the “short horizon bias” (Wu et al., 2018) of our one-step-ahead acquisition function. If so, compute budget can be used to approximate multi-step-ahead gains to help reduce this bias. Additionally, the ability to optimize within high-curvature high-variance regions could potentially be an advantage on problems with fewer local minima, yet this may not be the case for deep learning. 

## 8 Conclusion

We introduced an online gradient estimation framework for stochastic gradient-based optimization, which leverages Hessian-vector products and variance estimates to perform automatic online gradient estimation and step size selection. The result is a stochastic optimization algorithm that can self-tune many important parameters such as momentum and learning rate schedules, in an online fashion without checkpointing or expensive outer-loop optimization. 

While the required additional observables can be computed efficiently with recent advances in automatic differentiation tooling, they are of course not free, increasing computational cost and memory usage compared to SGD. What one gains in return is automation, so that it suffices to run the algorithm just once, without tedious tuning. Given the amount of human effort and computational resources currently invested into hyperparameter tuning, we believe our contributions are valuable steps towards fully-automated gradient-based optimization. 

## References



Ashish Agarwal and Igor Ganichev. Auto-vectorizing TensorFlow graphs: Jacobians, auto-batching and beyond. arXiv preprint arXiv:1903.04243, 2019. 





Luís B Almeida, Thibault Langlois, José D Amaral, and Alexander Plakhov. Parameter adaptation in stochastic optimization. In On-line learning in neural networks, pages 111–134. 1999. 





Shun-Ichi Amari. Natural gradient works efficiently in learning. Neural computation, 10(2):251–276, 1998. 





Sébastien Arnold, Pierre-Antoine Manzagol, Reza Babanezhad Harikandeh, Ioannis Mitliagkas, and Nicolas Le Roux. Reducing the variance in online optimization by transporting past gradients. In Advances in Neural Information Processing Systems 32. 2019a. 





Sébastien Arnold, Pierre-Antoine Manzagol, Reza Babanezhad Harikandeh, Ioannis Mitliagkas, and Nicolas Le Roux. Reducing the variance in online optimization by transporting past gradients. In Advances in Neural Information Processing Systems, pages 5391–5402, 2019b. 





Atilim Gunes Baydin, Robert Cornish, David Martinez Rubio, Mark Schmidt, and Frank Wood. Online learning rate adaptation with hypergradient descent. arXiv preprint arXiv:1703.04782, 2017. 





Leonard Berrada, Andrew Zisserman, and M Pawan Kumar. Training neural networks for and by interpolation. arXiv preprint arXiv:1906.05661, 2019. 





Barbara Bittner and Luc Pronzato. Kalman filtering in stochastic gradient algorithms: construction of a stopping rule. In 2004 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2004. 





Léon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. Siam Review, 60(2):223–311, 2018. 





James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, and Skye Wanderman-Milne. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/google/jax. 





Dami Choi, Christopher J Shallue, Zachary Nado, Jaehoon Lee, Chris J Maddison, and George E Dahl. On empirical comparisons of optimizers for deep learning. arXiv preprint arXiv:1910.05446, 2019. 





Felix Dangel, Frederik Kunstner, and Philipp Hennig. BackPACK: Packing more into backprop. In International Conference on Learning Representations, 2020. 





John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 2011. 





Agoston E Eiben and Selmar K Smit. Parameter tuning for configuring and analyzing evolutionary algorithms. Swarm and Evolutionary Computation, 1(1):19–31, 2011. 





Robert M Gower, Nicolas Le Roux, and Francis Bach. Tracking the gradients using the hessian: A new look at variance reducing stochastic methods. arXiv preprint arXiv:1710.07462, 2017. 





Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016a. 





Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In European conference on computer vision, pages 630–645. Springer, 2016b. 





Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In Advances in neural information processing systems, pages 315–323, 2013. 





Donald R Jones, Matthias Schonlau, and William J Welch. Efficient global optimization of expensive black-box functions. Journal of Global optimization, 1998. 





Rudolph Emil Kalman. A new approach to linear filtering and prediction problems. Transactions of the ASME–Journal of Basic Engineering, 82(Series D):35–45, 1960. 





Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 





Harold J Kushner. A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise. 1964. 





Maren Mahsereci. Probabilistic Approaches to Stochastic Optimization. PhD thesis, Eberhard Karls Universität Tübingen Tübingen, 2018. 





Maren Mahsereci and Philipp Hennig. Probabilistic line searches for stochastic optimization. The Journal of Machine Learning Research, 2017. 





James Martens. Deep learning via Hessian-free optimization. In Proceedings of the 27th International Conference on International Conference on Machine Learning, 2010. 





James Martens. New insights and perspectives on the natural gradient method. arXiv preprint arXiv:1412.1193, 2014. 





Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th international conference on machine learning (ICML-10), pages 807–814, 2010. 





Vivak Patel. Kalman-based stochastic gradient method with stop condition and insensitivity to conditioning. SIAM Journal on Optimization, 2016. 





VP Plagianakos, GD Magoulas, and MN Vrahatis. Learning rate adaptation in stochastic gradient descent. In Advances in convex analysis and global optimization, pages 433–444. Springer, 2001. 





Boris T Polyak. Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 1964. 





Michal Rolinek and Georg Martius. L4: Practical loss-based stepsize adaptation for deep learning. In Advances in Neural Information Processing Systems, pages 6433–6443, 2018. 





Simo Särkkä. Bayesian filtering and smoothing, volume 3. Cambridge University Press, 2013. 





Mark Schmidt, Nicolas Le Roux, and Francis Bach. Minimizing finite sums with the stochastic average gradient. Mathematical Programming, 162(1-2):83–112, 2017. 





Frank Schneider, Lukas Balles, and Philipp Hennig. Deepobs: A deep learning optimizer benchmark suite. arXiv preprint arXiv:1903.05499, 2019. 





Nicol N Schraudolph. Local gain adaptation in stochastic gradient descent. 1999. 





B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and N. de Freitas. Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE, 2016. 





Sharan Vaswani, Aaron Mishkin, Issam Laradji, Mark Schmidt, Gauthier Gidel, and Simon Lacoste Julien. Painless stochastic gradient: Interpolation, line-search, and convergence rates. In Advances in Neural Information Processing Systems 32. 2019. 





James Vuckovic. Kalman gradient descent: Adaptive variance reduction in stochastic optimization, 2018. 





Yuhuai Wu, Mengye Ren, Renjie Liao, and Roger Grosse. Understanding short-horizon bias in stochastic meta-optimization. arXiv preprint arXiv:1803.02021, 2018. 





Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European Conference on Computer Vision (ECCV), 2018. 





Xin-She Yang, Suash Deb, Martin Loomes, and Mehmet Karamanoglu. A framework for self-tuning optimization algorithm. Neural Computing and Applications, 23(7-8):2051–2057, 2013. 





Tjalling J Ypma. Historical development of the newton–raphson method. SIAM review, 37(4): 531–551, 1995. 





Jin Yu, Douglas Aberdeen, and Nicol N Schraudolph. Fast online policy gradient learning with smd gain vector adaptation. In Advances in neural information processing systems, pages 1185–1192, 2006. 





Matthew D Zeiler. Adadelta: an adaptive learning rate method. arXiv preprint arXiv:1212.5701, 2012. 

