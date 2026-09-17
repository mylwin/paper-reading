# Parameter-free Clipped Gradient Descent Meets Polyak

Yuki Takezawa<sup>1,2</sup>, Han Bao<sup>1,2</sup>, Ryoma Sato<sup>3</sup>, Kenta Niwa<sup>4</sup>, Makoto Yamada<sup>2</sup> <sup>1</sup>Kyoto University, <sup>2</sup>OIST, <sup>3</sup>NII, <sup>4</sup>NTT Communication Science Laboratories 

## Abstract

Gradient descent and its variants are de facto standard algorithms for training machine learning models. As gradient descent is sensitive to its hyperparameters, we need to tune the hyperparameters carefully using a grid search. However, the method is time-consuming, particularly when multiple hyperparameters exist. Therefore, recent studies have analyzed parameter-free methods that adjust the hyperparameters on the fly. However, the existing work is limited to investigations of parameter-free methods for the stepsize, and parameter-free methods for other hyperparameters have not been explored. For instance, although the gradient clipping threshold is a crucial hyperparameter in addition to the stepsize for preventing gradient explosion issues, none of the existing studies have investigated parameterfree methods for clipped gradient descent. Therefore, in this study, we investigate the parameter-free methods for clipped gradient descent. Specifically, we propose Inexact Polyak Stepsize, which converges to the optimal solution without any hyperparameters tuning, and its convergence rate is asymptotically independent of L under L-smooth and $( L _ { 0 } , L _ { 1 } )$ -smooth assumptions of the loss function, similar to that of clipped gradient descent with well-tuned hyperparameters. We numerically validated our convergence results using a synthetic function and demonstrated the effectiveness of our proposed methods using LSTM, Nano-GPT, and T5. 

## 1 Introduction

We consider the convex optimization problem: 

$$
\min _ {\boldsymbol {x} \in \mathbb {R} ^ {d}} f (\boldsymbol {x}),\tag{1}
$$

where the loss function f is convex and lower bounded. In this setting, gradient descent and its variants (Duchi et al., 2011; Kingma and Ba, 2015) are the de facto standard algorithms to minimize the loss function. The performance of the algorithm is highly sensitive to the hyperparameter settings, necessitating the careful tuning of the hyperparameters to achieve best performance. More specifically, when the loss function is L-smooth, gradient descent can achieve the optimal convergence rate $O ( \frac { L \| \pmb { x } _ { 0 } - \pmb { x } ^ { \star } \| ^ { 2 } } { T } )$ when we set the stepsize to $\frac { 1 } { L }$ where $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ is the initial parameter and $\mathbf { \Delta } \mathbf { x } ^ { \star }$ is the optimal solution (Nesterov, 2018). Unfortunately, parameter $L$ is problem-specific and unavailable in practice. Thus, gradient descent must be executed in many times with different hyperparameters to identify the good hyperparameter settings, which is a very time-consuming process. Notably, when multiple hyperparameters are under consideration, this hyperparameter search becomes computationally more demanding. 

Several recent studies have examined parameter-free methods for tuning hyperparameters on the fly (Berrada et al., 2020; Defazio and Mishchenko, 2023; Orvieto et al., 2022; Jiang and Stich, 2023; Ivgi et al., 2023; Khaled et al., 2023; Orabona and Tommasi, 2017; Carmon and Hinder, 2022).<sup>1</sup> 

These methods automatically adjust the stepsize during the training and are guaranteed to converge to the optimal solution without tuning the stepsize. In other words, the stepsize did not require tuning using the grid search. However, the existing parameter-free methods only focus on the stepsize, and parameter-free methods for other hyperparameters have not been explored. For example, in addition to the stepsize, the gradient clipping threshold is an important hyperparameter for training language models (Pascanu et al., 2013; Zhang et al., 2020a,b,c). 

Clipped gradient descent can achieve the convergence rate $\begin{array} { r } { \mathcal { O } ( \frac { L _ { 0 } \| \pmb { x } _ { 0 } - \pmb { x } ^ { \star } \| ^ { 2 } } { T } ) } \end{array}$ under the assumption that the loss function is $( L _ { 0 } , L _ { 1 } )$ -smooth when we use the optimal stepsize and gradient clipping threshold (Koloskova et al., 2023). In many cases, $L _ { 0 }$ is significantly smaller than L (Zhang et al., 2020b). Thus, by comparing with the convergence rate of gradient descent $O ( \frac { L \| \pmb { x } _ { 0 } - \pmb { x } ^ { \star } \| ^ { 2 } } { T } )$ , gradient clipping often allows gradient descent to converge faster. However, we must carefully tune two hyperparameters, stepsize and gradient clipping threshold, to achieve this convergence rate. If the gradient clipping threshold is too large, the gradient clipping fails to accelerate the convergence. Moreover, if the gradient clipping threshold is too small, gradient clipping deteriorates rather than accelerating the convergence rate. Can we develop a parameter-free method whose convergence rate is asymptotically independent ofL under $( L _ { 0 } , L _ { 1 } )$ -smoothness? 

In this study, we investigate a parameter-free method for clipped gradient descent. First, we provide the better convergence rate of Polyak stepsize (Polyak, 1987) under $( L _ { 0 } , L _ { 1 } )$ -smoothness. We discover that the convergence rate of Polyak stepsize matches that of clipped gradient descent with well-tuned stepsize and gradient clipping threshold. Although the convergence rate of Polyak stepsize is asymptotically independent of $L$ under $( L _ { 0 } , L _ { 1 } )$ -smooth assumption as clipped gradient descent, it still requires the minimum loss value, which is a problem-specific value. Thus, we make Polyak stepsize parameter-free without losing this property under $( L _ { 0 } , \bar { L } _ { 1 } )$ )-smoothness by proposing Inexact Polyak Stepsize, which converges to the optimal solution without any problem-specific parameters. We numerically evaluated Inexact Polyak Stepsize using a synthetic function and neural networks, validating our theory and demonstrating the effectiveness of Inexact Polyak Stepsize. 

## 2 Preliminary

## 2.1 Gradient descent & L-smoothness

One of the most fundamental algorithms for solving Eq. (1) represents the gradient descent: 

$$
\boldsymbol {x} _ {t + 1} = \boldsymbol {x} _ {t} - \eta_ {t} \nabla f (\boldsymbol {x} _ {t}),
$$

where $\pmb { x } _ { 0 } \in \mathbb { R } ^ { d }$ is the initial parameter and $\eta _ { t } > 0$ is the stepsize at t-th iteration. To ensure that gradient descent converges to the optimal solution quickly, we must carefully tune the stepsize $\eta _ { t }$ When the stepsize is too large, the training collapses. By contrast, when the stepsize is too small, the convergence rate becomes too slow. Thus, we must search for a proper stepsize as the following theorem indicates. 

Assumption 1 (L-smoothness). There exists a constant $L > 0$ that satisfies the following for all $\pmb { x } , \pmb { y } \in \mathbb { \bar { R } } ^ { d } .$ 

$$
\| \nabla f (\boldsymbol {x}) - \nabla f (\boldsymbol {y}) \| \leq L \| \boldsymbol {x} - \boldsymbol {y} \|.\tag{2}
$$

Theorem 1 (Nesterov (2018, Corollary 2.1.2)). Assume that $f$ is convex and L-smooth, and there exists an optimal solution $\begin{array} { r } { \pmb { x } ^ { \star } : = \arg \operatorname* { m i n } _ { \pmb { x } \in \mathbb { R } ^ { d } } f ( \pmb { x } ) } \end{array}$ ). Then, gradient descent with stepsize $\begin{array} { r } { \eta _ { t } = \frac { 1 } { L } } \end{array}$ satisfies 

$$
f (\bar {\boldsymbol {x}}) - f (\boldsymbol {x} ^ {\star}) \leq \mathcal {O} \left(\frac {L \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T}\right),\tag{3}
$$

where $\begin{array} { r } { \bar { \pmb { x } } : = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \pmb { x } _ { t } } \end{array}$ and T is the number ofiterations. 

## 2.2 Clipped gradient descent & $( L _ { 0 } , L _ { 1 } )$ -smoothness

Gradient clipping is widely used to stabilize and accelerate the training of gradient descent (Pascanu et al., 2013; Devlin et al., 2019). Let $c > 0$ be the threshold for gradient clipping. Clipped gradient descent is given by: 

$$
\boldsymbol {x} _ {t + 1} = \boldsymbol {x} _ {t} - \eta_ {t} \min \left\{1, \frac {c}{\| \nabla f (\boldsymbol {x} _ {t}) \|} \right\} \nabla f (\boldsymbol {x} _ {t}).\tag{4}
$$

Many prior studies investigated the theoretical benefits of gradient clipping (Koloskova et al., 2023; Zhang et al., 2020a,b,c; Li and Liu, 2022; Sadiev et al., 2023). Zhang et al. (2020b) experimentally found that the gradient Lipschitz constant decreases during the training of various neural networks and is highly correlated with gradient norm $\| \nabla f ( \pmb { x } ) \|$ . To describe this phenomenon, Zhang et al. (2020a) introduced a novel smoothness assumption called $( L _ { 0 } , L _ { 1 } )$ )-smoothness. Then, it has been experimentally demonstrated that the local gradient Lipschitz constant $L _ { 0 }$ is thousands of times smaller than the global gradient Lipschitz constant L. 

Assumption $\pmb { 2 } ( ( L _ { 0 } , L _ { 1 } )$ -smoothness). There exists constants $L _ { 0 } > 0$ and $L _ { 1 } > 0$ that satisfy the following for all $\begin{array} { r } { \pmb { x } , \pmb { y } \in \mathbb { R } ^ { d } w i t h \| \pmb { x } - \pmb { y } \| \leq \frac { 1 } { L _ { 1 } } , } \end{array}$ : 

$$
\left\| \nabla f (\boldsymbol {x}) - \nabla f (\boldsymbol {y}) \right\| \leq \left(L _ {0} + L _ {1} \| \nabla f (\boldsymbol {x}) \|\right) \| \boldsymbol {x} - \boldsymbol {y} \|.\tag{5}
$$

Note that $( L _ { 0 } , L _ { 1 } )$ )-smoothness is strictly weaker than L-smoothness because $( L _ { 0 } , L _ { 1 } )$ -smoothness covers L-smoothness by taking $L _ { 1 } = 0$ . Using the $( L _ { 0 } , L _ { 1 } )$ )-smoothness assumption, the convergence rate of clipped gradient descent was established as follows. 

Theorem 2 (Koloskova et al. (2023, Theorem 2.3)). Assume that f is convex, L-smooth, and $( L _ { 0 } , L _ { 1 } )$ -smooth, and there exists an optimal solution $\begin{array} { r } { \pmb { x } ^ { \star } : = \arg \operatorname* { m i n } _ { \pmb { x } \in \mathbb { R } ^ { d } } f ( \pmb { x } ) } \end{array}$ . Then, clipped gradient descent with $\begin{array} { r } { \eta _ { t } = \frac { 1 } { L _ { 0 } } } \end{array}$ and $\begin{array} { r } { c = \frac { L _ { 0 } } { L _ { 1 } } } \end{array}$ satisfies: 

$$
f (\bar {\boldsymbol {x}}) - f (\boldsymbol {x} ^ {\star}) \leq \mathcal {O} \left(\frac {L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T} + \frac {L L _ {1} ^ {2} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {4}}{T ^ {2}}\right),\tag{6}
$$

where $\begin{array} { r } { \bar { \pmb { x } } : = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \pmb { x } _ { t } } \end{array}$ and T is the number of iterations. 

When the number of iterations $T$ is large, the first term $\begin{array} { r } { \mathcal { O } ( \frac { L _ { 0 } \| \pmb { x } _ { 0 } - \pmb { x } ^ { \star } \| ^ { 2 } } { T } ) } \end{array}$ becomes dominant, and the convergence rate of clipped gradient descent is asymptotically independent of L. Gradient clipping allows for the use of a larger stepsize, and thus, gradient descent converges faster because of $L _ { 0 } \ll L$ . We can interpret $\begin{array} { r } { L ^ { ' } \simeq L _ { 0 } ^ { - } + L _ { 1 } \operatorname* { s u p } _ { \pmb { x } } \| \nabla \bar { f } ( \pmb { x } ) \| } \end{array}$ . The stepsize of gradient descent in Theorem 1 is $\frac { 1 } { L _ { 0 } + L _ { 1 } \operatorname* { s u p } _ { \mathbf { x } } \| \nabla f ( \mathbf { x } ) \| }$ , which is typically very small. By comparing with gradient descent, the coefficient multiplied by the gradient of clipped gradient descent in Theorem 2 is min $\Big \{ \frac { 1 } { L _ { 0 } } , \frac { 1 } { L _ { 1 } \| \nabla f ( { \pmb x } _ { t } ) \| } \Big \}$ , which is larger than $\frac { 1 } { L _ { 0 } + L _ { 1 } \operatorname* { s u p } _ { \mathbf { x } } \parallel \nabla f ( \mathbf { x } ) \parallel }$ . Specifically, when parameter x is close to the optimal solution $\pmb { x } ^ { \star } \left( \mathrm { i . e . , } \| \nabla f ( \pmb { x } ) \| \right.$ is small), clipped gradient descent can use a larger stepsize and then reach the optimal solution faster than gradient descent. 

## 2.3 Polyak stepsize

When f is convex, ${ \pmb x } _ { t + 1 }$ and $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ generated by gradient descent satisfy $\| { \pmb x } _ { t + 1 } - { \pmb x } ^ { \star } \| ^ { 2 } \leq \| { \pmb x } _ { t } - { \pmb x } ^ { \star } \| ^ { 2 } -$ $2 \eta _ { t } ( f ( \pmb { x } _ { t } ) - f ( \pmb { x } ^ { \star } ) ) + \eta _ { t } ^ { 2 } \| \nabla f ( \overline { { \pmb { x } } } _ { t } ) \| ^ { 2 }$ . By minimizing the right-hand side, we can derive well-known Polyak stepsize (Polyak, 1987): 

$$
\eta_ {t} = \frac {f (\pmb {x} _ {t}) - f ^ {\star}}{\| \nabla f (\pmb {x} _ {t}) \| ^ {2}},\tag{7}
$$

where $f ^ { \star } : = f ( { \pmb x } ^ { \star } )$ . When f is L-smooth, gradient descent with Polyak stepsize converges to the optimal solution as quickly as gradient descent with $\begin{array} { r } { \eta _ { t } = \frac { 1 } { L } } \end{array}$ 

Theorem 3 (Hazan and Kakade (2019, Theorem 1)). Assume that f is convex and L-smooth, and there exists an optimal solution $\begin{array} { r } { \pmb { x } ^ { \star } : = \arg \operatorname* { m i n } _ { \pmb { x } \in \mathbb { R } ^ { d } } f ( \pmb { x } ) } \end{array}$ . Then, gradient descent with Polyak stepsize Eq. (7) satisfies: 

$$
f (\bar {\boldsymbol {x}}) - f \left(\boldsymbol {x} ^ {\star}\right) \leq \mathcal {O} \left(\frac {L \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T}\right),\tag{8}
$$

where $\begin{array} { r } { \bar { \boldsymbol { \mathbf { r } } } : = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \quad } \end{array}$ x and T is the number ofiterations. 

In addition to the L-smooth setting, Polyak stepsize is known to cause gradient descent to converge to the optimal solution with the optimal rate among various settings, e.g., non-smooth convex, smooth convex, and strongly convex settings (Hazan and Kakade, 2019). 

## 3 Improved convergence result of Polyak stepsize

Before proposing a parameter-free method for clipped gradient descent, in this section, we present a new convergence analysis of Polyak stepsize under $( \mathbf { \breve { L } } _ { 0 } , L _ { 1 } )$ )-smoothness. Surprisingly, our new analysis reveals that Polyak stepsize achieves exactly the same convergence rate as clipped gradient descent with appropriate hyperparameters. A bunch of prior studies established the convergence rates of Polyak stepsize, and it is well-known that Polyak stepsize allows gradient descent to converge as fast as the optimal stepsize. However, our theorem finds that Polyak stepsize achieves a faster convergence rate than gradient descent with the optimal stepsize as clipped gradient descent, and none of the existing studies have found this favorable property of Polyak stepsize. 

## 3.1 Connection between Polyak stepsize and clipped gradient descent

Under $( L _ { 0 } , L _ { 1 } )$ )-smoothness, we can obtain the following results. 

Proposition 1. Assume that $f$ is convex and $( L _ { 0 } , L _ { 1 } )$ -smooth. Then, Polyak stepsize Eq. (7) satisfies: 

$$
\min \left\{\frac {1}{4 L _ {0}}, \frac {1}{4 L _ {1} \| \nabla f (\pmb {x} _ {t}) \|} \right\} \leq \frac {f (\pmb {x} _ {t}) - f ^ {\star}}{\| \nabla f (\pmb {x} _ {t}) \| ^ {2}}.\tag{9}
$$

Proof. Assumption 2 and Lemma 2 imply 

$$
\frac {f (\boldsymbol {x} _ {t}) - f ^ {\star}}{\| \nabla f (\boldsymbol {x} _ {t}) \| ^ {2}} \geq \frac {1}{2 (L _ {0} + L _ {1} \| \nabla f (\boldsymbol {x} _ {t}) \|)}.
$$

When $\begin{array} { r } { \| \nabla f ( \pmb { x } _ { t } ) \| < \frac { L _ { 0 } } { L _ { 1 } } } \end{array}$ , Polyak stepsize is bounded from below by $\frac { 1 } { 4 L _ { 0 } }$ . When $\begin{array} { r } { \| \nabla f ( \pmb { x } _ { t } ) \| \geq \frac { L _ { 0 } } { L _ { 1 } } } \end{array}$ , we have 

$$
\frac {1}{2 (L _ {0} + L _ {1} \| \nabla f (\boldsymbol {x} _ {t}) \|)} \geq \frac {1}{4 L _ {1} \| \nabla f (\boldsymbol {x} _ {t}) \|}.
$$

Therefore, we can conclude the statement. 

Under L-smoothness, the lower bound of Polyak stepsize was obtained as follows. 

Proposition 2 (Jiang and Stich (2023, Lemma 15)). Assume that f is convex and L-smooth. Then, Polyak stepsize Eq. (7) satisfies: 

$$
\frac {1}{2 L} \leq \frac {f (\boldsymbol {x} _ {t}) - f ^ {\star}}{\| \nabla f (\boldsymbol {x} _ {t}) \| ^ {2}}.\tag{10}
$$

By comparing Propositions 1 and 2, Proposition 1 shows that Polyak stepsize does not become excessively small when the parameter approaches the optimal solution $( \mathrm { i } . \mathrm { e } . , \| \nabla f ( \pmb { x } ) \|$ approaches zero), similar to clipped gradient descent. If we choose the stepsize and gradient clipping threshold as in Theorem 2, clipped gradient descent can be written as follows: 

$$
\boldsymbol {x} _ {t + 1} = \boldsymbol {x} _ {t} - \min \left\{\frac {1}{L _ {0}}, \frac {1}{L _ {1} \| \nabla f (\boldsymbol {x} _ {t}) \|} \right\} \nabla f (\boldsymbol {x} _ {t}).\tag{11}
$$

Thus, Proposition 1 implies that Polyak stepsize can be regarded as internally estimating the hyperparameters for clipped gradient descent, as shown in Theorem 2. 

## 3.2 Convergence analysis of Polyak stepsize under $( L _ { 0 } , L _ { 1 } )$ -smoothness

Based on the relationship between Polyak stepsize and clipped gradient descent in Sec. 3.1, we provide a new convergence result for Polyak stepsize under $( L _ { 0 } , L _ { 1 } )$ -smoothness. The proof is deferred to Sec. A. 

Theorem 4. Assume that f is convex, L-smooth, and $( L _ { 0 } , L _ { 1 } )$ -smooth, and there exists an optimal solution $x ^ { \star } : =$ arg $\scriptstyle \operatorname* { m i n } _ { \pmb { x } \in \mathbb { R } ^ { d } } f ( \pmb { x } )$ . Let T be the number of iterations and define $\tau : =$ $\begin{array} { r } { \arg \operatorname* { m i n } _ { 0 \leq t \leq T - 1 } f ( \pmb { x } _ { t } ) } \end{array}$ . Then, gradient descent with Polyak stepsize Eq. (7) satisfies: 

$$
f (\boldsymbol {x} _ {\tau}) - f (\boldsymbol {x} ^ {\star}) \leq \mathcal {O} \left(\frac {L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T} + \frac {L L _ {1} ^ {2} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {4}}{T ^ {2}}\right).\tag{12}
$$

By comparing Theorem 4 with Theorem 2, the convergence rate of Polyak stepsize is the same as that of clipped gradient descent. Thus, Polyak stepsize can converge faster than the optimal stepsize given in Theorem 1 when $L _ { 0 } \ll L$ . Many prior studies analyzed the convergence rate of Polyak stepsize and discussed the relationship between Polyak stepsize and gradient descent with the optimal stepsize (Polyak, 1987; Loizou et al., 2021; Galli et al., 2023; Berrada et al., 2020). However, they only recognized Polyak stepsize as making gradient descent converge with the same convergence rate as the optimal stepsize, and none of the prior studies have found this relationship between Polyak stepsize and clipped gradient descent. Our new convergence result is the first to discover that the Polyak stepsize can achieve the same convergence rate not only as gradient descent with an appropriate stepsize but also as clipped gradient descent with an appropriate stepsize and gradient clipping threshold. 

## 4 Making clipped gradient descent parameter-free

In the previous section, we found that the convergence rate of Polyak stepsize is asymptotically independent of $L$ under $( L _ { 0 } , L _ { 1 } )$ -smoothness as clipped gradient descent with appropriate hyperparameters. However, Polyak stepsize requires the minimum loss value $f ^ { \star }$ , which is a problemspecific parameter. In this section, we propose a method that can remove the prior knowledge of $f ^ { \star }$ from Polyak stepsize without losing the property of asymptotic independence of $L$ under $( L _ { 0 } , L _ { 1 } )$ -smoothness. 

## 4.1 Inexact Polyak Stepsize

To make Polyak stepsize parameter-free, several prior studies have proposed the use of lower bound of $f ^ { \star }$ instead of $f ^ { \star }$ (Loizou et al., 2021; Orvieto et al., 2022; Jiang and Stich, 2023). The loss functions commonly used in machine learning models are non-negative. Thus, the lower bound of $f ^ { \star }$ is trivially obtained as zero and is not a problem-specific parameter. By utilizing this lower bound, a straightforward approach to make Polyak stepsize independent of problem-specific parameters is replacing $f ^ { \star }$ in Polyak stepsize with the lower bound $l ^ { \star }$ as follows: 

$$
\eta_ {t} = \frac {f (\pmb {x} _ {t}) - l ^ {\star}}{\| \nabla f (\pmb {x} _ {t}) \| ^ {2}}.\tag{13}
$$

However, the stepsize in Eq. (13) becomes excessively large as the parameter approaches the optimal solution, and it does not lead to the optimal solution (Loizou et al., 2021). This is because $\| \nabla f ( \pmb { x } _ { t } ) \|$ approaches zero, while $f ( \pmb { x } _ { t } ) - l ^ { \star }$ approaches $f ^ { \star } - l ^ { \star } ( > 0 )$ , which makes the stepsize in Eq. (13) excessively large as the parameter approaches the optimal solution. To mitigate this issue, DecSPS (Orvieto et al., 2022) and AdaSPS (Jiang and Stich, 2023), which are parameter-free methods based on Polyak stepsize that use $l ^ { \star }$ instead of $f ^ { \star }$ , make the stepsize monotonically non-increasing to converge to the optimal solution. 

However, making the stepsize monotonically non-increasing loses the fruitful property that the convergence rate of Polyak stepsize is asymptotically independent of L as clipping gradient descent under $( L _ { 0 } , L _ { 1 } )$ -smoothness. This is because Polyak stepsize and clipped gradient descent make the convergence rate asymptotically independent of L by increasing the stepsize when the parameter approaches the optimal solution. In fact, we evaluated DecSPS and AdaSPS with a synthetic function in Sec. 6.1, demonstrating that the convergence deteriorates as L increases. 

Algorithm 1 Inexact Polyak Stepsize
1: Input: The number of iterations T and lower bound $l^{\star}$ .
2: $f^{\text{best}}, x^{\text{best}} \leftarrow f(x_0), x_0$ .
3: for $t = 0, 1, \cdots, T - 1$ do
4: $x_{t+1} \leftarrow x_t - \frac{f(x_t) - l^\star}{\sqrt{T}\|\nabla f(x_t)\|^2} \nabla f(x_t)$ .
5: if $f(x_{t+1}) \leq f^{\text{best}}$ then
6: $f^{\text{best}}, x^{\text{best}} \leftarrow f(x_{t+1}), x_{t+1}$ .
7: return $x^{best}$ . 

To address this issue, we propose Inexact Polyak Stepsize, whose details are described in Alg. 1. As discussed above, we cannot make the stepsize decrease to maintain the asymptotic independence of L under $( L _ { 0 } , L _ { 1 } )$ )-smoothness. Thus, we set the stepsize as follows: 

$$
\eta_ {t} = \frac {f (\pmb {x} _ {t}) - l ^ {\star}}{\sqrt {T} \| \nabla f (\pmb {x} _ {t}) \| ^ {2}},\tag{14}
$$

where $T$ denotes the number of iterations. Instead of making the stepsize decrease, we propose returning the parameter for which the lowest loss is achieved as the final parameter. 

## 4.2 Convergence analysis of Inexact Polyak Stepsize

The following theorem provides the convergence rate of Inexact Polyak Stepsize. The proof is deferred to Sec. B. 

Theorem 5. Assume that f is convex, L-smooth, and $( L _ { 0 } , L _ { 1 } )$ -smooth, and there exists an optimal solution $\begin{array} { r } { \pmb { x } ^ { \star } : = \arg \operatorname* { m i n } _ { \pmb { x } \in \mathbb { R } ^ { d } } f ( \pmb { x } ) } \end{array}$ . Let $T$ be the number ofiterations and $\sigma ^ { 2 } : = f ^ { \star } - l ^ { \star }$ . Then, x generated by Alg. 1 satisfies: 

$$
f (\boldsymbol {x}) - f (\boldsymbol {x} ^ {\star}) \leq \mathcal {O} \left(\frac {L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2} + \sigma^ {2}}{\sqrt {T}} + \frac {L L _ {1} ^ {2} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {4}}{T} + \frac {L _ {1} ^ {2} L \sigma^ {4}}{L _ {0} ^ {2} T}\right).\tag{15}
$$

Asymptotic independence of L: When the number of iterations $T$ is large, only the first term $\begin{array} { r } { \mathcal { O } \big ( \frac { L _ { 0 } \| \bar { \mathbf { x } } _ { 0 } - \mathbf { x } ^ { \star } \| ^ { 2 } + \sigma ^ { \bar { 2 } } } { \sqrt { T } } \big ) } \end{array}$ becomes dominant in the convergence rate, which does not depend on $L .$ Thus, Theorem $^ { 5 }$ shows that Inexact Polyak Stepsize successfully inherits the favorable property of Polyak stepsize under $( L _ { 0 } , L _ { 1 } )$ )-smoothness. In addition to Inexact Polyak Stepsieze, DecSPS (Orvieto et al., 2022) and AdaSPS (Jiang and Stich, 2023) have been proposed as parameter-free methods that use $l ^ { \star }$ instead of $f ^ { \star }$ in Polyak stepsize. However, these prior methods fail to inherit the favorable property of Polyak stepsize, and their convergence rates deteriorate when L is large because these methods decrease the stepsize during the training. In fact, we evaluated DecSPS and AdaSPS with a synthetic function in Sec. 6.1, demonstrating that convergence rates of DecSPS and AdaSPS are degraded when L becomes large, whereas the convergence rate of Inexact Polyak Stepsize does not depend on L. 

Removing dependence on $D _ { T } { \mathrm { : } }$ : The convergence rates of DecSPS and AdaSPS depend on $D _ { T } ( : =$ $\mathrm { m a x } _ { 0 \leq t \leq T } \| \bar { \mathbf { x } _ { t } } - \mathbf { x } ^ { \star } \| )$ . Thus, strictly speaking, these convergence rates cannot show that DecSPS and AdaSPS converge to the optimal solution because $D _ { T }$ may increase as the number of iterations T increases. For instance, if $D _ { T }$ increase with $\Omega ( T ^ { \frac { 1 } { 4 } } )$ , the convergence rate of AdaSPS is $\mathcal { O } ( L \sigma + L ^ { 2 } )$ which does not show that AdaSPS converges to the optimal solution. In contrast, the convergence rate in Eq. (15) depends on only $\| \pmb { x } _ { 0 } - \pmb { x } ^ { \star } \|$ . Theorem 5 indicates that Inexact Polyak Stepsize converges to the optimal solution. 

Convergence rate with respect to $T { \ : } $ Inexact Polyak Stepsize successfully achieves the asymptotic independence of $L ,$ while it slows down the convergence rate with respect to the number of iterations $T$ by comparing clipped gradient descent with proper hyperparameters. The convergence rate of Inexact Polyak Stepsize $\begin{array} { r l r } {  { \mathcal { O } \big ( \frac { L _ { 0 } } { \sqrt { T } } \big ) } } \end{array}$ is not optimal in terms of $T _ { \ast }$ , and there may be room to improve this rate. For instance, the adaptive methods proposed by Hazan and Kakade (2019) might be used to alleviate this issue. However, the parameter-free methods for clipped gradient descent have not been explored well in the existing studies. We believe that Inexact Polyak Stepsize is the important first step for developing parameter-free clipped gradient descent. 


Table 1: Summary of convergence rates of parameter-free methods based on Polyak stepsize. All convergence results are the ones under convex, L-smoothness, and $( L _ { 0 } , L _ { 1 } )$ -smoothness. We define $D _ { T } : = \mathrm { { m a x } } _ { 0 \leq t \leq T } \| \pmb { x } _ { t } - \pmb { x } ^ { \star } \|$


<table><tr><td>Algorithm</td><td>Convergence Rate</td><td>Assumption</td></tr><tr><td>DecSPS (Orvieto et al., 2022)<eq>^{(a)}</eq></td><td><eq>\mathcal{O}\left(\frac{\max\{L,\eta_0^{-1}\}D_T^2+\sigma^2}{\sqrt{T}}\right)</eq></td><td>1</td></tr><tr><td>AdaSPS (Jiang and Stich, 2023)<eq>^{(a)}</eq></td><td><eq>\mathcal{O}\left(\frac{LD_T^2\sigma}{\sqrt{T}}+\frac{L^2D_T^4}{T}\right)</eq></td><td>1</td></tr><tr><td>Inexact Polyak Stepsize (This work)</td><td><eq>\mathcal{O}\left(\frac{L_0\|\boldsymbol{x}_0-\boldsymbol{x}^\star\|^2+\sigma^2}{\sqrt{T}}+\frac{LL_1^2\|\boldsymbol{x}_0-\boldsymbol{x}^\star\|^4}{T}+\frac{L_1^2L\sigma^4}{L_0^2T}\right)</eq></td><td><eq>1,2^{(b)}</eq></td></tr></table>


(a) We present the convergence rates of DecSPS and AdaSPS in the deterministic setting to compare DecSPS, AdaSPS, and Inexact Polyak Stepsize in the same deterministic setting, while Orvieto et al. (2022) and Jiang and Stich (2023) also analyzed the rate rates in the stochastic setting. 


## 5 Related work

Gradient clipping: Gradient clipping was initially proposed to mitigate the gradient explosion problem for training RNN and LSTM (Mikolov et al., 2010; Merity et al., 2018) and is now widely used to accelerate and stabilize the training not only for RNN and LSTM, but also for various machine learning models, especially language models (Devlin et al., 2019; Raffel et al., 2019). Recently, many studies have investigated the theoretical benefits of gradient clipping and analyzed the convergence rate of clipped gradient descent under (1) $( L _ { 0 } , L _ { 1 } )$ -smoothness assumption (Koloskova et al., 2023; Zhang et al., 2020a,b) and (2) heavy-tailed noise assumption (Zhang et al., 2020c; Li and Liu, 2022; Sadiev et al., 2023). (1) Zhang et al. (2020b) found that the local gradient Lipschitz constant is correlated with the gradient norm. To describe this phenomenon, Zhang et al. (2020b), Zhang et al. (2020a), and Koloskova et al. (2023) introduced the new assumption, $( L _ { 0 } , L _ { 1 } )$ -smoothness, providing the convergence rate of clipped gradient descent under $( L _ { 0 } , L _ { 1 } )$ -smoothness. Then, they showed that gradient clipping can improve the convergence rate of gradient descent, as we introduced in Sec. 2.2. (2) Besides $\bar { ( L _ { 0 } , L _ { 1 } ) }$ -smoothness, Zhang et al. (2020c) pointed out that the distribution of stochastic gradient noise is heavy-tailed for language models. Then, it has been shown that gradient clipping can make the stochastic gradient descent robust against the heavy-tailed noise of stochastic gradient (Li and Liu, 2022; Sadiev et al., 2023; Zhang et al., 2020c). 

Parameter-free methods: Hyperparameter-tuning is one of the most time-consuming tasks for training machine learning models. To alleviate this issue, many parameter-free methods that adjust the stepsize on the fly have been proposed, e.g., Polyak-based stepsize (Berrada et al., 2020; Hazan and Kakade, 2019; Loizou et al., 2021; Mukherjee et al., 2023; Orvieto et al., 2022; Jiang and Stich, 2023), AdaGrad-based methods (Ivgi et al., 2023; Khaled et al., 2023), and Dual Averaging-based methods (Orabona and Tommasi, 2017; Defazio and Mishchenko, 2023). However, parameter-free methods for hyperparameters, except for stepsizes, have not been studied. In this work, we studied the parameter-free methods for two hyperparameters, the stepsize and gradient clipping threshold, and then proposed Inexact Polyak Stepsize, which converges to the optimal solution without tuning any hyperparameters and its convergence rate is asymptotically independent of $L$ as clipped gradient descent with well-tuned hyperparameters. 

## 6 Numerical evaluation

In this section, we evaluate our theory numerically. In Sec. 6.1, we evaluate Polyak stepsize and Inexact Polyak Stepsize using a synthetic function, varying that their convergence rates are asymptotically independent of $L .$ . In Sec. 6.2, we show the results obtained using neural networks. 

## 6.1 Synthetic function

Setting: In this section, we validate our theory for Polyak stepsize and Inexact Polyak Stepsize using a synthetic function. We set the loss function as $\begin{array} { r } { f ( x ) = \frac { L _ { 0 } L _ { 1 } ^ { 2 } } { 7 \mathcal { D } } x ^ { 4 } + \frac { L _ { 0 } } { 4 } x ^ { 2 } + f ^ { \star } } \end{array}$ , which is $( L _ { 0 } , L _ { 1 } )$ )-smooth for any $L _ { 0 } > 0$ and $L _ { 1 } > 0$ (See Proposition 3 in Appendix). We set $L _ { 0 }$ to $1 , \pmb { x } _ { 0 }$ to 5, $f ^ { \star } = 1$ , and $l ^ { \star } = 0$ and then evaluated various methods when varying $L _ { 1 }$ 

Results: We show the results in Fig. 1. The results indicate that gradient descent converges slowly when $L _ { 1 }$ is large, whereas Polyak stepsize and clipped gradient descent does not depend on $L _ { 1 }$ . These observations are consistent with those discussed in Sec. 3, which shows that the convergence rate of Polyak stepsize is asymptotically independent of L as in clipped gradient descent. By comparing DecSPS, AdaSPS, and Inexact Polyak Stepsize, which are parameter-free methods, the convergence rates of DecSPS and AdaSPS degrade as $L _ { 1 }$ increases. Thus, DecSPS and AdaSPS lose the favorable property of asymptotic independence of L under $( L _ { 0 } , L _ { 1 } )$ -smoothness. In contrast, the convergence behavior of Inexact Polyak Stepsize does not depend on $L _ { 1 } ,$ which is consistent with Theorem $^ { 5 , }$ and Inexact Polyak Stepsize successfully inherits the Polyak stepsize under $( L _ { 0 } , L _ { 1 } )$ -smoothness. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/403cc40d8e6836f8b8f7aefa5d864faf25a5d3d8c2279ae68be125060d221344.jpg)



(a) Gradient Descent


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/16d33830bf7b4f541bfcd51dfc207089a777a007e58f9eb0a565ac2e2f7d8f06.jpg)



(b) Clipped Gradient Descent


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/3712b39806f0a19bcaddc826a832652e1d2ab9dd55e82964edfba2c144b141d5.jpg)



(c) Polyak Stepsize


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/f87dd40e42fdc9d552b53c6c4fd1a02642f372d88fe39dbf4967e960898e58a9.jpg)



(d) DecSPS (Orvieto et al., 2022)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/ccfb5f659a06769fee3d9930fd606fc0c82eb33bfd144129832e1b996ed719c3.jpg)



(e) AdaSPS (Jiang and Stich, 2023)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/e139dac3ee10303d0f6b7d3d5ae4c38c3137b2459eeac900c95ae056564b11ee.jpg)



(f) Inexact Polyak Stepsize



Figure 1: Convergence behaviors of various methods with the synthetic function.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/95dda6fe37f41d44de86c9d9b65988d6333a3b779a8a1604bc36799f926b4059.jpg)



(a) LSTM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/d0bc5d7116e61ebea2c65c8b2eef31ae109919f4ab6abb971f52f34331e3e11a.jpg)



(b) Nano-GPT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/e9f4c3f7f3e3fc42d9ff0753420e62b1b6d45435554f880ec0ceb57dd110d291.jpg)



(c) T5



Figure 2: The final test loss with various hyperparameter settings. For T5, the results of DecSPS and AdaSPS were omitted because their final test loss was much larger than the others, as shown in Fig. 4. Furthermore, the results of SGD were also omitted when the final test loss became nan or infinity.


## 6.2 Neural networks

Setting: Next, we evaluated Inexact Polyak Stepsize using LSTM, Nano-GPT<sup>2</sup>, and T5 (Nawrot, 2023). For LSTM, Nano-GPT, and T5, we used the Penn Treebank, Shakespeare, and C4 as training datasets, respectively. For SGD and Clipped SGD, we tuned the stepsize and gradient clipping threshold on validation datasets. For Polyak stepsize, we showed the results when we set $f ^ { \star }$ to zero. For Inexact Polyak Stepsize, Theorem 4 requires the selection of the best parameters. However, we do not need to choose this for neural networks because the parameters only reach the stationary point and do not reach the global minima. See Sec. D for the detailed training configuration. For all experiments, we repeated with three different seed values and reported the average. 

Results: Figure 4 shows the loss curves, and Fig. 2 shows the final test losses for various hyperparameters. The results indicate that Inexact Polyak Stepsize consistently outperform DecSPS and AdaSPS for all neural network architectures. Although DoG performed the best for LSTM among the parameter-free methods, the training behavior of DoG was very unstable for Nano-GPT, and the loss values were much higher than those of the other methods. Similar to DoG, Polyak stepsize outperformed all parameter-free methods for T5, but the loss values of Polyak stepsize diverged for LSTM and Nano-GPT. Thus, Inexact Polyak Stepsize can consistently succeed in training models for all neural network architectures. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/f2597aa58428ac60252f4871c3cf135e8c21672c14b4aadfa24d3e703c8a7f69.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/016ef6637485a65e7dba956723ee9da9fa93d2dd3cd74a21d149973428d9708f.jpg)



(a) LSTM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/05f96a585c6eb276f1b69d508e5120e4fc27f077c7ccb4c45acff87924398a85.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/a73f1de67551e73e7334cdf755f8632b71262a09eea89d7cedd7623861567763.jpg)



(b) Nano-GPT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/4c91367716ff090a631b785df730f60e416f24456943c95bf58f16779799c077.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/f1fbc1123e933c6b5a4a7bcc937ec1215d2c648b6754112cb64f6b8150944648.jpg)



(c) T5



Figure 3: Loss curves for LSTM, Nano-GPT, and T5. We plotted the training loss per 100, 10, and 10 iterations for LSTM, Nano-GPT, and T5, respectively. We plotted the test loss per one epoch, 100 iterations, and 200 iterations, respectively. For LSTM and Nano-GPT, we found that Polyak stepsize does not converge, and its loss was much larger than that of other comparison methods. Thus, to make the figure easier to read, we omit the results of Polyak stepsize and provide the complete results, including Polyak stepsize in Sec. E.


## 7 Conclusion

In this study, we proposed Inexact Polyak Stepsize, which converges to the optimal solution without hyperparameter tuning at the convergence rate that is asymptotically independent of L under $( L _ { 0 } , \bar { L _ { 1 } } )$ -smoothness. Specifically, we first provided the novel convergence rate of Polyak stepsize under $( L _ { 0 } , L _ { 1 } )$ -smoothness, revealing that Polyak stepsize can achieve exactly the same convergence rate as clipped gradient descent. Although Polyak stepsize can improve the convergence under $( L _ { 0 } , L _ { 1 } )$ -smoothness, Polyak stepsize requires the minimum loss value, which is a problemspecific parameter. Then, we proposed Inexact Polyak Stepsize, which removes the problem-specific parameter from Polyak stepsize without losing the property of asymptotic independence of L under $( L _ { 0 } , L _ { 1 } )$ )-smoothness. We numerically validated our convergence results and demonstrated the effectiveness of Inexact Polyak Stepsize. 

## Acknowledgement

Y.T. was supported by KAKENHI Grant Number 23KJ1336. H.B. and M.Y. were supported by MEXT KAKENHI Grant Number 24K03004. We thank Satoki Ishikawa for his helpful comments on our experiments. 

## References



Berrada, L., Zisserman, A., and Kumar, M. P. (2020). Training neural networks for and by interpolation. In International Conference on Machine Learning. 





Carmon, Y. and Hinder, O. (2022). Making SGD parameter-free. In Conference on Learning Theory. 





Defazio, A. and Mishchenko, K. (2023). Learning-rate-free learning by D-adaptation. In International Conference on Machine Learning. 





Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In Associationfor Computational Linguistics. 





Duchi, J., Hazan, E., and Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. In Journal of Machine Learning Research. 





Galli, L., Rauhut, H., and Schmidt, M. (2023). Don't be so monotone: Relaxing stochastic line search in over-parameterized models. In Advances in Neural Information Processing Systems. 





Garrigos, G. and Gower, R. M. (2023). Handbook of convergence theorems for (stochastic) gradient methods. In arXiv. 





Hazan, E. and Kakade, S. M. (2019). Revisiting the Polyak step size. In arXiv. 





Ivgi, M., Hinder, O., and Carmon, Y. (2023). DoG is SGD’s best friend: A parameter-free dynamic step size schedule. In International Conference on Machine Learning. 





Jiang, X. and Stich, S. U. (2023). Adaptive SGD with Polyak stepsize and line-search: Robust convergence and variance reduction. In Advances in Neural Information Processing Systems. 





Khaled, A., Mishchenko, K., and Jin, C. (2023). DoWG unleashed: An efficient universal parameterfree gradient descent method. In Advances in Neural Information Processing Systems. 





Kingma, D. and Ba, J. (2015). Adam: A method for stochastic optimization. In International Conference on Learning Representations. 





Koloskova, A., Hendrikx, H., and Stich, S. U. (2023). Revisiting gradient clipping: Stochastic bias and tight convergence guarantees. In International Conference on Machine Learning. 





Li, S. and Liu, Y. (2022). High probability guarantees for nonconvex stochastic gradient descent with heavy tails. In International Conference on Machine Learning. 





Loizou, N., Vaswani, S., Hadj Laradji, I., and Lacoste-Julien, S. (2021). Stochastic Polyak step-size for SGD: An adaptive learning rate for fast convergence. In International Conference on Artificial Intelligence and Statistics. 





Merity, S., Keskar, N. S., and Socher, R. (2018). Regularizing and optimizing LSTM language models. In International Conference on Learning Representations. 





Mikolov, T., Karafiat, M., Burget, L., Cernocky, J. H., and Khudanpur, S. (2010). Recurrent neural network based language model. In Interspeech. 





Mukherjee, S., Loizou, N., and Stich, S. U. (2023). Locally adaptive federated learning via stochastic polyak stepsizes. In arXiv. 





Nawrot, P. (2023). NanoT5: A pytorch framework for pre-training and fine-tuning t5-style models with limited resources. In arXiv. 





Nesterov, Y. (2018). Lectures on Convex Optimization. Springer. 





Orabona, F. and Tommasi, T. (2017). Training deep networks without learning rates through coin betting. In Advances in Neural Information Processing Systems. 





Orvieto, A., Lacoste-Julien, S., and Loizou, N. (2022). Dynamics of SGD with stochastic Polyak stepsizes: Truly adaptive variants and convergence to exact solution. In Advances in Neural Information Processing Systems. 





Pascanu, R., Mikolov, T., and Bengio, Y. (2013). On the difficulty of training recurrent neural networks. In International Conference on Machine Learning. 





Polyak, B. (1987). Introduction to Optimization. Optimization Software. 





Raffel, C., Shazeer, N. M., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. (2019). Exploring the limits of transfer learning with a unified text-to-text transformer. In Journal ofMachine Learning Research. 





Sadiev, A., Danilova, M., Gorbunov, E., Horváth, S., Gidel, G., Dvurechensky, P., Gasnikov, A., and Richtárik, P. (2023). High-probability bounds for stochastic optimization and variational inequalities: the case of unbounded variance. In International Conference on Machine Learning. 





Zhang, B., Jin, J., Fang, C., and Wang, L. (2020a). Improved analysis of clipping algorithms for non-convex optimization. In Advances in Neural Information Processing Systems. 





Zhang, J., He, T., Sra, S., and Jadbabaie, A. (2020b). Why gradient clipping accelerates training: A theoretical justification for adaptivity. In International Conference on Learning Representations. 





Zhang, J., Karimireddy, S. P., Veit, A., Kim, S., Reddi, S., Kumar, S., and Sra, S. (2020c). Why are adaptive methods good for attention models? In Advances in Neural Information Processing Systems. 



## A Proof of Theorem 4

Lemma 1. IfAssumption 1 holds, thefollowing holdsfor any x $\in \mathbb { R } ^ { d } .$ 

$$
\frac {1}{2 L} \| \nabla f (\boldsymbol {x}) \| ^ {2} \leq f (\boldsymbol {x}) - f (\boldsymbol {x} ^ {\star}).\tag{16}
$$

Proof. See Lemma 2.28 in (Garrigos and Gower, 2023). 

Lemma 2. If Assumption 2 holds, the following holds for any $\pmb { x } \in \mathbb { R } ^ { d } .$ 

$$
\frac {1}{2 (L _ {0} + L _ {1} \| \nabla f (\boldsymbol {x}) \|)} \| \nabla f (\boldsymbol {x}) \| ^ {2} \leq f (\boldsymbol {x}) - f (\boldsymbol {x} ^ {\star}).\tag{17}
$$

Proof. See Lemma A.2 in (Koloskova et al., 2023). 

Lemma 3. Assume that f is convex and Assumption 1 and 2 hold. Let T be the number ofiterations and define $\tau : = \arg \operatorname* { m i n } _ { 0 \leq t \leq T - 1 } f ( \pmb { x } _ { t } )$ . Then, gradient descent with Polyak stepsize Eq. (7) satisfies: 

$$
f (\boldsymbol {x} _ {\tau}) - f (\boldsymbol {x} ^ {\star}) \leq \frac {8 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T} + \frac {6 4 L L _ {1} ^ {2} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {4}}{T ^ {2}}.
$$

Proof. We have 

$$
\begin{array}{r l} & {\| \pmb {x} _ {t + 1} - \pmb {x} ^ {\star} \| ^ {2} = \| \pmb {x} _ {t} - \pmb {x} ^ {\star} \| ^ {2} - 2 \eta_ {t} \langle \nabla f (\pmb {x} _ {t}), \pmb {x} _ {t} - \pmb {x} ^ {\star} \rangle + \eta_ {t} ^ {2} \| \nabla f (\pmb {x}) \| ^ {2}} \\ & {\qquad \leq \| \pmb {x} _ {t} - \pmb {x} ^ {\star} \| ^ {2} - 2 \eta_ {t} (f (\pmb {x} _ {t}) - f (\pmb {x} ^ {\star})) + \eta_ {t} ^ {2} \| \nabla f (\pmb {x}) \| ^ {2},} \end{array}
$$

where we use the convexity of f in the inequality. 

Case when $\begin{array} { r } { \| \nabla f ( \pmb { x } _ { t } ) \| \le \frac { L _ { 0 } } { L _ { 1 } } } \end{array}$ : Substituting the stepsize, we get 

$$
\begin{array}{l} \| \boldsymbol {x} _ {t + 1} - \boldsymbol {x} ^ {\star} \| ^ {2} \leq \| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \frac {f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})}{\| \nabla f (\boldsymbol {x} _ {t}) \| ^ {2}} (f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})) \\ \quad \leq \| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \frac {1}{2 (L _ {0} + L _ {1} \| \nabla f (\boldsymbol {x} _ {t}) \|)} (f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})) \\ \quad \leq \| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \frac {1}{4 L _ {0}} (f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})), \end{array}
$$

where we use Lemma 2 in the second inequality. Unrolling the above inequality, we obtain 

$$
f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star}) \leq 4 L _ {0} \left(\| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \| \boldsymbol {x} _ {t + 1} - \boldsymbol {x} ^ {\star} \| ^ {2}\right).
$$

Case when $\begin{array} { r } { \| \nabla f ( \pmb { x } _ { t } ) \| > \frac { L _ { 0 } } { L _ { 1 } } } \end{array}$ : Substituting the stepsize, we get 

$$
\begin{array}{l} \| \boldsymbol {x} _ {t + 1} - \boldsymbol {x} ^ {\star} \| ^ {2} \leq \| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \frac {(f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})) ^ {2}}{\| \nabla f (\boldsymbol {x} _ {t}) \| ^ {2}} \\ \qquad \leq \| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \sqrt {\frac {f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})}{2 L}} \frac {f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})}{\| \nabla f (\boldsymbol {x} _ {t}) \|}, \end{array}
$$

where we use Lemmas 1 in the last inequality. Then $\begin{array} { r } { \| \nabla f ( \pmb { x } _ { t } ) \| > \frac { L _ { 0 } } { L _ { 1 } } } \end{array}$ implies 

$$
\frac {L _ {0} + L _ {1} \| \nabla f (\pmb {x} _ {t}) \|}{2 L _ {1} \| \nabla f (\pmb {x} _ {t}) \|} <   1.
$$

Thus, we get 

$$
\begin{array}{c} \| \boldsymbol {x} _ {t + 1} - \boldsymbol {x} ^ {\star} \| ^ {2} \leq \| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \sqrt {\frac {f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})}{2 L}} \frac {f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})}{\| \nabla f (\boldsymbol {x} _ {t}) \| ^ {2}} \frac {L _ {0} + L _ {1} \| \nabla f (\boldsymbol {x} _ {t}) \|}{2 L _ {1}} \\ \leq \| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \frac {1}{4 L _ {1}} \sqrt {\frac {f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})}{2 L}}, \end{array}
$$

where we use Lemma 2 in the last inequality. Unrolling the above inequality and multiplying $L _ { 0 }$ on both sides, we get 

$$
\frac {L _ {0}}{L _ {1}} \sqrt {\frac {f (\boldsymbol {x}) - f (\boldsymbol {x} ^ {\star})}{2 L}} \leq 4 L _ {0} \left(\| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \| \boldsymbol {x} _ {t + 1} - \boldsymbol {x} ^ {\star} \| ^ {2}\right).
$$

Summing the two cases: Define $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ as follows: 

$$
\mathcal {T} _ {1} := \left\{t \middle | \| \nabla f (\boldsymbol {x} _ {t}) \| \leq \frac {L _ {0}}{L _ {1}} \right\}, \mathcal {T} _ {2} := \left\{t \middle | \| \nabla f (\boldsymbol {x} _ {t}) \| > \frac {L _ {0}}{L _ {1}} \right\}.
$$

We obtain 

$$
\sum_ {t \in \mathcal {T} _ {1}} (f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})) + \frac {L _ {0}}{L _ {1}} \sum_ {t \in \mathcal {T} _ {2}} \sqrt {\frac {f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})}{2 L}} \leq 4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}.
$$

Then, the above inequality implies 

$$
\frac {1}{T} \sum_ {t \in \mathcal {T} _ {1}} f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star}) \leq \frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T},
$$

$$
\frac {1}{T} \sum_ {t \in \mathcal {T} _ {2}} \sqrt {f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})} \leq \frac {4 L _ {1} \sqrt {2 L} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T}.
$$

Using $a ^ { 2 } \geq 2 a b - b ^ { 2 }$ , we obtain for any $b \in \mathbb { R }$ 

$$
\frac {1}{T} \sum_ {t \in \mathcal {T} _ {1}} \left(2 b \sqrt {f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})} - b ^ {2}\right) \leq \frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T}.
$$

Thus, when $b > 0$ , we obtain 

$$
\frac {1}{T} \sum_ {t \in \mathcal {T} _ {1}} \sqrt {f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})} \leq \frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{2 b T} + \frac {b}{2}.
$$

Choosing $\begin{array} { r } { b = \sqrt { \frac { 4 L _ { 0 } \left\| \pmb { x } _ { 0 } - \pmb { x } ^ { \star } \right\| ^ { 2 } } { T } } } \end{array}$ , we get 

$$
\frac {1}{T} \sum_ {t \in \mathcal {T} _ {1}} \sqrt {f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})} \leq \sqrt {\frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T}}.
$$

Thus, we get 

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \sqrt {f (\boldsymbol {x} _ {t}) - f (\boldsymbol {x} ^ {\star})} \leq \sqrt {\frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T}} + \frac {4 L _ {1} \sqrt {2 L} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T}.
$$

Defining $\tau : = \arg \operatorname* { m i n } _ { t } f ( \pmb { x } _ { t } )$ , we get 

$$
\sqrt {f (\boldsymbol {x} _ {\tau}) - f (\boldsymbol {x} ^ {\star})} \leq \sqrt {\frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T}} + \frac {4 L _ {1} \sqrt {2 L} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T}.
$$

Squaring the both sides, and using $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ for all $a , b \in \mathbb { R }$ , we obtain 

$$
f (\boldsymbol {x} _ {\tau}) - f (\boldsymbol {x} ^ {\star}) \leq \frac {8 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{T} + \frac {6 4 L L _ {1} ^ {2} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {4}}{T ^ {2}}.
$$

This concludes the statement. 

## B Proof of Theorem 5

Lemma 4. Assume that f is convex and Assumptions 1 and 2 hold. Let T be the number ofiterations and define $\begin{array} { r } { \tau : = \arg \operatorname* { m i n } _ { 0 \leq t \leq T - 1 } f ( { \pmb x } _ { t } ) . { \cal I } f f ( { \pmb x } _ { t } ) - f ^ { \star } \geq \frac { \sigma ^ { 2 } } { \sqrt { T } } } \end{array}$ for all t, then gradient descent with stepsize Eq. (14) satisfies: 

$$
f (\boldsymbol {x} _ {\tau}) - f ^ {\star} \leq \frac {8 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2} + 2 \sigma^ {2}}{\sqrt {T}} + \frac {1 2 8 L _ {1} ^ {2} L \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {4}}{T} + \frac {8 L _ {1} ^ {2} \sigma^ {4} L}{L _ {0} ^ {2} T}.
$$

where $\begin{array} { r } { \pmb { x } ^ { \star } : = \arg \operatorname* { m i n } _ { \pmb { x } } f ( \pmb { x } ) a n d \sigma ^ { 2 } : = f ^ { \star } - l ^ { \star } } \end{array}$ 

Proof. By the convexity of $f ,$ we have 

$$
\begin{array}{c} \| \boldsymbol {x} _ {t + 1} - \boldsymbol {x} ^ {\star} \| ^ {2} = \| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - 2 \eta_ {t} \langle \nabla f (\boldsymbol {x} _ {t}), \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \rangle + \eta_ {t} ^ {2} \| \nabla f (\boldsymbol {x} _ {t}) \| ^ {2} \\ \leq \| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - 2 \eta_ {t} (f (\boldsymbol {x} _ {t}) - f ^ {\star}) + \eta_ {t} ^ {2} \| \nabla f (\boldsymbol {x} _ {t}) \| ^ {2}. \end{array}
$$

Substituting the stepsize Eq. (14), we get 

$$
\begin{array}{r l} & {\| \boldsymbol {x} _ {t + 1} - \boldsymbol {x} ^ {\star} \| ^ {2} \leq \| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - 2 \eta_ {t} (f (\boldsymbol {x} _ {t}) - f ^ {\star}) + \frac {\eta_ {t}}{\sqrt {T}} (f (\boldsymbol {x} _ {t}) - l ^ {\star})} \\ & {\qquad \leq \| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \eta_ {t} (2 - \frac {1}{\sqrt {T}}) (f (\boldsymbol {x} _ {t}) - f ^ {\star}) + \frac {\eta_ {t} \sigma^ {2}}{\sqrt {T}}} \\ & {\qquad \leq \| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \eta_ {t} (f (\boldsymbol {x} _ {t}) - f ^ {\star}) + \frac {\eta_ {t} \sigma^ {2}}{\sqrt {T}},} \end{array}\tag{18}
$$

where we use $T \geq 1$ in the last inequality. Unrolling the above inequality and dividing by $\eta _ { t }$ , we obtain 

$$
f (\boldsymbol {x} _ {t}) - f ^ {\star} \leq \frac {\| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \| \boldsymbol {x} _ {t + 1} - \boldsymbol {x} ^ {\star} \| ^ {2}}{\eta_ {t}} + \frac {\sigma^ {2}}{\sqrt {T}}.\tag{19}
$$

Case when $\begin{array} { r } { \| \nabla f ( \pmb { x } _ { t } ) \| \leq \frac { L _ { 0 } } { L _ { 1 } } ; } \end{array}$ From $\begin{array} { r } { f ( \pmb { x } _ { t } ) - f ^ { \star } \geq \frac { \sigma ^ { 2 } } { \sqrt { T } } } \end{array}$ and Eq. (18), we obtain 

$$
\left\| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \right\| ^ {2} - \left\| \boldsymbol {x} _ {t + 1} - \boldsymbol {x} ^ {\star} \right\| ^ {2} \geq 0.\tag{20}
$$

Thus, we get 

$$
\begin{array}{l} f (\boldsymbol {x} _ {t}) - f ^ {\star} \leq 2 (L _ {0} + L _ {1} \| \nabla f (\boldsymbol {x} _ {t}) \|) \sqrt {T} (\| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \| \boldsymbol {x} _ {t + 1} - \boldsymbol {x} ^ {\star} \| ^ {2}) + \frac {\sigma^ {2}}{\sqrt {T}} \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \leq 4 L _ {0} \sqrt {T} (\| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \| \boldsymbol {x} _ {t + 1} - \boldsymbol {x} ^ {\star} \| ^ {2}) + \frac {\sigma^ {2}}{\sqrt {T}}, \end{array}
$$

where we use $f ^ { \star } \geq l ^ { \star }$ and Lemma 2 for the first inequality and use $\begin{array} { r } { \| \nabla f ( \pmb { x } _ { t } ) \| \le \frac { L _ { 0 } } { L _ { 1 } } } \end{array}$ for the last inequality. 

Case when $\begin{array} { r } { \| \nabla f ( \pmb { x } _ { t } ) \| > \frac { L _ { 0 } } { L _ { 1 } } } \end{array}$ : From Lemma 2, we have 

$$
\eta_ {t} \geq \frac {f (\boldsymbol {x} _ {t}) - f ^ {\star}}{\sqrt {T} \| \nabla f (\boldsymbol {x} _ {t}) \| ^ {2}} \geq \frac {1}{2 (L _ {0} + L _ {1} \| \nabla f (\boldsymbol {x} _ {t}) \|) \sqrt {T}}.
$$

Then, we obtain 

$$
\eta_ {t} \geq \frac {1}{4 L _ {1} \| \nabla f (\boldsymbol {x} _ {t}) \| \sqrt {T}} \geq \frac {1}{4 L _ {1} \sqrt {2 L T (f (\boldsymbol {x} _ {t}) - f ^ {\star})}},
$$

where we use $\begin{array} { r } { \| \nabla f ( \pmb { x } _ { t } ) \| > \frac { L _ { 0 } } { L _ { 1 } } } \end{array}$ for the first inequality, and Lemma 1 for the last inequality. Combining Eqs. (19) and (20), we obtain 

$$
f (\boldsymbol {x} _ {t}) - f ^ {\star} \leq 4 L _ {1} \sqrt {2 L T (f (\boldsymbol {x} _ {t}) - f ^ {\star})} (\| \boldsymbol {x} _ {t} - \boldsymbol {x} ^ {\star} \| ^ {2} - \| \boldsymbol {x} _ {t + 1} - \boldsymbol {x} ^ {\star} \| ^ {2}) + \frac {\sigma^ {2}}{\sqrt {T}}.
$$

Furthermore, from $\begin{array} { r } { \| \nabla f ( \pmb { x } _ { t } ) \| > \frac { L _ { 0 } } { L _ { 1 } } } \end{array}$ and Lemma 1, we obtain 

$$
\sqrt {f (\pmb {x} _ {t}) - f ^ {\star}} \geq \frac {L _ {0}}{L _ {1}} \sqrt {\frac {f (\pmb {x} _ {t}) - f ^ {\star}}{\| \nabla f (\pmb {x} _ {t}) \| ^ {2}}} \geq \frac {L _ {0}}{L _ {1}} \sqrt {\frac {1}{2 L}}.
$$

Thus, we get 

$$
f (\boldsymbol {x} _ {t}) - f ^ {\star}
$$

$$
\leq 4 L _ {1} \sqrt {2 L T (f (\pmb {x} _ {t}) - f ^ {\star})} (\| \pmb {x} _ {t} - \pmb {x} ^ {\star} \| ^ {2} - \| \pmb {x} _ {t + 1} - \pmb {x} ^ {\star} \| ^ {2}) + \frac {L _ {1} \sigma^ {2}}{L _ {0} \sqrt {T}} \sqrt {2 L (f (\pmb {x} _ {t}) - f ^ {\star})}.
$$

Dividing by $\frac { L _ { 1 } \sqrt { 2 L ( f ( \pmb { x } _ { t } ) - f ^ { \star } ) } } { L _ { 0 } }$ , we get 

$$
\frac {L _ {0}}{L _ {1}} \sqrt {\frac {f (\pmb {x} _ {t}) - f ^ {\star}}{2 L}} \leq 4 L _ {0} \sqrt {T} (\| \pmb {x} _ {t} - \pmb {x} ^ {\star} \| ^ {2} - \| \pmb {x} _ {t + 1} - \pmb {x} ^ {\star} \| ^ {2}) + \frac {\sigma^ {2}}{\sqrt {T}}.
$$

Summing the two cases: Define $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ as follows: 

$$
\mathcal {T} _ {1} := \left\{t \middle | \| \nabla f (\boldsymbol {x} _ {t}) \| \leq \frac {L _ {0}}{L _ {1}} \right\}, \mathcal {T} _ {2} := \left\{t \middle | \| \nabla f (\boldsymbol {x} _ {t}) \| > \frac {L _ {0}}{L _ {1}} \right\}.
$$

We obtain 

$$
\frac {1}{T} \left(\sum_ {t \in \mathcal {T} _ {1}} (f (\boldsymbol {x} _ {t}) - f ^ {\star}) + \frac {L _ {0}}{L _ {1}} \sum_ {t \in \mathcal {T} _ {2}} \sqrt {\frac {f (\boldsymbol {x}) - f ^ {\star}}{2 L}}\right) \leq \frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2} + \sigma^ {2}}{\sqrt {T}}.
$$

The above inequality implies that 

$$
\frac {1}{T} \sum_ {t \in \mathcal {T} _ {1}} (f (\boldsymbol {x} _ {t}) - f ^ {\star}) \leq \frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2} + \sigma^ {2}}{\sqrt {T}},
$$

$$
\frac {1}{T} \sum_ {t \in \mathcal {T} _ {2}} \sqrt {f (\boldsymbol {x}) - f ^ {\star}} \leq \frac {4 L _ {1} \sqrt {2 L} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{\sqrt {T}} + \frac {L _ {1} \sigma^ {2} \sqrt {2 L}}{L _ {0} \sqrt {T}}.
$$

Using $a ^ { 2 } \geq 2 a b - b ^ { 2 }$ , we obtain for any $b \in \mathbb { R }$ 

$$
\frac {1}{T} \sum_ {t \in \mathcal {T} _ {1}} \left(2 b \sqrt {f (\boldsymbol {x} _ {t}) - f ^ {\star}} - b ^ {2}\right) \leq \frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2} + \sigma^ {2}}{\sqrt {T}}.
$$

Thus, when $b > 0$ , we obtain 

$$
\frac {1}{T} \sum_ {t \in \mathcal {T} _ {1}} \sqrt {f (\boldsymbol {x} _ {t}) - f ^ {\star}} \leq \frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2} + \sigma^ {2}}{2 b \sqrt {T}} + \frac {b}{2}.
$$

Choosing $\begin{array} { r } { b = \sqrt { \frac { 4 L _ { 0 } \| \pmb { x } _ { 0 } - \pmb { x } ^ { \star } \| ^ { 2 } + \sigma ^ { 2 } } { \sqrt { T } } } } \end{array}$ , we get 

$$
\frac {1}{T} \sum_ {t \in \mathcal {T} _ {1}} \sqrt {f (\boldsymbol {x} _ {t}) - f ^ {\star}} \leq \sqrt {\frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2} + \sigma^ {2}}{\sqrt {T}}}.
$$

Thus, we get 

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \sqrt {f (\boldsymbol {x} _ {t}) - f ^ {\star}} \leq \sqrt {\frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2} + \sigma^ {2}}{\sqrt {T}}} + \frac {4 L _ {1} \sqrt {2 L} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{\sqrt {T}} + \frac {L _ {1} \sigma^ {2} \sqrt {2 L}}{L _ {0} \sqrt {T}}.
$$

Defining $\tau : = \arg \operatorname* { m i n } _ { t } f ( \pmb { x } _ { t } )$ , we get 

$$
\sqrt {f (\boldsymbol {x} _ {\tau}) - f ^ {\star}} \leq \sqrt {\frac {4 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2} + \sigma^ {2}}{\sqrt {T}}} + \frac {4 L _ {1} \sqrt {2 L} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2}}{\sqrt {T}} + \frac {L _ {1} \sigma^ {2} \sqrt {2 L}}{L _ {0} \sqrt {T}}.
$$

Squaring the both sides, and using $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ for all $a , b \in \mathbb { R }$ , we obtain 

$$
f (\boldsymbol {x} _ {\tau}) - f ^ {\star} \leq \frac {8 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2} + 2 \sigma^ {2}}{\sqrt {T}} + \frac {1 2 8 L _ {1} ^ {2} L \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {4}}{T} + \frac {8 L _ {1} ^ {2} \sigma^ {4} L}{L _ {0} ^ {2} T}.
$$

This concludes the statement. 

Lemma 5. Assume that f is convex and Assumptions 1 and 2 hold. Let T be the number ofiterations and define $\tau : = \arg \operatorname* { m i n } _ { 0 \leq t \leq T - 1 } f ( \pmb { x } _ { t } )$ . Then, gradient descent with stepsize Eq. (14) satisfies: 

$$
f (\boldsymbol {x} _ {\tau}) - f (\boldsymbol {x} ^ {\star}) \leq \mathcal {O} \left(\frac {L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2} + \sigma^ {2}}{\sqrt {T}} + \frac {L L _ {1} ^ {2} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {4}}{T} + \frac {L _ {1} ^ {2} L \sigma^ {4}}{L _ {0} ^ {2} T}\right),\tag{21}
$$

where $\boldsymbol { x } ^ { \star } : =$ arg min<sub>x</sub> f(x) and $\sigma ^ { 2 } : = f ^ { \star } - l ^ { \star }$ 

Proof. If there exists t such that $\begin{array} { r } { f ( \pmb { x } _ { t } ) - f ^ { \star } < \frac { \sigma ^ { 2 } } { \sqrt { T } } } \end{array}$ , we have 

$$
f (\pmb {x} _ {\tau}) - f ^ {\star} \leq f (\pmb {x} _ {t}) - f ^ {\star} <   \frac {\sigma^ {2}}{\sqrt {T}}.
$$

Then, if $\begin{array} { r } { f ( \pmb { x } _ { t } ) - f ^ { \star } \geq \frac { \sigma ^ { 2 } } { \sqrt { T } } } \end{array}$ for all t, Lemma 4 shows that 

$$
f (\boldsymbol {x} _ {\tau}) - f ^ {\star} \leq \frac {8 L _ {0} \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {2} + 2 \sigma^ {2}}{\sqrt {T}} + \frac {1 2 8 L _ {1} ^ {2} L \| \boldsymbol {x} _ {0} - \boldsymbol {x} ^ {\star} \| ^ {4}}{T} + \frac {8 L _ {1} ^ {2} \sigma^ {4} L}{L _ {0} ^ {2} T}.
$$

By combining the above two cases, we have the desired statement. 

## C Additional theoretical result

Lemma 6. Let f be afunction such that $\| \nabla ^ { 2 } f ( \pmb { x } ) \| \leq L _ { 0 } + L _ { 1 } \| \nabla f ( \pmb { x } ) \|$ holdsfor any x. For any x, y such that $\begin{array} { r } { \| \dot { \pmb { x } } - \pmb { y } \| \le \frac { 1 } { L _ { 0 } } } \end{array}$ , we have 

$$
\left\| \nabla f (\boldsymbol {x}) - \nabla f (\boldsymbol {y}) \right\| \leq 2 (L _ {0} + L _ {1} \| \nabla f (\boldsymbol {x}) \|) \| \boldsymbol {x} - \boldsymbol {y} \|.
$$

Proof. See Lemma A.2 in (Zhang et al., 2020a). 

Proposition 3. For any $L _ { 0 } \geq 0$ and $\begin{array} { r } { L _ { 1 } \ge 0 , f ( x ) : = \frac { L _ { 0 } L _ { 1 } ^ { 2 } } { 7 2 } x ^ { 4 } + \frac { L _ { 0 } } { 4 } x ^ { 2 } \mathrm { ~ } i s \left( L _ { 0 } , L _ { 1 } \right) \cdot s m o o t h } \end{array}$ 

Proof. Since f(x) is twice differentiable, we have 

$$
| \nabla^ {2} f (x) | = \frac {L _ {0} L _ {1} ^ {2}}{6} x ^ {2} + \frac {L _ {0}}{2}.
$$

Using $\begin{array} { r } { \frac { L _ { 1 } } { 6 } x ^ { 2 } + \frac { 3 } { 2 L _ { 1 } } \geq | x | } \end{array}$ , we obtain 

$$
\begin{array}{c} | \nabla^ {2} f (x) | \leq \frac {L _ {0} L _ {1} ^ {2}}{6} \left(\frac {L _ {1}}{6} x ^ {2} + \frac {3}{2 L _ {1}}\right) | x | + L _ {0} \\ = \frac {L _ {1}}{2} \left| \frac {L _ {0} L _ {1} ^ {2}}{1 8} x ^ {3} + \frac {L _ {0}}{2} x \right| + \frac {L _ {0}}{2} \\ = \frac {L _ {1}}{2} | \nabla f (x) | + \frac {L _ {0}}{2}. \end{array}
$$

From Lemma 6, we have the desired statement. 

## D Hyperparameter settings

## D.1 Synthetic function

In our experiments, we ran the clipped gradient descent with the following hyperparameters and tuned the hyperparameters by grid search. 


Table 2: Hyperparameter settings for clipped gradient descent.


<table><tr><td>Learning Rate</td><td><eq>\{1, 1.0 \times 10^{-1}, \cdots, 1.0 \times 10^{-8}\}</eq></td></tr><tr><td>Gradient Clipping Threshold</td><td><eq>\{0.01, 0.1, 1, 5, 10, 15, 20, \infty\}</eq></td></tr></table>


Table 3: Hyperparameters selected by grid search.


<table><tr><td rowspan="2"></td><td rowspan="2">Gradient Descent Learning Rate</td><td colspan="2">Clipped Gradient Descent</td></tr><tr><td>Learning Rate</td><td>Gradient Clipping Threshold</td></tr><tr><td><eq>L_1 = 1</eq></td><td><eq>1.0 \times 10^{-1}</eq></td><td>0.1</td><td>20</td></tr><tr><td><eq>L_1 = 10</eq></td><td><eq>1.0 \times 10^{-3}</eq></td><td>0.1</td><td>10</td></tr><tr><td><eq>L_1 = 100</eq></td><td><eq>1.0 \times 10^{-5}</eq></td><td>0.1</td><td>10</td></tr><tr><td><eq>L_1 = 1000</eq></td><td><eq>1.0 \times 10^{-7}</eq></td><td>0.1</td><td>10</td></tr></table>

## D.2 Neural networks

In our experiments, we used the following training configuration: 

• LSTM: https://github.com/salesforce/awd-lstm-lm 

• Nano-GPT: https://github.com/karpathy/nanoGPT 

• T5: https://github.com/PiotrNawrot/nanoT5 

We ran all experiments on an A100 GPU. For Clipped SGD and SGD, we tuned the stepsize and gradient clipping threshold using the grid search. See Tables 4, 5, and 6 for detailed hyperparameter settings, and see Table 7 for the selected hyperparameters. 


Table 4: Hyperparameter settings for LSTM.


<table><tr><td>Learning Rate</td><td>{100, 50, 10, 1, 0.1, 0.01}</td></tr><tr><td>Gradient Clipping Threshold</td><td>{0.5, 1,···,4.5, 5,∞}</td></tr><tr><td>Batch Size</td><td>80</td></tr></table>


Table 5: Hyperparameter settings for Nano-GPT.


<table><tr><td>Learning Rate</td><td>{1,0.5,0.1,···,0.0005,0.0001}</td></tr><tr><td>Gradient Clipping Threshold</td><td>{1,2,···,9,10,∞}</td></tr><tr><td>Batch Size</td><td>64</td></tr></table>

<table><tr><td colspan="2">Table 6: Hyperparameter settings for T5.</td></tr><tr><td>Learning Rate</td><td>{5.0, 1.0, 0.5, 0.1, 0.05}</td></tr><tr><td>Gradient Clipping Threshold</td><td>{1, 2, 3,∞}</td></tr><tr><td>Batch Size</td><td>128</td></tr></table>


Table 7: Hyperparameters selected by grid search. Three values correspond to the selected hyperparameters for different seed values.


<table><tr><td rowspan="2"></td><td rowspan="2">Gradient Descent Learning Rate</td><td colspan="2">Clipped Gradient Descent</td></tr><tr><td>Learning Rate</td><td>Gradient Clipping Threshold</td></tr><tr><td>LSTM</td><td>10 / 10 / 10</td><td>10 / 50 / 50</td><td>0.5 / 1 / 0.5</td></tr><tr><td>Nano-GPT</td><td>0.001 / 0.001 / 0.001</td><td>0.001 / 0.001 / 0.001</td><td>∞ / 10 / 10</td></tr><tr><td>T5</td><td>0.1 / 0.1 / 0.05</td><td>1 / 1 / 1</td><td>2 / 2 / 2</td></tr></table>

## E Additional numerical evaluation

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/e7437904cc3e9f0f6ae631ad35761c573b64961602abd7341da89dab9f809e5f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/e75886ff857f33f6659cd1df6be6711873efeb6815efb7acfb49ebd76c933127.jpg)



(a) LSTM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/e9d6f80f12e85c062360d0ff5216d266a4d26455772eddae938ebc339aa3ffb4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/ea4e0d80f730433239c8fa8e25b8d2947cceedd5ca1e7108bc9271675a1a1bce.jpg)



(b) Nano-GPT



Figure 4: Loss curves for LSTM and Nano-GPT. We plotted the training loss per 100, 10, and 10 iterations for LSTM, Nano-GPT, and T5, respectively. We plotted the test loss per one epoch, 100 iterations, and 200 iterations, respectively.


Clipped SGD SGD DoG AdaSPS DecSPS Inexact Polyak Stepsize 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/43b0bae97307e9038de5bd5cdd06821f2cdfdc382e229f7be269eb0d8bd61031.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/4e1b8b9bb75084daffbbad0f5a2df70e2dd71322ac22bd51e1795969d877d50f.jpg)



(a) $T = 2 5 0 0$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/a4e3299b378026ad986fb93494e989baec8ac6b162385042cc6316b095faf137.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-10/9d78ec41-a695-40d9-8a6b-4b383f5b3d04/cce4d2cb2186f0cce77ad874a2c6cec6ac50edba61f95cbec2bfcd0ea8a2bf24.jpg)



Clipped SGD SGD DoG AdaSPS DecSPS Inexact Polyak Stepsize



(b) $T = 7 5 0 0$



Figure 5: Loss curves for Nano-GPT with different T.


## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? 

Answer: [Yes] 

Justification: Our main claims are clearly discussed in Sec. 1. 

Guidelines: 

• The answer NA means that the abstract and introduction do not include the claims made in the paper. 

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. 

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. 

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? 

Answer: [Yes] 

Justification: See Sec. 6.2. 

Guidelines: 

• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. 

• The authors are encouraged to create a separate "Limitations" section in their paper. 

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. 

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. 

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. 

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. 

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. 

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations. 

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? 

Answer: [Yes] 

Justification: All proofs are provided in Sec. A and B. 

Guidelines: 

• The answer NA means that the paper does not include theoretical results. 

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced. 

• All assumptions should be clearly stated or referenced in the statement of any theorems. 

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. 

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. 

• Theorems and Lemmas that the proof relies upon should be properly referenced. 

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? 

Answer: [Yes] 

Justification: All training configuration and hyperparameter setting are provided in Sec. D. Guidelines: 

• The answer NA means that the paper does not include experiments. 

• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. 

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. 

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. 

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example 

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. 

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. 

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). 

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? 

Answer: [Yes] 

Justification: Our code is contained in the supplementary material. 

## Guidelines:

• The answer NA means that paper does not include experiments requiring code. 

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details. 

• While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). 

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details. 

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. 

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. 

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). 

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? 

Answer: [Yes] 

Justification: Our training configuration is provided in Sec. D. 

Guidelines: 

• The answer NA means that the paper does not include experiments. 

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. 

• The full details can be provided either with the code, in appendix, or as supplemental material. 

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? 

Answer: [No] 

Justification: We repeated the experiments in Sec. 6.2 with three different seed values and reported the average, while we did not report error bars to make the figure easy to read. 

Guidelines: 

• The answer NA means that the paper does not include experiments. 

• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. 

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). 

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) 

• The assumptions made should be given (e.g., Normally distributed errors). 

• It should be clear whether the error bar is the standard deviation or the standard error of the mean. 

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. 

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). 

• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? 

Answer: [Yes] 

Justification: See Sec. D. 

Guidelines: 

• The answer NA means that the paper does not include experiments. 

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. 

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. 

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper). 

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? 

Answer: [Yes] 

Justification: The authors read and complied with the code of ethics. 

Guidelines: 

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 

• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. 

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? 

Answer: [Yes] 

Justification: The motivation and its impact of our study are clearly discussed in Sec. 1. 

Guidelines: 

• The answer NA means that there is no societal impact of the work performed. 

• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. 

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. 

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. 

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. 

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? 

Answer: [NA] 

Justification: Our study does not provide any new dataset or pre-trained models. 

Guidelines: 

• The answer NA means that the paper poses no such risks. 

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. 

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. 

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? 

Answer: [Yes] 

Justification: See Sec. D. 

Guidelines: 

• The answer NA means that the paper does not use existing assets. 

• The authors should cite the original paper that produced the code package or dataset. 

• The authors should state which version of the asset is used and, if possible, include a URL. 

• The name of the license (e.g., CC-BY 4.0) should be included for each asset. 

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. 

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. 

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. 

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators. 

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? 

Answer: [Yes] 

Justification: Our code is provided in the supplementary material with MIT license. 

Guidelines: 

• The answer NA means that the paper does not release new assets. 

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. 

• The paper should discuss whether and how consent was obtained from people whose asset is used. 

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? 

Answer: [NA] 

Justification: Our experiments do not use any crowdsourcing service. 

Guidelines: 

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. 

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? 

Answer: [NA] 

Justification: Our experiments do not use any crowdsourcing service. 

Guidelines: 

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. 

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. 

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. 

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 