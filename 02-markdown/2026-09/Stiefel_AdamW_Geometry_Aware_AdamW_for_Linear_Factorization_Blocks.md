# STIEFEL-ADAMW: GEOMETRY-AWARE ADAMW FOR LINEAR FACTORIZATION BLOCKS

EMANUELE ZANGRANDO<sup>∗</sup> , MARCO SUTTI<sup>†</sup> AND FRANCESCO TUDISCO<sup>∗‡</sup> 

## Abstract

A pervasive structural pattern in modern deep learning is the linear factorization block: a submodule of the form W = BA in which two parameter matrices are multiplied directly, with no intervening nonlinearity. Such blocks appear in LoRA adapters, low-rank compressed layers, query-key products of self-attention, and share a common pathology: the factorization is non-unique, which can destabilize training and limit usable learning rates. Despite this, factorization blocks are typically optimized with standard Euclidean methods that ignore the underlying geometry. We introduce Stiefel-AdamW, a near drop-in replacement for AdamW for use wherever such blocks appear. By constraining one factor on the Stiefel manifold while leaving the other Euclidean, Stiefel-AdamW relaxes the full $\operatorname{GL}(\mathbb{R}^r)$ gauge symmetry to a compact orthogonal symmetry, ruling out factor blow-up while retaining the coordinate-wise diagonal preconditioning that gives AdamW its practical strength. Moment estimation is performed in the ambient Euclidean space, with geometry entering only through a tangent-space projection and a manifold retraction. The implementation overhead over AdamW is minimal, and we show that the resulting optimizer inherits both the stability benefits of Riemannian methods and standard convergence guarantees. We validate Stiefel-AdamW on LoRA-style fine-tuning of GPT2, ViT, and Mistral 7B and on full pretraining of $\mathrm { G P T 2 }$ on OpenWebText, showing consistent improvements over strong baselines at essentially no additional cost over AdamW. 

## 1. Introduction

A pervasive structural pattern in modern deep-learning architectures is the presence of linear factorization blocks: trainable submodules of the form $W = B A$ , in which two parameter matrices are multiplied directly, with no intervening nonlinearity. Such blocks arise in many guises across the modern model zoo. In LoRA-style parameter-eficient fine-tuning [HSW<sup>+</sup>22, ZCB<sup>+</sup>23, HGY24, ZP24, ZZC<sup>+</sup>24, LMSR24, SZC<sup>+</sup>25], weight correctors are parametrized as $\varDelta W = B A$ , with $B \in \mathbb { R } ^ { m \times r }$ and $A \in \mathbb { R } ^ { r \times n }$ . In low-rank pretraining [WAP21, KTMF21, SKK25] and network compression [VKJ19, SSP23, MHP25, SZK<sup>+</sup>22], weight matrices are similarly factorized into two trainable factors. The same structure also appears inside standard architectures: in self-attention $[ \mathrm { V S P ^ { + } 1 7 } ]$ , the score matrix $W _ { Q } W _ { K } ^ { \top }$ factorizes through the rank-r key/query head dimension, and analogous matrix-matrix factorizations appear in the recurrence kernels of several structured state-space models (SSMs). In all these cases, two parameter blocks combine multiplicatively, without an intervening nonlinearity, to produce a single efective linear operator. 

These blocks share a common geometric feature: the product map $\Phi ( A , B ) = B A$ is highly non-unique. For any $M \in \operatorname { G L } ( \mathbb { R } ^ { r } ) , \Phi ( A , B ) = \Phi ( M ^ { - 1 } A , B M )$ , so each efective weight W corresponds to a continuous family of parameter pairs. This gauge symmetry has direct consequences for optimization. If the loss L, viewed as a function of the efective matrix W, has a stationary point, then $\mathcal { L } \circ$ Φ has an entire orbit of equivalent stationary points parametrized by $\operatorname{GL}(\mathbb{R}^r)$. More worryingly, training along these blocks can be numerically unstable even without spurious minimizers: a sequence $\left( M _ { n } ^ { - 1 } A _ { n } , B _ { n } M _ { n } \right)$ with $M _ { n } \to 0$ produces a bounded product $W _ { n }$ even though one factor diverges. These issues are well documented in practice [MMBS14, SZK<sup>+</sup>22, ZSK<sup>+</sup>24] and become especially pronounced under unbalanced initializations, such as those typical of LoRA. 

Despite the prevalence of this structure, standard practice for training linear factorization blocks is to apply Euclidean optimizers, most commonly AdamW [LH19], directly to the unconstrained pair (A, B), ignoring the underlying geometry entirely. 

The natural geometric remedy is to optimize on the fixed-rank matrix manifold 

$$
\mathcal {M} _ {r} = \mathbb {R} _ {*} ^ {r \times n} \times \mathbb {R} _ {*} ^ {m \times r} / \mathrm{GL} (\mathbb {R} ^ {r}),
$$

viewed as a quotient [MMBS14]. Here, $\mathbb { R } _ { * } ^ { m \times r }$ denotes the set of full-rank matrices of size $m \times r$ . While geometrically clean, this approach is at odds with adaptive optimizers such as Adam, since the full $\operatorname { G L } ( \mathbb { R } ^ { r } )$ invariance forces nonlinear adaptive updates to satisfy strong equivariance constraints that, in general, cannot be reconciled with the coordinate-wise diagonal preconditioning that gives AdamW much of its practical strength. As a result, prior Riemannian adaptive methods either give up coordinate-wise adaptivity in favor of scalar preconditioners, or modify the gradient structure rather than the preconditioner itself [BG19, SI25, SKK25, BZL<sup>+</sup>25]. 

In this work, we adopt a diferent trade-of. Rather than enforcing full quotient invariance, we relax it to an orthogonal symmetry by restricting to the product manifold $\mathrm { S t } ( n , r ) \times \mathbb { R } _ { * } ^ { m \times r }$ on which Φ is invariant only under $\mathrm { O } ( r )$ 

$$
\Phi (A, B) = \Phi (Q ^ {\top} A, B Q), \quad \forall Q \in \mathrm{O} (r).
$$

This relaxation accomplishes two things at once. Geometrically, it makes the fibers of Φ compact, which rules out the kind of factor blow-up illustrated above and stabilizes training under aggressive learning rates. Algorithmically, because the constrained factor lives on an embedded submanifold and the other factor in a flat Euclidean space, we can perform all moment accumulation in the ambient space and only project onto the tangent space of the Stiefel manifold immediately before the update. This preserves AdamW’s coordinate-wise diagonal preconditioning essentially unchanged. 

The resulting algorithm, Stiefel-AdamW, is best understood as a small, structural modification of AdamW. The Euclidean factor is updated by the standard AdamW step; the Stiefel factor is updated by computing the first and second moments in the ambient space, projecting the preconditioned direction onto the tangent space, and retracting back to the manifold. The retraction step is treated as a modular component: any eficient retraction on the Stiefel manifold can be used, including the Cayley transform, QR-based retraction, polar decomposition, or Newton–Schulz iteration [Kov70, BB71, Hig08]. In our experiments, the choice of retraction has only a marginal efect on final performance, with the best option mildly modeland problem-dependent; see Appendix B.1 and Table 3 for a comparison. The overall implementation overhead over standard AdamW is minimal, yet the resulting optimizer inherits the principal benefits of geometric optimization: numerical stability, parameter invariance to orthogonal reparametrization, low memory footprint, and provable convergence under standard assumptions. 

We emphasize that Stiefel-AdamW is not a method specific to LoRA or to any particular architectural family. It is a simple, near drop-in replacement for AdamW that can (and should) 

be used on any factorization block of the form W = BA in which the two factors are not separated by a nonlinearity, regardless of where such a block appears in the model. This includes LoRA adapters, low-rank compressed layers, and query/key projections in attention blocks. In all these cases, applying Stiefel-AdamW rather than vanilla AdamW replaces a Euclidean parametrization with hidden gauge symmetry by one in which (most of) the symmetry has been quotiented out, and does so essentially for free. 

Contributions. Our contributions are as follows. 

1. We identify factorization blocks W = BA ubiquitous in modern deep-learning architectures (LoRA, low-rank compressed layers, attention, SSMs) as a unifying setting in which geometric optimization can be applied transparently, and we propose Stiefel-AdamW, a near drop-in replacement for AdamW that should be used wherever such blocks appear. 

2. We propose a structured relaxation of the full $\operatorname{GL}(\mathbb{R}^r)$ quotient invariance to orthogonal invariance, working on the product manifold $\mathrm { S t } ( n , r ) \times \mathbb { R } _ { * } ^ { m \times r }$ . This is precisely the relaxation that allows full coordinate-wise Adam-style adaptive preconditioning to coexist with geometry-aware updates, in contrast to prior Riemannian Adam-type methods that rely on simple scalar preconditioners. 

3. We design the algorithm so that all adaptive moment estimation is performed in ambient Euclidean space, with geometric corrections (tangent projection and retraction) applied only at the update step. As a consequence, the per-step cost and memory footprint are essentially those of AdamW. 

4. We treat the Stiefel retraction as a modular, plug-and-play component: any eficient retraction (Cayley transform, QR, polar decomposition, Newton–Schulz) can be used, with only marginal diferences in practice. Approximate retractions, when second-order accurate, remain compatible with the standard convergence theory. 

5. We establish theoretical guarantees of boundedness of gradients and convergence of the regret function under standard assumptions. 

6. We validate Stiefel-AdamW empirically across a range of representative tasks, from LoRA-style fine-tuning (GPT2, ViT, Mistral 7B) to full LLM pretraining (GPT2 on OpenWebText), demonstrating that the additional cost over AdamW is minimal while consistently improving stability and final performance. 

## 2. Related Work

Riemannian Optimization. Riemannian gradient methods on embedded and quotient manifolds, including the Stiefel and Grassmann manifolds, have a long history in numerical optimization [Lue72, GL76, EAS98, Yan07], and have since been extended to trust-region [ABG07], quasi-Newton [RW12], and conjugate-gradient [SI15, Sat16, Sat22] methods. Key tools including retraction mappings [AM12, AO15], eficient preconditioners [VV10, BA15], and quotient geometries [MMBS14] have been thoroughly developed; comprehensive treatments can be found in [AMS08, Sat21, Bou23]. 

Adaptive Methods and their Riemannian Extensions. In deep learning, adaptive optimizers such as AdaGrad [DHS11], RMSProp [HSS12], Adam [KB15], AMSGrad [RKK18], and AdamW [LH19], are the de facto standard, combining fast convergence, robustness, and low memory overhead. Extending them to manifold-constrained settings is nontrivial, because adaptive moments and preconditioning must respect the underlying geometry. Stochastic Riemannian optimization began with Riemannian SGD [Bon13] and variance-reduced variants [ZJRS16, KSM18]. More recent work has proposed Riemannian AdaGrad and AMS-Grad [BG19], modified AMSGrad schemes [SI22], RASA [KJM19], and Riemannian adaptive gradient methods with theoretical guarantees [BZL<sup>+</sup>25, SI25]. A common limitation of these approaches is that they rely on scalar or geometry-compatible preconditioners; the fully coordinate-wise diagonal preconditioning central to AdamW is generally incompatible with strict manifold invariance, which is the gap Stiefel-AdamW is designed to close. 

Linear Factorization Blocks in Deep Learning. Matrix factorization blocks of the form $W = B A$ , in which two parameter matrices multiply directly without an intervening nonlinearity, appear throughout modern architectures. They arise explicitly in LoRA-style parameter-eficient fine-tuning [HSW<sup>+</sup>22] and its variants [HGY24, ZP24, ZWGC24, WLH<sup>+</sup>25, $\mathrm { S Z C ^ { + } 2 5 ] }$ , in network compression [VKJ19, SSP23, $\mathrm { S Z K ^ { + } 2 2 } |$ , and implicitly inside standard architectures such as the query-key product in self-attention $[ \mathrm { V S P ^ { + } 1 7 } ]$ . The non-uniqueness of such factorizations and its consequences for training stability have been studied through dynamical low-rank approximation [KL07, HKK<sup>+</sup>26, ZSK<sup>+</sup>24] and quotient-manifold optimization [MMBS14]. Methods such as GeoLoRA [SZC<sup>+</sup>25] have explicitly exploited Grassmannian geometry in low-rank adaptation, and recent work has combined momentum and adaptivity with Riemannian updates [SKK25, SI25]. 

Relation to Prior Work. The closest methods to Stiefel-AdamW are GeoLoRA $\mathrm { [ S Z C ^ { + } 2 5 ] }$ and RAdam [BG19] or its Stiefel-specific version, Cayley Adam [LLT20]. GeoLoRA enforces geometry on the full factorization via a quotient-manifold formulation, which precludes coordinate-wise adaptive preconditioning. RAdam defines a Riemannian Adam on the Stiefel manifold but uses a scalar preconditioner, departing from AdamW’s diagonal adaptivity. In contrast, Stiefel-AdamW constrains only one factor to the Stiefel manifold while leaving the other in Euclidean space. This product-manifold structure is precisely what makes full coordinate-wise adaptive preconditioning tractable: moment accumulation is performed in the ambient space for both factors, and geometry enters only through one factor via a tangentspace projection followed by a retraction on the Stiefel manifold. The result is an optimizer that inherits the stability benefits of Riemannian methods and the practical performance of AdamW, at minimal additional cost. 

## 3. The Proposed Method: Stiefel-AdamW

3.1. Problem Setup. We consider a trainable linear factorization block of the form $W =$ BA, where $B \in \mathbb { R } ^ { m \times r }$ and $A \in \mathbb { R } ^ { r \times n }$ . In order to avoid potential instabilities due to nonuniqueness of this representation, we impose a row-orthonormality constraint on $A .$ , namely $A A ^ { \top } = I _ { r }$ , i.e., we require $A ^ { \top }$ to lie on the Stiefel manifold $\operatorname { S t } ( n , r ) = \{ X \in \mathbb { R } ^ { n \times r } \colon X ^ { \top } X = I _ { r } \}$}, while leaving B unconstrained. This restriction reduces the invariance group from $\operatorname { G L } ( \mathbb { R } ^ { r } )$ to the compact group $\mathrm { O } ( r )$ , making the fibers of $\Phi$ compact and ruling out the factor blow-up illustrated in Section 1; see also Section 4.2 for further details. The resulting product-manifold structure $\mathrm { S t } ( n , r ) \times \mathbb { R } ^ { m \times r }$ enables an eficient fully coordinate-wise adaptive update: moment accumulation is performed in the ambient Euclidean space for both factors, with geometric corrections entering only through a tangent-space projection and a retraction onto the Stiefel manifold for the A factor. 

Algorithm 1 Single iteration of Stiefel-AdamW.

Require: $A_{t}$ with $A_{t}A_{t}^{\top}=I_{r}, B_{t}, M_{t-1}^{A}, M_{t-1}^{B}, V_{t-1}^{A}, V_{t-1}^{B}, \eta_{t}, \beta_{1}, \beta_{2}, \varepsilon$

1: $G_{t}^{B} \leftarrow \nabla_{B}\mathcal{L}(B_{t}A_{t}) A_{t}^{\top}$ ▷ Euclidean gradient w.r.t. B

2: $G_{t}^{A} \leftarrow B_{t}^{\top}\nabla_{A}\mathcal{L}(B_{t}A_{t})$ ▷ Euclidean gradient w.r.t. A

3: $M_{t}^{B} \leftarrow \beta_{1}M_{t-1}^{B} + (1 - \beta_{1}) G_{t}^{B}$ ▷ First moment, B

4: $M_{t}^{A} \leftarrow \beta_{1}M_{t-1}^{A} + (1 - \beta_{1}) G_{t}^{A}$ ▷ First moment, A

5: $V_{t}^{B} \leftarrow \beta_{2}V_{t-1}^{B} + (1 - \beta_{2})(G_{t}^{B})^{\circ2}$ ▷ Second moment, B

6: $V_{t}^{A} \leftarrow \beta_{2}V_{t-1}^{A} + (1 - \beta_{2})(G_{t}^{A})^{\circ2}$ ▷ Second moment, A

7: $B_{t+1} \leftarrow B_{t} - \eta_{t}\left(M_{t}^{B}/\left(\sqrt{V_{t}^{B} + \varepsilon}\right) + \lambda B_{t}\right)$ ▷ Standard AdamW step on B

8: $X_{t} \leftarrow A_{t}^{\top}$ ▷ Column convention, $X_{t} \in \mathrm{St}(n, r)$

9: $D_{t} \leftarrow M_{t}^{A}/(\sqrt{V_{t}^{A} + \varepsilon})$ ▷ Preconditioned direction

10: $\xi_{t} \leftarrow -\eta_{t}\mathrm{P}_{X_{t}}(D_{t})$ ▷ Project onto $T_{X_{t}}\mathrm{St}(n,r), P_{X}(Z) = X\operatorname{skew}(X^{\top}Z) + (I - XX^{\top})Z$

11: $X_{t+1} \leftarrow \mathrm{Retr}_{X_t}(\xi_t), A_{t+1} \leftarrow X_{t+1}^{\top}$ ▷ Retract to $\mathrm{St}(n,r)$ ; see Section 3.3

3.2. Description of the Algorithm. In this section, we describe one iteration of Stiefel-AdamW; the pseudocode is given in Algorithm 1. For simplicity of exposition, we omit bias correction and explicit weight decay; both can be incorporated straightforwardly as in AdamW. 

Let $W _ { t } \in \mathbb { R } ^ { m \times n }$ be a weight matrix at iteration t, represented by the factorization $W _ { t } =$ $B _ { t } A _ { t } ,$ where $A _ { t } \in \mathbb { R } ^ { r \times n }$ and $B _ { t } \in \mathbb { R } ^ { m \times r }$ . The objective function is evaluated as $\mathcal { L } ( B _ { t } A _ { t } )$ . Next, the algorithm computes the Euclidean gradients $G _ { t } ^ { B } = \nabla _ { B } \mathcal { L } ( B _ { t } A _ { t } )$ and $G _ { t } ^ { A } = \nabla _ { A } \mathcal { L } ( B _ { t } A _ { t } )$ As in Adam and AdamW, Stiefel-AdamW then computes the first and second moments for both factors, i.e., 

$$
\left\{ \begin{array}{l} M _ {t} ^ {B} = \beta_ {1} M _ {t - 1} ^ {B} + (1 - \beta_ {1}) G _ {t} ^ {B}, \\ M _ {t} ^ {A} = \beta_ {1} M _ {t - 1} ^ {A} + (1 - \beta_ {1}) G _ {t} ^ {A}, \end{array} \right. \quad \left\{ \begin{array}{l} V _ {t} ^ {B} = \beta_ {2} V _ {t - 1} ^ {B} + (1 - \beta_ {2}) (G _ {t} ^ {B}) ^ {\circ 2}, \\ V _ {t} ^ {A} = \beta_ {2} V _ {t - 1} ^ {A} + (1 - \beta_ {2}) (G _ {t} ^ {A}) ^ {\circ 2}, \end{array} \right.
$$

where $^ { \hphantom { 0 } 2 }$ denotes elementwise squaring. 

Up to this step, both factors are treated in the same way; however, the subsequent update steps do difer. Indeed, since $B _ { t }$ is unconstrained, the algorithm performs the usual AdamW update, i.e., $B _ { t + 1 } = B _ { t } - \eta _ { t } \Big ( M _ { t } ^ { B } / ( \sqrt { V _ { t } ^ { B } + \varepsilon } ) + \lambda B _ { t } \Big )$ , where $\eta _ { t }$ is the learning rate, / indicates elementwise division, the square root is also meant to be performed componentwise, and the $\varepsilon > 0$ is a small constant to avoid blowup of the metric. This factor requires no Riemannian machinery, which keeps the method simple and eficient. 

For the orthonormal factor, we employ a retraction-based Riemannian update. For convenience, we switch to a column-orthonormal representation by defining $X _ { t } : = A _ { t } ^ { \top } \in \mathbb { R } ^ { n \times r }$ , so that $X _ { t } \in \mathrm { S t } ( n , r ) , \mathrm { i . e . , } X _ { t } ^ { \top } X _ { t } = I _ { r }$ . A key feature of Stiefel-AdamW is that it performs adaptive moment estimation in the ambient Euclidean space before projecting onto the tangent space. This allows us to use coordinate-wise preconditioning, as in AdamW. In contrast, many existing Riemannian adaptive methods restrict the preconditioner to be scalar or geometrycompatible to preserve invariance, thereby limiting their practical efectiveness. More precisely, the Euclidean adaptive direction is first formed as $D _ { t } = M _ { t } ^ { A } / ( \sqrt { V _ { t } ^ { A } + \varepsilon } )$ , and then projected onto the tangent space $\mathrm { T } _ { X _ { t } } \mathrm { S t } ( n , r )$ to obtain the direction $\xi _ { t } = - \eta _ { t } \mathrm { P } _ { X _ { t } } ( D _ { t } )$ , where $\mathrm { P } _ { X _ { t } } \colon \mathbb { R } ^ { n \times r }  \mathrm { T } _ { X _ { t } } \mathrm { S t } ( n , r )$ is the orthogonal projection onto the tangent space to $\operatorname { S t } ( n , r )$ at 

$X _ { t } , \operatorname* { P } _ { X } ( Z ) = X { \mathrm { s k e w } } ( X ^ { \top } Z ) + \left( I - X X ^ { \top } \right) Z ,$ , with skew $( M ) = ( M - M ^ { \top } ) / 2$ . See Appendix A.1 for more details on the geometry of the Stiefel manifold. 

3.3. Choice of Retraction. To map a tangent vector $\xi _ { t }$ back onto the manifold, we need to apply a retraction mapping, $X _ { t + 1 } = \operatorname { R e t r } _ { X _ { t } } ( \xi _ { t } )$ . While the exponential map provides the most geometrically accurate geodesic path, it is often computationally prohibitive for large-scale problems because it requires full eigenvalue decompositions or matrix exponentials [AMS08]. In practice, a retraction is any mapping that agrees to first order with the exponential map $( \mathrm { i . e . , }$ is centered at the point and has the diferential at the origin equal to the identity map). 

For the Stiefel manifold $\operatorname { S t } ( n , r )$ , several eficient retractions exist with a computational complexity of $O ( n r ^ { 2 } + r ^ { 3 } )$ , which is ideal for settings where $r \ll n \colon$ 

• QR Decomposition: A standard choice that performs a QR factorization of $X _ { t } + D _ { t }$ and extracts the orthogonal factor $Q \ [ \mathrm { A M S 0 8 }$ , (4.8)]. 

• Polar Decomposition: Maps the tangent vector to the manifold by finding the closest orthogonal matrix in the Frobenius norm, typically implemented via iterative Newton– Schulz methods [ZS20]. 

• Cayley Transform: An algebraic alternative using a skew-symmetric mapping. When implemented with the Sherman–Morrison–Woodbury (SMW) identity, it avoids large matrix inversions, reducing the cost to a $2 r \times 2 r$ system [WY13, §2.2]. It can also be computed implicitly via the fixed-point iteration $\begin{array} { r } { Y _ { k + 1 } = X + \frac { \alpha } { 2 } \varOmega \bigl ( X + Y _ { k } \bigr ) } \end{array}$ , which converges quadratically as $o ( \alpha ^ { 2 + k } )$ ). 

In Section 3.3 and in the right part of Table 3, we present numerical results comparing different kinds of retractions. In the remaining numerical experiments, we use the Cayley retraction as the standard choice, approximated via fixed-point iteration, because of its simplicity of implementation and good performance in the comparison tests. Moreover, our analysis shows that the framework is robust to approximate retractions: as established in Theorem 4.2, the introduction of a maximal Frobenius error δ in the retraction mapping merely adds a manageable linear term to the regret bound. This theoretical guarantee justifies using truncated or iterative retraction methods that can run for only a few iterations without reaching machine precision, while ofering significant speedups. 

## 4. Theoretical Guarantees

4.1. Regret Analysis. In this section, we present a convex regret analysis for Algorithm 1. Regret analysis is a standard tool in convex optimization that quantifies how much an optimization algorithm, when running dynamically on a family of convex objective functions $\mathcal { L } _ { t }$ , is suboptimal with respect to the optimal objective ahead of time. In particular, given a sequence of iterates $\{ W _ { t } \} _ { t = 1 , \dots , T }$ , we define the regret function as 

$$
R (T) := \sum_ {t = 1} ^ {T} \mathcal {L} _ {t} (W _ {t}) - \min _ {W} \sum_ {t = 1} ^ {T} \mathcal {L} _ {t} (W).
$$

We recall that an algorithm is said to be zero regret if $R ( T ) / T \to 0 { \mathrm { ~ a s ~ } } T \to + \infty$ . Our aim is to show that Algorithm 1 indeed produces arbitrarily small regret for a small enough learning rate and retraction error. To prove this result, we will make the following assumptions: 

Assumptions 4.1 (Setting of regret analysis). 

(H1) The family $\mathcal { L } _ { t } \colon \mathbb { R } ^ { m \times r } \times \mathrm { S t } ( n , r ) \to \mathbb { R }$ is the restriction of a family of Euclidean strictly convex functions defined on $\mathbb { R } ^ { r \times m } \times \mathbb { R } ^ { r \times n }$ . By a small abuse of notation, we will also denote the extension family with $\mathcal { L } _ { t }$ , and we will denote the minimizer with $( B ^ { * } , A ^ { * } )$ 

(H2) The first momentum coeficients $\beta _ { 1 , t } = \beta _ { 1 } b ^ { t }$ decrease geometrically in time for a constant $0 < b < 1$ , and with $\beta _ { 1 } < \sqrt { \beta _ { 2 } }$ 

(H3) The iterates $B _ { t }$ stay bounded, i.e., $\operatorname* { s u p } _ { t } \| B _ { t } \| _ { \operatorname* { m a x } } \leq D _ { \infty }$ 

(H4) The Euclidean gradient $\nabla { \mathcal { L } } _ { t } ( B _ { t } , A _ { t } )$ stays bounded, i.e., $\operatorname* { s u p } _ { t } \| \nabla \mathcal L _ { t } ( B _ { t } , A _ { t } ) \| _ { \operatorname* { m a x } } \leq G _ { \infty }$ 

(H5) The second momentum update in Algorithm 1 is followed by an entrywise maximum, i.e., $V _ { t + 1 } = \operatorname* { m a x } ( \beta _ { 2 } V _ { t - 1 } + \left( 1 - \beta _ { 2 } \right) G _ { t } ^ { 2 } , \ V _ { t - 1 } )$ , as in AMSGrad [RKK18, Algorithm 2]. 

(H6) The projected direction is aligned with the globally correct direction, $\langle \xi _ { t } , A ^ { * } - A _ { t } \rangle \geq 0$ 

We emphasize that Assumptions 4.1 are fairly standard assumptions used to study convergence of Adam-like algorithms, and they were already employed in, e.g., [RKK18]. Assumption (H6) is a hypothesis often used in Euclidean optimizers to ensure that the current local descent direction is aligned with the global direction to the minimizer. 

Theorem 4.2. (Regret bound) Under Assumptions 4.1, consider the sequence of iterates produced by Algorithm 1 with decreasing learning rates $\eta _ { t } = \eta / \sqrt { t }$ , and no weight decay. Then, 

$$
R (T) \leq C _ {1} + C _ {2} \sqrt {T} + C _ {3} \sqrt {1 + \log T} + C _ {4} \log T + C _ {5} T ^ {- 1 / 2},\tag{4.1}
$$

where $C _ { 1 } , C _ { 2 } , C _ { 3 } , C _ { 4 } , C _ { 5 }$ are constants independent of T. In particular, lim $\scriptstyle { \mathsf { l } } _ { T \to + \infty } { \cal R } ( T ) / T = 0$ 

We note that, although the theoretical result requires an AMSGrad-like assumption (H5), in practice the algorithm can be used without the max update with no loss in performance. The proof of Theorem 4.2 can be found in Appendix B. 

In most practical implementations, the retraction is computed only approximately via a numerical algorithm, such as the fixed-point method used in most of our experiments; see also Appendix A.2.2. The proof of Theorem 4.2 above extends straightforwardly to that case: assuming that the computed retraction has an error of $\delta > 0$ , then a term $C _ { 6 } \delta$ has to be added to (4.1), without significantly afecting the main result of the theorem. 

4.2. Gradient Boundedness and Stability to Large Learning Rates. Working with an orthonormal factor yields a method with bounded gradients, potentially improving stability at large learning rates. Here, we make this point more concrete with an example. Let us consider the rank-r recovery problem in $\mathbb { R } ^ { n \times n }$ ， 

$$
\mathcal {L} (A, B) = \frac {1}{2} \| B A - \alpha I _ {n} \| _ {\mathrm{F}} ^ {2}, \qquad B \in \mathbb {R} ^ {n \times r}, A \in \mathbb {R} ^ {r \times n},
$$

trained by plain gradient descent from the canonical LoRA initialization $B _ { 0 } = 0$ , A arbitrary. The first GD step gives $B _ { 1 } = \eta \alpha A _ { 0 } ^ { \top }$ and $A _ { 1 } \ = \ A _ { 0 }$ , so for the scaling $\alpha = 1 / \eta$ the two factors align already after one step. Specifically, we have $B _ { 1 } = A _ { 1 } ^ { \top }$ and a direct induction argument shows that the iterates preserve $B _ { k } = A _ { k } ^ { \top }$ thereafter. Setting $X _ { k } : = B _ { k } = A _ { k } ^ { \top }$ , the dynamics take the form $X _ { k + 1 } = X _ { k } - \eta ( X _ { k } X _ { k } ^ { \top } - \eta ^ { - \tilde { 1 } } I ) X _ { k } = ( 2 I - \eta X _ { k } X _ { k } ^ { \top } ) X _ { k }$ , thus, using the SVD $X _ { k } = U _ { k } \itSigma _ { k } V _ { k } ^ { \top }$ , the dynamics decouple across singular values into the scalar recursion $\sigma _ { i , k + 1 } = \sigma _ { i , k } \big ( 2 - \eta \sigma _ { i , k } ^ { 2 } \big )$ , which diverges as soon $| 2 - \eta \sigma _ { i , k } ^ { 2 } | > 1$ , i.e., when some $\sigma _ { i , k } > \sqrt { 3 / \eta }$ The largest stable learning rate is therefore dictated by the largest singular value of the iterate, a property of the parameterization, not of the underlying optimization landscape. 

If instead the constraint $A _ { k } A _ { k } ^ { \top } = I _ { r }$ is enforced, the gradient descent update on B simplifies to the afine recursion $B _ { k + 1 } = { \ o \sp { . } { ( 1 - \eta ) } } B _ { k } + \eta \alpha A _ { k } ^ { \top }$ , which is bounded for every $0 < \eta < 2$ as $\| A _ { k } \| = 1$ , regardless of singular-value scale, removing the dependence of the stable learning rate on the iterate. This phenomenon is well documented in the Riemannian optimization literature [SKK25], and we provide a more precise result in the next Proposition 4.3, whose proof can be found in Appendix C. 

Proposition 4.3 (Gradient boundedness on fibers). Consider the maps $\Phi \colon \mathbb { R } ^ { m \times r } \times \mathbb { R } ^ { r \times n } $ $\mathbb { R } ^ { m \times n }$ given by $\Phi ( B , A ) = B A$ and the map $\widetilde { \Phi } = \Phi | _ { \mathrm { S t } ( n , r ) \times \mathbb { R } ^ { m \times r } }$ . Let $W \in \mathbb { R } ^ { m \times n }$ be a fixed matrix with rank(W) ≤ r such that $\nabla \mathcal { L } ( W ) \neq 0$ . Let $\mathcal { F } : = \Phi ^ { - 1 } ( W )$ and $\widetilde { \mathcal { F } } : = \widetilde { \Phi } ^ { - 1 } ( W )$ . Then, 

$$
\| \nabla (\mathcal {L} \circ \Phi) \| _ {L ^ {\infty} (\mathcal {F})} = + \infty , \quad \| \nabla (\mathcal {L} \circ \widetilde {\Phi}) \| _ {L ^ {\infty} (\widetilde {\mathcal {F}})} <   + \infty .
$$

In particular, it is known that the boundedness of $\nabla ( \mathcal { L } \circ \Phi )$ is closely related to the range of stable learning rates. This suggests that optimization algorithms on the parameterization $\widetilde { \Phi }$ are more stable than the ones on the more redundant representation Φ. 

## 5. Numerical Experiments

In this section, to show the efectiveness and scalability of Stiefel-AdamW, we present several numerical experiments for both fine-tuning pretrained models with LoRA adapters $\mathrm { [ H S W ^ { + } 2 2 ] }$ and LLM pretraining. We compare against six baselines: standard AdamW [LH19]; Scaled AdamW [ZP24], which introduces a coupled preconditioner accounting for the product structure; GeoLoRA [SZC<sup>+</sup>25], which uses two Stiefel representations; LoRA-RITE [YSM<sup>+</sup>25] and LoRA-Pro [WLH<sup>+</sup>25], which modify the gradient structure to reduce sensitivity to the non-uniqueness of the factorization; and Cayley Adam [LLT20], which uses the Riemannian Adam variant proposed in [BG19] on the Stiefel manifold. We emphasize that the latter method uses a scalar preconditioner, whereas Stiefel-AdamW uses a diagonal preconditioner. 

## 5.1. LoRA Fine-Tuning.

GPT2 In this experiment, we tested Stiefel-AdamW for fine-tuning GPT2 on the E2E Natural Language Generation challenge [NDR17] with LoRA of rank 4. We report the results in Table 1. In all experiments, we trained the models for 5 epochs with a batch size of 8. For details on the hyperparameter settings, see Table 5. As shown in Table 3, Stiefel-AdamW outperforms all baselines across all tasks except ROUGE-L. Interestingly, we observe that, consistently with the findings of [ZP24] for Scaled AdamW, Stiefel-AdamW achieves better performance when using more aggressive moving average parameters $\beta _ { 1 } , \beta _ { 2 }$ . This suggests that part of AdamW’s update may be spent along invariant directions, while Riemannian-informed approaches avoid this and allow greater emphasis to be placed on the current gradient direction. All numerical experiments were performed on a single NVIDIA A100 80GB, except for GPT2 pretraining, which was performed on two NVIDIA H100 80GB via Modal. 


Table 1: Fine-tuning performance with low-rank adapters. Best results highlighted in bold. Left: GPT2 on the E2E Natural Language Generation challenge, with rank = 4. AdamW and Scaled AdamW are reported from [ZP24]. Right: ViT-Base on CIFAR-10, for three diferent choices of adapter rank. We report with ±σ the standard deviation over 5 random initializations.


| Method | BLEU | NIST | MET | ROUGE-L | CIDEr | Rank 32 | Rank 64 | Rank 128 |
|---|---|---|---|---|---|---|---|---|
| AdamW [LH19] | $68.41 \pm 0.4950$ | $8.65 \pm 0.04$ | $46.38 \pm 0.12$ | $71.13 \pm 0.17$ | $2.51 \pm 0.001$ | $95.6 \pm 0.2$ | $95.55 \pm 0.15$ | $95.82 \pm 0.29$ |
| Scaled AdamW [ZP24] | $\textbf{69.17} \pm 0.43$ | $8.72 \pm 0.058$ | $46.44 \pm 0.16$ | $\textbf{71.57} \pm 0.235$ | $2.51 \pm 0.005$ | $92.25 \pm 0.28$ | $94.83 \pm 0.31$ | $95.30 \pm 0.12$ |
| Stiefel-AdamW | $69.20 \pm 0.964$ | $\textbf{8.74} \pm 0.102$ | $\textbf{46.48} \pm 0.259$ | $71.40 \pm 0.44$ | $\textbf{2.51} \pm 0.02$ | $95.91 \pm 0.19$ | $96.04 \pm 0.15$ | $96.41 \pm 0.10$ |
| GeoLoRA [SKK25] | $68.11 \pm 0.271$ | $8.60 \pm 0.07$ | $45.82 \pm 0.24$ | $70.47 \pm 0.388$ | $2.41 \pm 0.02$ | $95.53 \pm 0.44$ | $95.12 \pm 0.37$ | $95.27 \pm 0.23$ |
| LoRA-RITE [YSM+25] | $68.98 \pm 1.03$ | $8.69 \pm 0.11$ | $46.35 \pm 0.219$ | $71.17 \pm 0.410$ | $2.48 \pm 0.05$ | $94.55 \pm 0.32$ | $94.98 \pm 0.10$ | $94.89 \pm 0.24$ |
| LoRA-Pro [WLH+25] | $68.12 \pm 0.25$ | $8.61 \pm 0.06$ | $45.82 \pm 0.250$ | $70.46 \pm 0.382$ | $2.42 \pm 0.02$ | $90.17 \pm 2.30$ | $94.37 \pm 0.26$ | $94.32 \pm 0.21$ |
| Cayley Adam [LLT20] | $68.97 \pm 1.04$ | $8.69 \pm 0.11$ | $46.36 \pm 0.215$ | $71.16 \pm 0.415$ | $2.48 \pm 0.05$ | $95.44 \pm 0.92$ | $95.71 \pm 0.17$ | $96.12 \pm 0.11$ |

![image](images/Stiefel_AdamW_Geometry_Aware_AdamW_for_Linear_Factorization_Blocks/fig1.jpg)


![image](images/Stiefel_AdamW_Geometry_Aware_AdamW_for_Linear_Factorization_Blocks/fig2.jpg)


![image](images/Stiefel_AdamW_Geometry_Aware_AdamW_for_Linear_Factorization_Blocks/fig3.jpg)



Figure 1: Left and center: Loss function descent for ViT Base on CIFAR-10 LoRA finetuning (LoRA rank 64 and learning rate $1 0 ^ { - 3 } )$ . Right: Learning rate versus best training loss for diferent optimizers when fine-tuning ViT Base on CIFAR-10. In this experiment, we fixed the LoRA rank at 32 and trained all models for 50 epochs.



Vision Transformers In this experiment, we fine-tuned the base Vision Transformer from [DBK<sup>+</sup>21] on CIFAR-10 [KH09], with results shown in the right part of Table 1. In Figure 1, we compare loss descent and time per iteration against the best loss achieved. All models have been trained for 50 epochs with a batch size of 64, LoRA alpha 32, learning rate $1 0 ^ { - 3 }$ and no scheduler. For all optimizers, we used weight decay of $1 0 ^ { - 5 }$ on all adapters, applied to the key-query attention matrices, attention projection, and the last two fully connected layers. We did not optimize biases and left them as in the pretrained model. As we can observe from the results in the right panel of Table 3 and Figure 1, Stiefel-AdamW is able to outperform all baselines in terms of performance, with convergence speed comparable to that of Scaled AdamW [ZP24] and AdamW [LH19].


Mistral 7B In this experiment, we tested the efectiveness of Stiefel-AdamW for fine-tuning Mistral 7B $[ \mathrm { J } \mathrm { S } \mathrm { M } ^ { + } 2 3 ]$ on the GLUE benchmark [WSM<sup>+</sup>19] for natural language understanding, following the implementation in [ZP24]. LoRA adapters of rank 16 have been applied to all query, key, value projection, and gate matrices of multihead attention. We did not train biases, and for all optimizers, we set the LoRA alpha learning-rate scaling parameter to 16. We used mixed precision for all optimizers: the base model was loaded in its 4-bit quantized version, optimizer states were in float32, and operations were performed in mixed-precision bfloat16. We trained all models using the codebase of [ZP24], with a dropout of 0.1 and a batch size of 8 across all models. For AdamW, Scaled AdamW, and GeoLoRA, we used the optimal hyperparameters (learning rate and $\beta _ { 1 } , \beta _ { 2 } )$ from [ZP24]; for Stiefel-AdamW, we used $\beta _ { 1 } = \beta _ { 2 } = 0 . 9 5$ . For each GLUE task, we report the value of the standard test metric (either accuracy or correlation) and the percentage deviation from the best performer among all optimizers. As shown in Table 2, Stiefel-AdamW outperforms all baselines in terms of average score. On single tasks, Stiefel-AdamW outperforms all baselines on MNLI, MRPC, STS-B, and WNLI, while maintaining a competitive performance on all other tasks. 


Table 2: Scores for rank 16 LoRA fine-tuning of the 4-bit quantized Mistral 7B model on the GLUE benchmark for Natural Language Understanding (NLU) challenges with diferent optimizers (best results highlighted in bold). In parentheses, we report, for each task, the percentage deviation from the best performer. SGD, Scaled GD, and Scaled AdamW results reported from [ZP24, Table 2].


| Method | MNLI | SST-2 | MRPC | CoLA | QNLI | QQP | RTE | STS-B | WNLI | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|
| SGD | 88.15(-4.14%) | 96.10(-1.18%) | 70.10(-22.07%) | 55.89(-22.23%) | 94.22(-1.21%) | 88.59(-3.94%) | 50.90(-44.27%) | 47.64(-48.37%) | 49.30(-43.54%) | 71.21 |
| Scaled GD [ZP24] | 90.21(-1.89%) | 96.90(-0.36%) | 81.62(-9.26%) | 68.17(-5.14%) | 94.40(-1.01%) | 91.15(-1.16%) | 54.15(-40.72%) | 90.31(-2.13%) | 56.34(-35.48%) | 80.36 |
| AdamW | 91.64(-0.34%) | 97.25 | 87.01(-3.27%) | 71.87 | 94.79(-0.61%) | 91.81(-0.44%) | 90.25(-1.19%) | 90.51(-1.92%) | 85.91(-1.61%) | 89.00 |
| Scaled AdamW [ZP24] | 90.68(-1.38%) | 97.25 | 89.46(-0.55%) | 71.30(-0.79%) | 94.67(-0.73%) | 92.22 | 91.34 | 91.10(-1.28%) | 83.10(-4.83%) | 89.01 |
| Stiefel-AdamW | 91.95 | 96.79(-0.47%) | 89.95 | 70.61(-1.75%) | 94.78(-0.62%) | 91.83(-0.42%) | 90.61(-0.80%) | 92.28 | 87.32 | 89.57 |
| GeoLoRA [SZC+25] | 91.30(-0.71%) | 94.61(-2.71%) | 87.26(-2.99%) | 69.78(-2.91%) | 95.37 | 90.80(-1.54%) | 88.81(-2.77%) | 91.45(-0.90%) | 87.32 | 88.52 |

5.2. GPT2 Pretraining. As the proposed method works for all problems in which the manifold of rank-r matrices appears, it also applies directly to pretraining transformer-based architectures. In particular, self-attention naturally respects this structure, as the image of the map $( W _ { Q } , W _ { K } ) \in \mathbb { R } ^ { n \times r } \times \mathbb { R } ^ { n \times r } \mapsto W _ { Q } W _ { K } ^ { \top }$ is exactly the set $\mathcal { M } _ { r }$ . Despite this, the parametrization map is highly non-injective, and therefore the problem could be restated equivalently by minimizing on $\mathcal { M } _ { r } \cong \mathrm { S t } ( n , r ) \times \mathbb { R } _ { * } ^ { n \times r } / \mathrm { O } ( r )$ instead of $\mathbb { R } ^ { n \times r } \times \mathbb { R } ^ { n \times r }$ . In contrast, we optimize all other parameters, such as biases or non-structured matrices, with the standard AdamW step. In Table 3 we present the results for pretraining GPT2 [RWC<sup>+</sup>19] on OpenWebText [GC19] using Karpathy’s reproduction<sup>1</sup>. We reproduced the pretraining for Stiefel-AdamW using exactly the same AdamW hyperparameters from the repository. We trained both AdamW and Stiefel-AdamW for 7000 iterations. As shown, Stiefel-AdamW produces results comparable to standard AdamW [LH19] in both performance and peak GPU memory usage. We performed no hyperparameter tuning, and Stiefel-AdamW uses the same hyperparameters as AdamW, as shown in the reproduced repository. This experiment highlights that Stiefel-AdamW is not limited to fine-tuning scenarios such as LoRA, but can be applied directly in pretraining settings where low-rank structure arises naturally. 

5.3. Qwen2 Pretraining. To showcase compatibility of our proposed approach with the presence of positional embeddings such as RoPE [SAL<sup>+</sup>24], in Table 4 we also present numerical results for Qwen2 [YYH<sup>+</sup>24] LoRA pretraining on WikiText-103 [MXBS16]. The model with the adapters has 100M parameters; it was trained for 20K steps with a batch size of 16 and two steps of gradient accumulation. As shown in Table 4, Stiefel-AdamW consistently outperforms the Euclidean version of AdamW in both mean and variance. 


Table 3: Left: Full GPT2 pretraining on OpenWeb Text for 6000 iterations. Right: Ablation over diferent retractions on ViT Base.


| Method | Test Loss $\pm \sigma$ | Peak Memory (GB) |
|---|---|---|
| AdamW | $3.260 \pm 0.05$ | 13.8 |
| Stiefel-AdamW | $3.236 \pm 0.07$ | 13.8 |

| Retraction | Test acc. ($r = 32$) | Test acc. ($r = 64$) | Test acc. ($r = 128$) |
|---|---|---|---|
| Cayley FP | 96.19 | 96.25 | 96.45 |
| Cayley SMW | 96.11 | 96.09 | 96.52 |
| Cayley direct | 96.11 | 96.22 | 96.61 |
| QR | 95.41 | 94.83 | 93.71 |
| Polar | 96.11 | 96.22 | 96.61 |
| Newton–Schulz | 96.11 | 96.20 | 96.52 |


Table 4: Qwen2 pretraining on Wikitext-103, standard deviation reported over 5 random seeds.


| Method | Loss $\pm \sigma$ |
|---|---|
| AdamW | $2.66 \pm 0.05$ |
| Stiefel-AdamW | $2.59 \pm 0.004$ |

5.4. Stepsize Stability. Motivated by Proposition 4.3, we show the stability of Stiefel-AdamW with respect to the learning rate, numerically demonstrating that methods with compact fibers are more stable with respect to learning-rate size. 

We fine-tuned multiple vision transformers (ViT Base) on CIFAR-10 for diferent learning rates, keeping all other hyperparameters fixed as in Section 5.1. In the right panel of Figure 1, we plot the learning rate against the best loss obtained during training. As expected, the results in Figure 1 show that “pure” Riemannian methods such as GeoLoRA [SZC<sup>+</sup>25] and the proposed Stiefel-AdamW are more stable with respect to learning-rate magnitude. 

In particular, Riemannian methods in which the fiber of Φ is not compact (such as AdamW and Scaled AdamW, which are defined in $\mathbb { R } ^ { r \times n } \times \mathbb { R } ^ { m \times r } )$ , appear to be less stable with respect to larger learning rates, despite the preconditioning (i.e., diferent metric) employed in Scaled AdamW [ZP24]. 

## 6. Conclusions, Limitations, and Future Work

In this work, we presented Stiefel-AdamW, a stochastic Riemannian variant of AdamW that naturally provides convergence guarantees. The compactness of the space in which one factor lives helps avoid potential numerical instabilities that can arise from unbalanced initializations. One key advantage of the proposed method is its simplicity of implementation, which requires only minimal machinery from Riemannian optimization theory while maintaining guarantees. We demonstrated the method’s efectiveness and scalability across a range of problems, from fine-tuning to pretraining LLMs. One limitation of the current method is that it still retains a set of orthogonal invariances, which stems from the entrywise nature of the Adam algorithm. While this can be solved by imposing a gauge condition and working on horizontal spaces on the quotient space $\mathcal { M } _ { r }$ , it requires additional computational efort, as the gradient needs to be computed using joint information on the pair (B, A), $( \nabla _ { B } \mathcal { L } , \nabla _ { A } \mathcal { L } )$ , and therefore does not allow for full parallelization. Future research could extend the current method to diferent preconditioners and propose a similarly simple version that is fully invariant on the manifold of fixed-rank matrices and can partially mitigate this extra computational efort. 

## References



[ABG07] P.-A. Absil, C. G. Baker, and K. A. Gallivan. Trust-region methods on Riemannian manifolds. Found. of Comput. Math., 7:303–330, 2007. 





[AM12] P.-A. Absil and J. Malick. Projection-like Retractions on Matrix Manifolds. SIAM J. Optim., 22(1):135–158, 2012. 





[AMS08] P.-A. Absil, R. Mahony, and R. Sepulchre. Optimization Algorithms on Matrix Manifolds. Princeton University Press, Princeton, NJ, 2008. 





[AO15] P.-A. Absil and I. V. Oseledets. Low-rank retractions: a survey and new results. Computational Optimization and Applications, 62(1):5–29, Sep 2015. 





[BA15] Nicolas Boumal and P.-A. Absil. Low-rank matrix completion via preconditioned optimization on the Grassmann manifold. Linear Algebra and its Applications, 475:200–239, 2015. 





[BB71] Å. Björck and C. Bowie. An Iterative Algorithm for Computing the Best Estimate of an Orthogonal Matrix. SIAM Journal on Numerical Analysis, 8(2):358–364, 1971. 





[BG19] Gary Bécigneul and Octavian-Eugen Ganea. Riemannian adaptive optimization methods. In International Conference on Learning Representations (ICLR 2019), volume 9, pages 6384–6399, 2019. 





[Bon13] Silvère Bonnabel. Stochastic Gradient Descent on Riemannian Manifolds. IEEE Transactions on Automatic Control, 58(9):2217–2229, 2013. 





[Bou23] Nicolas Boumal. An Introduction to Optimization on Smooth Manifolds. Cambridge University Press, 2023. 





[BZL<sup>+</sup>25] Fengmiao Bian, Jinyang Zheng, Ziyun Liu, Jianzhou Luo, and Jian-Feng Cai. Finding Low-Rank Matrix Weights in DNNs via Riemannian Optimization: RAdaGrad and RAdamW. In The 39th Annual Conference on Neural Information Processing Systems (NeurIPS), 2025. 





[DBK<sup>+</sup>21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations, 2021. 





[DHS11] John Duchi, Elad Hazan, and Yoram Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12(61):2121–2159, 2011. 





[EAS98] Alan Edelman, Tomás A. Arias, and Steven T. Smith. The geometry of algorithms with orthogonality constraints. SIAM J. Matrix Anal. Appl., 20(2):303–353, 1998. 





[GC19] Aaron Gokaslan and Vanya Cohen. OpenWebText Corpus. http://Skylion007.github. io/OpenWebTextCorpus, 2019. 





[GL76] Daniel Gabay and David G. Luenberger. Eficiently Converging Minimization Methods Based on the Reduced Gradient. SIAM Journal on Control and Optimization, 14(1):42– 61, 1976. 





[HGY24] Soufiane Hayou, Nikhil Ghosh, and Bin Yu. LoRA+: Eficient low rank adaptation of large models. In Proceedings of the 41st International Conference on Machine Learning, pages 17783–17806, 2024. 





[Hig08] Nicholas J. Higham. Functions of Matrices. Society for Industrial and Applied Mathematics, 2008. 





[HKK<sup>+</sup>26] Arsen Hnatiuk, Jonas Kusch, Lisa Kusch, Nicolas R. Gauger, and Andrea Walther. Stochastic Dynamical Low-Rank Approximation in the Context of Machine Learning. Journal of Optimization Theory and Applications, 208(1):1–33, January 2026. 





[HSS12] Geofrey Hinton, Nitish Srivastava, and Kevin Swersky. Neural Networks for Machine Learning Lecture 6a Overview of mini-batch gradient descent. https://www.cs.toronto. edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf, 2012. 





[HSW<sup>+</sup>22] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR), 2022. 





[JSM<sup>+</sup>23] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7B, 2023. 





[KB15] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In International Conference on Learning Representations (ICLR), 2015. 





[KH09] Alex Krizhevsky and Geofrey Hinton. Learning multiple layers of features from tiny images. Technical Report 0, University of Toronto, Toronto, Ontario, 2009. 





[KJM19] Hiroyuki Kasai, Pratik Jawanpuria, and Bamdev Mishra. Riemannian adaptive stochastic gradient algorithms on matrix manifolds. In International conference on machine learning, pages 3262–3271. PMLR, 2019. 





[KL07] Othmar Koch and Christian Lubich. Dynamical Low-Rank Approximation. SIAM Journal on Matrix Analysis and Applications, 29(2):434–454, 2007. 





[Kov70] Zdislav V. Kovarik. Some Iterative Methods for Improving Orthonormality. SIAM Journal on Numerical Analysis, 7:386–389, 1970. 





[KSM18] Hiroyuki Kasai, Hiroyuki Sato, and Bamdev Mishra. Riemannian Stochastic Recursive Gradient Algorithm. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 2516–2524. PMLR, 10–15 Jul 2018. 





[KTMF21] Mikhail Khodak, Neil A. Tenenholtz, Lester Mackey, and Nicolo Fusi. Initialization and Regularization of Factorized Neural Layers. In International Conference on Learning Representations, 2021. 





[LH19] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization, 2019. 





[LLT20] Jun Li, Fuxin Li, and Sinisa Todorovic. Eficient Riemannian Optimization on the Stiefel Manifold via the Cayley Transform. In International Conference on Learning Representations (ICLR), 2020. 





[LMSR24] Vladislav Lialin, Sherin Muckatira, Namrata Shivagunde, and Anna Rumshisky. ReLoRA: High-Rank Training Through Low-Rank Updates. In The Twelfth International Conference on Learning Representations, 2024. 





[Lue72] David G. Luenberger. The Gradient Projection Method along Geodesics. Manage. Sci., 18(11):620–631, 1972. 





[MHP25] Zhanfeng Mo, Long-Kai Huang, and Sinno Jialin Pan. Parameter and Memory Eficient Pretraining via Low-rank Riemannian Optimization. In The Thirteenth International Conference on Learning Representations, 2025. 





[MMBS14] Bamdev Mishra, Gilles Meyer, Silvère Bonnabel, and Rodolphe Sepulchre. Fixed-rank matrix factorizations and Riemannian low-rank optimization. Computational Statistics, 29(3):591–621, Jun 2014. 





[MXBS16] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016. 





[NDR17] Jekaterina Novikova, Ondrej Dušek, and Verena Rieser. The E2E Dataset: New Challenges for End-to-End Generation. In Proceedings of the 18th Annual Meeting of the Special Interest Group on Discourse and Dialogue, Saarbrücken, Germany, 2017. arXiv:1706.09254. 





[RKK18] Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the Convergence of Adam and Beyond. In International Conference on Learning Representations, 2018. 





[RW12] W. Ring and B. Wirth. Optimization Methods on Riemannian Manifolds and Their Application to Shape Space. SIAM J. Optim., 22(2):596–627, 2012. 





[RWC<sup>+</sup>19] Alec Radford, Jef Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language Models are Unsupervised Multitask Learners. Technical report, OpenAI, 2019. 





[SAL<sup>+</sup>24] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 





[Sat16] Hiroyuki Sato. A Dai–Yuan-type Riemannian conjugate gradient method with the weak Wolfe conditions. Comput. Optim. Appl., 64(1):101–118, May 2016. 





[Sat21] Hiroyuki Sato. Riemannian Optimization and Its Applications. Springer International Publishing, 2021. 





[Sat22] Hiroyuki Sato. Riemannian Conjugate Gradient Methods: General Framework and Specific Algorithms with Convergence Analyses. SIAM J. Optim., 32(4):2690–2717, 2022. 





[SI15] Hiroyuki Sato and Toshihiro Iwai. A new, globally convergent Riemannian conjugate gradient method. Optimization, 64(4):1011–1031, 2015. 





[SI22] Hiroyuki Sakai and Hideaki Iiduka. Riemannian Adaptive Optimization Algorithm and its Application to Natural Language Processing. IEEE Transactions on Cybernetics, 52(8):7328–7339, 2022. 





[SI25] Hiroyuki Sakai and Hideaki Iiduka. A general framework of Riemannian adaptive optimization methods with a convergence analysis. Transactions on Machine Learning Research, page n/a, 2025. Reproducibility Certification. 





[SKK25] Stefen Schotthöfer, Timon Klein, and Jonas Kusch. A geometric framework for momentum-based optimizers for low-rank training. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 





[SSP23] Rajarshi Saha, Varun Srivastava, and Mert Pilanci. Matrix Compression via Randomized Low Rank and Low Precision Factorization. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 





[SZC<sup>+</sup>25] Stefen Schotthöfer, Emanuele Zangrando, Gianluca Ceruti, Francesco Tudisco, and Jonas Kusch. GeoLoRA: Geometric integration for parameter eficient fine-tuning. In The Thirteenth International Conference on Learning Representations, 2025. 





[SZK<sup>+</sup>22] Stefen Schotthöfer, Emanuele Zangrando, Jonas Kusch, Gianluca Ceruti, and Francesco Tudisco. Low-rank lottery tickets: finding eficient low-rank neural networks via matrix diferential equations. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, pages 20051–20063, Red Hook, NY, USA, 2022. Curran Associates Inc. 





[VKJ19] Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. 





[VSP<sup>+</sup>17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is All You Need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. 





[VV10] Bart Vandereycken and Stefan Vandewalle. A Riemannian Optimization Approach for Computing Low-Rank Solutions of Lyapunov Equations. SIAM Journal on Matrix Analysis and Applications, 31(5):2553–2579, 2010. 





[WAP21] Hongyi Wang, Saurabh Agarwal, and Dimitris Papailiopoulos. Puferfish: Communicationeficient Models At No Extra Cost. In A. Smola, A. Dimakis, and I. Stoica, editors, Proceedings of Machine Learning and Systems, volume 3, pages 365–386, 2021. 





[WLH<sup>+</sup>25] Zhengbo Wang, Jian Liang, Ran He, Zilei Wang, and Tieniu Tan. LoRA-Pro: Are Low-Rank Adapters Properly Optimized? In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Representation Learning, pages 93787–93808, 2025. 





[WSM<sup>+</sup>19] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding, 2019. 





[WY13] Zaiwen Wen and Wotao Yin. A feasible method for optimization with orthogonality constraints. Mathematical Programming, 142(1):397–434, Dec 2013. 





[Yan07] Y. Yang. Globally Convergent Optimization Algorithms on Riemannian Manifolds: Uniform Framework for Unconstrained and Constrained Optimization. Journal of Optimization Theory and Applications, 132(2):245–265, Feb 2007. 





[YSM<sup>+</sup>25] Jui-Nan Yen, Si Si, Zhao Meng, Felix Yu, Sai Surya Duvvuri, Inderjit S Dhillon, Cho-Jui Hsieh, and Sanjiv Kumar. LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization. In The Thirteenth International Conference on Learning Representations, 2025. 





[YYH<sup>+</sup>24] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin 





Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. 





[ZCB<sup>+</sup>23] Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. AdaLoRA: Adaptive Budget Allocation for Parameter-Eficient Fine-Tuning. In The Eleventh International Conference on Learning Representations, 2023. 





[ZJRS16] Hongyi Zhang, Sashank J. Reddi, and Suvrit Sra. Riemannian SVRG: Fast Stochastic Optimization on Riemannian Manifolds. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016. 





[ZP24] Fangzhao Zhang and Mert Pilanci. Riemannian preconditioned LoRA for fine-tuning foundation models. In Proceedings of the 41st International Conference on Machine Learning, ICML’24, pages 59641–59669. JMLR.org, 2024. 





[ZS20] Xiaojing Zhu and Hiroyuki Sato. Riemannian conjugate gradient methods with inverse retraction. Computational Optimization and Applications, 77(3):779–810, Dec 2020. 





[ZSK<sup>+</sup>24] Emanuele Zangrando, Stefen Schotthöfer, Jonas Kusch, Gianluca Ceruti, and Francesco Tudisco. Geometry-aware training of factorized layers in tensor Tucker format. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 





[ZWGC24] Zhenyu Zhu, Yongtao Wu, Quanquan Gu, and Volkan Cevher. Imbalance-Regularized LoRA: A Plug-and-Play Method for Improving Fine-Tuning of Foundation Models. In Adaptive Foundation Models: Evolving AI for Personalized and Eficient Learning, 2024. 





[ZZC<sup>+</sup>24] Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian. GaLore: Memory-Eficient LLM Training by Gradient Low-Rank Projection. In 5th Workshop on practical ML for limited/low resource settings, 2024. 



## A. Appendices

A.1. Geometry of the Stiefel Manifold. The (column-orthonormal) Stiefel manifold is defined as 

$$
\operatorname{St} (n, r) := \left\{X \in \mathbb {R} ^ {n \times r}: X ^ {\top} X = I _ {r} \right\}.
$$

It is a smooth embedded submanifold of $\mathbb { R } ^ { n \times r }$ of dimension $n r - { \textstyle { \frac { 1 } { 2 } } } r ( r + 1 )$ 

The tangent space at a point $X \in \mathrm { S t } ( n , r )$ is given by 

$$
\mathrm{T} _ {X} \mathrm{St} (n, r) = \left\{\xi \in \mathbb {R} ^ {n \times r} \colon X ^ {\top} \xi + \xi^ {\top} X = 0 \right\}.
$$

Equivalently, any tangent vector $\xi \in \mathrm { T } _ { X } \mathrm { S t } ( n , r )$ can be decomposed as 

$$
\xi = X \Omega + X _ {\perp} K,
$$

where $\varOmega \in \mathbb { R } ^ { r \times r }$ is skew-symmetric, $X _ { \perp } \in \mathbb { R } ^ { n \times ( n - r ) }$ satisfies $[ X X \lrcorner ] \in { \mathrm { O } } ( n ) , { \mathrm { O } } ( n )$ being the orthogonal group, and $K \in \mathbb { R } ^ { ( n - r ) \times r }$ 

A.1.1. Orthogonal Projection onto $\mathrm { T } _ { X } \mathrm { S t } ( n , r )$ . For any matrix $\xi \in  { \mathbb { R } } ^ { n \times r }$ , its orthogonal projection onto $\mathrm { T } _ { X } \mathrm { S t } ( n , r )$ with respect to the Euclidean inner product is given by 

$$
\mathrm{P} _ {X} (\xi) = X \operatorname{skew} (X ^ {\top} \xi) + (I _ {n} - X X ^ {\top}) \xi ,\tag{A.1}
$$

where skew $( M ) : = { \textstyle { \frac { 1 } { \gamma } } } ( M - M ^ { \top } )$ denotes the skew-symmetric part of a square matrix. 

Equation (A.1) admits the equivalent and more compact expression 

$$
\mathrm{P} _ {X} (\xi) = \xi - X \mathrm{sym} (X ^ {\top} \xi),
$$

where sym $( M ) : = { \textstyle { \frac { 1 } { 2 } } } ( M + M ^ { \top } )$ denotes the symmetric part. This formula is commonly used in Riemannian optimization on the Stiefel manifold; see, e.g., [AMS08, Prop. 3.6.1]. 

A.1.2. Row-Orthonormal Stiefel Manifold. In this work, we enforce a row-orthonormal constraint on the factor $A \in \mathbb { R } ^ { r \times n }$ , namely, 

$$
A A ^ {\top} = I _ {r}.
$$

This corresponds to working with the transpose variable $X : = A ^ { \top } \in \operatorname { S t } ( n , r )$ . All Riemannian operations (projection, retraction, and gradient computation) are therefore performed on X, and the updated factor is recovered as $A = X ^ { \top }$ 

A.2. Cayley Transform and Retraction. A commonly used retraction on the Stiefel manifold $\operatorname { S t } ( n , r ) = \{ X \in \mathbb { R } ^ { n \times r } \colon X ^ { \top } X = I _ { r } \}$ is based on the Cayley transform. The Cayley transform generates a smooth curve on the Stiefel manifold by exponentiating a skew-symmetric matrix in a rational form, thereby avoiding explicit matrix exponentials. 

The closed-form Cayley retraction at a point $X \in \mathrm { S t } ( n , r )$ is defined as 

$$
Y (\alpha) = \left(I _ {n} - \frac {\alpha}{2} \Omega\right) ^ {- 1} \left(I _ {n} + \frac {\alpha}{2} \Omega\right) X,\tag{A.2}
$$

where $\varOmega \in \mathbb { R } ^ { n \times n }$ is a skew-symmetric matrix and $\alpha \geq 0$ is a step-size parameter. The curve satisfies 

$$
Y (0) = X, \qquad \frac {\mathrm{d}}{\mathrm{d} \alpha} Y (\alpha) \big | _ {\alpha = 0} = \varOmega X,
$$

and therefore defines a valid first-order retraction on the Stiefel manifold [AMS08]. 

Computing the closed-form expression (A.2) requires solving a linear system involving an $n \times n$ matrix, which can be computationally expensive for large n. A fixed-point approximation of the Cayley transform is given by 

$$
Y (\alpha) = X + \frac {\alpha}{2} \Omega \big (X + Y (\alpha) \big).\tag{A.3}
$$

Starting from the initialization $Y _ { 0 } = ( I + \alpha \varOmega ) X$ , this fixed-point equation can be solved with a small number of iterations, each involving only matrix multiplications. In practice, only a few iterations are suficient to obtain an accurate approximation of the exact Cayley retraction. 

A.2.1. Cayley Retraction in Our Setting. As mentioned in the main text, in the factorization of a weight matrix, we work with the row-orthonormal factor of size r-by-n satisfying $A A ^ { \top } =$ $I _ { r }$ . For convenience, we switch to a column-orthonormal representation so that $X : = A ^ { \top } \in$ $\operatorname { S t } ( n , r )$ 

Let $\xi \in \mathrm { T } _ { X _ { t } } \mathrm { S t } ( n , r )$ be a tangent vector at $X _ { t } : = A _ { t } ^ { \top }$ . Following [LLT20, Eq. (2)], the Cayley retraction used in our algorithm is defined as 

$$
X _ {t + 1} = \left(I _ {n} - \frac {1}{2} \varOmega\right) ^ {- 1} \left(I _ {n} + \frac {1}{2} \varOmega\right) X _ {t}, \qquad A _ {t + 1} := X _ {t + 1} ^ {\top},
$$

where the skew-symmetric matrix $\varOmega$ is constructed as 

$$
\widehat {\varOmega} = \xi X _ {t} ^ {\top} - \frac {1}{2} X _ {t} (X _ {t} ^ {\top} \xi X _ {t} ^ {\top}), \qquad \varOmega = \widehat {\varOmega} - \widehat {\varOmega} ^ {\top}.
$$

In practice, we compute the Cayley retraction using the fixed-point iteration $\mathrm { ( A . 3 ) }$ . This yields an eficient and numerically stable retraction satisfying the standard first-order retraction conditions $\mathrm { R e t r } _ { x } ( 0 _ { x } ) = x$ and $\mathrm { D R e t r } _ { x } ( 0 _ { x } ) =$ Id required for the convergence analysis of Riemannian optimization methods; see, $\mathrm { e . g . , [ A M S 0 8 , \ S 4 . 1 ] }$ 

A.2.2. Fixed-Point Approximation of the Cayley Retraction. In this section, we analyze the fixed-point iteration used to approximate the Cayley retraction and show that it geometrically converges to the exact Cayley transform. Most importantly, we further show that the resulting approximate mapping retains the retraction properties required by the convergence theory. 

Let $\{ Y _ { i } \} _ { i \ge 0 }$ be the sequence of fixed-point iterates defined by 

$$
Y _ {i + 1} = \mathcal {F} (Y _ {i}),\tag{A.4}
$$

where $\mathcal { F }$ is the fixed-point map of $\mathrm { ( A . 3 ) }$ , namely, $\begin{array} { r } { \mathcal { F } ( Y ) : = X + \frac { \alpha } { 2 } \varOmega ( X + Y ) } \end{array}$ , and the initialization is $\begin{array} { r } { Y _ { 0 } = ( I + \alpha \varOmega ) X } \end{array}$ . We recall the contraction property of a fixed-point map. 

Lemma A.1 (Contraction of the fixed-point map). Assume that $\alpha \| \varOmega \| _ { 2 } < 2 \ ( \alpha \geq 0 )$ . Then $\mathcal { F }$ is a contraction mapping on $\mathbb { R } ^ { n \times r }$ with contraction factor 

$$
\rho := \frac {\alpha}{2} \| \Omega \| _ {2} <   1.
$$

Proof. For any $Y _ { 1 } , Y _ { 2 } \in \mathbb { R } ^ { n \times r }$ ， 

$$
\| \mathcal {F} (Y _ {1}) - \mathcal {F} (Y _ {2}) \| = \left\| \frac {\alpha}{2} \Omega Y _ {1} - \frac {\alpha}{2} \Omega Y _ {2} \right\| \leq \frac {\alpha}{2} \| \Omega \| _ {2} \| Y _ {1} - Y _ {2} \| <   \| Y _ {1} - Y _ {2} \|,
$$

which shows that $\mathcal { F }$ is a contraction mapping. 

Theorem A.2 (Geometric convergence to the Cayley retraction). Let $Y ^ { \star } : = \operatorname { R e t r } _ { X } ( \alpha \xi )$ be the exact Cayley retraction of $\alpha \xi$ at $X$ . Under the assumptions of Lemma A.1, the fixed-point iterates defined by (A.4) satisfy 

$$
\forall i \geq 0, \qquad \| Y _ {i} - Y ^ {\star} \| \leq \rho^ {i} \| Y _ {0} - Y ^ {\star} \|.\tag{A.5}
$$

Proof. Since by Lemma A.1 F is a contraction mapping, the Banach fixed-point theorem applies, with the exact retraction $Y ^ { \star } = \operatorname { R e t r } _ { X } ( \alpha \xi )$ being the unique fixed point of $\mathcal { F }$ . The theorem’s result (A.5) follows directly. □ 

We are now in the position to state the properties of the approximate retraction, which we denote by 

$$
\widetilde {\mathrm{Retr}} _ {X} (\alpha \xi) := Y _ {s},
$$

where s is the total number of fixed-point iterations performed. 

Lemma A.3 (Second-order accuracy). For suficiently small α and any fixed number of iterations s, the approximate Cayley retraction satisfies 

$$
\widetilde {\mathrm{Retr}} _ {X} (\alpha \xi) = X + \alpha \xi + \mathcal {O} (\alpha^ {2}) + \mathcal {O} (\rho^ {s}),
$$

and converges to the exact Cayley retraction as $s \to \infty$ 

In particular, for fixed s and suficiently small step size α, the approximation error remains of higher order and does not dominate the first-order behavior of the update. This justifies using a finite number of fixed-point iterations in practice. 

Proof. The exact Cayley transform (A.2) admits the expansion 

$$
\mathrm{Retr} _ {X} (\alpha \xi) = X + \alpha \xi + \mathcal {O} (\alpha^ {2}).\tag{A.6}
$$

The initialization of the fixed-point iteration method is 

$$
Y _ {0} = (I + \alpha \Omega) X = X + \alpha \Omega X = X + \alpha \xi ,
$$

which is clearly a first-order approximation of Retr (αξ). Therefore 

$$
\left\| Y _ {0} - Y ^ {\star} \right\| = \mathcal {O} (\alpha^ {2}).\tag{A.7}
$$

By Theorem A.2, we have the contraction estimate (A.5). Inserting (A.7) into the (A.5), with $i = s .$ , we obtain 

$$
\| Y _ {s} - Y ^ {\star} \| \leq \rho^ {s} \mathcal {O} (\alpha^ {2}).
$$

Since α is fixed within one update, the factor $\mathcal { O } ( \alpha ^ { 2 } )$ can be absorbed into the constant, i.e., 

$$
\left\| Y _ {s} - Y ^ {\star} \right\| \leq \mathcal {O} \left(\rho^ {s}\right).
$$

or, equivalently, $Y _ { s } = \mathrm { R e t r } _ { X } ( \alpha \xi ) + { \mathcal O } ( \rho ^ { s } )$ . Combining this with the expansion of the exact retraction (A.6), the result of the lemma follows immediately. □ 

In particular, Lemma A.3 shows that the approximate Cayley mapping satisfies the firstorder retraction conditions up to a controllable error. Such inexact retractions preserve the convergence guarantees of Riemannian first-order methods provided the approximation error is suficiently small; see, e.g., standard analyses of inexact retraction schemes. 

Lemma A.3 shows that for any fixed s and suficiently small α, the approximate Cayley transform defined by the fixed-point iteration still satisfies the first-order retraction condition. The term $\mathcal { O } ( \rho ^ { s } )$ is the numerical approximation error, controlled by the number of iterations s of the fixed-point approximation method. In particular, for a fixed α and large enough $s ,$ the approximate Cayley transform converges to the exact Cayley map. 

## B. Proof of Theorem 4.2

Proof. Consider the sequence $\mathcal { L } _ { t }$ of convex loss functions and define 

$$
R (T) := \sum_ {t = 1} ^ {T} \mathcal {L} _ {t} (B _ {t}, A _ {t}) - \mathcal {L} _ {t} (B ^ {*}, A ^ {*}),
$$

where $( B _ { t } , A _ { t } )$ is the iteration in Algorithm 1, and $( B ^ { * } , A ^ { * } )$ is a minimizer of $\textstyle \sum _ { t = 1 } ^ { T } { \mathcal { L } } _ { t } ( B , A )$ Using the convexity of $\mathcal { L } _ { t }$ , and defining $\boldsymbol { G } _ { t } ^ { A , B } = \nabla _ { A , B } \mathcal { L } _ { t } ( B _ { t } , A _ { t } )$ , we get 

$$
R (T) \leq \sum_ {t = 1} ^ {T} \langle B _ {t} - B ^ {*}, G _ {t} ^ {B} \rangle + \sum_ {t = 1} ^ {T} \langle A _ {t} - A ^ {*}, G _ {t} ^ {A} \rangle .\tag{B.1}
$$

We now bound the first term on the right-hand side of (B.1), i.e., the Euclidean term $\langle B _ { t }$ $B ^ { * } , G _ { t } ^ { B } \rangle$ . Let $H _ { t } ^ { B } \colon \mathbb { R } ^ { m \times r }  \mathbb { R } ^ { m \times r }$ the linear operator defined by $H _ { t } ^ { B } ( X ) = ( V _ { t } ^ { B } + \varepsilon ) \odot X$ We bound the norm $\| B _ { t + 1 } - B ^ { * } \| _ { ( H _ { t } ^ { B } ) ^ { 1 / 2 } } ^ { 2 }$ (see Definition B.1): 

$$
\begin{array}{r l} & {\| B _ {t + 1} - B ^ {*} \| _ {(H _ {t} ^ {B}) ^ {1 / 2}} ^ {2} = \| B _ {t} - \eta_ {t} (H _ {t} ^ {B}) ^ {- 1 / 2} M _ {t} ^ {B} - B ^ {*} \| _ {(H _ {t} ^ {B}) ^ {1 / 2}} ^ {2} = \| B _ {t} - B ^ {*} \| _ {(H _ {t} ^ {B}) ^ {1 / 2}} ^ {2} + \qquad \mathrm{(B}} \\ & {\qquad + \eta_ {t} ^ {2} \| (H _ {t} ^ {B}) ^ {- 1 / 2} M _ {t} ^ {B} \| _ {(H _ {t} ^ {B}) ^ {1 / 2}} ^ {2} - 2 \eta_ {t} \langle B _ {t} - B ^ {*}, (H _ {t} ^ {B}) ^ {- 1 / 2} M _ {t} ^ {B} \rangle_ {(H _ {t} ^ {B}) ^ {1 / 2}}} \\ & {\qquad = \quad \| B _ {t} - B ^ {*} \| _ {(H _ {t} ^ {B}) ^ {1 / 2}} ^ {2} + \eta_ {t} ^ {2} \| M _ {t} ^ {B} \| _ {(H _ {t} ^ {B}) ^ {- 1 / 2}} ^ {2} - 2 \eta_ {t} \langle B _ {t} - B ^ {*}, M _ {t} ^ {B} \rangle ,} \\ & {\qquad \mathrm{Lemma~B.2}} \end{array}\tag{B.2}
$$

where the last inner product is in the Frobenius norm (for clarity, we always omit the subscript). By rearranging (B.2) (bringing the inner product on the left-hand side and the norm on the right-hand side), we get: 

$$
2 \eta_ {t} \langle B _ {t} - B ^ {*}, M _ {t} ^ {B} \rangle = \underbrace {\| B _ {t} - B ^ {*} \| _ {(H _ {t} ^ {B}) ^ {1 / 2}} ^ {2} - \| B _ {t + 1} - B ^ {*} \| _ {(H _ {t} ^ {B}) ^ {1 / 2}} ^ {2}} _ {=: \varDelta_ {t} ^ {B}} + \eta_ {t} ^ {2} \| M _ {t} ^ {B} \| _ {(H _ {t} ^ {B}) ^ {- 1 / 2}} ^ {2}.\tag{B.3}
$$

Using the definition of $M _ { t } ^ { B } = \beta _ { 1 } M _ { t - 1 } ^ { B } + ( 1 - \beta _ { 1 } ) G _ { t } ^ { B }$ in (B.3), and by defining $\varDelta _ { t } ^ { B } : = \| B _ { t } -$ $B ^ { * } \| _ { ( H _ { t } ^ { B } ) ^ { 1 / 2 } } ^ { 2 } - \| B _ { t + 1 } - B ^ { * } \| _ { ( H _ { t } ^ { B } ) ^ { 1 / 2 } } ^ { 2 } ,$ , we get 

$$
\begin{array}{l} \langle B _ {t} - B ^ {*}, G _ {t} ^ {B} \rangle = \frac {1}{2 (1 - \beta_ {1}) \eta_ {t}} \varDelta_ {t} ^ {B} + \frac {\eta_ {t}}{2 (1 - \beta_ {1})} \| M _ {t} ^ {B} \| _ {(H _ {t} ^ {B}) ^ {- 1 / 2}} ^ {2} - \frac {\beta_ {1}}{1 - \beta_ {1}} \langle B _ {t} - B ^ {*}, M _ {t - 1} ^ {B} \rangle \\ \qquad \leq \frac {1}{2 (1 - \beta_ {1}) \eta_ {t}} \varDelta_ {t} ^ {B} + \frac {\eta_ {t}}{2 (1 - \beta_ {1})} \| M _ {t} ^ {B} \| _ {(H _ {t} ^ {B}) ^ {- 1 / 2}} ^ {2} + \frac {\beta_ {1}}{1 - \beta_ {1}} \left| \langle B _ {t} - B ^ {*}, M _ {t - 1} ^ {B} \rangle \right| \\ \qquad \leq \frac {\varDelta_ {t} ^ {B}}{\texttt {L e m m a B . 3}} \frac {\varDelta_ {t} ^ {B}}{2 (1 - \beta_ {1}) \eta_ {t}} + \frac {\eta_ {t}}{2 (1 - \beta_ {1})} \| M _ {t} ^ {B} \| _ {(H _ {t} ^ {B}) ^ {- 1 / 2}} ^ {2} + \frac {\beta_ {1}}{2 (1 - \beta_ {1}) \alpha_ {t} ^ {2}} \| M _ {t - 1} ^ {B} \| _ {(H _ {t} ^ {B}) ^ {- 1 / 2}} ^ {2} \\ \qquad + \frac {\beta_ {1} \alpha_ {t} ^ {2}}{2 (1 - \beta_ {1})} \| B _ {t} - B ^ {*} \| _ {(H _ {t} ^ {B}) ^ {1 / 2}} ^ {2}. \end{array}\tag{B.4}
$$

Apart from the retraction, the term $\langle A _ { t } - A ^ { * } , G _ { t } ^ { A } \rangle$ is similar to (B.4), i.e., 

$$
A _ {t + 1} = \operatorname{Retr} _ {A _ {t}} \left(- \eta_ {t} \mathrm{P} _ {A _ {t}} \left(\left(H _ {t} ^ {A}\right) ^ {- 1 / 2} M _ {t} ^ {A}\right)\right).
$$

We define 

$$
E _ {t} := \mathrm{Retr} _ {A _ {t}} (- \eta_ {t} \mathrm{P} _ {A _ {t}} (H _ {t} ^ {A}) ^ {- 1 / 2} M _ {t} ^ {A}) - (A _ {t} - \eta_ {t} \mathrm{P} _ {A _ {t}} (H _ {t} ^ {A}) ^ {- 1 / 2} M _ {t} ^ {A}),
$$

and the invertible linear operator $\Gamma _ { t } = \mathrm { P } _ { A _ { t } } ( H _ { t } ^ { A } ) ^ { - 1 / 2 } \mathrm { P } _ { A _ { t } } \colon \mathrm { T } _ { A _ { t } } \mathrm { S t } ( n , r )  \mathrm { T } _ { A _ { t } } \mathrm { S t } ( n , r )$ . We define $\bar { \Gamma } _ { t }$ as an extension of the previous map on the whole space, $\bar { \Gamma } _ { t } = \Gamma _ { t } + \gamma _ { t } ( I - \mathrm { P } _ { A _ { t } } )$ where $\gamma _ { t } > 0$ is a scalar. With a small abuse of notation, we will still denote by $\Gamma _ { t }$ the map $\bar { \Gamma } _ { t }$ when there is no risk of confusion. Let $D _ { t } : = \mathrm { P } _ { A _ { t } } ( H _ { t } ^ { A } ) ^ { - 1 / 2 } M _ { t } ^ { A } \in \mathrm { T } _ { A _ { t } } \mathrm { S t } ( n , r )$ , and consider the norm 

$$
\begin{array}{r l} & {\| A _ {t + 1} - A ^ {*} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2} = \| A _ {t} - \eta_ {t} D _ {t} + E _ {t} - A ^ {*} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2}} \\ & {\qquad = \| A _ {t} - A ^ {*} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2} - 2 \eta_ {t} \langle \Gamma_ {t} ^ {- 1} D _ {t}, A _ {t} - A ^ {*} \rangle + \eta_ {t} ^ {2} \| D _ {t} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2}} \\ & {\qquad \qquad - 2 \eta_ {t} \langle E _ {t}, \Gamma_ {t} ^ {- 1} D _ {t} \rangle + 2 \langle \Gamma_ {t} ^ {- 1} E _ {t}, A _ {t} - A ^ {*} \rangle + \| E _ {t} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2}} \\ & {\qquad \leq \| A _ {t} - A ^ {*} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2} - 2 \eta_ {t} \langle \Gamma_ {t} ^ {- 1} D _ {t}, A _ {t} - A ^ {*} \rangle + \eta_ {t} ^ {2} \| D _ {t} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2} + \underbrace {- 2 \eta_ {t} \langle E _ {t} , \Gamma_ {t} ^ {- 1} D _ {t} \rangle + \| E _ {t} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2}} _ {=: \delta_ {t}}.} \end{array}
$$

namely, 

$$
\| A _ {t + 1} - A ^ {*} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2} \leq \| A _ {t} - A ^ {*} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2} - 2 \eta_ {t} \left\langle \Gamma_ {t} ^ {- 1} D _ {t}, A _ {t} - A ^ {*} \right\rangle + \eta_ {t} ^ {2} \| D _ {t} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2} + \delta_ {t}.\tag{B.5}
$$

Bringing the inner product in (B.5) to the left-hand side and the norm to the right-hand side leads to 

$$
2 \eta_ {t} \left\langle \Gamma_ {t} ^ {- 1} D _ {t}, A _ {t} - A ^ {*} \right\rangle \leq \| A _ {t} - A ^ {*} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2} - \| A _ {t + 1} - A ^ {*} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2} + \eta_ {t} ^ {2} \| D _ {t} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2} + \delta_ {t}.
$$

Defining $\begin{array} { r } { \varDelta _ { t } ^ { A } : = \| A _ { t } - A ^ { * } \| _ { \Gamma _ { t } ^ { - 1 } } ^ { 2 } - \| A _ { t + 1 } - A ^ { * } \| _ { \Gamma _ { t } ^ { - 1 } } ^ { 2 } } \end{array}$ , and dividing by $2 \eta _ { t }$ , we can write 

$$
\langle \Gamma_ {t} ^ {- 1} D _ {t}, A _ {t} - A ^ {*} \rangle \leq \frac {1}{2 \eta_ {t}} \Delta_ {t} ^ {A} + \frac {\eta_ {t}}{2} \| D _ {t} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2} + \frac {1}{2 \eta_ {t}} \delta_ {t}.
$$

We now notice that, by definition of $\Gamma _ { t }$ , we have $\Gamma _ { t } ^ { - 1 } D _ { t } = M _ { t } ^ { A }$ , and, by using again Young inequality (Lemma B.3), we get 

$$
\begin{array}{l} \langle G _ {t} ^ {A}, A _ {t} - A ^ {*} \rangle \leq \frac {1}{2 (1 - \beta_ {1}) \eta_ {t}} \varDelta_ {t} ^ {A} + \frac {\eta_ {t}}{2 (1 - \beta_ {1})} \| D _ {t} \| _ {\Gamma_ {t} ^ {- 1}} ^ {2} + \frac {1}{2 (1 - \beta_ {1}) \eta_ {t}} \delta_ {t} \\ \quad + \frac {\beta_ {1} \alpha_ {t} ^ {2}}{2 (1 - \beta_ {1})} \| A _ {t} - A ^ {*} \| _ {\mathrm{F}} ^ {2} + \frac {\beta_ {1}}{2 (1 - \beta_ {1}) \alpha_ {t} ^ {2}} \| M _ {t - 1} ^ {A} \| _ {\mathrm{F}} ^ {2} \\ = \frac {1}{2 (1 - \beta_ {1}) \eta_ {t}} \varDelta_ {t} ^ {A} + \frac {\eta_ {t}}{2 (1 - \beta_ {1})} \left\| (H _ {t} ^ {A}) ^ {- 1 / 4} (H _ {t} ^ {A}) ^ {1 / 4} \Gamma_ {t} ^ {1 / 2} \Gamma_ {t} ^ {- 1} D _ {t} \right\| _ {\mathrm{F}} ^ {2} + \frac {1}{2 (1 - \beta_ {1}) \eta_ {t}} \delta_ {t} \\ \quad + \frac {\beta_ {1} \alpha_ {t} ^ {2}}{2 (1 - \beta_ {1})} \left\| (H _ {t} ^ {A}) ^ {- 1 / 4} (H _ {t} ^ {A}) ^ {1 / 4} (A _ {t} - A ^ {*}) \right\| _ {\mathrm{F}} ^ {2} + \frac {\beta_ {1}}{2 (1 - \beta_ {1}) \alpha_ {t} ^ {2}} \| (H _ {t} ^ {A}) ^ {- 1 / 4} (H _ {t} ^ {A}) ^ {1 / 4} M _ {t - 1} ^ {A} \| _ {\mathrm{F}} ^ {2} \\ = \frac {1}{2 (1 - \beta_ {1}) \eta_ {t}} \varDelta_ {t} ^ {A} + \frac {\eta_ {t}}{2 (1 - \beta_ {1})} \| (H _ {t} ^ {A}) ^ {1 / 4} \Gamma_ {t} ^ {1 / 2} M _ {t} ^ {A} \| _ {(H _ {t} ^ {A}) ^ {- 1 / 2}} ^ {2} + \frac {1}{2 (1 - \beta_ {1}) \eta_ {t}} \delta_ {t} \\ \quad + \frac {\beta_ {1} \alpha_ {t} ^ {2}}{2 (1 - \beta_ {1})} \| (H _ {t} ^ {A}) ^ {1 / 4} (A _ {t} - A ^ {*}) \| _ {(H _ {t} ^ {A}) ^ {- 1 / 2}} ^ {2} + \frac {\beta_ {1}}{2 (1 - \beta_ {1}) \alpha_ {t} ^ {2}} \| (H _ {t} ^ {A}) ^ {- 1 / 4} M _ {t - 1} ^ {A} \| _ {(H _ {t} ^ {A}) ^ {1 / 2}} ^ {2} \\ \leq \frac {\varDelta_ {t} ^ {A}}{2 (1 - \beta_ {1}) \eta_ {t}} + \frac {\eta_ {t} \| (H _ {t} ^ {A}) ^ {1 / 4} \Gamma_ {t} ^ {1 / 2} \| _ {\mathrm{op}} ^ {2}}{2 (1 - \beta_ {1})} \| M _ {t} ^ {A} \| _ {(H _ {t} ^ {A}) ^ {- 1 / 2}} ^ {2} + \frac {\delta_ {t}}{2 (1 - \beta_ {1}) \eta_ {t}} \\ \quad + \frac {\beta_ {1} \alpha_ {t} ^ {2} \| (H _ {t} ^ {A}) ^ {1 / 4} \| _ {\mathrm{op}} ^ {2}}{2 (1 - \beta_ {1})} \| A _ {t} - A ^ {*} \| _ {(H _ {t} ^ {A}) ^ {- 1 / 2}} ^ {2} + \frac {\beta_ {1} \| (H _ {t} ^ {A}) ^ {- 1 / 4} \| _ {\mathrm{op}} ^ {2}}{2 (1 - \beta_ {1}) \alpha_ {t} ^ {2}} \| M _ {t - 1} ^ {A} \| _ {(H _ {t} ^ {A}) ^ {1 / 2}} ^ {2}. \end{array}\tag{B.6}
$$

The term $\delta _ { t }$ is the only one structurally diferent from the ones in (B.4). Thus, (B.6) can be bounded using the definition of $E _ { t }$ (retraction error) with Lagrange remainder error 

$$
\| E _ {t} \| \leq C \| \mathrm{D} ^ {2} \mathrm{Retr} _ {A _ {t}} (\zeta) \| \| \eta_ {t} D _ {t} \| ^ {2} \leq \tilde {C} \eta_ {t} ^ {2} \| D _ {t} \| ^ {2},
$$

as 

$$
\begin{array}{r l} \frac {\delta_ {t}}{2 (1 - \beta_ {1}) \eta_ {t}} & \leq \frac {1}{2 (1 - \beta_ {1}) \eta_ {t}} \Big [ 2 \eta_ {t} \| E _ {t} \| \| M _ {t} ^ {A} \| + \| \Gamma_ {t} ^ {- 1} \| _ {\mathrm{op}} \| E _ {t} \| ^ {2} \Big ] \\ & \leq \frac {1}{2 (1 - \beta_ {1})} \Big [ 2 \| E _ {t} \| \| M _ {t} ^ {A} \| + \frac {1}{\eta_ {t}} \| \Gamma_ {t} ^ {- 1} \| _ {\mathrm{op}} \| E _ {t} \| ^ {2} \Big ] \\ & \leq \frac {1}{2 (1 - \beta_ {1})} \Big [ 2 \eta_ {t} ^ {2} \| D _ {t} \| ^ {2} \| M _ {t} ^ {A} \| + \eta_ {t} ^ {3} \| \Gamma_ {t} ^ {- 1} \| _ {\mathrm{op}} \| D _ {t} \| ^ {4} \Big ]. \end{array}
$$

Using the definition 

$$
\| M _ {t} ^ {A} \| := \| (1 - \beta_ {1}) \sum_ {s = 1} ^ {t} \beta_ {1} ^ {t - s} G _ {s} ^ {A} \| \leq G _ {\infty},
$$

and the fact that $\| D _ { t } \| \lesssim G _ { \infty } , \| \Gamma _ { t } ^ { - 1 } \| _ { \mathrm { o p } }$ bounded, we get 

$$
\sum_ {t = 1} ^ {T} \frac {\delta_ {t}}{2 (1 - \beta_ {1}) \eta_ {t}} \lesssim \frac {1}{2 (1 - \beta_ {1})} \Big [ G _ {\infty} ^ {3} \sum_ {t = 1} ^ {T} \eta_ {t} ^ {2} + G _ {\infty} ^ {4} \sum_ {t = 1} ^ {T} \eta_ {t} ^ {3} \Big ].\tag{B.7}
$$

By combining equations (B.1), (B.4), (B.6) and (B.7), we get 

$$
\begin{array}{l} R (T) \leq \sum_ {t = 1} ^ {T} \left(\frac {\varDelta_ {t} ^ {B}}{2 (1 - \beta_ {1}) \eta_ {t}} + \frac {\eta_ {t} \| M _ {t} ^ {B} \| _ {(H _ {t} ^ {B}) ^ {- 1 / 2}} ^ {2}}{2 (1 - \beta_ {1})} + \frac {\beta_ {1} \| M _ {t - 1} ^ {B} \| _ {(H _ {t} ^ {B}) ^ {- 1 / 2}} ^ {2}}{2 (1 - \beta_ {1}) \alpha_ {t} ^ {2}} + \frac {\beta_ {1} \alpha_ {t} ^ {2} \| B _ {t} - B ^ {*} \| _ {(H _ {t} ^ {B}) ^ {1 / 2}} ^ {2}}{2 (1 - \beta_ {1})}\right) \\ \quad + \sum_ {t = 1} ^ {T} \left(\frac {\varDelta_ {t} ^ {A}}{2 (1 - \beta_ {1}) \eta_ {t}} + \frac {\eta_ {t} K _ {1} \| M _ {t} ^ {A} \| _ {(H _ {t} ^ {A}) ^ {- 1 / 2}} ^ {2}}{2 (1 - \beta_ {1})} + \frac {\beta_ {1} K _ {2} \| M _ {t - 1} ^ {A} \| _ {(H _ {t} ^ {A}) ^ {1 / 2}} ^ {2}}{2 (1 - \beta_ {1}) \alpha_ {t} ^ {2}} + \frac {\beta_ {1} \alpha_ {t} ^ {2} K _ {3} \| A _ {t} - A ^ {*} \| _ {(H _ {t} ^ {A}) ^ {- 1 / 2}} ^ {2}}{2 (1 - \beta_ {1})}\right) \\ \quad + \frac {1}{2 (1 - \beta_ {1})} \left[ G _ {\infty} ^ {3} \sum_ {t = 1} ^ {T} \eta_ {t} ^ {2} + G _ {\infty} ^ {4} \sum_ {t = 1} ^ {T} \eta_ {t} ^ {3} \right], \end{array}\tag{B.8}
$$

where $K _ { 1 } , K _ { 2 } , K _ { 3 }$ are three constants bounding the operator norms in (B.6). Notice that up to these constants, the terms in A and B are similar to each other; therefore, we focus on bounding the terms in B (the ones in A yield analogous bounds). 

Now, we observe that, thanks to [RKK18, Lemma 2], for $\eta _ { t } = \eta / \sqrt { t }$ we obtain the following bounds 

$$
\sum_ {t = 1} ^ {T} \eta_ {t} \| M _ {t} ^ {B} \| _ {(H _ {t} ^ {B}) ^ {- 1 / 2}} ^ {2} \leq \frac {\eta \| G ^ {B} \| _ {L ^ {1} L ^ {2} ([ 0 , T ])} \sqrt {1 + \log T}}{(1 - \beta_ {1}) (1 - \beta_ {1} / \sqrt {\beta_ {2}}) \sqrt {1 - \beta_ {2}}},
$$

where $\| G ^ { B } \| _ { L ^ { 1 } L ^ { 2 } ( [ 0 , T ] ) }$ is the $L ^ { 1 } - L ^ { 2 }$ norm defined by 

$$
\| G ^ {B} \| _ {L ^ {1} L ^ {2} ([ 0, T ])} := \sum_ {i, j} \left(\sum_ {t = 1} ^ {T} \left| (G _ {t} ^ {B}) _ {i j} \right| ^ {2}\right) ^ {1 / 2}.
$$

Moreover, 

$$
\sum_ {t = 1} ^ {T} \eta_ {t} \| M _ {t} ^ {A} \| _ {(H _ {t} ^ {A}) ^ {- 1 / 2}} ^ {2} \leq \frac {\eta \| G ^ {A} \| _ {L ^ {1} L ^ {2} ([ 0 , T ])} \sqrt {1 + \log T}}{(1 - \beta_ {1}) (1 - \beta_ {1} / \sqrt {\beta_ {2}}) \sqrt {1 - \beta_ {2}}},
$$

where 

$$
\sum_ {t = 1} ^ {T} \eta_ {t} ^ {2} = \eta^ {2} \sum_ {t = 1} ^ {T} \frac {1}{t} \leq \eta^ {2} (1 + \log T), \quad \mathrm{and} \quad \sum_ {t = 1} ^ {T} \eta_ {t} ^ {3} = \sum_ {t = 1} ^ {T} \frac {\eta}{t ^ {3 / 2}} \leq 1 + \int_ {1} ^ {T} t ^ {- 3 / 2} \mathrm{d} t = 3 - \frac {2}{\sqrt {T}}.\tag{B.9}
$$

Similarly, we can bound the third term on the right-hand side of (B.8), i.e., 

$$
\begin{array}{l} \| M _ {t - 1} ^ {B} \| _ {(H _ {t} ^ {B}) ^ {- 1 / 2}} ^ {2} = \left\| (H _ {t} ^ {B}) ^ {- 1 / 4} (H _ {t - 1} ^ {B}) ^ {1 / 4} (H _ {t - 1} ^ {B}) ^ {- 1 / 4} (M _ {t - 1} ^ {B}) \right\| _ {\mathrm{F}} ^ {2} \\ \leq \left\| (H _ {t} ^ {B}) ^ {- 1 / 4} (H _ {t - 1} ^ {B}) ^ {1 / 4} \right\| _ {2 \to 2} ^ {2} \| M _ {t - 1} ^ {B} \| _ {(H _ {t - 1} ^ {B}) ^ {- 1 / 2}} ^ {2} \\ \leq \left(\frac {1}{\beta_ {2}}\right) ^ {1 / 4} \| M _ {t - 1} ^ {B} \| _ {(H _ {t - 1} ^ {B}) ^ {- 1 / 2}} ^ {2}, \end{array}\tag{B.10}
$$

where the last inequality is given by the fact that 

$$
\begin{array}{r} \left\| (H _ {t} ^ {B}) ^ {- 1 / 4} (H _ {t - 1} ^ {B}) ^ {1 / 4} \right\| _ {2 \to 2} ^ {4} = \left\| \left(\frac {V _ {t - 1} ^ {B}}{V _ {t} ^ {B}}\right) ^ {1 / 4} \right\| _ {L ^ {\infty}} ^ {4} = \left\| \frac {V _ {t - 1} ^ {B}}{\beta_ {2} V _ {t - 1} ^ {B} + (1 - \beta_ {2}) (G _ {t} ^ {B}) ^ {\circ 2}} \right\| _ {L ^ {\infty}} \\ = \left\| 1 + \frac {(1 - \beta_ {2}) V _ {t - 1} ^ {B}}{\beta_ {2} V _ {t - 1} ^ {B} + (1 - \beta_ {2}) (G _ {t} ^ {B}) ^ {\circ 2}} \right\| _ {L ^ {\infty}} \leq 1 + \frac {1 - \beta_ {2}}{\beta_ {2}}. \end{array}
$$

The result in (B.10) allows to apply [RKK18, Lemma 2] (or, equivalently, using (B.9)) also on the time-shifted term $\| M _ { t - 1 } ^ { B } \| _ { ( H _ { t } ^ { B } ) ^ { - 1 / 2 } }$ , which for $1 / \alpha _ { t } ^ { 2 } = \alpha / \sqrt { t }$ gives 

$$
\begin{array}{r l} \sum_ {t = 1} ^ {T} \frac {\| M _ {t - 1} ^ {B} \| _ {(H _ {t} ^ {B}) ^ {- 1 / 2}} ^ {2}}{\alpha_ {t} ^ {2}} & \leq \sum_ {t = 1} ^ {T} \frac {\alpha}{\sqrt {t}} \Big (\frac {1}{\beta_ {2}} \Big) ^ {1 / 4} \| M _ {t - 1} ^ {B} \| _ {(H _ {t - 1} ^ {B}) ^ {- 1 / 2}} ^ {2} \\ & \leq \frac {\alpha \| G ^ {B} \| _ {L ^ {1} L ^ {2} ([ 0 , T ])} \sqrt {1 + \log T}}{\beta_ {2} ^ {1 / 4} (1 - \beta_ {1}) (1 - \beta_ {1} / \sqrt {\beta_ {2}}) \sqrt {1 - \beta_ {2}}}. \end{array}\tag{B.11}
$$

Using assumptions (H2) that $\beta _ { 1 , t } = \beta _ { 1 } b ^ { t }$ , and (H3) that $\operatorname* { s u p } _ { t } \| B _ { t } - B ^ { * } \| _ { \operatorname* { m a x } } \leq D _ { \infty }$ (the same quantity can be bounded by 2 for the term in A, given that Stiefel is compact and has finite diameter), together with the last bound from [RKK18, Theorem 4], and combining the results from (B.9), (B.11),(B.8) (for $A _ { t }$ they are similar), we get 

$$
\begin{array}{l}R (T) \leq \frac {D _ {\infty} ^ {2} \left\| (H _ {T} ^ {B}) ^ {1 / 4} \right\| _ {2 \rightarrow 2} ^ {2}}{2 \eta_ {T} (1 - \beta_ {1})} + \frac {D _ {\infty} ^ {2}}{2 (1 - \beta_ {1})} \sum_ {t = 1} ^ {T} \beta_ {1, t} \alpha_ {t} ^ {2} \left\| (H _ {t} ^ {B}) ^ {1 / 4} \right\| _ {2 \rightarrow 2} ^ {2}\\\quad + \frac {\mathrm{diam} _ {\infty} (\mathrm{St} (n , r)) ^ {2} \left\| (H _ {T} ^ {A}) ^ {1 / 4} \right\| _ {2 \rightarrow 2} ^ {2}}{2 \eta_ {T} (1 - \beta_ {1})} + \frac {\mathrm{diam} _ {\infty} (\mathrm{St} (n , r)) ^ {2}}{2 (1 - \beta_ {1})} \sum_ {t = 1} ^ {T} \beta_ {1, t} \alpha_ {t} ^ {2} \left\| (H _ {t} ^ {A}) ^ {1 / 4} \right\| _ {2 \rightarrow 2} ^ {2}\\\quad + \frac {(\eta + 2 \alpha \beta_ {1} ^ {3 / 4}) \sqrt {1 + \log T}}{2 (1 - \beta_ {1}) ^ {2} (1 - \beta_ {1} / \sqrt {\beta_ {2}}) \sqrt {1 - \beta_ {2}}} \left\| G ^ {A} \right\| _ {L ^ {1} L ^ {2} ([ 0, T ])}\\\quad + \frac {(\eta + 2 \alpha \beta_ {1} ^ {3 / 4}) \sqrt {1 + \log T}}{2 (1 - \beta_ {1}) ^ {2} (1 - \beta_ {1} / \sqrt {\beta_ {2}}) \sqrt {1 - \beta_ {2}}} \left\| G ^ {B} \right\| _ {L ^ {1} L ^ {2} ([ 0, T ])}\\\quad + \frac {1}{2 (1 - \beta_ {1})} \left[ G _ {\infty} ^ {3} \eta^ {2} (1 + \log T) + G _ {\infty} ^ {4} \eta^ {3} \left(3 - \frac {2}{\sqrt {T}}\right) \right].\end{array}\tag{B.12}
$$

To conclude, we bound the two terms $\textstyle \sum _ { t = 1 } ^ { T } \beta _ { 1 , t } \alpha _ { t } ^ { 2 } \| ( H _ { t } ^ { B } ) ^ { 1 / 4 } \| _ { 2 \to 2 } ^ { 2 }$ uniformly in t, for the choice 

$\beta _ { 1 , t } = \beta _ { 1 } b ^ { t }$ , and $\alpha _ { t } ^ { 2 } = \sqrt { t } / \alpha$ , we get the upper bound 

$$
\begin{array}{l} \sum_ {t = 1} ^ {T} \beta_ {1, t} \alpha_ {t} ^ {2} \| (H _ {t} ^ {B}) ^ {1 / 4} \| _ {2 \to 2} ^ {2} \leq \frac {\beta_ {1}}{\alpha} \| \| (H ^ {B}) ^ {1 / 4} \| _ {2 \to 2} ^ {2} \| _ {L _ {t} ^ {\infty}} \sum_ {t = 1} ^ {T} \sqrt {t} b ^ {t} \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qend{array}
$$

This gives the final bound 

$$
\begin{array}{r l} & R (T) \lesssim \frac {D _ {\infty} ^ {2} \| (H _ {T} ^ {B}) ^ {1 / 4} \| _ {2 \to 2} ^ {2}}{2 \eta_ {T} (1 - \beta_ {1})} + \frac {D _ {\infty} ^ {2}}{2 (1 - \beta_ {1})} \frac {\beta_ {1}}{\alpha} \| \| (H ^ {B}) ^ {1 / 4} \| _ {2 \to 2} ^ {2} \| _ {L _ {t} ^ {\infty}} \frac {b}{(1 - b) ^ {3 / 2}} \\ & \qquad + \frac {(\eta + 2 \alpha \beta_ {1} ^ {3 / 4}) \sqrt {1 + \log T}}{2 (1 - \beta_ {1}) ^ {2} (1 - \beta_ {1} / \sqrt {\beta_ {2}}) \sqrt {1 - \beta_ {2}}} \| G ^ {B} \| _ {L ^ {1} L ^ {2} ([ 0, T ])} + \frac {\mathrm{diam} _ {\infty} (\mathrm{St} (n , r)) ^ {2} \| (H _ {T} ^ {A}) ^ {1 / 4} \| _ {2 \to 2} ^ {2}}{2 \eta_ {T} (1 - \beta_ {1})} \\ & \qquad + \frac {\mathrm{diam} _ {\infty} (\mathrm{St} (n , r)) ^ {2}}{2 (1 - \beta_ {1})} \frac {\beta_ {1}}{\alpha} \| \| (H ^ {A}) ^ {1 / 4} \| _ {2 \to 2} ^ {2} \| _ {L _ {t} ^ {\infty}} \frac {b}{(1 - b) ^ {3 / 2}} \\ & \qquad + \frac {(\eta + 2 \alpha \beta_ {1} ^ {3 / 4}) \sqrt {1 + \log T}}{2 (1 - \beta_ {1) 2} (1 - \beta_ {1} / \sqrt {\beta_ {2}}) \sqrt {1 - \beta_ {2}}} \| G ^ {A} \| _ {L ^ {1} L ^ {2} ([ 0, T ])} \\ & \qquad + \frac {1}{2 (1 - \beta_ {1})} \left[ G _ {\infty} ^ {3} \eta^ {2} (1 + \log T) + G _ {\infty} ^ {4} \eta^ {3} \left(3 - \frac {2}{\sqrt T}\right) \right]. \end{array}
$$

By using the fact that $\eta _ { T } ^ { - 1 }$ , we can collect the constant $C _ { 1 }$ of the terms of order 1, the constant $C _ { 2 }$ of order $\sqrt { T }$ , the constant $C _ { 3 }$ of the terms of order $\sqrt { 1 + \log T }$ , and $C _ { 4 }$ of the terms of order log T and with $C _ { 5 }$ the constant of the term of order $T ^ { - 1 / 2 }$ to get the final bound. In particular, the order of the bound is 

$$
R (T) \leq C _ {1} \sqrt {T} + C _ {2} + C _ {3} \sqrt {1 + \log T} + C _ {4} \log T + C _ {5} T ^ {- 1 / 2}.
$$

Definition B.1. (H-norm induced by a full-rank operator) Consider a self-adjoint positive definite operator $H \colon V  V$ defined on a finite-dimensional real Hilbert space $( V , g )$ ). We define the H-weighted inner product as 

$$
g _ {H} (v, w) := g (H ^ {1 / 2} v, H ^ {1 / 2} w),
$$

and denote the corresponding norm as 

$$
\| x \| _ {H} = \| H ^ {1 / 2} x \| _ {g}.
$$

Lemma B.2. (Properties of H-induced inner products) Let H and V be as in Definition B.1. Then the following holds: 

$$
\bullet g _ {H ^ {\gamma}} (H ^ {\alpha} v, H ^ {\beta} w) = g _ {H ^ {\gamma}} (v, H ^ {\alpha + \beta} w) = g _ {H ^ {\alpha + \beta + \gamma}} (v, w),
$$

$$
\bullet \| H ^ {\alpha} x \| _ {H ^ {\gamma}} = \| H ^ {\alpha + \gamma / 2} x \| _ {g} = \| x \| _ {H ^ {2 \alpha + \gamma}}.
$$

Proof. The first point follows from the definition, self-adjointness $g ( H v , w ) = g ( v , H w )$ (which holds for any power), and the fact that powers commute $( H ^ { \alpha } H ^ { \beta } = H ^ { \beta } H ^ { \alpha }$ for all $\alpha , \beta )$ 

$$
\begin{array}{c} g _ {H ^ {\gamma}} (H ^ {\alpha} v, H ^ {\beta} w) = g (H ^ {\alpha + \gamma / 2} v, H ^ {\beta + \gamma / 2} w) = g (v, H ^ {\alpha + \beta + \gamma} w) = g (H ^ {(\alpha + \beta + \gamma) / 2} v, H ^ {(\alpha + \beta + \gamma) / 2} w) \\ = g _ {H ^ {\alpha + \beta + \gamma}} (v, w). \end{array}
$$

The first equality is similar 

$$
g _ {H ^ {\gamma}} (H ^ {\alpha} v, H ^ {\beta} w) = g (H ^ {\alpha} H ^ {\gamma / 2} v, H ^ {\beta} H ^ {\gamma / 2} w) = g (H ^ {\gamma / 2} v, H ^ {\gamma / 2} H ^ {\alpha + \beta} w) = g _ {H ^ {\gamma}} (v, H ^ {\alpha + \beta} w).
$$

The norm equality follows immediately from the definition 

$$
\| H ^ {\alpha} x \| _ {H ^ {\gamma}} ^ {2} = g _ {H ^ {\gamma}} (H ^ {\alpha} x, H ^ {\alpha} x) = g (H ^ {\alpha + \gamma / 2} x, H ^ {\alpha + \gamma / 2} x) = \| H ^ {\alpha + \gamma / 2} x \| _ {g} ^ {2} = \| x \| _ {H ^ {2 \alpha + \gamma}} ^ {2}.
$$

Lemma B.3. (Young inequality for dual H norms) Let H, V be as in Definition B.1 and let $u , v \in V , \zeta \neq 0$ , and $\alpha \in \mathbb { R }$ . Then, we have 

$$
g (u, v) \leq \frac {1}{2 \zeta^ {2}} \| u \| _ {H ^ {- \alpha}} ^ {2} + \frac {\zeta^ {2}}{2} \| v \| _ {H ^ {\alpha}} ^ {2}.
$$

Proof. Consider the expansion 

$$
0 \leq \| u - v \| _ {g} ^ {2} = g (u - v, u - v) = \| u \| _ {g} ^ {2} + \| v \| _ {g} ^ {2} - 2 g (u, v),
$$

which implies 

$$
g (u, v) \leq \frac {1}{2} \| u \| _ {g} ^ {2} + \frac {1}{2} \| v \| _ {g} ^ {2}.
$$

Now, since H is self-adjoint and positive definite with respect to the inner product $^ { g , }$ we have that $\zeta H ^ { \alpha }$ is too, and therefore we get 

$$
g (\zeta^ {- 1} H ^ {- \alpha / 2} u, \zeta H ^ {\alpha / 2} v) \leq \frac {1}{2} \| \zeta^ {- 1} H ^ {- \alpha / 2} u \| _ {g} ^ {2} + \frac {1}{2} \| \zeta H ^ {\alpha / 2} v \| _ {g} ^ {2} = \frac {1}{2 \zeta^ {2}} \| u \| _ {H ^ {- \alpha}} ^ {2} + \frac {\zeta^ {2}}{2} \| v \| _ {H ^ {\alpha}} ^ {2}.
$$

B.1. Additional Results on Diferent Numerical Retractions. In this section, we present numerical results comparing several possible retraction choices on the Stiefel manifold. In Figure 2, we compare matrix size against GPU wall-clock time and final error. In particular, we compare QR decomposition, polar decomposition, a direct solver for the Cayley linear system (Cayley-Direct), an iterative method for the Cayley linear system that employs the Sherman–Morrison–Woodbury formula (Cayley-SMW), the fixed-point iteration (Cayley-FP), and the Newton–Schulz iteration. 

## C. Proof of Proposition 4.3

Let $( B , A ) \in { \mathcal { F } }$ and consider 

$$
\begin{array}{r l} & {\| \nabla (\mathcal {L} \circ \Phi) (t B, t ^ {- 1} A) \| ^ {2} = \| \mathrm{D} \Phi (t B, t ^ {- 1} A) ^ {\top} \nabla \mathcal {L} (W) \| ^ {2}} \\ & {\qquad = \| t B ^ {\top} \nabla \mathcal {L} (W) \| ^ {2} + \| \nabla \mathcal {L} (W) A ^ {\top} t ^ {- 1} \| ^ {2} \underset {t \to 0} {\longrightarrow} + \infty ,} \end{array}
$$

![image](images/Stiefel_AdamW_Geometry_Aware_AdamW_for_Linear_Factorization_Blocks/fig4.jpg)


![image](images/Stiefel_AdamW_Geometry_Aware_AdamW_for_Linear_Factorization_Blocks/fig5.jpg)



Figure 2: Comparison of diferent numerical retraction methods.


Table 5: Hyperparameters for GPT2 LoRA fine-tuning on E2E.

| Configuration | Stiefel-AdamW | AdamW | Scaled AdamW | GeoLoRA |
|---|---|---|---|---|
| Learning rate | $8 \times 10^{-3}$ | $8 \times 10^{-3}$ | $8 \times 10^{-3}$ | $8 \times 10^{-3}$ |
| Weight decay | $10^{-4}$ | $10^{-2}$ | $10^{-2}$ | $10^{-4}$ |
| Learning rate schedule ($\beta_1, \beta_2$) | Linear(0.98, 0.98) | Linear(0.9, 0.999) | Linear(0.7, 0.8) | Linear(0.98, 0.98) |


and therefore $\| \nabla ( \mathcal { L } \circ \Phi ) \| _ { L ^ { \infty } ( \mathcal { F } ) } = + \infty$ 

For the second claim, fix A and B such that $\tilde { \Phi } ( B , A ) = W$ . Then, we have 

$$
\sup _ {(B ^ {\prime}, A ^ {\prime}) \in \widetilde {\mathcal {F}}} \| \nabla (\mathcal {L} \circ \widetilde {\Phi}) (B ^ {\prime}, A ^ {\prime}) \| ^ {2} = \sup _ {O \in \mathrm{St} (r, r)} \| \nabla (\mathcal {L} \circ \widetilde {\Phi}) (B O, O ^ {\top} A) \| ^ {2} <   + \infty ,
$$

because of compactness of $\mathrm { S t } ( r , r )$ and continuity. 

## D. Additional Experimental Details

D.1. GPT2 E2E Fine-Tuning. We use hyperparameters tuned as in [ZP24], as reported in Table 5. 