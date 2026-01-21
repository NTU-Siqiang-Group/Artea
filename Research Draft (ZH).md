# 研究草稿

## (1) Preliminaries

### (1.1) 问题定义

给定一个度量空间 $(\mathcal{M,\delta})$，其中 $\mathcal{M}$ 是一个集合（通常是无限的），$\delta$ 表示距离计算函数，对于 $\mathcal{M}$ 中任意两个点 $p_1,p_2$ 满足交换律和三角不等式：

- $\delta(p_1,p_2) = \delta(p_2,p_1)$;
- $\delta(p_1,p_2)\le \delta(p_1,p_3) + \delta(p_2,p_3)$.

同时，我们也定义点和一个集合之间的距离为点和它在这个集合中的最近点之间的距离，即 $\delta(q, Q) = \delta(Q,q) = \min_{v \in Q} \delta(q,v)$。

令 $P$ 为包含 $n$ 个点的数据集。给定 query 点 $q\in \mathcal{M}$，$q$ 的最近邻定义为：$\mathop{\mathrm{argmin}}\limits_{p\in P}\ \delta(p,q)$。

我们要研究的问题为：给定值 $\epsilon, \tau > 0$，对于任意 $q\in \mathcal{M}$，若满足 $\delta(q,P)\le \tau$，ANN Query 能返回一个点 $p$ 满足：
$$
\delta(p,q) \le (1+\epsilon)\cdot \delta(p^*,q)
$$
其中 $p^*$ 表示点 $q$ 在数据集 $P$ 中的最近邻（i.e. $p^* = \mathop{\mathrm{argmin}}\limits_{p\in P}\ \delta(p,q)$）。

### (1.2) $r$-net

假设我们要研究的数据集为点集 $P$。给定一个数值 $r > 0$， $P$ 的 **$r$-net**是 $P$ 的一个子集 $V_r \subseteq P$，且满足：

*   **（Separation Property）** 对于任意两个不同的点 $v_1, v_2 \in V_r$，有 $\delta(v_1, v_2) \ge r$；
*   **（Covering Property）** $P \subseteq \bigcup_{v \in V_r} B(v, r)$，即对于 $\forall p \in P$，$\exists$ 一个点 $v \in V_r$ 使得 $\delta(p, v) \le r$。

---

## (2) Artea Graph 理论原型

在本节，我们提出一种名为 Artea Graph 的新的 Proximity Graph，它改进现有的所有 Proximity Graph 的时间复杂度到 $O(\log\Delta)$，和 Navigating Nets 一致，同时，我们的算法在实践上对比 Navigating Nets 有更高的效率。构建 Artea Graph 的时间复杂度为 [TODO]，在 [TODO] 节，我们会提出一个实践上的版本，支持对 Artea Graph 进行高效的构建和更新。

### (2.1) 数据结构

我们假设本文研究的数据集为 $P$，它有有限的 Doubling Dimension 记作 $\lambda = O(d)$，而 $P$ 的 Aspect Ratio 可以记作：
$$
\Delta_P = \frac{D_{\max}(P)}{D_{\min}(P)}
$$
为了简化，我们在下文直接假定，数据集 $P$ 的最短两点间距离记作 $D_{\min} = D_{\min}(P)$，最远两点间距离记为 $D_{\max}=D_{\max}(P)$

给定参数 $\beta$，Artea Graph 是一个多层级结构 $\mathcal{G}(\beta)=(\mathcal{V},\mathcal{E})$，我们将该多层级图的最底层记为 $\mathcal{G}_0=(\mathcal{V}_0,\mathcal{E}_0)$，其中 $\mathcal{V}_0 = P$。

除了 $\mathcal{V}_0$ 外，Artea Graph 的第 $i$ 层可以看作是下一层（第 $i-1$ 层）的一个 $(\beta^i \cdot \rho)$-net。即：
$$
\mathcal{V}_i = \text{a }(\beta^i \cdot D_{\min})\text{-net of }\ \mathcal{V}_{i-1}, \forall\ i = 1, 2,\cdots i_{\max}
$$
其中，$i_\max\approx \log_{\beta}\Delta = \log_{\beta}(\frac{D_{\max}}{D_{\min}})$。同时，我们保证 $\mathcal{V_i} \subset \mathcal{V_{i-1}} \subset \cdots \subset \mathcal{V_1} \subset \mathcal{V}_0$..

显然，$\mathcal{V}_i$ 中任意两点间的最短距离为 $\beta^i\cdot D_{\min}(P)$.

现在，我们已经定义了 Artea Graph 每一层的点集 $\mathcal{V}_i$，下面，我们进一步定义 Artea Graph 每一层的边集 $\mathcal{E}_i$：

### () Covering Property of Artea Graph

Artea Graph 的任意一层 $\mathcal{V}_{i}$，对数据集 $P$ 都是 $\beta^{i+1}\cdot D_{\min}$-covering 的。即：
$$
\forall\ p\in P,\ \exists\ v\in \mathcal{V_i},\ \delta(p,v)\le \beta^{i+1} D_{\min}
$$
首先，我们考虑 **base case**，$\mathcal{V}_0$ 即包含

### () Entry point 性质





