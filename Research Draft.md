# Research Draft

## (1) Problem Setting

Considering a metric space $(\mathcal{M},\delta)$ where $\mathcal{M}$ is a (possibly infinite) set, and $\delta$ is a function that given two points $p_1,p_2\in \mathcal{M}$，computes in constant time a non-negative real value as their distance, denoted as $\delta(p_1,p_2)$.

The function $\delta$ satisfices:

- identity of indiscernible: $\delta(p_1, p_2)=0$ if and only if $p_1 = p_2$;
- symmetry: $\delta(p_1,p_2) = \delta(p_2,p_1)$;
- triangle inequality: $\delta(p_1,p_2)\le \delta(p_1,p_3) + \delta(p_2,p_3)$.

Let $P$ be a set of $n\ge 2$ points from $\mathcal{M}$, which we refer to as the data points. Given a point $q\in\mathcal{M}$, a point $p*$ is a nearest neighbor (NN) of $q$ if $\delta(p^*,q)\le \delta(p,q)$ holds for all $p\in P$. For a value $\epsilon > 0$, a point $p\in P$ is called a <u>**$(1+\epsilon)$-approximate nearest neighbor**</u> (ANN) of $q$ if $\delta(p,q)\le (1+\epsilon)\cdot \delta(p^*,q)$; 

---

## (2) Theoretical Prototype of Artea Graph

We propose a theoretical prototype of Artea Graph, which is a hierarchical structure consisting of multiple layers of proximity graphs.

Assuming that the minimum distance between any two data points in $P$ is at least $\tau$. The bottom layer of the Artea Graph is a proximity graph $G_0$ constructed over all data points in $P$. Apparently, the minimum distance between any two points in $G_0$ is at least $\tau$. And we can construct higher layers $G_1, G_2, \ldots, G_h$ of the Artea Graph. For layer $G_i$, the minimum distance between any two points is at least $2^i \cdot \tau$. Each layer $G_i$ is constructed over a subset of data points $P_i \subseteq P$, where $P_0 = P$ and $P_{i+1} \subseteq P_i$.



