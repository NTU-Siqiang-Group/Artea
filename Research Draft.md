# Research Draft

## Problem Setting

Considering a metric space $(\mathcal{M},\delta)$ where $\mathcal{M}$ is a (possibly infinite) set, and $\delta$ is a function that given two points $p_1,p_2\in \mathcal{M}$，computes in constant time a non-negative real value as their distance, denoted as $\delta(p_1,p_2)$.

The function $\delta$ satisfices:

- identity of indiscernible: $\delta(p_1, p_2)=0$ if and only if $p_1 = p_2$;
- symmetry: $\delta(p_1,p_2) = \delta(p_2,p_1)$;
- triangle inequality: $\delta(p_1,p_2)\le \delta(p_1,p_3) + \delta(p_2,p_3)$.

Let $P$ be a set of $n\ge 2$ points from $\mathcal{M}$, which we refer to as the data points. Given a point $q\in\mathcal{M}$, a point $p*$ is a nearest neighbor (NN) of $q$ if $\delta(p^*,q)\le \delta(p,q)$ holds for all $p\in P$. For a value $\epsilon > 0$, a point $p\in P$ is called a <u>**$(1+\epsilon)$-approximate nearest neighbor**</u> (ANN) of $q$ if $\delta(p,q)\le (1+\epsilon)\cdot \delta(p^*,q)$; 

## A Theoretical Prototype of Artea Graph

