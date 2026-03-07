# Bottom Layer Radius Determination

## (1) Preliminaries: Quantile Estimation

### (1.1) Problem Definition and Variables

Let $X$ be a random variable with an unknown distribution function $F(x)$. We aim to estimate the quantile $\xi_p$ such that $F(\xi_p) = p$.

- **$p$**: The target probability associated with the quantile $P_s$.
    >   *Note: If $P_s$ is the $0.1$-th percentile ($P_{0.1}$), then $p = 0.001$.*
- **$s$**: The sample size to be determined.
- **$\hat{p}$**: The empirical proportion of samples less than or equal to the true quantile $\xi_p$ (e.g., if the target is $p=0.001$ and you collect $s=10,000$ samples, finding that $12$ of them are below the true threshold value, then $\hat{p} = \frac{12}{10,000} = 0.0012$).
- **$(1 - \alpha)$**: The desired confidence level (e.g., 0.95 or 95%).
- **$\delta$**: The allowable **relative error** (margin of error expressed as a percentage of $p$).

### (1.2) Statistical Foundation

Let $Y$ be the number of observations in a random sample of size $s$ that fall below the true quantile $\xi_p$. Since each observation independently falls below $\xi_p$ with probability $p$, $Y$ follows a Binomial distribution:
$$Y \sim \text{Binomial}(s, p)$$

The sample proportion is given by $\hat{p} = Y/s$. According to the **Central Limit Theorem**, for a sufficiently large $s$, the distribution of $\hat{p}$ can be approximated by a Normal distribution:

$$\hat{p} \sim \mathcal{N}\left(p, \frac{p(1-p)}{s}\right)$$

### (1.3) Formulating and Derivation of the Relative Error Constraint

We require that the estimated probability $\hat{p}$ does not deviate from the true probability $p$ by more than a fraction $\delta$ (relative error), with a confidence level of $1 - \alpha$.

Mathematically, this condition is expressed as:

$$P\left( |\hat{p} - p| \le \delta \cdot p \right) \ge 1 - \alpha$$

Here, $\epsilon = \delta \cdot p$ represents the **absolute error margin**. Standardize the inequality using the Z-score transformation ($Z = ({\hat{p} - \mu}) / {\sigma}$):

$$P\left( \left| \frac{\hat{p} - p}{\sqrt{{p(1-p)} / {s}}} \right| \le \frac{\delta \cdot p}{\sqrt{{p(1-p)} / {s}}} \right) \ge 1 - \alpha$$

Let $Z_{\alpha/2}$ be the critical value from the standard normal distribution (e.g., $1.96$ for $95\%$ confidence). To satisfy the condition, the boundary of the standardized interval must be at least $Z_{\alpha/2}$:

$$\frac{\delta \cdot p}{\sqrt{\frac{p(1-p)}{s}}} = Z_{\alpha/2}$$

Now, we solve for $s$. First, square both sides:

$$\frac{\delta^2 p^2}{{p(1-p)}/{s}} = Z_{\alpha/2}^2 \implies \boxed{s = \frac{Z_{\alpha/2}^2 (1-p)}{p \delta^2}}$$

### (1.4) Examples

Consider the case that estimating $P_{0.1}$ (The 0.1-th percentile) with confidence  ($1-\alpha$) = $99\%$ and relative error ($\delta$) = $10\%$ ($0.1$). For the specific case $P_{0.1}$, $p$ is extremely small ($p = 0.001 \ll 1$). Therefore, we can approximate $(1 - p) \approx 1$. The formula simplifies to:

$$s \approx {Z_{\alpha/2}^2} / \left( {p \cdot \delta^2} \right)$$

For $99\%$ confidence, $Z_{\alpha/2} = Z_{0.005} = 2.576$. Substituting the values:

$$s \approx \frac{2.576^2}{0.001 \times 0.1^2} = \frac{6.635}{0.00001} = 663,500$$

Therefore, approximately **663,500 samples** are required to estimate $P_{0.1}$ with $99\%$ confidence and $10\%$ relative error.

#### Additional Examples

| Target Percentile | $p$ | Confidence $(1-\alpha)$ | $Z_{\alpha/2}$ | Relative Error $\delta$ | Required Samples $s$ |
|-------------------|-----|-------------------------|----------------|-------------------------|---------------------|
| $P_{0.1}$ (0.1%)  | 0.001 | 95% | 1.96 | 10% | $\approx 384,160$ |
| $P_{0.1}$ (0.1%)  | 0.001 | 99% | 2.576 | 10% | $\approx 663,500$ |
| $P_{0.1}$ (0.1%)  | 0.001 | 99% | 2.576 | 20% | $\approx 165,875$ |
| $P_{0.05}$ (0.05%) | 0.0005 | 95% | 1.96 | 10% | $\approx 768,320$ |
| $P_{0.05}$ (0.05%) | 0.0005 | 99% | 2.576 | 10% | $\approx 1,327,000$ |
| $P_{0.01}$ (0.01%) | 0.0001 | 95% | 1.96 | 10% | $\approx 3,841,600$ |
| $P_{0.01}$ (0.01%) | 0.0001 | 99% | 2.576 | 10% | $\approx 6,635,000$ |


**Key Observations:**
- Higher confidence levels increase sample requirements by approximately $(Z_{99\%}/Z_{95\%})^2 \approx 1.73\times$
- Tighter relative error (smaller $\delta$) increases sample requirements quadratically

---

## (2) RadiusProber Mathematical Derivation

### (2.1) Problem Statement

Given a dataset $\mathcal{D}$ of $n$ vectors, we want to estimate the $p$-th quantile of the pairwise distance distribution. The RadiusProber algorithm:

1. Samples $s$ vectors from $\mathcal{D}$
2. Computes all $\binom{s}{2} = \frac{s(s-1)}{2}$ pairwise distances
3. Returns the empirical $p$-th quantile of these distances

### (2.2) Number of Distance Computations

For $s$ sampled vectors, the number of unique pairwise distances is:

$$m = \binom{s}{2} = \frac{s(s-1)}{2}$$

**Examples:**

| Sampled Vectors $s$ | Pairwise Distances $m$ | Computational Cost |
|---------------------|------------------------|-------------------|
| 100 | 4,950 | ~5K |
| 500 | 124,750 | ~125K |
| 1,000 | 499,500 | ~500K |
| 2,000 | 1,999,000 | ~2M |
| 5,000 | 12,497,500 | ~12.5M |
| 10,000 | 49,995,000 | ~50M |

### (2.3) Statistical Properties

Let $D_{ij}$ denote the distance between vectors $i$ and $j$ in the full dataset $\mathcal{D}$. The true $p$-th quantile of the distance distribution is $\xi_p$.

When we sample $s$ vectors and compute $m = \binom{s}{2}$ pairwise distances, we obtain an empirical quantile $\hat{\xi}_p$. The question is: **How many vector samples $s$ do we need to ensure $\hat{\xi}_p$ is close to $\xi_p$?**

### (2.4) Sample Size Determination

From Section (1), we know that to estimate a quantile with:
- Confidence level $(1-\alpha)$
- Relative error $\delta$

We need $m$ **distance samples** where:

$$m = \frac{Z_{\alpha/2}^2 (1-p)}{p \delta^2}$$

Since RadiusProber computes $\binom{s}{2} = \frac{s(s-1)}{2}$ distances from $s$ vectors, we need:

$$\frac{s(s-1)}{2} \ge m$$

Solving for $s$:

$$s(s-1) \ge 2m$$

$$s^2 - s - 2m \ge 0$$

Using the quadratic formula:

$$s \ge \frac{1 + \sqrt{1 + 8m}}{2}$$

For large $m$, this approximates to:

$$\boxed{s \approx \sqrt{2m}}$$

### (2.5) Practical Examples

#### Example: Estimating $P_{0.1}$ (0.1% percentile) with 99% confidence and 10% relative error

From Section (1.4), we need $m \approx 663,500$ distance samples.

Required vector samples:
$$s \approx \sqrt{2 \times 663,500} = \sqrt{1,327,000} \approx 1,152$$

**Verification:** $\binom{1,152}{2} = \frac{1,152 \times 1,151}{2} = 663,576 \approx 663,500$ ✓

**Summary table**:

| Target | $p$ | Confidence | $\delta$ | Distance Samples $m$ | Vector Samples $s$ | Actual Distances |
|--------|-----|------------|----------|---------------------|-------------------|------------------|
| $P_{0.1}$ | 0.001 | 95% | 10% | 384,160 | 877 | 384,126 |
| $P_{0.1}$ | 0.001 | 99% | 10% | 663,500 | 1,152 | 663,576 |
| $P_{0.1}$ | 0.001 | 99% | 20% | 165,875 | 576 | 165,600 |
| $P_{0.05}$ | 0.0005 | 95% | 10% | 768,320 | 1,240 | 768,780 |
| $P_{0.05}$ | 0.0005 | 99% | 10% | 1,327,000 | 1,629 | 1,325,406 |
| $P_{0.01}$ | 0.0001 | 95% | 10% | 3,841,600 | 2,772 | 3,842,406 |
| $P_{0.01}$ | 0.0001 | 99% | 10% | 6,635,000 | 3,643 | 6,636,663 |

### (2.7) Implementation Notes

The RadiusProber implementation in `radius_prober.hpp`:

1. **Sampling** (lines 132-169): Uses `RandomSeq` to sample $s$ vectors uniformly at random;
2. **Distance Computation** (lines 177-238): Computes all $\binom{s}{2}$ pairwise distances in parallel using TBB;
3. **Quantile Extraction** (lines 104-111): Sorts distances and returns the empirical $p$-th quantile;

**Time Complexity:** $O(s^2 \cdot d)$ where $d$ is the vector dimension

**Space Complexity:** $O(s^2)$ for storing all pairwise distances

**Parallelization:** Both sampling and distance computation are parallelized using TBB for efficiency.

### (2.8) Parameter Recommendations

For ANNS applications targeting $P_{0.1}$ (0.1% percentile):

- **Standard estimation** (95% confidence, 10% error): $s \approx 877$ → $m \approx 384,000$ distances
- **High-confidence estimation** (99% confidence, 10% error): $s \approx 1,152$ → $m \approx 664,000$ distances
- **Relaxed-error estimation** (99% confidence, 20% error): $s \approx 576$ → $m \approx 166,000$ distances

**Trade-off:** More samples provide better quantile estimates but increase computational cost quadratically.
