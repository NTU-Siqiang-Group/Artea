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

For $99\%$ confidence, $Z_{\alpha/2} = Z_{0.005} = 2.5758$. Substituting the values:

$$s \approx \frac{2.5758^2}{0.001 \times 0.1^2} = \frac{6.6348}{0.00001} = 663,480$$

Therefore, approximately **663,000 samples** are required to estimate $P_{0.1}$ with $99\%$ confidence and $10\%$ relative error.

**Note:** The implementation uses precise Z-values from Boost.Math (e.g., $Z_{99\%} = 2.5758$) rather than approximations (2.576).

#### Additional Examples

| Target Percentile | $p$ | Confidence $(1-\alpha)$ | $Z_{\alpha/2}$ | Relative Error $\delta$ | Required Samples $s$ |
|-------------------|-----|-------------------------|----------------|-------------------------|---------------------|
| $P_{0.1}$ (0.1%)  | 0.001 | 95% | 1.9600 | 10% | $\approx 384,000$ |
| $P_{0.1}$ (0.1%)  | 0.001 | 99% | 2.5758 | 10% | $\approx 663,000$ |
| $P_{0.1}$ (0.1%)  | 0.001 | 99% | 2.5758 | 20% | $\approx 166,000$ |
| $P_{0.05}$ (0.05%) | 0.0005 | 95% | 1.9600 | 10% | $\approx 768,000$ |
| $P_{0.05}$ (0.05%) | 0.0005 | 99% | 2.5758 | 10% | $\approx 1,326,000$ |
| $P_{0.01}$ (0.01%) | 0.0001 | 95% | 1.9600 | 10% | $\approx 3,841,000$ |
| $P_{0.01}$ (0.01%) | 0.0001 | 99% | 2.5758 | 10% | $\approx 6,634,000$ |


**Key Observations:**
- Higher confidence levels increase sample requirements by approximately $(Z_{99\%}/Z_{95\%})^2 \approx 1.73\times$
- Tighter relative error (smaller $\delta$) increases sample requirements quadratically

---

## (2) RadiusProber Mathematical Derivation

### (2.1) Problem Statement and Independence Requirement

Given a dataset $\mathcal{D}$ of $n$ vectors, we want to estimate the $p$-th quantile of the pairwise distance distribution.

**Critical Statistical Requirement:** To apply the quantile estimation theory from Section (1), we need **independent and identically distributed (i.i.d.)** distance samples. However, if we sample $s$ vectors and compute all $\binom{s}{2}$ pairwise distances, these distances **share endpoints** and are therefore **not independent**.

**Solution:** The RadiusProber algorithm ensures independence by:

1. Sampling $m$ vectors uniformly at random as the **first endpoints** (with replacement)
2. Sampling $m$ vectors uniformly at random as the **second endpoints** (with replacement)
3. Computing distances between corresponding pairs: $d_i = \text{dist}(\text{vec}_1[i], \text{vec}_2[i])$ for $i = 1, \ldots, m$

This produces $m$ **independent** distance samples, allowing us to directly apply the formulas from Section (1).

### (2.2) Sample Size Determination

From Section (1.3), to estimate a quantile with:
- Target quantile $p$ (e.g., $p = 0.0005$ for $P_{0.05}$)
- Confidence level $(1-\alpha)$ (e.g., 95% or 99%)
- Relative error $\delta$ (e.g., 10%)

We need $m$ **independent distance samples** where:

$$\boxed{m = \frac{Z_{\alpha/2}^2 (1-p)}{p \delta^2}}$$

**Key Simplification:** Unlike the old approach, we no longer need to convert between vector samples and distance samples. The parameter $m$ directly specifies the number of distance samples to collect.

### (2.3) Practical Examples

#### Example: Estimating $P_{0.1}$ (0.1% percentile) with 99% confidence and 10% relative error

From Section (1.4), we need $m \approx 662,827$ distance samples.

**With the new independent sampling approach:** Simply set `num_distances_to_sample = 662,827`.

**Summary table**:

| Target | $p$ | Confidence | $\delta$ | Required Distance Samples $m$ |
|--------|-----|------------|----------|-------------------------------|
| $P_{0.1}$ | 0.001 | 95% | 10% | 383,762 |
| $P_{0.1}$ | 0.001 | 99% | 10% | 662,827 |
| $P_{0.1}$ | 0.001 | 99% | 20% | 165,707 |
| $P_{0.05}$ | 0.0005 | 95% | 10% | 767,908 |
| $P_{0.05}$ | 0.0005 | 99% | 10% | 1,326,316 |
| $P_{0.01}$ | 0.0001 | 95% | 10% | 3,841,076 |
| $P_{0.01}$ | 0.0001 | 99% | 10% | 6,634,234 |

### (2.4) Implementation Notes

The RadiusProber implementation in `radius_prober.hpp`:

1. **Independent Sampling**: Samples two independent sets of $m$ vector IDs using `RandomSeq` (with replacement)
2. **Paired Distance Computation**: Computes $m$ distances between corresponding pairs in parallel using TBB
3. **Quantile Extraction**: Sorts distances and returns the empirical $p$-th quantile

**Time Complexity:** $O(m \cdot d)$ where $d$ is the vector dimension

**Space Complexity:** $O(m)$ for storing distance samples

**Parallelization:** Both sampling and distance computation are parallelized using TBB for efficiency.

**Key Advantage:** This approach ensures statistical independence of distance samples, which is required for the quantile estimation formulas to be valid.

### (2.5) Parameter Recommendations

For ANNS applications targeting $P_{0.1}$ (0.1% percentile):

- **Standard estimation** (95% confidence, 10% error): $m \approx 384,000$ distances
- **High-confidence estimation** (99% confidence, 10% error): $m \approx 663,000$ distances
- **Relaxed-error estimation** (99% confidence, 20% error): $m \approx 166,000$ distances

**Trade-off:** More samples provide better quantile estimates but increase computational cost linearly (not quadratically as in the old approach).
