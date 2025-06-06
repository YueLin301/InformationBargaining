"""
End‑to‑end script:
1.  Merge solver outputs, read ground‑truth and hypothesis labels.
2.  Compute Pearson correlations plus Fisher‑z statistics.
3.  Auto‑generate plain‑English conclusions about:
      • Solver validation quality
      • Support for the hypothesis
      • Whether the two correlations differ significantly
All messages and comments are in English.
"""

import pandas as pd
from fractions import Fraction
from math import log, sqrt, exp
from scipy.stats import norm

# --------------------------- File paths ---------------------------
# model_name = "gpt-4o-mini"
# model_name = "gpt-4.1-mini-2025-04-14"
# model_name = "o3-2025-04-16"

# model_name = "deepseek-chat"  # V3
model_name = "deepseek-reasoner"  # R1

csv1a_path = f"results/{model_name}/bargaining_statistics_summary.csv"
csv1b_path = f"results/{model_name}/signaling_statistics_summary.csv"
csv2_path  = "task_truth_n_hypothesis.csv"
combined_path = f"results/{model_name}/combined_statistics_summary.csv"

# --------------------------- 0. Merge solver results ---------------------------
df_a = pd.read_csv(csv1a_path)          # 72 rows
df_b = pd.read_csv(csv1b_path)          # 15 rows
combined = pd.concat([df_a, df_b], ignore_index=True)
combined.to_csv(combined_path, index=False)
print(f"[INFO] Merged {len(combined)} rows  ➜  {combined_path}")

# --------------------------- 1. Solver outputs ---------------------------
result_col = "last_proposer_payoff_mean"
solver_results = combined[result_col].astype(float)

# --------------------------- 2. Ground‑truth & hypothesis ---------------------------
labels = pd.read_csv(csv2_path)
gt_raw = labels["last_proposer_payoff_ground_truth"].astype(str)
hs_raw = labels["last_proposer_payoff_hypothesis"].astype(str)

def parse_number(text: str):
    """Convert '', 'None', fractions such as '1/3', or plain numbers to float / None."""
    text = (text or "").strip().lower()
    if text in {"", "none", "nan"}:
        return None
    if "/" in text:
        try:
            return float(Fraction(text))
        except Exception:
            return None
    try:
        return float(text)
    except Exception:
        return None

gt_vals = gt_raw.map(parse_number)
hs_vals = hs_raw.map(parse_number)

mask_gt = gt_vals.notna()
mask_hs = hs_vals.notna()

print(f"[INFO] valid rows – ground_truth: {mask_gt.sum()} / {len(gt_vals)}, "
      f"hypothesis: {mask_hs.sum()} / {len(hs_vals)}")

# --------------------------- 3. Pearson correlations ---------------------------
r_gt = solver_results[mask_gt].corr(gt_vals[mask_gt])
r_hs = solver_results[mask_hs].corr(hs_vals[mask_hs])

print("\n=== Pearson correlation ===")
print(f"  solver vs ground_truth : r = {r_gt:.4f}")
print(f"  solver vs hypothesis   : r = {r_hs:.4f}")

# --------------------------- 4. Fisher‑z helpers ---------------------------
def fisher_z(r):                # r‑to‑z conversion
    return 0.5 * log((1 + r) / (1 - r))

def ci_from_r(r, n, alpha=0.05):   # two‑sided 1‑alpha CI for r
    z, se = fisher_z(r), 1 / sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    lo, hi = z - z_crit * se, z + z_crit * se
    r_lo = (exp(2 * lo) - 1) / (exp(2 * lo) + 1)
    r_hi = (exp(2 * hi) - 1) / (exp(2 * hi) + 1)
    return r_lo, r_hi

def p_two_tailed_from_r(r, n):     # H0: ρ = 0
    z = fisher_z(r) * sqrt(n - 3)
    return 2 * (1 - norm.cdf(abs(z)))

# --------------------------- 5. CIs & p‑values ---------------------------
n_gt, n_hs = mask_gt.sum(), mask_hs.sum()
ci_gt, ci_hs = ci_from_r(r_gt, n_gt), ci_from_r(r_hs, n_hs)
p_gt, p_hs   = p_two_tailed_from_r(r_gt, n_gt), p_two_tailed_from_r(r_hs, n_hs)

print("\n=== Individual tests (H0: ρ = 0) ===")
print(f"  ground_truth : 95% CI = [{ci_gt[0]:.3f}, {ci_gt[1]:.3f}],  p = {p_gt:.4g}")
print(f"  hypothesis   : 95% CI = [{ci_hs[0]:.3f}, {ci_hs[1]:.3f}],  p = {p_hs:.4g}")

# --------------------------- 6. Difference test between two r's ---------------------------
z_gt, z_hs = fisher_z(r_gt), fisher_z(r_hs)
se_diff = sqrt(1 / (n_gt - 3) + 1 / (n_hs - 3))
z_stat  = (z_hs - z_gt) / se_diff
p_diff  = 2 * (1 - norm.cdf(abs(z_stat)))

print("\n=== Difference test (r_hypothesis vs r_ground_truth) ===")
print(f"  z = {z_stat:.3f},  p = {p_diff:.4g}")

# --------------------------- 7. Automated conclusion ---------------------------
def corr_strength(r):
    if abs(r) >= .9:    return "very strong"
    if abs(r) >= .8:    return "strong"
    if abs(r) >= .5:    return "moderate"
    if abs(r) >= .3:    return "weak"
    return "negligible"

print("\n=== Conclusion ===")

# Solver validation
if p_gt < 0.05 and r_gt >= .8:
    print(f"- Solver validation: SUCCESS. The correlation with ground‑truth is {corr_strength(r_gt)} "
          f"(r={r_gt:.3f}, n={n_gt}), explaining ≈{r_gt**2*100:.1f}% of the variance.")
else:
    print("- Solver validation: INCONCLUSIVE or FAILED. Correlation with ground‑truth is not sufficiently strong "
          f"(r={r_gt:.3f}, p={p_gt:.4g}).")

# Hypothesis support
if n_hs < 5:
    size_note = "Sample size is very small; interpret with caution."
else:
    size_note = ""

if p_hs < 0.05:
    print(f"- Hypothesis: PRELIMINARY SUPPORT. The solver aligns {corr_strength(r_hs)}ly with the hypothesized values "
          f"(r={r_hs:.3f}, n={n_hs}). {size_note}")
else:
    print(f"- Hypothesis: NO EVIDENCE of alignment (r={r_hs:.3f}, p={p_hs:.4g}). {size_note}")

# Difference between correlations
if p_diff < 0.05:
    print("- The two correlations differ significantly (p < 0.05), indicating unequal predictive performance.")
else:
    print("- No significant difference between the two correlations (p ≥ 0.05); "
          "the solver relates to the hypothesis about as strongly as to the ground‑truth.")
