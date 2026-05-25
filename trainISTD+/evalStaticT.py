import pandas as pd
from scipy import stats

ours = pd.read_excel(r"D:\welemon2077\my_cv_storyCVPR\_2_SPL_PhasorFormer-Neurocomputing\Neurocomputing\组员任务\统计数据\SRD_ours.xlsx", sheet_name="Per Image")
shadowformer = pd.read_excel(r"D:\welemon2077\my_cv_storyCVPR\_2_SPL_PhasorFormer-Neurocomputing\Neurocomputing\组员任务\统计数据\SRD_OmniSR.xlsx", sheet_name="Per Image")

# Find common filenames
common_files = set(ours["filename"]).intersection(set(shadowformer["filename"]))
print(f"Number of common files: {len(common_files)}")
print(f"Ours file count: {len(ours)}")
print(f"ShadowFormer file count: {len(shadowformer)}")

# Filter and sort by common files
ours_filtered = ours[ours["filename"].isin(common_files)].sort_values("filename").reset_index(drop=True)
shadow_filtered = shadowformer[shadowformer["filename"].isin(common_files)].sort_values("filename").reset_index(drop=True)

# Verify exact filename match
assert ours_filtered["filename"].equals(shadow_filtered["filename"]), "Filenames do not match"

metrics = ["full_psnr", "full_ssim", "full_mae"]
direction = {"full_psnr": "greater", "full_ssim": "greater", "full_mae": "less"}

print("\n=== Paired Significance Test Results (One-tailed) ===")
for metric in metrics:
    ours_vals = ours_filtered[metric]
    sf_vals = shadow_filtered[metric]

    # Paired t-test (converted to one-tailed)
    t_stat, p_two = stats.ttest_rel(ours_vals, sf_vals)
    if direction[metric] == "greater":
        p_ttest = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    else:  # less
        p_ttest = p_two / 2 if t_stat < 0 else 1 - p_two / 2

    # Wilcoxon one-tailed test
    w_stat, p_wilcox = stats.wilcoxon(ours_vals, sf_vals, alternative=direction[metric])

    print(f"\n{metric}:")
    print(f"  Paired t-test: t = {t_stat:.4f}, p = {p_ttest:.2e} -> {'Significantly better' if p_ttest < 0.05 else 'Not significant'}")
    print(f"  Wilcoxon: W = {w_stat:.1f}, p = {p_wilcox:.2e} -> {'Significantly better' if p_wilcox < 0.05 else 'Not significant'}")

print("\n=== Paired Significance Test Results (Two-tailed) ===")
for metric in metrics:
    ours_vals = ours_filtered[metric]
    sf_vals = shadow_filtered[metric]

    # Paired t-test (two-tailed)
    t_stat, p_ttest = stats.ttest_rel(ours_vals, sf_vals)

    # Wilcoxon two-tailed test
    w_stat, p_wilcox = stats.wilcoxon(ours_vals, sf_vals)

    print(f"\n{metric}:")
    print(f"  Paired t-test: t = {t_stat:.4f}, p = {p_ttest:.2e} -> {'Significant difference' if p_ttest < 0.05 else 'No significant difference'}")
    print(f"  Wilcoxon: W = {w_stat:.1f}, p = {p_wilcox:.2e} -> {'Significant difference' if p_wilcox < 0.05 else 'No significant difference'}")

print("\n=== Mean and Standard Deviation ===")
for metric in metrics:
    ours_mean = ours_filtered[metric].mean()
    ours_std  = ours_filtered[metric].std()
    sf_mean   = shadow_filtered[metric].mean()
    sf_std    = shadow_filtered[metric].std()
    print(f"{metric}:")
    print(f"  Ours:   {ours_mean:.4f} ± {ours_std:.4f}")
    print(f"  Shadow: {sf_mean:.4f} ± {sf_std:.4f}")