import matplotlib
matplotlib.use("Agg")
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = load_from_disk("codenet_balanced_hf")

bucket_labels = sorted(
    set(dataset["bucket"]), key=lambda b: dataset["bucket"].index(b)
)
bucket_to_locs = {bucket: [] for bucket in bucket_labels}
for row in dataset:
    bucket_to_locs[row["bucket"]].append(row["c_loc"])

percentile_levels = [25, 50, 75, 90, 95, 99]
rows = []
for bucket in bucket_labels:
    locs = np.asarray(bucket_to_locs[bucket])
    row = {
        "bucket": bucket,
        "count": len(locs),
        "mean": locs.mean(),
        "min": int(locs.min()),
        "max": int(locs.max()),
    }
    for p, v in zip(percentile_levels, np.percentile(locs, percentile_levels)):
        row[f"p{p}"] = v
    rows.append(row)

summary = pd.DataFrame(rows).set_index("bucket")
print(summary.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
x = np.arange(len(bucket_labels))
width = 0.13
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(percentile_levels)))
for i, p in enumerate(percentile_levels):
    offset = (i - (len(percentile_levels) - 1) / 2) * width
    ax.bar(x + offset, summary[f"p{p}"].values, width, label=f"p{p}", color=colors[i])
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(bucket_labels)
ax.set_xlabel("Bucket")
ax.set_ylabel("Lines of C Code (log scale)")
ax.set_title("C LOC percentiles by bucket")
ax.legend(ncol=3, fontsize=9)
ax.grid(axis="y", which="both", alpha=0.3)

ax = axes[1]
data = [bucket_to_locs[b] for b in bucket_labels]
bp = ax.boxplot(
    data,
    tick_labels=bucket_labels,
    showmeans=True,
    meanline=True,
    patch_artist=True,
    flierprops=dict(marker=".", markersize=3, alpha=0.3),
)
for patch in bp["boxes"]:
    patch.set_facecolor("#cfe2ff")
    patch.set_edgecolor("#1f4e8a")
ax.set_yscale("log")
ax.set_xlabel("Bucket")
ax.set_ylabel("Lines of C Code (log scale)")
ax.set_title("C LOC distribution by bucket (log scale)")
ax.grid(axis="y", which="both", alpha=0.3)

plt.tight_layout()
out = "c_loc_by_bucket.png"
plt.savefig(out, dpi=140)
print(f"\nsaved: {out}")
