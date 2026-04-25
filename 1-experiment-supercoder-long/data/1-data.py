# Claude: keep my comments here please lol.
#  pull a dataset (ours, extended dataset)
# https://huggingface.co/datasets/KrishPS/codenet-accepted-c

# pull dataset from (super coder paper) https://huggingface.co/datasets/random1123anonymized/supercoder

# get the average length of programs from supercode (C and Assembly length)
# bucketize the programs into even dataset sizes

# my guess is that super coder only does small C programs

# create a new dataset with the same structure as super coder, but with the programs bucketized into even sizes, also include wether or not the data is in the train or val split of supercoder already


from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
import json
from collections import defaultdict


def safe_extra(item):
    """Safely extract extra_info dict from supercoder row (can be dict or list-of-dicts)."""
    extra = item.get("extra_info", {})
    if isinstance(extra, list) and extra:
        first = extra[0]
        if isinstance(first, dict):
            return first
    if isinstance(extra, dict):
        return extra
    return {}


# ── Load supercoder ───────────────────────────────────────────────────────────
print("Loading supercoder dataset...")
supercoder = load_dataset("random1123anonymized/supercoder")
print(supercoder)

# Supercoder splits: train / val / fewshot
train_split = supercoder["train"]
val_split   = supercoder["val"]

# Inspect one example
ex0 = safe_extra(train_split[0])
print("Supercoder extra_info keys:", list(ex0.keys()))

# Measure C and assembly lengths (LOC = lines of code, matching the paper's metric)
def loc(text):
    return len((text or "").splitlines())

c_locs   = []
asm_locs = []
for row in train_split:
    extra = safe_extra(row)
    c_locs.append(loc(extra.get("c_code", "")))
    asm_locs.append(loc(extra.get("unoptimized_assembly", "")))

# Paper reports: C LOC ~22.3 train, ASM LOC ~130.3 train
print(f"\nSupercoder C LOC   — mean: {np.mean(c_locs):.1f}, "
      f"median: {np.median(c_locs):.1f}, "
      f"min: {np.min(c_locs)}, max: {np.max(c_locs)}")
print(f"Supercoder ASM LOC — mean: {np.mean(asm_locs):.1f}, "
      f"median: {np.median(asm_locs):.1f}, "
      f"min: {np.min(asm_locs)}, max: {np.max(asm_locs)}")

# Build lookup sets (by c_code content) for split tracking
supercoder_train_codes = set()
supercoder_val_codes   = set()
for row in train_split:
    code = safe_extra(row).get("c_code", "")
    if code:
        supercoder_train_codes.add(code.strip())
for row in val_split:
    code = safe_extra(row).get("c_code", "")
    if code:
        supercoder_val_codes.add(code.strip())

print(f"\nSupercoder train: {len(supercoder_train_codes)} unique C programs")
print(f"Supercoder val:   {len(supercoder_val_codes)} unique C programs")


# ── Bucket boundaries from supercoder C LOC quartiles ────────────────────────
bucket_edges = np.percentile(c_locs, [0, 25, 50, 75, 100])
bucket_labels = ["short", "medium_short", "medium_long", "long"]
print(f"\nBucket edges (quartiles of supercoder C LOC): {bucket_edges.astype(int)}")

def assign_bucket(c_loc):
    if c_loc <= bucket_edges[1]:
        return 0
    elif c_loc <= bucket_edges[2]:
        return 1
    elif c_loc <= bucket_edges[3]:
        return 2
    else:
        return 3


# ── Load CodeNet ──────────────────────────────────────────────────────────────
print("\nLoading KrishPS/codenet-accepted-c dataset...")
codenet = load_dataset("KrishPS/codenet-accepted-c")
print(codenet)

cn_example = codenet["train"][0]
print("CodeNet columns:", list(cn_example.keys()))

codenet_locs = [loc(row.get("code", "")) for row in codenet["train"]]
print(f"\nCodeNet C LOC — mean: {np.mean(codenet_locs):.1f}, "
      f"median: {np.median(codenet_locs):.1f}, "
      f"min: {np.min(codenet_locs)}, max: {np.max(codenet_locs)}")


# ── Build new dataset ─────────────────────────────────────────────────────────
print("\nBuilding new dataset...")
records = []
bucket_counts = defaultdict(int)

for row in codenet["train"]:
    c_code    = row.get("code", "") or ""
    prob_id   = row.get("problem_id", "")
    sub_id    = row.get("submission_id", "")
    test_cases_raw = row.get("test_cases", "[]") or "[]"

    try:
        test_cases = json.loads(test_cases_raw)
    except Exception:
        test_cases = []

    c_loc_val  = loc(c_code)
    bucket_idx = assign_bucket(c_loc_val)
    bucket     = bucket_labels[bucket_idx]
    bucket_counts[bucket] += 1

    c_stripped = c_code.strip()
    records.append({
        "problem_id":           prob_id,
        "submission_id":        sub_id,
        "c_code":               c_code,
        "c_loc":                c_loc_val,
        "test_cases":           test_cases,
        "num_test_cases":       row.get("num_test_cases", len(test_cases)),
        "bucket":               bucket,
        "bucket_idx":           bucket_idx,
        "in_supercoder_train":  c_stripped in supercoder_train_codes,
        "in_supercoder_val":    c_stripped in supercoder_val_codes,
    })

print(f"\nFull dataset size: {len(records)}")
print("Bucket distribution:")
for label in bucket_labels:
    count = bucket_counts[label]
    print(f"  {label}: {count:>6}  ({100*count/len(records):.1f}%)")

# How many came from supercoder?
n_train = sum(1 for r in records if r["in_supercoder_train"])
n_val   = sum(1 for r in records if r["in_supercoder_val"])
print(f"\nOverlap with supercoder train: {n_train}")
print(f"Overlap with supercoder val:   {n_val}")


# ── Balance buckets to equal sizes ────────────────────────────────────────────
min_size = min(bucket_counts[b] for b in bucket_labels)
print(f"\nBalancing to {min_size} samples per bucket "
      f"({min_size * len(bucket_labels)} total)...")

per_bucket = defaultdict(list)
for r in records:
    per_bucket[r["bucket"]].append(r)

rng = np.random.default_rng(42)
balanced = []
for label in bucket_labels:
    bucket_recs = per_bucket[label]
    idxs = rng.choice(len(bucket_recs), size=min_size, replace=False)
    balanced.extend(bucket_recs[i] for i in idxs)

print(f"Balanced dataset size: {len(balanced)}")


# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(balanced)
# test_cases is a list — store as JSON string for parquet compatibility
df["test_cases"] = df["test_cases"].apply(json.dumps)
df.to_parquet("codenet_balanced.parquet", index=False)
print("\nSaved -> codenet_balanced.parquet")

hf_dataset = Dataset.from_pandas(df)
hf_dataset.save_to_disk("codenet_balanced_hf")
print("Saved -> codenet_balanced_hf/")
