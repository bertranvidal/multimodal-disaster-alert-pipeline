from datasets import load_dataset, concatenate_datasets
import pandas as pd
import random
from collections import Counter
from pathlib import Path
import os

SEED = 42
SAMPLES_PER_CLASS = 100

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "SA_data"


def load_full_damage_dataset():
    train_ds = load_dataset("QCRI/CrisisMMD", "damage", split="train")
    dev_ds = load_dataset("QCRI/CrisisMMD", "damage", split="dev")
    test_ds = load_dataset("QCRI/CrisisMMD", "damage", split="test")

    full_ds = concatenate_datasets([train_ds, dev_ds, test_ds])
    df = full_ds.to_pandas()

    if "image" in df.columns:
        df = df.drop(columns=["image"])

    return df


def sample_balanced_total(df, samples_per_class=100, seed=42):
    parts = []

    for label in sorted(df["label"].unique()):
        label_df = df[df["label"] == label]

        if len(label_df) < samples_per_class:
            raise ValueError(
                f"No hay suficientes muestras para label={label}. "
                f"Disponibles: {len(label_df)}, pedidas: {samples_per_class}"
            )

        sampled = label_df.sample(n=samples_per_class, random_state=seed)
        parts.append(sampled)

    balanced_df = pd.concat(parts, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return balanced_df


def split_by_tweet_id(df, seed=42):
    random.seed(seed)

    groups = [g.copy() for _, g in df.groupby("tweet_id")]
    random.shuffle(groups)

    groups.sort(key=lambda g: (-len(g), -g["label"].nunique()))

    split_targets = {
        "train": {"total": 210, "labels": {0: 70, 1: 70, 2: 70}},
        "dev": {"total": 45, "labels": {0: 15, 1: 15, 2: 15}},
        "test": {"total": 45, "labels": {0: 15, 1: 15, 2: 15}},
    }

    assigned = {split: [] for split in split_targets}
    current_totals = {split: 0 for split in split_targets}
    current_labels = {split: Counter() for split in split_targets}

    def penalty(split_name, group_df):
        group_size = len(group_df)
        label_counts = group_df["label"].value_counts().to_dict()

        total_after = current_totals[split_name] + group_size
        total_target = split_targets[split_name]["total"]

        p = 0

        if total_after > total_target:
            p += (total_after - total_target) * 1000

        for label in [0, 1, 2]:
            after = current_labels[split_name][label] + label_counts.get(label, 0)
            target = split_targets[split_name]["labels"][label]

            if after > target:
                p += (after - target) * 100

            p += abs(target - after) * 0.1

        return p

    for group_df in groups:
        group_counts = group_df["label"].value_counts().to_dict()

        valid_splits = []
        for split_name in split_targets:
            fits_total = current_totals[split_name] + len(group_df) <= split_targets[split_name]["total"]
            fits_labels = all(
                current_labels[split_name][label] + group_counts.get(label, 0)
                <= split_targets[split_name]["labels"][label]
                for label in [0, 1, 2]
            )

            if fits_total and fits_labels:
                valid_splits.append(split_name)

        if valid_splits:
            chosen_split = min(valid_splits, key=lambda s: penalty(s, group_df))
        else:
            chosen_split = min(split_targets.keys(), key=lambda s: penalty(s, group_df))

        assigned[chosen_split].append(group_df)
        current_totals[chosen_split] += len(group_df)

        for label, count in group_counts.items():
            current_labels[chosen_split][label] += count

    split_dfs = {}
    for split_name in split_targets:
        if assigned[split_name]:
            split_dfs[split_name] = pd.concat(assigned[split_name], ignore_index=True)
        else:
            split_dfs[split_name] = pd.DataFrame(columns=df.columns)

    return split_dfs


def rewrite_image_paths(df, split_name):
    df = df.copy()
    df["image_path"] = df["image_path"].apply(
        lambda p: str(Path("images") / split_name / Path(str(p)).name)
    )
    return df


def print_split_stats(split_name, df):
    print(f"\n{split_name.upper()}")
    print(f"Samples: {len(df)}")
    print("Labels:", df["label"].value_counts().sort_index().to_dict())
    print("Unique tweet_id:", df["tweet_id"].nunique())


def check_no_tweet_overlap(train_df, dev_df, test_df):
    train_ids = set(train_df["tweet_id"])
    dev_ids = set(dev_df["tweet_id"])
    test_ids = set(test_df["tweet_id"])

    print("\nOVERLAP CHECK")
    print("train-dev:", len(train_ids & dev_ids))
    print("train-test:", len(train_ids & test_ids))
    print("dev-test:", len(dev_ids & test_ids))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_df = load_full_damage_dataset()

    balanced_df = sample_balanced_total(
        full_df,
        samples_per_class=SAMPLES_PER_CLASS,
        seed=SEED
    )

    splits = split_by_tweet_id(balanced_df, seed=SEED)

    train_df = rewrite_image_paths(splits["train"], "train")
    dev_df = rewrite_image_paths(splits["dev"], "dev")
    test_df = rewrite_image_paths(splits["test"], "test")

    train_df.to_csv(OUTPUT_DIR / "crisismmd_damage_train.csv", index=False)
    dev_df.to_csv(OUTPUT_DIR / "crisismmd_damage_dev.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "crisismmd_damage_test.csv", index=False)

    print_split_stats("train", train_df)
    print_split_stats("dev", dev_df)
    print_split_stats("test", test_df)

    check_no_tweet_overlap(train_df, dev_df, test_df)

    print(f"\nArchivos guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()