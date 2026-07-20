import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_records(root):
    for path in sorted(Path(root).glob("vitb16/imagenet/*shots/seed*/cross_dataset/**/eval_log.jsonl")):
        parts = path.parts
        seed = next(part for part in parts if part.startswith("seed"))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                record["_seed"] = seed
                record["_path"] = str(path)
                yield record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="output root, e.g. output_cross_dataset_ori")
    parser.add_argument("--seeds", nargs="+", default=["seed1", "seed2", "seed3"],
                        help="expected seed names used for missing-seed warnings")
    args = parser.parse_args()

    values = defaultdict(dict)
    for record in read_records(args.root):
        if record.get("event") != "transfer_eval":
            continue
        dataset = record["dataset"]
        seed = record["_seed"]
        values[dataset][seed] = float(record["test_accuracy"])

    if not values:
        raise SystemExit(f"No transfer_eval records found under {args.root}")

    print(f"Transfer results from {args.root}")
    print("dataset\tmean\tseed_values")
    means = []
    expected_seeds = set(args.seeds)
    missing = {}
    for dataset in sorted(values):
        seed_values = values[dataset]
        ordered = [seed_values[seed] for seed in sorted(seed_values)]
        mean = sum(ordered) / len(ordered)
        means.append(mean)
        formatted = ", ".join(
            f"{seed}={seed_values[seed]:.2f}"
            for seed in sorted(seed_values))
        print(f"{dataset}\t{mean:.2f}\t{formatted}")
        missing_seeds = sorted(expected_seeds - set(seed_values))
        if missing_seeds:
            missing[dataset] = missing_seeds

    print(f"average\t{sum(means) / len(means):.2f}\t{len(means)} dataset(s)")
    if missing:
        print("\nWARNING: missing expected seeds for at least one dataset:")
        for dataset in sorted(missing):
            print(f"{dataset}: {', '.join(missing[dataset])}")


if __name__ == "__main__":
    main()
