"""Small ExtraTrees tuning sweep with visible progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sweep_models import load_cached_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="cs-3780-5780-spring-2026-all-about-birds")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument("--seed", type=int, default=3780)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--quick", action="store_true", help="Run only the strongest short tree configuration.")
    args = parser.parse_args()

    x, labels, x_test, sample = load_cached_features(Path(args.data_dir), Path(args.cache_dir), False)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=args.validation_size, random_state=args.seed, stratify=y
    )

    if args.quick:
        configs = [
            {"n_estimators": 600, "max_features": 0.35, "min_samples_leaf": 1, "class_weight": "balanced"},
        ]
    else:
        configs = [
            {"n_estimators": 1000, "max_features": "sqrt", "min_samples_leaf": 1, "class_weight": "balanced"},
            {"n_estimators": 1000, "max_features": 0.35, "min_samples_leaf": 1, "class_weight": "balanced"},
            {"n_estimators": 1000, "max_features": "sqrt", "min_samples_leaf": 2, "class_weight": "balanced"},
            {"n_estimators": 1000, "max_features": 0.5, "min_samples_leaf": 1, "class_weight": None},
        ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, config in enumerate(configs, start=1):
        print(f"\n=== trees config {i}/{len(configs)} {config} ===", flush=True)
        model = ExtraTreesClassifier(
            **config,
            n_jobs=-1,
            random_state=args.seed,
            verbose=1,
        )
        model.fit(x_train, y_train)
        acc = accuracy_score(y_val, model.predict(x_val))
        print(f"validation_accuracy={acc:.6f}", flush=True)

        model.fit(x, y)
        submission = sample.copy()
        submission["label"] = encoder.inverse_transform(model.predict(x_test))
        out = output_dir / f"extra_trees_tuned_{i}_{acc:.5f}.csv"
        submission.to_csv(out, index=False)
        print(f"wrote {out}", flush=True)
        results.append({"config": config, "validation_accuracy": float(acc), "submission": str(out)})

    results.sort(key=lambda row: row["validation_accuracy"], reverse=True)
    (output_dir / "tree_tuning_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nBest:", results[0], flush=True)


if __name__ == "__main__":
    main()
