"""Run a fast model sweep on cached bird spectrogram features.

AI/tools disclosure: This experiment runner was authored with assistance from
OpenAI Codex to compare several classical image classifiers and generate
Kaggle-ready submissions. Review the code and results before submitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from train_submit import FEATURE_VERSION, build_model, load_test, load_train, resolve_data_dir


def cache_path(data_dir: Path, cache_dir: Path) -> Path:
    resolved = resolve_data_dir(data_dir)
    name = resolved.parent.parent.name if resolved.parent.parent.name else resolved.name
    return cache_dir / f"{name}_{FEATURE_VERSION}_features.npz"


def load_cached_features(data_dir: Path, cache_dir: Path, rebuild: bool):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(data_dir, cache_dir)
    if path.exists() and not rebuild:
        data = np.load(path, allow_pickle=True)
        sample = pd.DataFrame({"file_name": data["test_files"].astype(str), "label": ""})
        return data["x_train"], data["labels"], data["x_test"], sample

    x_train, labels, _ = load_train(data_dir)
    x_test, sample, _ = load_test(data_dir)
    np.savez(
        path,
        x_train=x_train.astype(np.float32),
        labels=labels,
        x_test=x_test.astype(np.float32),
        test_files=sample["file_name"].astype(str).to_numpy(dtype=object),
    )
    return x_train, labels, x_test, sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="cs-3780-5780-spring-2026-all-about-birds")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3780)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "linear_svc", "logistic", "extra_trees", "hard_ensemble"],
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, labels, x_test, sample = load_cached_features(Path(args.data_dir), Path(args.cache_dir), args.rebuild_cache)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=y,
    )

    results = []
    for model_name in args.models:
        print(f"\n=== {model_name} ===", flush=True)
        model = build_model(model_name, args.seed)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        acc = accuracy_score(y_val, pred)
        print(f"validation_accuracy={acc:.6f}", flush=True)
        print(classification_report(y_val, pred, target_names=encoder.classes_, zero_division=0), flush=True)

        model.fit(x, y)
        submission = sample.copy()
        submission["label"] = encoder.inverse_transform(model.predict(x_test))
        out = output_dir / f"{model_name}_{acc:.5f}.csv"
        submission.to_csv(out, index=False)
        print(f"wrote {out}", flush=True)
        results.append({"model": model_name, "validation_accuracy": float(acc), "submission": str(out)})

    results = sorted(results, key=lambda row: row["validation_accuracy"], reverse=True)
    (output_dir / "sweep_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nBest:", results[0], flush=True)


if __name__ == "__main__":
    main()
