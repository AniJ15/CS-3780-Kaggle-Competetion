"""Progress-visible training for faster bird spectrogram submissions.

AI/tools disclosure: OpenAI Codex helped create this progress-visible training
script and experiment workflow. Review the code and results before submitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from train_submit import FEATURE_VERSION, load_test, load_train, resolve_data_dir


def cache_path(data_dir: Path, cache_dir: Path) -> Path:
    resolved = resolve_data_dir(data_dir)
    return cache_dir / f"{resolved.parent.parent.name}_{FEATURE_VERSION}_features.npz"


def load_cached(data_dir: Path, cache_dir: Path, rebuild: bool):
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


def make_model(loss: str, alpha: float, seed: int, class_weight: dict[int, float]) -> SGDClassifier:
    return SGDClassifier(
        loss=loss,
        alpha=alpha,
        penalty="l2",
        class_weight=class_weight,
        learning_rate="optimal",
        random_state=seed,
        n_jobs=-1,
        average=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="cs-3780-5780-spring-2026-all-about-birds")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output", default="submissions/submission_progress.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3780)
    parser.add_argument("--loss", choices=["hinge", "log_loss", "modified_huber"], default="log_loss")
    parser.add_argument("--alpha", type=float, default=0.0002)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    x, labels, x_test, sample = load_cached(Path(args.data_dir), Path(args.cache_dir), args.rebuild_cache)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_all = scaler.fit_transform(x)
    x_test_scaled = scaler.transform(x_test)

    classes = np.arange(len(encoder.classes_))
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    model = make_model(args.loss, args.alpha, args.seed, class_weight)
    best = {"epoch": 0, "accuracy": -1.0}

    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(args.seed + epoch).permutation(len(x_train))
        model.partial_fit(x_train[order], y_train[order], classes=classes)
        pred = model.predict(x_val)
        acc = accuracy_score(y_val, pred)
        print(f"epoch {epoch:02d}/{args.epochs} validation_accuracy={acc:.6f}", flush=True)
        if acc > best["accuracy"]:
            best = {"epoch": epoch, "accuracy": float(acc)}

    print(f"best_validation_accuracy={best['accuracy']:.6f} at epoch {best['epoch']}", flush=True)

    final_model = make_model(args.loss, args.alpha, args.seed, class_weight)
    for epoch in range(1, best["epoch"] + 1):
        order = np.random.default_rng(args.seed + epoch).permutation(len(x_all))
        final_model.partial_fit(x_all[order], y, classes=classes)
        print(f"refit epoch {epoch:02d}/{best['epoch']}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission = sample.copy()
    submission["label"] = encoder.inverse_transform(final_model.predict(x_test_scaled))
    submission.to_csv(output, index=False)
    output.with_suffix(".metadata.json").write_text(json.dumps(vars(args) | best, indent=2), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
