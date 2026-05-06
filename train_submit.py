"""Train a bird spectrogram classifier and write a Kaggle submission.

AI/tools disclosure: This baseline was authored with assistance from OpenAI
Codex. Codex helped scaffold the data-loading, feature-extraction, validation,
and submission-writing workflow; the student should inspect, run, and document
the final chosen experiments before submitting code on Canvas.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
FEATURE_VERSION = "dct_v1"


@dataclass(frozen=True)
class RunConfig:
    data_dir: str
    output: str
    model: str
    validation_size: float
    seed: int


def read_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return arr


def image_features(path: Path) -> np.ndarray:
    arr = read_grayscale(path)
    flat = arr.reshape(-1)

    row_mean = arr.mean(axis=1)
    row_std = arr.std(axis=1)
    col_mean = arr.mean(axis=0)
    col_std = arr.std(axis=0)

    dx = np.diff(arr, axis=1)
    dy = np.diff(arr, axis=0)
    grad_stats = np.array(
        [dx.mean(), dx.std(), np.abs(dx).mean(), dy.mean(), dy.std(), np.abs(dy).mean()],
        dtype=np.float32,
    )

    h, w = arr.shape
    quadrants = [
        arr[: h // 2, : w // 2],
        arr[: h // 2, w // 2 :],
        arr[h // 2 :, : w // 2],
        arr[h // 2 :, w // 2 :],
    ]
    quadrant_stats = np.array(
        [stat for quad in quadrants for stat in (quad.mean(), quad.std(), quad.max())],
        dtype=np.float32,
    )

    dct = dctn(arr, type=2, norm="ortho")
    dct_low = dct[:16, :16].reshape(-1)
    dct_mid = dct[:24, :24].reshape(-1)
    dct_abs = np.log1p(np.abs(dct_mid))
    dct_stats = np.array(
        [
            dct[0, 0],
            dct_low.mean(),
            dct_low.std(),
            np.abs(dct_low).mean(),
            dct_abs.mean(),
            dct_abs.std(),
            dct_abs.max(),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            flat,
            row_mean,
            row_std,
            col_mean,
            col_std,
            grad_stats,
            quadrant_stats,
            dct_low,
            dct_abs,
            dct_stats,
        ]
    )


def load_train(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    data_dir = resolve_data_dir(data_dir)
    train_dir = data_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Expected training folder at {train_dir}")

    paths: list[Path] = []
    labels: list[str] = []
    for class_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        for image_path in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            paths.append(image_path)
            labels.append(class_dir.name)

    if not paths:
        raise FileNotFoundError(f"No training images found below {train_dir}")

    x = np.vstack([image_features(path) for path in paths])
    return x, np.asarray(labels), paths


def load_test(data_dir: Path) -> tuple[np.ndarray, pd.DataFrame, list[Path]]:
    data_dir = resolve_data_dir(data_dir)
    sample_path = find_sample_submission(data_dir)
    test_dir = data_dir / "test"
    if not sample_path.exists():
        raise FileNotFoundError(f"Expected sample submission at {sample_path}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Expected test folder at {test_dir}")

    sample = pd.read_csv(sample_path)
    if list(sample.columns) != ["file_name", "label"]:
        raise ValueError("sample_submission.csv must have columns: file_name,label")

    paths = [test_dir / file_name for file_name in sample["file_name"].astype(str)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"Some sample_submission files are missing from test/:\n{preview}")

    x = np.vstack([image_features(path) for path in paths])
    return x, sample, paths


def resolve_data_dir(data_dir: Path) -> Path:
    if (data_dir / "train").is_dir() and (data_dir / "test").is_dir():
        return data_dir

    matches = [
        path
        for path in data_dir.rglob("train")
        if path.is_dir() and (path.parent / "test").is_dir()
    ]
    if len(matches) == 1:
        return matches[0].parent
    if matches:
        return sorted(path.parent for path in matches)[0]
    return data_dir


def find_sample_submission(data_dir: Path) -> Path:
    candidates = [
        data_dir / "sample_submission.csv",
        data_dir.parent / "sample_submission.csv",
        data_dir.parent.parent / "sample_submission.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(data_dir.rglob("sample_submission.csv"))
    if matches:
        return matches[0]
    matches = sorted(data_dir.parent.rglob("sample_submission.csv"))
    if matches:
        return matches[0]
    return data_dir / "sample_submission.csv"


def build_model(kind: str, seed: int):
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    class_weight="balanced",
                    penalty="l2",
                    max_iter=900,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
    trees = ExtraTreesClassifier(
        n_estimators=700,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
        verbose=1,
    )
    linear_svc = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    C=0.35,
                    class_weight="balanced",
                    dual="auto",
                    max_iter=6000,
                    random_state=seed,
                ),
            ),
        ]
    )
    ridge = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", RidgeClassifier(alpha=2.0, class_weight="balanced")),
        ]
    )
    sgd_hinge = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="hinge",
                    alpha=0.0002,
                    class_weight="balanced",
                    max_iter=80,
                    tol=1e-4,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
    hist_gb = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=220,
        l2_regularization=0.02,
        random_state=seed,
        verbose=1,
    )

    if kind == "logistic":
        return logistic
    if kind == "linear_svc":
        return linear_svc
    if kind == "ridge":
        return ridge
    if kind == "sgd_hinge":
        return sgd_hinge
    if kind == "extra_trees":
        return trees
    if kind == "hist_gb":
        return hist_gb
    if kind == "ensemble":
        return VotingClassifier(
            estimators=[("logistic", logistic), ("extra_trees", trees), ("hist_gb", hist_gb)],
            voting="soft",
            weights=[3, 2, 1],
            n_jobs=-1,
        )
    if kind == "hard_ensemble":
        return VotingClassifier(
            estimators=[("logistic", logistic), ("linear_svc", linear_svc), ("ridge", ridge), ("extra_trees", trees)],
            voting="hard",
            weights=[3, 2, 2, 2],
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/birds")
    parser.add_argument("--output", default="submissions/submission.csv")
    parser.add_argument(
        "--model",
        choices=[
            "logistic",
            "linear_svc",
            "ridge",
            "sgd_hinge",
            "extra_trees",
            "hist_gb",
            "ensemble",
            "hard_ensemble",
        ],
        default="ensemble",
    )
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3780)
    args = parser.parse_args()

    config = RunConfig(**vars(args))
    data_dir = Path(args.data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    x, labels, train_paths = load_train(data_dir)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    stratify = y if np.min(np.bincount(y)) >= 2 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=stratify,
    )

    model = build_model(args.model, args.seed)
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_accuracy:.5f}")
    print(classification_report(y_val, val_pred, target_names=encoder.classes_, zero_division=0))

    model.fit(x, y)
    x_test, sample, test_paths = load_test(data_dir)
    test_pred = encoder.inverse_transform(model.predict(x_test))

    submission = sample.copy()
    submission["label"] = test_pred
    submission.to_csv(output, index=False)
    print(f"Wrote {output} with {len(submission)} rows.")

    metadata = {
        "config": asdict(config),
        "validation_accuracy": float(val_accuracy),
        "n_train": len(train_paths),
        "n_test": len(test_paths),
        "classes": encoder.classes_.tolist(),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
