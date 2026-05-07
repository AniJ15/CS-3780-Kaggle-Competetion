# CS 3780/5780 All About Birds

Code and generated submissions for the Kaggle bird spectrogram image competition.

## Best Completed Submission

Submit:

```text
submissions/submission.csv
```

This file is copied from:

```text
submissions/extra_trees_tuned_2_0.42854.csv
```

Best local validation accuracy from completed runs: `0.42854`

Model: `ExtraTreesClassifier` with `1000` trees, `max_features=0.35`, `min_samples_leaf=1`, and balanced class weights.

## Data Layout

The downloaded competition files are nested in this repo as:

```text
cs-3780-5780-spring-2026-all-about-birds/
  sample_submission.csv
  All_about_Birds/All_about_Birds/
    train/<bird_label>/*.png
    test/*.png
```

The scripts also support a simpler layout:

```text
data/birds/
  train/<bird_label>/*.png
  test/*.png
  sample_submission.csv
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

If the dataset is not already present, authenticate Kaggle first by creating an API token from Kaggle Account Settings and placing it at `~/.kaggle/kaggle.json`.

```bash
python3 download_data.py
```

## Reproduce The Best Completed Run

The best completed CSV came from the tree tuning script. To rerun the full tree sweep:

```bash
python3 tune_trees.py --data-dir cs-3780-5780-spring-2026-all-about-birds
```

The strongest completed configuration was the second tree tuning run:

```text
n_estimators=1000
max_features=0.35
min_samples_leaf=1
class_weight=balanced
```

Tree training prints progress such as completed tree counts.

## Faster Progress-Visible Runs

For quick experiments with per-epoch validation output:

```bash
python3 train_progress.py \
  --data-dir cs-3780-5780-spring-2026-all-about-birds \
  --loss log_loss \
  --alpha 0.00005 \
  --epochs 20 \
  --output submissions/sgd_log_loss_a00005.csv
```

This finished quickly and reached `0.35822` validation accuracy before DCT features were added.

## Feature Extraction

```bash
python3 train_submit.py \
  --data-dir cs-3780-5780-spring-2026-all-about-birds \
  --model extra_trees \
  --output submissions/extra_trees.csv
```

The shared feature extractor uses:

- raw grayscale pixels from each `64 x 32` image
- row and column means/stds
- simple horizontal/vertical gradient statistics
- quadrant statistics
- low-frequency and log-magnitude 2D DCT coefficients

The DCT linear run completed at `0.35525`; the interrupted DCT tree run did not finish, so the best completed submission remains `submissions/submission.csv`.

All submission CSVs contain exactly:

```text
file_name,label
```

## AI/Agent Tools Note

This code was developed with assistance from OpenAI Codex. I provided the competition details, the initial KaggleHub download snippet, the dataset/submission requirements, and follow-up modeling ideas such as using DCT coefficients and adding progress-visible training. Codex helped debug the local setup, fill out the implementation, add feature extraction and model training scripts, run validation experiments, generate candidate submission files, and update this documentation. The final submitted code and Kaggle submission should be reviewed and understood by the student before submitting on Canvas.
