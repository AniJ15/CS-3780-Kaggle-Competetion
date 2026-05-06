# CS 3780/5780 All About Birds

Baseline code for the Kaggle bird spectrogram image competition. The expected data layout is:

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

Authenticate Kaggle first by creating an API token from Kaggle Account Settings and placing it at `~/.kaggle/kaggle.json`.

```bash
python3 download_data.py
```

## Train And Submit

```bash
python3 train_submit.py --data-dir data/birds --model ensemble --output submissions/submission.csv
```

The script prints a local validation accuracy, retrains on all training images, and writes a Kaggle-ready CSV with exactly:

```text
file_name,label
```

## AI/Agent Tools Note

This code was scaffolded with assistance from OpenAI Codex. Codex was used to create the Kaggle download wrapper, a baseline feature extraction and classification pipeline, and project documentation. The final experiments, selected model, generated submission, and Canvas code submission should be reviewed and understood by the student.

