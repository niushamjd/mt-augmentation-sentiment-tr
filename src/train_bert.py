"""
Fine-tune dbmdz/bert-base-turkish-cased for TR sentiment classification
(PROJECT_INSTRUCTIONS.md Section 1.5 training protocol, Section 4
"Niyousha, bert-classifier").

Fully fine-tuned (no frozen layers), max sequence length 128, early
stopping on dev macro-F1 with patience 3 evaluations (one eval per
epoch), max 5 epochs, linear warmup. Logs via src/eval.py so results land
in the shared results.csv schema.

Usage (condition C1, seed 42):
    python src/train_bert.py --condition C1 --seed 42

Usage (C2/C3/C2b, once synthetic files exist): pass --extra-train with the
synthetic TSV to concatenate onto the real training set, and set
--mt-system/--synth-ratio so the logged row is correct.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import compute_metrics, evaluate_and_log  # noqa: E402

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LENGTH = 128
MAX_EPOCHS = 5
PATIENCE = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=MAX_LENGTH
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_tsv(path: Path) -> pd.DataFrame:
    # keep_default_na=False: an empty TSV cell is a genuine empty-string
    # translation (real, unfiltered MT output for C2 -- not missing data),
    # and pandas' default NaN-coercion on blank cells crashes the tokenizer.
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    assert set(df.columns) >= {"text", "label"}, df.columns
    df["label"] = df["label"].astype(int)
    return df


@torch.no_grad()
def predict(model, loader, device) -> list:
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=["C1", "C2", "C3", "C2b"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--train", default="data/sentiment/real_train.tsv")
    ap.add_argument("--dev", default="data/sentiment/real_dev.tsv")
    ap.add_argument("--test", default="data/sentiment/real_test.tsv")
    ap.add_argument("--extra-train", default=None, help="synthetic TSV to concat for C2/C3/C2b")
    ap.add_argument("--mt-system", default="none", choices=["none", "final", "early", "pretrained"])
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--results-csv", default="results/results_niyousha.csv")
    ap.add_argument("--notes", default="")
    ap.add_argument(
        "--dev-only", action="store_true",
        help="for hyperparameter search: train + pick best epoch on dev, skip test "
             "entirely (Section 1.2: no hyperparameter decision on the test set). "
             "Prints a DEV_RESULT line instead of logging to results.csv.",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    train_df = load_tsv(Path(args.train))
    n_real = len(train_df)
    n_synth = 0
    if args.extra_train:
        synth_df = load_tsv(Path(args.extra_train))
        n_synth = len(synth_df)
        train_df = pd.concat([train_df, synth_df], ignore_index=True)
    dev_df = load_tsv(Path(args.dev))
    test_df = None if args.dev_only else load_tsv(Path(args.test))
    synth_ratio = (n_synth / n_real) if n_real else 0.0

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train_ds = SentimentDataset(train_df["text"], train_df["label"], tokenizer)
    dev_ds = SentimentDataset(dev_df["text"], dev_df["label"], tokenizer)

    # shuffling controlled by the same seed set above -> generator makes the
    # DataLoader's own shuffling reproducible independent of global torch state
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_dev_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_run = 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
        epochs_run = epoch

        dev_preds = predict(model, dev_loader, device)
        dev_metrics = compute_metrics(dev_df["label"].tolist(), dev_preds)
        print(
            f"epoch {epoch}: train_loss={total_loss / len(train_loader):.4f} "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}"
        )

        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            print(f"early stopping: no dev improvement in {PATIENCE} evaluations")
            break

    train_time_s = time.time() - t0

    print(f"restoring best checkpoint from epoch {best_epoch} (dev macro-F1={best_dev_f1:.4f})")
    model.load_state_dict(best_state)
    model.to(device)

    if args.dev_only:
        print(
            f"DEV_RESULT lr={args.lr} batch_size={args.batch_size} "
            f"best_epoch={best_epoch} epochs_run={epochs_run} "
            f"best_dev_f1={best_dev_f1:.4f} train_time_s={train_time_s:.1f}"
        )
        return

    test_ds = SentimentDataset(test_df["text"], test_df["label"], tokenizer)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    test_preds = predict(model, test_loader, device)
    run_id = f"bert_{args.condition}_{args.mt_system}_{args.seed}"
    evaluate_and_log(
        test_df["label"].tolist(), test_preds,
        run_id=run_id, model="bert", condition=args.condition, seed=args.seed,
        n_real=n_real, n_synth=n_synth, synth_ratio=synth_ratio, mt_system=args.mt_system,
        epochs_run=epochs_run, best_epoch=best_epoch, lr=args.lr, batch_size=args.batch_size,
        dev_macro_f1=best_dev_f1, train_time_s=train_time_s,
        texts=test_df["text"].tolist(), results_csv=args.results_csv, notes=args.notes,
    )


if __name__ == "__main__":
    main()
