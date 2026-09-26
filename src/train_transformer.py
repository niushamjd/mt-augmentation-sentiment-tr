"""Train and evaluate our own Transformer sentiment classifier

Model:
    8,000 x 256 embedding -> sinusoidal positional encoding -> 2 encoder layers
    (d_model 256, 4 heads, feed-forward 512, dropout 0.1) -> mean pooling over the
    real tokens -> linear layer to 2 classes.  About 1.05M non-embedding parameters.

Examples:

    # tuning on C1: only looks at dev, never at test
    python src/train_transformer.py --condition C1 --seed 42 --lr 5e-4 --tune

    # a real run: trains, evaluates ONCE on test, writes results/results_buse.csv
    python src/train_transformer.py --condition C1 --seed 42
    python src/train_transformer.py --condition C3 --seed 1337
    python src/train_transformer.py --condition C2 --seed 42 --mt-system early      # RQ4
    python src/train_transformer.py --condition C2 --seed 42 --synth-n 2000         # RQ3, 1x
"""
import argparse
import copy
import math
import os
import random
import time

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eval import compute_metrics, evaluate_and_log
from text_utils import normalise_tr

MAX_LEN = 128
SPM_MODEL = "data/spm/tr_sp8k.model"
DATA = "data/sentiment"

# which synthetic file goes with which condition / MT system
SYNTH_FILES = {
    ("C2", "final"): "synth_all.tsv",
    ("C3", "final"): "synth_clean.tsv",
    ("C2b", "final"): "synth_matched.tsv",
    ("C2", "early"): "synth_all_early.tsv",
    ("C2", "pretrained"): "synth_all_pretrained.tsv",
}


def set_seed(s):
    # from the project instructions, Section 1.4
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)  # also seeds MPS
    torch.cuda.manual_seed_all(s)


def read_tsv(name):
    # pandas default quoting (the synthetic files are written with it), no NA guessing
    return pd.read_csv(f"{DATA}/{name}", sep="\t", keep_default_na=False)


# ---------------------------------------------------------------- data

def load_training_data(args):
    real = read_tsv("real_train.tsv")
    n_synth = 0
    if args.condition == "C1":
        train = real
    else:
        synth = read_tsv(SYNTH_FILES[(args.condition, args.mt_system)])
        if args.synth_n is not None:
            # RQ3: nested subsamples, always the same shuffle (seed 42), so the
            # 1x set is part of the 5x set, which is part of the 10x set
            synth = synth.sample(frac=1, random_state=42).iloc[:args.synth_n]
        n_synth = len(synth)
        train = pd.concat([real, synth], ignore_index=True)
    return train, len(real), n_synth


def encode(texts, sp):
    ids = []
    for t in texts:
        piece_ids = sp.encode(normalise_tr(t))[:MAX_LEN]
        if len(piece_ids) == 0:
            piece_ids = [sp.unk_id()]  # should not happen, but an empty review would crash
        ids.append(piece_ids)
    return ids


def make_batches(ids, labels, batch_size, shuffle, pad_id, generator=None):
    data = list(zip(ids, labels))

    def collate(batch):
        seqs = [b[0] for b in batch]
        longest = max(len(s) for s in seqs)
        x = torch.full((len(seqs), longest), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return x, y

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=generator)


# ---------------------------------------------------------------- model

class PositionalEncoding(nn.Module):
    """Sinusoidal positions like in "Attention is all you need" (no parameters)."""

    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # buffer = saved, but not trained

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, pad_id, d_model=256, n_heads=4, ff_dim=512,
                 n_layers=2, dropout=0.1, n_classes=2):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        pad_mask = x == self.pad_id                       # True where padding
        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.dropout(self.pos(h))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        # mean over the real tokens only (padding does not count)
        keep = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))


def count_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = model.embedding.weight.numel()
    print(f"parameters: embedding {emb:,} | non-embedding {total - emb:,} | total {total:,}")
    return emb, total


# ---------------------------------------------------------------- training

def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(device))
            preds += logits.argmax(dim=-1).cpu().tolist()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["C1", "C2", "C3", "C2b"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mt-system", default=None, choices=["final", "early", "pretrained"],
                    help="default: final (none for C1)")
    ap.add_argument("--synth-n", type=int, default=None, help="RQ3: use only this many synthetic examples")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=200, help="same number of steps in every condition")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--tune", action="store_true", help="dev only: no test evaluation, log to results/tuning_buse.csv")
    ap.add_argument("--results-csv", default="results/results_buse.csv")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    if args.condition == "C1":
        args.mt_system = "none"
    elif args.mt_system is None:
        args.mt_system = "final"
    if args.condition != "C1" and (args.condition, args.mt_system) not in SYNTH_FILES:
        ap.error(f"no synthetic file for {args.condition} with --mt-system {args.mt_system}")

    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    # data
    sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)
    pad_id = sp.pad_id()
    train, n_real, n_synth = load_training_data(args)
    dev = read_tsv("real_dev.tsv")
    test = read_tsv("real_test.tsv")
    print(f"train: {len(train)} ({n_real} real + {n_synth} synthetic) | dev: {len(dev)} | test: {len(test)}")

    g = torch.Generator()
    g.manual_seed(args.seed)  # batch shuffling depends on the seed too
    train_loader = make_batches(encode(train["text"], sp), train["label"].tolist(),
                                args.batch_size, True, pad_id, generator=g)
    dev_loader = make_batches(encode(dev["text"], sp), dev["label"].tolist(),
                              args.batch_size, False, pad_id)
    test_loader = make_batches(encode(test["text"], sp), test["label"].tolist(),
                               args.batch_size, False, pad_id)

    # model
    model = TransformerClassifier(sp.get_piece_size(), pad_id, dropout=args.dropout).to(device)
    count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # linear warm-up for a fixed number of steps, then constant learning rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / args.warmup_steps))
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_epoch, best_state = -1.0, 0, None
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(y)

        # evaluate on dev once per epoch, with the shared metric code
        dev_preds = predict(model, dev_loader, device)
        dev_f1 = compute_metrics(dev["label"].tolist(), dev_preds)["macro_f1"]
        print(f"epoch {epoch:2d} | train loss {total_loss / len(train):.4f} | "
              f"dev macro-F1 {dev_f1:.4f} | {time.time() - epoch_start:.0f}s")

        if dev_f1 > best_f1:
            best_f1, best_epoch = dev_f1, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping: no improvement for {args.patience} epochs")
                break

    epochs_run = epoch
    train_time = time.time() - start
    model.load_state_dict(best_state)  # restore the best checkpoint before testing
    print(f"best epoch {best_epoch}, dev macro-F1 {best_f1:.4f}, training took {train_time:.0f}s")

    run_id = f"transformer_{args.condition}_{args.mt_system}_{args.seed}"
    if args.synth_n is not None:
        run_id += f"_n{args.synth_n}"  # RQ3 runs would otherwise overwrite the C2 predictions

    if args.tune:
        # tuning: record dev only, never look at test (Section 1.2)
        row = pd.DataFrame([{
            "run_id": run_id, "lr": args.lr, "batch_size": args.batch_size, "dropout": args.dropout,
            "warmup_steps": args.warmup_steps, "epochs_run": epochs_run, "best_epoch": best_epoch,
            "dev_macro_f1": round(best_f1, 4), "train_time_s": round(train_time), "notes": args.notes,
        }])
        path = "results/tuning_buse.csv"
        os.makedirs("results", exist_ok=True)
        row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"tuning run logged to {path} (test set not used)")
        return

    test_preds = predict(model, test_loader, device)
    evaluate_and_log(
        test["label"].tolist(), test_preds,
        run_id=run_id, model="transformer", condition=args.condition, seed=args.seed,
        n_real=n_real, n_synth=n_synth, synth_ratio=round(n_synth / n_real, 4),
        mt_system=args.mt_system, epochs_run=epochs_run, best_epoch=best_epoch,
        lr=args.lr, batch_size=args.batch_size, dev_macro_f1=round(best_f1, 4),
        train_time_s=round(train_time), texts=test["text"].tolist(),
        results_csv=args.results_csv, notes=args.notes,
    )


if __name__ == "__main__":
    main()
