"""Train and evaluate the self-built LSTM sentiment classifier (Ipek).

Model:
    8,000 x 256 embedding -> 1 bidirectional LSTM layer, hidden 256 ->
    mean over the REAL tokens (padding excluded) -> linear layer to 2 classes.
    About 1.05M non-embedding parameters.

Changes from the first version (27.09):
  - padding is ignored: sequences are packed (pack_padded_sequence), so the LSTM
    never reads padding, the mean pooling only averages real tokens, and the
    embedding has padding_idx so the padding vector is not trained;
  - every text goes through text_utils.normalise_tr (agreed rule for all models);
  - the training file is chosen from --condition / --mt_system (and --rq3), and
    n_synth / synth_ratio are computed, not typed in;
  - RQ3 runs get a suffix (_1x, _5x) so they no longer overwrite the C2 run's
    predictions; they use the shared nested files synth_1x.tsv / synth_5x.tsv;
  - C2b (size-matched control) is supported;
  - --tune: dev only, never test, logged to results/tuning_ipek.csv;
  - dev macro-F1 uses the shared eval.compute_metrics;
  - batches are padded only to their longest review (faster).

Examples (repo root, nn2026 environment):
    python src/train_lstm.py --condition C1 --seed 42 --tune --lr 1e-3
    python src/train_lstm.py --condition C1 --seed 42
    python src/train_lstm.py --condition C2b --seed 1337
    python src/train_lstm.py --condition C2 --seed 42 --mt_system early     # RQ4
    python src/train_lstm.py --condition C2 --seed 42 --rq3 1x              # RQ3
"""
import argparse
import copy
import os
import random
import time

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

import eval as shared_eval
from text_utils import normalise_tr

MAX_LEN = 128
DATA = "data/sentiment"
SYNTH_FILES = {
    ("C2", "final"): "synth_all.tsv",
    ("C3", "final"): "synth_clean.tsv",
    ("C2b", "final"): "synth_matched.tsv",
    ("C2", "early"): "synth_all_early.tsv",
    ("C2", "pretrained"): "synth_all_pretrained.tsv",
}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def read_tsv(name):
    # default pandas quoting (how all sentiment TSVs are written), no NA guessing
    return pd.read_csv(os.path.join(DATA, name), sep="\t", keep_default_na=False)


def load_training_data(args):
    real = read_tsv("real_train.tsv")
    if args.condition == "C1":
        return real, len(real), 0
    if args.rq3:
        synth = read_tsv(f"synth_{args.rq3}.tsv")   # shared nested RQ3 subsets
    else:
        synth = read_tsv(SYNTH_FILES[(args.condition, args.mt_system)])
    return pd.concat([real, synth], ignore_index=True), len(real), len(synth)


def encode(texts, sp):
    out = []
    for t in texts:
        ids = sp.encode(normalise_tr(t), out_type=int)[:MAX_LEN]
        out.append(ids if ids else [sp.unk_id()])  # never an empty sequence
    return out


def make_loader(ids, labels, batch_size, shuffle, pad_id, generator=None):
    data = list(zip(ids, labels))

    def collate(batch):
        lengths = torch.tensor([len(b[0]) for b in batch], dtype=torch.long)
        x = torch.full((len(batch), int(lengths.max())), pad_id, dtype=torch.long)
        for i, (seq, _) in enumerate(batch):
            x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return x, lengths, y

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=generator)


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, pad_id, embed_size=256, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
        # one bidirectional LSTM layer, hidden size 256
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x, lengths):
        embeds = self.embedding(x)
        # packing: the LSTM only sees the real tokens of every review
        packed = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True,
                                     total_length=x.size(1))   # padding positions are 0
        # mean over the real tokens only
        mask = (torch.arange(x.size(1), device=x.device)[None, :]
                < lengths.to(x.device)[:, None]).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / lengths.to(x.device).unsqueeze(1).float()
        return self.fc(pooled)


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, lengths, _ in loader:
            logits = model(x.to(device), lengths)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds


def main():
    ap = argparse.ArgumentParser(description="Train self-built LSTM on Turkish sentiment")
    ap.add_argument("--condition", required=True, choices=["C1", "C2", "C3", "C2b"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mt_system", default=None, choices=["final", "early", "pretrained"],
                    help="default: final (none for C1)")
    ap.add_argument("--rq3", default=None, choices=["1x", "5x"],
                    help="RQ3: use data/sentiment/synth_{1x,5x}.tsv (with --condition C2)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                    help="cpu is the safe default for packed LSTMs; try mps if it is faster")
    ap.add_argument("--tune", action="store_true",
                    help="dev only (no test), logged to results/tuning_ipek.csv")
    ap.add_argument("--results_csv", default="results/results_ipek.csv")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    if args.condition == "C1":
        args.mt_system = "none"
        if args.rq3:
            ap.error("--rq3 needs --condition C2")
    elif args.mt_system is None:
        args.mt_system = "final"
    if args.rq3 and (args.condition != "C2" or args.mt_system != "final"):
        ap.error("--rq3 is only defined for --condition C2 with mt_final")
    if args.condition != "C1" and not args.rq3 and (args.condition, args.mt_system) not in SYNTH_FILES:
        ap.error(f"no synthetic file for {args.condition} with --mt_system {args.mt_system}")

    set_seed(args.seed)
    device = torch.device(args.device)
    print("device:", device)

    sp = spm.SentencePieceProcessor(model_file="data/spm/tr_sp8k.model")
    pad_id = sp.pad_id()  # 0 in tr_sp8k (unk is 1): never hard-code it

    train, n_real, n_synth = load_training_data(args)
    dev = read_tsv("real_dev.tsv")
    test = read_tsv("real_test.tsv")
    print(f"train: {len(train)} ({n_real} real + {n_synth} synthetic) | dev: {len(dev)} | test: {len(test)}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = make_loader(encode(train["text"], sp), train["label"].astype(int).tolist(),
                               args.batch_size, True, pad_id, generator=g)
    dev_loader = make_loader(encode(dev["text"], sp), dev["label"].tolist(),
                             args.batch_size, False, pad_id)
    test_loader = make_loader(encode(test["text"], sp), test["label"].tolist(),
                              args.batch_size, False, pad_id)

    model = SentimentLSTM(sp.get_piece_size(), pad_id).to(device)
    embed_params = model.embedding.weight.numel()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: embedding {embed_params:,} | non-embedding {total_params - embed_params:,} | total {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_dev_f1, best_epoch, best_state = -1.0, 0, None
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_start = time.time()
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, lengths), y)
            loss.backward()
            optimizer.step()

        dev_preds = predict(model, dev_loader, device)
        dev_f1 = shared_eval.compute_metrics(dev["label"].tolist(), dev_preds)["macro_f1"]
        print(f"Epoch {epoch}/{args.max_epochs} | Dev Macro-F1: {dev_f1:.4f} | {time.time() - epoch_start:.0f}s")

        if dev_f1 > best_dev_f1:
            best_dev_f1, best_epoch = dev_f1, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    epochs_run = epoch
    train_time_s = int(time.time() - start_time)
    model.load_state_dict(best_state)   # best dev checkpoint before testing
    print(f"best epoch {best_epoch}, dev macro-F1 {best_dev_f1:.4f}, {train_time_s}s")

    run_id = f"lstm_{args.condition}_{args.mt_system}_{args.seed}"
    if args.rq3:
        run_id += f"_{args.rq3}"   # same suffix convention as the BERT runs

    if args.tune:
        os.makedirs("results", exist_ok=True)
        path = "results/tuning_ipek.csv"
        row = pd.DataFrame([{
            "run_id": run_id, "lr": args.lr, "batch_size": args.batch_size,
            "epochs_run": epochs_run, "best_epoch": best_epoch,
            "dev_macro_f1": round(best_dev_f1, 4), "train_time_s": train_time_s, "notes": args.notes,
        }])
        row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"tuning run logged to {path} (test set not used)")
        return

    test_preds = predict(model, test_loader, device)
    shared_eval.evaluate_and_log(
        y_true=test["label"].tolist(), y_pred=test_preds,
        run_id=run_id, model="lstm", condition=args.condition, seed=args.seed,
        n_real=n_real, n_synth=n_synth, synth_ratio=round(n_synth / n_real, 4),
        mt_system=args.mt_system, epochs_run=epochs_run, best_epoch=best_epoch,
        lr=args.lr, batch_size=args.batch_size, dev_macro_f1=round(best_dev_f1, 4),
        train_time_s=train_time_s, texts=test["text"].tolist(),
        results_csv=args.results_csv, notes=args.notes,
    )
    print(f"Finished {run_id}. Results appended to {args.results_csv}.")


if __name__ == "__main__":
    main()
