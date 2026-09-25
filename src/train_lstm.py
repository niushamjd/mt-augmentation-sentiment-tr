import os
import argparse
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from sklearn.metrics import f1_score


import eval as shared_eval 

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

class TrendyolDataset(Dataset):
    def __init__(self, data_path, sp_model, max_len=128):
        self.data = pd.read_csv(data_path, sep='\t')
        self.sp_model = sp_model
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label = int(self.data.iloc[idx]['label'])
        
        # Tokenize and pad/truncate
        tokens = self.sp_model.encode(text, out_type=int)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.sp_model.pad_id()] * (self.max_len - len(tokens))
            
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size=8000, embed_size=256, hidden_size=256):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # One bidirectional LSTM layer, hidden size 256
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, 
                            bidirectional=True, batch_first=True)
        # Mean-over-time pooling will output hidden_size * 2 (512)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds) # [batch_size, seq_len, 512]
        
        # Mean-over-time pooling across the sequence length (dim=1)
        pooled = lstm_out.mean(dim=1)   # [batch_size, 512]
        logits = self.fc(pooled)        # [batch_size, 2]
        return logits

def evaluate_dev(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return f1_score(all_labels, all_preds, average='macro')

def main():
    parser = argparse.ArgumentParser(description="Train self-built LSTM on Turkish sentiment")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data TSV")
    parser.add_argument("--dev_file", type=str, default="data/sentiment/real_dev.tsv")
    parser.add_argument("--test_file", type=str, default="data/sentiment/real_test.tsv")
    parser.add_argument("--spm_model", type=str, default="data/spm/tr_sp8k.model")
    
    # Tracking arguments for results.csv
    parser.add_argument("--condition", type=str, required=True, choices=["C1", "C2", "C3", "C2b"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--synth_ratio", type=float, required=True)
    parser.add_argument("--mt_system", type=str, default="none", choices=["none", "final", "early"])
    parser.add_argument("--notes", type=str, default="")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    # Ensure pad token is defined; if not, we assume index 0 is <pad>
    if sp.pad_id() == -1:
        sp.set_default_extra_options(':bos_id=-1:eos_id=-1:unk_id=0:pad_id=0')
    vocab_size = sp.vocab_size()
    
    # 2. Prepare Data
    train_dataset = TrendyolDataset(args.train_file, sp)
    dev_dataset = TrendyolDataset(args.dev_file, sp)
    test_dataset = TrendyolDataset(args.test_file, sp)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize Model & Count Parameters
    model = SentimentLSTM(vocab_size=vocab_size).to(device)
    
    embed_params = sum(p.numel() for n, p in model.named_parameters() if 'embedding' in n)
    non_embed_params = sum(p.numel() for n, p in model.named_parameters() if 'embedding' not in n)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Embedding parameters: {embed_params:,}")
    print(f"Non-embedding parameters: {non_embed_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # 4. Training Loop setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    max_epochs = 20
    patience = 3
    best_dev_f1 = -1.0
    epochs_no_improve = 0
    best_model_path = f"results/best_lstm_{args.condition}_{args.seed}.pt"
    best_epoch = 0
    
    start_time = time.time()
    
    # 5. Train
    for epoch in range(1, max_epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
        dev_f1 = evaluate_dev(model, dev_loader, device)
        print(f"Epoch {epoch}/{max_epochs} | Dev Macro-F1: {dev_f1:.4f}")
        
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
    train_time_s = int(time.time() - start_time)
    
    # 6. Test & Evaluate (using Niyousha's eval framework)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            test_preds.extend(preds.cpu().tolist())
            test_labels.extend(labels.tolist())
            
    # Format properties to pass to Niyousha's eval.py
    # Format properties to pass to Niyousha's eval.py
    run_id = f"lstm_{args.condition}_{args.mt_system}_{args.seed}"
    n_real = 2000
    n_synth = len(train_dataset) - n_real if args.condition != "C1" else 0
    
    # Extract original text so eval.py can include it in the predictions TSV
    test_texts = test_dataset.data['text'].tolist()
    
    # Call the shared evaluation script
    shared_eval.evaluate_and_log(
        y_true=test_labels,
        y_pred=test_preds,
        run_id=run_id,
        model='lstm',
        condition=args.condition,
        seed=args.seed,
        n_real=n_real,
        results_csv='results/results_ipek.csv',  # Writing to your designated file
        n_synth=n_synth,
        synth_ratio=args.synth_ratio,
        mt_system=args.mt_system,
        epochs_run=epoch,
        best_epoch=best_epoch,
        lr=1e-3,
        batch_size=batch_size,
        dev_macro_f1=best_dev_f1,
        train_time_s=train_time_s,
        texts=test_texts,
        notes=args.notes
    )
    print(f"Finished {run_id}. Results appended to results_ipek.csv.")

if __name__ == "__main__":
    main()