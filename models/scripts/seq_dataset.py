import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from scripts.fasta_utils import read_fasta, load_labels, ENCODE

class SeqDataset(Dataset):
    def __init__(self, headers, sequences, labels):
        self.headers = headers
        self.sequences = sequences
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.headers[idx], self.sequences[idx].encode('ascii', 'ignore'), self.labels[idx]
    
def collate_pad(batch):
    headers, seq_bytes, labels = zip(*batch)
    idx_list = [ENCODE[np.frombuffer(sb, dtype=np.uint8)] for sb in seq_bytes]
    max_length = max(x.size for x in idx_list)
    batch_size = len(idx_list)
    
    X = torch.zeros(batch_size, 5, max_length, dtype=torch.float32)   # channels: A, C, G, T, N
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)      # padding mask
    
    for i, idx_np in enumerate(idx_list):
        length = idx_np.size
        idx = torch.from_numpy(idx_np).long()
        X[i, idx, torch.arange(length)] = 1.0
        mask[i, :length] = (idx != 4)   # invalid positions: N or everything beyond the length
    
    Y = torch.tensor(labels, dtype=torch.float32)
    return headers, X, mask, Y

def prepare_data(fasta_path, label_path, subset_size=None, random_state=42):
    headers, sequences = read_fasta(fasta_path)
    label_dict = load_labels(label_path)
    labels = [label_dict[h] for h in headers]
    
    total_sequences = len(sequences)
    if subset_size is not None and subset_size < total_sequences:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(total_sequences)
        sel_idx = np.sort(perm[:subset_size])
        headers = [headers[i] for i in sel_idx]
        sequences = [sequences[i] for i in sel_idx]
        labels = [labels[i] for i in sel_idx]
        print(f"Using random subset of {len(headers)} sequences (from {total_sequences}) with random_state={random_state}.")
    else:
        print(f"Using all {total_sequences} sequences for training.")

    idx_tr, idx_te = train_test_split(
        np.arange(len(sequences)), test_size=0.2, train_size=0.6, stratify=labels, random_state=42
    )
    idx_val = np.setdiff1d(np.arange(len(sequences)), np.concatenate([idx_tr, idx_te]))
    ds_tr = SeqDataset([headers[i] for i in idx_tr], [sequences[i] for i in idx_tr], [labels[i] for i in idx_tr])
    ds_val = SeqDataset([headers[i] for i in idx_te], [sequences[i] for i in idx_te], [labels[i] for i in idx_te])
    ds_te = SeqDataset([headers[i] for i in idx_val], [sequences[i] for i in idx_val], [labels[i] for i in idx_val])
    
    return ds_tr, ds_val, ds_te