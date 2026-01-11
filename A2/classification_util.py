# Classification utilities for HW2
# This module contains helper functions for text classification tasks.

import time
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import gensim.downloader
from nltk.tokenize import word_tokenize
from datasets import load_dataset

# Padding token index
PAD = 0


def get_device():
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(text):
    """Simple whitespace tokenizer with lowercasing."""
    return word_tokenize(text.lower())


def load_sst2_splits():
    """The original `sst2` datatest split does not include ground-truth labels. Instead, we create our own **labeled test set** by splitting the original training set into **train** and **test** with a 80%-20% portion. We still use the official **validation** split for model selection and tuning. Finally, we lower case all texts for consistency."""
    sst2 = load_dataset("glue", "sst2")
    val_data   = sst2["validation"]

    # Create train/test split from original training data
    split = sst2["train"].train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column="label"
    )
    train_data = split["train"]
    test_data = split["test"]
    return train_data, val_data, test_data

def preprocess_sst2(data_split):
    """
    Convert an SST-2 dataset split into model inputs and labels.
    Returns lowercased sentence strings (X) and binary sentiment labels (y).
    """
    df = pd.DataFrame(data_split)
    X = df["sentence"].astype(str).str.lower()
    y = df["label"].astype(int)
    return X, y

def load_snli_splits(seed=0):
    """
    Load the SNLI dataset (train/validation/test) and apply basic cleaning.

    Returns:
        train_data (List[dict]): Cleaned SNLI training split. Each dict contains
             {"premise": str, "hypothesis": str, "label": int, ...}.
        val_data (List[dict]): Cleaned SNLI validation split (same format).
        test_data (List[dict]): Cleaned SNLI test split (same format).
    """
    snli = load_dataset("snli")

    def clean_split(split):
        return [
            {**ex,
             "premise": ex["premise"].lower(),
             "hypothesis": ex["hypothesis"].lower()}
            for ex in split
            if ex["label"] != -1 and ex["premise"] and ex["hypothesis"]
        ]

    return clean_split(snli["train"]), clean_split(snli["validation"]), clean_split(snli["test"])

def preprocess_snli(data_split):
    """
    Inputs:
      - data_split: a list of SNLI examples (e.g., train/val/test split).

    Outputs:
      - X: a pandas Series of combined text strings, where each string is
           "premise [SEP] hypothesis"
      - y: a pandas Series of integer labels for NLI classification
    """
    # SNLI sometimes has -1 for label (no consensus), let's filter those
    df = pd.DataFrame(data_split)
    
    # Concatenate premise and hypothesis with a separator
    df['combined_text'] = df['premise'] + " [SEP] " + df['hypothesis']
    return df['combined_text'], df['label']


def load_with_retries(name, retries=8, base_sleep=10):
    """
    Download a pretrained embedding model from `gensim.downloader` with retry logic.

    Args:
        name (str): The model name to load from `gensim.downloader`.
        retries (int): Maximum number of download attempts before giving up.
        base_sleep (int or float): Base number of seconds to wait between retries.

    Returns:
        model: The downloaded Gensim model object (typically a `KeyedVectors` instance).

    Raises:
        RuntimeError: If the download fails after all retry attempts.
    """
    for i in range(retries):
        try:
            return gensim.downloader.load(name)
        except (HTTPError, URLError) as e:
            print(f"{type(e).__name__}: {e} (try {i+1}/{retries})")
            time.sleep(base_sleep * (i + 1))
    raise RuntimeError(f"Failed to download {name} after {retries} retries.")



def get_embedding_matrix_and_word2idx(wv):
    """
    Given a gensim KeyedVectors object, build a PAD-safe embedding matrix
    and a word2idx mapping with PAD at index 0.

    Args:
        wv: gensim KeyedVectors object with pretrained word vectors

    Returns:
        embedding_matrix: np.ndarray of shape (V+1, D) where V is vocab size
                          and D is embedding dimension. Row 0 is all zeros for PAD.
        word2idx: dict mapping word -> index (1 to V), with 0 reserved for PAD.
    """
    D = wv.vectors.shape[1]
    embedding_matrix = np.vstack([np.zeros((1, D), dtype=wv.vectors.dtype), wv.vectors])
    word2idx = {w: (i + 1) for w, i in wv.key_to_index.items()}
    return embedding_matrix, word2idx


class SentenceIdDataset(Dataset):
    """
    Dataset that converts sentences to variable-length sequences of word IDs.
    Padding is handled at the batch level by collate functions.
    """
    def __init__(self, data_split, word2idx, tokens_to_ids_fn, max_len=50):
        self.data = data_split
        self.word2idx = word2idx
        self.tokens_to_ids = tokens_to_ids_fn
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        toks = tokenize(ex["sentence"])
        ids = self.tokens_to_ids(toks, self.word2idx, max_len=self.max_len)
        ids = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(int(ex["label"]), dtype=torch.long)
        return ids, y


class SNLIPairIdDataset(Dataset):
    """
    Dataset for SNLI that returns (premise_ids, hypothesis_ids, label).
    """
    def __init__(self, data_split, word2idx, tokens_to_ids_fn,  max_len=50):
        self.data = data_split
        self.word2idx = word2idx
        self.tokens_to_ids = tokens_to_ids_fn
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        p_toks = tokenize(ex["premise"])
        h_toks = tokenize(ex["hypothesis"])
        p_ids = self.tokens_to_ids(p_toks, self.word2idx, max_len=self.max_len)
        h_ids = self.tokens_to_ids(h_toks, self.word2idx, max_len=self.max_len)
        p_ids = torch.tensor(p_ids, dtype=torch.long)
        h_ids = torch.tensor(h_ids, dtype=torch.long)
        y = torch.tensor(int(ex["label"]), dtype=torch.long)
        return p_ids, h_ids, y


@torch.no_grad()
def eval_acc(model, loader, snli_mode=False):
    """
    Evaluate model accuracy on a data loader.

    Args:
        model: The model to evaluate
        loader: DataLoader yielding batches
        snli_mode: if True, expects (p_ids, h_ids, labels) batches for SNLI

    Returns:
        Accuracy as a float
    """
    device = get_device()
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        if snli_mode:
            p_ids, h_ids, yb = batch
            p_ids, h_ids, yb = p_ids.to(device), h_ids.to(device), yb.to(device)
            logits = model(p_ids, h_ids)
        else:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total if total else 0.0


def train_dan(model, train_loader, val_loader, epochs, lr=1e-4, weight_decay=1e-4, snli_mode=False):
    """
    Train a DAN model and report validation accuracy each epoch.

    Args:
        model: The DAN model to train (SSTDANClassifier or SNLIDANClassifier)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: number of training epochs
        lr: learning rate for AdamW optimizer
        weight_decay: weight decay for AdamW optimizer
        snli_mode: if True, expects (p_ids, h_ids, labels) batches

    Returns:
        model: The trained model
    """
    device = get_device()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running = 0.0

        for batch in train_loader:
            if snli_mode:
                p_ids, h_ids, yb = batch
                p_ids, h_ids, yb = p_ids.to(device), h_ids.to(device), yb.to(device)
                logits = model(p_ids, h_ids)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)

            opt.zero_grad(set_to_none=True)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            running += loss.item()

        val_acc = eval_acc(model, val_loader, snli_mode=snli_mode)
        print(f"epoch {epoch+1:02d} | train loss={running/len(train_loader):.4f} | val acc={val_acc:.4f}")

    return model
