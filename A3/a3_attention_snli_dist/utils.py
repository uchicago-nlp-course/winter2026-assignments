# Utilities for A3 Attention-based SNLI
# Adapted from A2 classification utilities for consistency across assignments.

import time
from urllib.error import HTTPError, URLError

import numpy as np
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


def load_snli_splits(seed=0):
    """
    Load the SNLI dataset (train/validation/test) and apply basic cleaning.

    Args:
        seed: Random seed for reproducibility (default 0, same as A2).

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


def load_with_retries(name, retries=8, base_sleep=10):
    """
    Download a pretrained embedding model from gensim.downloader with retry logic.

    Args:
        name (str): The model name to load (e.g., "glove-wiki-gigaword-100")
        retries (int): Maximum number of download attempts
        base_sleep (int): Base seconds to wait between retries (increases linearly)

    Returns:
        The downloaded gensim model (KeyedVectors instance)

    Raises:
        RuntimeError: If download fails after all retry attempts
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


def tokens_to_ids(tokens, word2idx, max_len=50):
    """Convert tokens to IDs, truncating to max_len."""
    ids = [word2idx.get(t, 0) for t in tokens[:max_len]]
    return ids


class SNLIPairIdDataset(Dataset):
    """
    Dataset for SNLI that returns (premise_ids, hypothesis_ids, label).
    Uses lazy tokenization for memory efficiency.
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
        p_toks = tokenize(ex["premise"])
        h_toks = tokenize(ex["hypothesis"])
        p_ids = self.tokens_to_ids(p_toks, self.word2idx, max_len=self.max_len)
        h_ids = self.tokens_to_ids(h_toks, self.word2idx, max_len=self.max_len)
        p_ids = torch.tensor(p_ids, dtype=torch.long)
        h_ids = torch.tensor(h_ids, dtype=torch.long)
        y = torch.tensor(int(ex["label"]), dtype=torch.long)
        return p_ids, h_ids, y


def collate_pair_pad(batch):
    """
    Collate function for sentence-pair datasets (SNLI) with dynamic padding.

    Input:
        batch: list of (premise_ids, hypothesis_ids, label) tuples
    Output:
        p_padded: LongTensor of shape (B, Tp_max)
        h_padded: LongTensor of shape (B, Th_max)
        labels: LongTensor of shape (B,)
    """
    p_list, h_list, labels = zip(*batch)
    p_padded = pad_sequence(p_list, batch_first=True, padding_value=PAD)
    h_padded = pad_sequence(h_list, batch_first=True, padding_value=PAD)
    labels = torch.stack(labels)
    return p_padded, h_padded, labels


@torch.no_grad()
def eval_accuracy(model, loader):
    """
    Evaluate model accuracy on a data loader.

    Args:
        model: The model to evaluate (expects model(p_ids, h_ids))
        loader: DataLoader yielding (p_ids, h_ids, labels) batches

    Returns:
        Accuracy as a float
    """
    device = get_device()
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        p_ids, h_ids, yb = batch
        p_ids, h_ids, yb = p_ids.to(device), h_ids.to(device), yb.to(device)
        logits = model(p_ids, h_ids)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total if total else 0.0


def train_loop(model, train_loader, val_loader, epochs, lr=1e-3, weight_decay=1e-4):
    """
    Train a model and report validation accuracy each epoch.

    Args:
        model: The model to train (expects model(p_ids, h_ids))
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: number of training epochs
        lr: learning rate for Adam optimizer
        weight_decay: weight decay for Adam optimizer

    Returns:
        model: The trained model
    """
    device = get_device()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running = 0.0

        for i, batch in enumerate(train_loader):
            p_ids, h_ids, yb = batch
            p_ids, h_ids, yb = p_ids.to(device), h_ids.to(device), yb.to(device)
            logits = model(p_ids, h_ids)

            opt.zero_grad()
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            running += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss {loss.item():.4f}")

        val_acc = eval_accuracy(model, val_loader)
        print(f"Epoch {epoch+1}, Loss {running/len(train_loader):.3f}, Val Acc {val_acc:.3f}")

    return model
