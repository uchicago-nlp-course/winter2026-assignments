import torch
from torch.utils.data import DataLoader
from utils import (
    load_snli_splits,
    load_with_retries,
    get_embedding_matrix_and_word2idx,
    SNLIPairIdDataset,
    collate_pair_pad,
    tokens_to_ids,
    train_loop,
    eval_accuracy,
    get_device,
)
from model import DecomposableAttention

# Hyperparameters
BATCH_SIZE = 256
MAX_LEN = 50
EMBED_SIZE = 100
NUM_HIDDENS = 200
LEARNING_RATE = 0.001
NUM_EPOCHS = 3
DEVICE = get_device()


def train():
    print(f"Training on {DEVICE}...")

    # 1. Load Data
    print("Loading SNLI dataset...")
    train_data, val_data, test_data = load_snli_splits()
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 2. Load GloVe embeddings
    print("Loading GloVe embeddings via gensim...")
    wv = load_with_retries("glove-wiki-gigaword-100")
    embedding_matrix, word2idx = get_embedding_matrix_and_word2idx(wv)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")

    # 3. Build datasets and dataloaders
    print("Building datasets...")
    train_ds = SNLIPairIdDataset(train_data, word2idx, tokens_to_ids, max_len=MAX_LEN)
    val_ds = SNLIPairIdDataset(val_data, word2idx, tokens_to_ids, max_len=MAX_LEN)
    test_ds = SNLIPairIdDataset(test_data, word2idx, tokens_to_ids, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pair_pad)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pair_pad)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pair_pad)

    # 4. Initialize Model
    vocab_size = embedding_matrix.shape[0]
    net = DecomposableAttention(vocab_size, EMBED_SIZE, NUM_HIDDENS)

    # Copy pretrained embeddings into model
    net.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
    net = net.to(DEVICE)

    # 5. Train
    print("Starting training...")
    net = train_loop(net, train_loader, val_loader, epochs=NUM_EPOCHS, lr=LEARNING_RATE)

    # 6. Evaluate on test set
    test_acc = eval_accuracy(net, test_loader)
    print(f"Test Accuracy: {test_acc:.3f}")

    # 7. Prediction Example
    print("\nRunning a sample prediction...")
    predict_snli(net, word2idx, premise=['he', 'is', 'good', '.'], hypothesis=['he', 'is', 'bad', '.'])

    return net


def predict_snli(net, word2idx, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    net.eval()
    device = get_device()

    # Preprocess input (assumes list of tokens)
    p_ids = [word2idx.get(token, 0) for token in premise]
    h_ids = [word2idx.get(token, 0) for token in hypothesis]

    # Reshape to (1, len) for batch processing
    p_tensor = torch.tensor(p_ids, device=device).reshape(1, -1)
    h_tensor = torch.tensor(h_ids, device=device).reshape(1, -1)

    # Hugging Face SNLI label order
    label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

    with torch.no_grad():
        output = net(p_tensor, h_tensor)
        pred_idx = output.argmax(dim=1).item()

    result = label_map[pred_idx]
    print(f"Premise: {' '.join(premise)}")
    print(f"Hypothesis: {' '.join(hypothesis)}")
    print(f"Prediction: {result}")
    return result


if __name__ == "__main__":
    train()
