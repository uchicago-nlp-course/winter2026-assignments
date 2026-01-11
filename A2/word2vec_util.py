# utils functions used for word2vec evaluation
import urllib.request
import os

def download_analogy_dataset(
    url: str = "https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt",
    out_file: str = "questions-words.txt",
    overwrite: bool = False,
) -> str:
    """
    Download the Google analogy dataset (questions-words.txt) if needed.
    Returns the local path to the file.
    """
    if overwrite or (not os.path.exists(out_file)):
        urllib.request.urlretrieve(url, out_file)
    return out_file

def load_analogy_dataset(filepath):
    """Load Google analogy dataset."""

    analogies = {}
    current_category = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().lower()
            if line.startswith(':'):
                current_category = line[2:]
                analogies[current_category] = []
            elif line and current_category:
                parts = line.split()
                if len(parts) == 4:
                    analogies[current_category].append(parts)
    return analogies

def evaluate_analogies_matrix(vectors, word2idx, idx2word, analogy_dataset):
    """
    Evaluate analogies using a word vector matrix.
    For a:b::c:d, compute b - a + c and find nearest neighbor.
    """
    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors_norm = vectors / norms
    
    results = {}
    
    for category, questions in analogy_dataset.items():
        correct = 0
        total = 0
        skipped = 0
        
        for a, b, c, expected in questions:
            # Check if all words are in vocabulary
            if not all(w in word2idx for w in [a, b, c, expected]):
                skipped += 1
                continue
            
            total += 1
            
            # Get indices
            a_idx, b_idx, c_idx = word2idx[a], word2idx[b], word2idx[c]
            expected_idx = word2idx[expected]
            
            # Compute b - a + c
            query = vectors_norm[b_idx] - vectors_norm[a_idx] + vectors_norm[c_idx]
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            
            # Find most similar (excluding a, b, c)
            similarities = vectors_norm @ query_norm
            similarities[[a_idx, b_idx, c_idx]] = -np.inf  # exclude input words
            
            predicted_idx = np.argmax(similarities)
            
            if predicted_idx == expected_idx:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        results[category] = {
            'correct': correct,
            'total': total,
            'skipped': skipped,
            'accuracy': accuracy
        }
    
    return results

def evaluate_analogies_gensim(wv, analogy_dataset):
    """Evaluate analogies using gensim KeyedVectors."""
    results = {}
    
    for category, questions in analogy_dataset.items():
        correct = 0
        total = 0
        skipped = 0
        
        for a, b, c, expected in questions:
            if not all(w in wv for w in [a, b, c, expected]):
                skipped += 1
                continue
            
            total += 1
            
            try:
                predicted = wv.most_similar(positive=[b, c], negative=[a], topn=1)[0][0]
                if predicted == expected:
                    correct += 1
            except:
                pass
        
        accuracy = correct / total if total > 0 else 0
        results[category] = {
            'correct': correct,
            'total': total,
            'skipped': skipped,
            'accuracy': accuracy
        }
    
    return results

def summarize_results(results, name=None):
    """
    Summarize analogy evaluation results.

    Args:
        results (dict): output of evaluate_analogies_gensim
        name (str, optional): model name or identifier

    Returns:
        summary (dict): result summary
    """
    total_correct = sum(r["correct"] for r in results.values())
    total_attempted = sum(r["total"] for r in results.values())
    total_skipped = sum(r["skipped"] for r in results.values())

    overall_acc = (
        total_correct / total_attempted if total_attempted > 0 else 0.0
    )

    summary = {
        "name": name,
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_attempted": total_attempted,
        "total_skipped": total_skipped,
        "coverage": (
            total_attempted / (total_attempted + total_skipped)
            if (total_attempted + total_skipped) > 0
            else 0.0
        ),
        "by_category": results,  
    }

    return summary
