"""
Data Collection for A3 Q2 (Attention Coding)

Run this script to generate A3-Q2.json for Gradescope submission.
No grading is performed here - only data collection.

Usage:
    python collect_data.py

This will create A3-Q2.json in the same directory.
Submit both model.py and A3-Q2.json to Gradescope.
"""

import json
import torch
from pathlib import Path
import traceback

from model import Attend, Compare, Aggregate, DecomposableAttention


def count_parameters(model):
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Global dictionary to store test results
test_results = {}


def collect_attend_data():
    """Collect data about Attend module implementation."""
    torch.manual_seed(42)
    
    result = {
        "module_name": "Attend",
        "input_shapes": {"A": [2, 5, 100], "B": [2, 6, 100]},
        "output_shapes": None,
        "param_count": None,
        "has_beta_output": False,
        "has_alpha_output": False,
        "error": None,
        "traceback": None
    }
    
    try:
        # Create module
        attend = Attend(num_inputs=100, num_hiddens=200)
        
        # Count parameters
        result["param_count"] = count_parameters(attend)
        
        # Create test inputs
        A = torch.randn(2, 5, 100)
        B = torch.randn(2, 6, 100)
        
        # Forward pass
        output = attend(A, B)
        
        # Collect output information
        if output is not None:
            if isinstance(output, tuple) and len(output) == 2:
                beta, alpha = output
                result["has_beta_output"] = beta is not None
                result["has_alpha_output"] = alpha is not None
                
                if beta is not None:
                    result["output_shapes"] = {"beta": list(beta.shape)}
                if alpha is not None:
                    if result["output_shapes"] is None:
                        result["output_shapes"] = {}
                    result["output_shapes"]["alpha"] = list(alpha.shape)
            else:
                result["output_shapes"] = {"output": list(output.shape) if hasattr(output, 'shape') else str(type(output))}
        
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    test_results["attend_module"] = result


def collect_compare_data():
    """Collect data about Compare module implementation."""
    torch.manual_seed(42)
    
    result = {
        "module_name": "Compare",
        "input_shapes": {"A": [2, 5, 100], "B": [2, 6, 100], "beta": [2, 5, 100], "alpha": [2, 6, 100]},
        "output_shapes": None,
        "param_count": None,
        "has_v_a_output": False,
        "has_v_b_output": False,
        "error": None,
        "traceback": None
    }
    
    try:
        # Create module
        compare = Compare(num_inputs=200, num_hiddens=200)
        
        # Count parameters
        result["param_count"] = count_parameters(compare)
        
        # Create test inputs
        A = torch.randn(2, 5, 100)
        B = torch.randn(2, 6, 100)
        beta = torch.randn(2, 5, 100)
        alpha = torch.randn(2, 6, 100)
        
        # Forward pass
        output = compare(A, B, beta, alpha)
        
        # Collect output information
        if output is not None:
            if isinstance(output, tuple) and len(output) == 2:
                V_A, V_B = output
                result["has_v_a_output"] = V_A is not None
                result["has_v_b_output"] = V_B is not None
                
                if V_A is not None:
                    result["output_shapes"] = {"V_A": list(V_A.shape)}
                if V_B is not None:
                    if result["output_shapes"] is None:
                        result["output_shapes"] = {}
                    result["output_shapes"]["V_B"] = list(V_B.shape)
            else:
                result["output_shapes"] = {"output": list(output.shape) if hasattr(output, 'shape') else str(type(output))}
        
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    test_results["compare_module"] = result


def collect_aggregate_data():
    """Collect data about Aggregate module implementation."""
    torch.manual_seed(42)
    
    result = {
        "module_name": "Aggregate",
        "input_shapes": {"V_A": [2, 5, 200], "V_B": [2, 6, 200]},
        "output_shapes": None,
        "param_count": None,
        "error": None,
        "traceback": None
    }
    
    try:
        # Create module
        aggregate = Aggregate(num_inputs=400, num_hiddens=200, num_outputs=3)
        
        # Count parameters
        result["param_count"] = count_parameters(aggregate)
        
        # Create test inputs
        V_A = torch.randn(2, 5, 200)
        V_B = torch.randn(2, 6, 200)
        
        # Forward pass
        logits = aggregate(V_A, V_B)
        
        # Collect output information
        if logits is not None:
            result["output_shapes"] = {"logits": list(logits.shape)}
        
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    test_results["aggregate_module"] = result


def collect_decomposable_attention_data():
    """Collect data about DecomposableAttention module implementation."""
    torch.manual_seed(42)
    
    result = {
        "module_name": "DecomposableAttention",
        "input_shapes": {"premises": [2, 5], "hypotheses": [2, 6]},
        "output_shapes": None,
        "param_count": None,
        "vocab_size": 1000,
        "embed_size": 100,
        "has_embedding": False,
        "embedding_num_embeddings": None,
        "embedding_dim": None,
        "has_attend": False,
        "has_compare": False,
        "has_aggregate": False,
        "error": None,
        "traceback": None
    }
    
    try:
        # Create model
        vocab_size = 1000
        model = DecomposableAttention(
            vocab_size=vocab_size,
            embed_size=100,
            num_hiddens=200,
            num_inputs_attend=100,
            num_inputs_compare=200,
            num_inputs_agg=400
        )
        
        # Count parameters
        result["param_count"] = count_parameters(model)
        
        # Check architecture
        result["has_embedding"] = hasattr(model, 'embedding')
        if result["has_embedding"]:
            result["embedding_num_embeddings"] = model.embedding.num_embeddings
            result["embedding_dim"] = model.embedding.embedding_dim
        
        result["has_attend"] = hasattr(model, 'attend')
        result["has_compare"] = hasattr(model, 'compare')
        result["has_aggregate"] = hasattr(model, 'aggregate')
        
        # Create test inputs
        premises = torch.randint(0, vocab_size, (2, 5))
        hypotheses = torch.randint(0, vocab_size, (2, 6))
        
        # Forward pass
        logits = model(premises, hypotheses)
        
        # Collect output information
        if logits is not None:
            result["output_shapes"] = {"logits": list(logits.shape)}
        
    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    test_results["decomposable_attention"] = result


def main():
    """Main entry point for data collection."""
    print("Collecting A3 Q2 Data...")
    print("="*60)
    
    collect_attend_data()
    status = "✓" if test_results['attend_module']['error'] is None else "✗"
    print(f"Attend module: {status}")
    
    collect_compare_data()
    status = "✓" if test_results['compare_module']['error'] is None else "✗"
    print(f"Compare module: {status}")
    
    collect_aggregate_data()
    status = "✓" if test_results['aggregate_module']['error'] is None else "✗"
    print(f"Aggregate module: {status}")
    
    collect_decomposable_attention_data()
    status = "✓" if test_results['decomposable_attention']['error'] is None else "✗"
    print(f"DecomposableAttention module: {status}")
    
    # Save results
    output_path = Path(__file__).parent / "A3-Q2.json"
    with open(output_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("="*60)
    print(f"Results saved to: {output_path}")
    print("="*60)
    print("\nSubmit both model.py and A3-Q2.json to Gradescope")


if __name__ == "__main__":
    main()
