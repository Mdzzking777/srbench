"""
Summarize EQL discovered Lorenz equation results
"""
import json
import os

print("="*80)
print("EQL Lorenz Equation Discovery Results Summary")
print("="*80)

equations = ['dx', 'dy', 'dz']
theoretical = {
    'dx': 'dx/dt = 10(y - x)  [σ=10]',
    'dy': 'dy/dt = x(28 - z) - y  [ρ=28]',
    'dz': 'dz/dt = xy - (8/3)z  [β≈2.67]'
}

results = []

for eq in equations:
    print(f"\n{'='*80}")
    print(f"Lorenz {eq.upper()} Equation")
    print(f"{'='*80}")

    # Read EQL result JSON
    result_path = f'outputs/lorenz_{eq}_eql/eql_result.json'

    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            data = json.load(f)

        print(f"\nTheoretical formula: {theoretical[eq]}")
        print(f"\nEQL discovered formula:")
        print(f"{'─'*80}")

        # Extract model and performance metrics
        model_str = data.get('symbolic_model', 'N/A')
        print(f"Model: {model_str}")

        # Performance metrics
        if 'mse_test' in data:
            print(f"\nTest MSE: {data['mse_test']:.6e}")
        if 'r2_test' in data:
            print(f"Test R²: {data['r2_test']:.6f}")
        if 'mae_test' in data:
            print(f"Test MAE: {data['mae_test']:.6e}")

        # Training info
        if 'symbolic_complexity' in data:
            print(f"\nComplexity: {data['symbolic_complexity']}")
        if 'training_time' in data:
            print(f"Training time: {data['training_time']:.2f} seconds")

        # Best hyperparameters
        if 'best_params' in data:
            print(f"\nBest hyperparameters:")
            for param, value in data['best_params'].items():
                print(f"  {param}: {value}")

        results.append({
            'equation': eq,
            'model': model_str,
            'mse': data.get('mse_test', float('inf')),
            'r2': data.get('r2_test', 0),
            'complexity': data.get('symbolic_complexity', 'N/A')
        })
    else:
        print(f"Result file not found: {result_path}")

# Summary table
print(f"\n{'='*80}")
print("Overall Summary")
print(f"{'='*80}")
print(f"{'Equation':<10} {'MSE':<15} {'R²':<10} {'Complexity':<15} {'Model'}")
print(f"{'─'*80}")

for r in results:
    model_preview = r['model'][:50] + "..." if len(r['model']) > 50 else r['model']
    print(f"{r['equation']:<10} {r['mse']:<15.6e} {r['r2']:<10.6f} {str(r['complexity']):<15} {model_preview}")

# Theoretical comparison
print(f"\n{'='*80}")
print("Theoretical Formula Comparison")
print(f"{'='*80}")

comparisons = {
    'dx': {
        'theoretical': '10(y - x)',
        'note': 'Linear multiplication form'
    },
    'dy': {
        'theoretical': 'x(28 - z) - y',
        'note': 'Bilinear with subtraction'
    },
    'dz': {
        'theoretical': 'xy - 2.67z',
        'note': 'Bilinear with coefficient 8/3'
    }
}

for eq in equations:
    comp = comparisons[eq]
    print(f"\n{eq.upper()}:")
    print(f"  Theoretical: {comp['theoretical']}")
    print(f"  Note:        {comp['note']}")

    # Find corresponding result
    result = next((r for r in results if r['equation'] == eq), None)
    if result:
        print(f"  Discovered:  {result['model']}")
        print(f"  MSE:         {result['mse']:.6e}")
        print(f"  R²:          {result['r2']:.6f}")

print(f"\n{'='*80}")
print("Algorithm Characteristics")
print(f"{'='*80}")
print("EQL (Equation Learner) is a neural network-based symbolic regression method.")
print("Key features:")
print("  • Uses JAX for automatic differentiation")
print("  • Learns symbolic expressions through gradient descent")
print("  • Enabled functions: sin, cos, exp, log, sqrt, div, mul")
print("  • Addition/subtraction implemented via network weights")
print("  • Grid search over regularization, layers, and function sets")

print(f"\n{'='*80}")
print("Summary")
print(f"{'='*80}")
print("EQL attempted to discover the Lorenz attractor equations from data.")
print("Performance depends on:")
print("  • Network architecture (layers and activation functions)")
print("  • Regularization strength")
print("  • Training iterations (50,000 per configuration)")
print("  • Grid search over hyperparameter space")
print(f"{'='*80}\n")
