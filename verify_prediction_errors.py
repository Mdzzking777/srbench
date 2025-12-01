"""
Verify prediction errors: Single-point prediction vs trajectory integration
"""

import numpy as np
import pandas as pd
import json
import os

# Load real data
def load_lorenz_data():
    """Load real Lorenz data and derivatives"""
    data = {}
    for eq in ['dx', 'dy', 'dz']:
        df = pd.read_csv(f'data/lorenz/lorenz_{eq}.tsv.gz', sep='\t', compression='gzip')
        data[eq] = df
    return data

# Load equations
def load_equations(eq_type):
    """Load equations from JSON"""
    equations = {}
    # Use equations that match Lorenz structure (not the most complex ones)
    target_complexity = {'dx': 5, 'dy': 9, 'dz': 7}

    for eq in ['dx', 'dy', 'dz']:
        if eq_type == 'pysr':
            csv_path = f'outputs/lorenz_{eq}_pysr/hall_of_fame.csv'
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Get equation with target complexity (matches Lorenz structure)
                target_row = df[df['Complexity'] == target_complexity[eq]]
                if not target_row.empty:
                    equations[eq] = target_row.iloc[0]['Equation']
                else:
                    equations[eq] = None
        elif eq_type == 'eql':
            json_path = f'outputs/lorenz_{eq}_eql/eql_result.json'
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    equations[eq] = data.get('symbolic_model', None)
        elif eq_type == 'eql_simplified':
            json_path = f'outputs/lorenz_{eq}_eql_simplified/eql_simplified_result.json'
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    equations[eq] = data.get('symbolic_model', None)
    return equations

# Evaluate equation on data points
def evaluate_equation(equation, x, y, z):
    """Evaluate discovered equation on data points"""
    namespace = {
        'x': x, 'y': y, 'z': z,
        'sin': np.sin, 'cos': np.cos,
        'exp': np.exp, 'log': np.log,
        'sqrt': np.sqrt, 'abs': np.abs
    }
    try:
        result = eval(equation, {"__builtins__": {}}, namespace)
        return result
    except Exception as e:
        print(f"Error evaluating equation: {e}")
        return None

def verify_errors():
    """Verify single-point prediction errors"""
    print("="*80)
    print("Single-Point Prediction Error Verification")
    print("="*80)

    # Load data
    data = load_lorenz_data()

    # Get x, y, z coordinates
    x = data['dx']['x'].values
    y = data['dx']['y'].values
    z = data['dx']['z'].values

    # Load all equations
    pysr_eq = load_equations('pysr')
    eql_eq = load_equations('eql')
    eql_simp_eq = load_equations('eql_simplified')

    print(f"\nData points: {len(x)}")
    print(f"Using all {len(x)} points for evaluation\n")

    for eq_name in ['dx', 'dy', 'dz']:
        print("="*80)
        print(f"{eq_name.upper()}/dt Equation")
        print("="*80)

        # True derivatives (column name is 'target')
        y_true = data[eq_name]['target'].values

        print(f"\nTrue {eq_name}/dt statistics:")
        print(f"  Mean: {np.mean(y_true):.6e}")
        print(f"  Std:  {np.std(y_true):.6e}")
        print(f"  Min:  {np.min(y_true):.6e}")
        print(f"  Max:  {np.max(y_true):.6e}")

        # PySR predictions
        if pysr_eq.get(eq_name):
            print(f"\n--- PySR ---")
            print(f"Equation: {pysr_eq[eq_name]}")
            y_pred_pysr = evaluate_equation(pysr_eq[eq_name], x, y, z)
            if y_pred_pysr is not None:
                mse = np.mean((y_true - y_pred_pysr)**2)
                mae = np.mean(np.abs(y_true - y_pred_pysr))
                r2 = 1 - np.sum((y_true - y_pred_pysr)**2) / np.sum((y_true - np.mean(y_true))**2)
                print(f"MSE:  {mse:.6e}")
                print(f"MAE:  {mae:.6e}")
                print(f"R²:   {r2:.10f}")
                print(f"Max error: {np.max(np.abs(y_true - y_pred_pysr)):.6e}")

        # EQL Complete predictions
        if eql_eq.get(eq_name):
            print(f"\n--- EQL Complete ---")
            print(f"Equation: {eql_eq[eq_name]}")
            y_pred_eql = evaluate_equation(eql_eq[eq_name], x, y, z)
            if y_pred_eql is not None:
                mse = np.mean((y_true - y_pred_eql)**2)
                mae = np.mean(np.abs(y_true - y_pred_eql))
                r2 = 1 - np.sum((y_true - y_pred_eql)**2) / np.sum((y_true - np.mean(y_true))**2)
                print(f"MSE:  {mse:.6e}")
                print(f"MAE:  {mae:.6e}")
                print(f"R²:   {r2:.10f}")
                print(f"Max error: {np.max(np.abs(y_true - y_pred_eql)):.6e}")

        # EQL Simplified predictions
        if eql_simp_eq.get(eq_name):
            print(f"\n--- EQL Simplified ---")
            print(f"Equation: {eql_simp_eq[eq_name]}")
            y_pred_eql_simp = evaluate_equation(eql_simp_eq[eq_name], x, y, z)
            if y_pred_eql_simp is not None:
                mse = np.mean((y_true - y_pred_eql_simp)**2)
                mae = np.mean(np.abs(y_true - y_pred_eql_simp))
                r2 = 1 - np.sum((y_true - y_pred_eql_simp)**2) / np.sum((y_true - np.mean(y_true))**2)
                print(f"MSE:  {mse:.6e}")
                print(f"MAE:  {mae:.6e}")
                print(f"R²:   {r2:.10f}")
                print(f"Max error: {np.max(np.abs(y_true - y_pred_eql_simp)):.6e}")

        print()

    print("="*80)
    print("Summary")
    print("="*80)
    print("\nKey Points:")
    print("1. Single-point prediction MSE (above) measures accuracy at each data point")
    print("2. Trajectory integration error (in residual plot) measures cumulative error")
    print("   after integrating the ODE for 40 seconds")
    print("3. In chaotic systems, even tiny single-point errors (10^-11) can lead to")
    print("   large trajectory errors (10^0 - 10^1) due to exponential divergence")
    print("="*80)

if __name__ == "__main__":
    verify_errors()
