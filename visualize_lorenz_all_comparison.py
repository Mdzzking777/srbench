"""
Comprehensive Lorenz Attractor Visualization:
Compare Ground Truth vs PySR vs EQL Complete vs EQL Simplified
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import json
import os

# Load real data
def load_lorenz_data():
    """Load real Lorenz data"""
    df = pd.read_csv('data/lorenz/lorenz_dx.tsv.gz', sep='\t', compression='gzip')
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    return x, y, z

# Load PySR results
def load_pysr_equations():
    """Load PySR discovered equations from hall_of_fame.csv"""
    equations = {}
    # Use equations that match Lorenz structure (not the most complex ones)
    target_complexity = {'dx': 5, 'dy': 9, 'dz': 7}

    for eq in ['dx', 'dy', 'dz']:
        csv_path = f'outputs/lorenz_{eq}_pysr/hall_of_fame.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Get equation with target complexity (matches Lorenz structure)
            target_row = df[df['Complexity'] == target_complexity[eq]]
            if not target_row.empty:
                best_equation = target_row.iloc[0]['Equation']
                equations[eq] = best_equation
                print(f"PySR {eq}/dt (complexity {target_complexity[eq]}): {best_equation}")
            else:
                equations[eq] = None
                print(f"Warning: Target complexity {target_complexity[eq]} not found for {eq}")
        else:
            equations[eq] = None
            print(f"Warning: PySR result for {eq} not found")
    return equations

# Load EQL complete results
def load_eql_complete_equations():
    """Load EQL complete discovered equations from JSON results"""
    equations = {}
    for eq in ['dx', 'dy', 'dz']:
        result_path = f'outputs/lorenz_{eq}_eql/eql_result.json'
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                data = json.load(f)
                equations[eq] = data.get('symbolic_model', None)
            print(f"EQL Complete {eq}/dt: {equations[eq]}")
        else:
            equations[eq] = None
            print(f"Warning: EQL complete result for {eq} not found")
    return equations

# Load EQL simplified results
def load_eql_simplified_equations():
    """Load EQL simplified discovered equations from JSON results"""
    equations = {}
    for eq in ['dx', 'dy', 'dz']:
        result_path = f'outputs/lorenz_{eq}_eql_simplified/eql_simplified_result.json'
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                data = json.load(f)
                equations[eq] = data.get('symbolic_model', None)
            print(f"EQL Simplified {eq}/dt: {equations[eq]}")
        else:
            equations[eq] = None
            print(f"Warning: EQL simplified result for {eq} not found")
    return equations

# Create Lorenz function from discovered equations
def create_lorenz_function(equations, name="Model"):
    """
    Create a Lorenz system function from discovered equations
    Uses eval() to execute the symbolic string expressions
    """
    def lorenz_discovered(state, t):
        x, y, z = state

        # Create namespace for eval with math functions
        namespace = {
            'x': x, 'y': y, 'z': z,
            'sin': np.sin, 'cos': np.cos,
            'exp': np.exp, 'log': np.log,
            'sqrt': np.sqrt, 'abs': np.abs
        }

        try:
            # Evaluate discovered equations
            dx_dt = eval(equations['dx'], {"__builtins__": {}}, namespace) if equations['dx'] else 10.0 * (y - x)
            dy_dt = eval(equations['dy'], {"__builtins__": {}}, namespace) if equations['dy'] else x * (28.0 - z) - y
            dz_dt = eval(equations['dz'], {"__builtins__": {}}, namespace) if equations['dz'] else x * y - (8.0/3.0) * z
        except Exception as e:
            print(f"Warning: Failed to evaluate {name} equations: {e}")
            # Fallback to theoretical equations if parsing fails
            dx_dt = 10.0 * (y - x)
            dy_dt = x * (28.0 - z) - y
            dz_dt = x * y - (8.0/3.0) * z

        return [dx_dt, dy_dt, dz_dt]

    return lorenz_discovered

# True Lorenz equations (Ground Truth)
def lorenz_true(state, t, sigma=10, rho=28, beta=8/3):
    """
    Theoretical Lorenz equations (Ground Truth):
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    """
    x, y, z = state

    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    return [dx_dt, dy_dt, dz_dt]

# Generate trajectory
def generate_trajectory(initial_state, t_span, equations, name="Model"):
    """Generate trajectory using given equations"""
    print(f"Generating trajectory for {name}...")
    t = np.linspace(0, t_span, 10000)
    try:
        trajectory = odeint(equations, initial_state, t)
        return trajectory
    except Exception as e:
        print(f"Error generating trajectory for {name}: {e}")
        return None

# Visualization
def visualize_all_comparison():
    """Create comprehensive 3D comparison visualization"""
    print("="*60)
    print("Loading all model equations...")
    print("="*60)

    # Load all equations
    pysr_equations = load_pysr_equations()
    eql_complete_equations = load_eql_complete_equations()
    eql_simplified_equations = load_eql_simplified_equations()

    print("\n" + "="*60)
    print("Generating trajectories...")
    print("="*60)

    # Load real data for initial condition
    x_real, y_real, z_real = load_lorenz_data()
    initial_state = [x_real[0], y_real[0], z_real[0]]
    t_span = 40  # Time span

    # Generate all trajectories
    trajectory_true = generate_trajectory(initial_state, t_span, lorenz_true, "Ground Truth")

    pysr_func = create_lorenz_function(pysr_equations, "PySR")
    trajectory_pysr = generate_trajectory(initial_state, t_span, pysr_func, "PySR")

    eql_complete_func = create_lorenz_function(eql_complete_equations, "EQL Complete")
    trajectory_eql_complete = generate_trajectory(initial_state, t_span, eql_complete_func, "EQL Complete")

    eql_simplified_func = create_lorenz_function(eql_simplified_equations, "EQL Simplified")
    trajectory_eql_simplified = generate_trajectory(initial_state, t_span, eql_simplified_func, "EQL Simplified")

    # Create 3D figure with 3 subplots
    print("\n" + "="*60)
    print("Creating visualization...")
    print("="*60)

    fig = plt.figure(figsize=(18, 6))

    # Set viewing angle (matching visualize_lorenz_pysr.py - matplotlib default)
    elev_angle = 30
    azim_angle = -60

    # Subplot 1: Theoretical vs PySR
    ax1 = fig.add_subplot(131, projection='3d')
    if trajectory_true is not None:
        ax1.plot(trajectory_true[:, 0], trajectory_true[:, 1], trajectory_true[:, 2],
                 'gray', linewidth=2, alpha=0.7, label='Theoretical (σ=10, ρ=28, β=8/3)')
    if trajectory_pysr is not None:
        ax1.plot(trajectory_pysr[:, 0], trajectory_pysr[:, 1], trajectory_pysr[:, 2],
                 'red', linewidth=1.5, alpha=0.8, label='PySR Discovered')
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('Z', fontsize=10)
    ax1.set_title('PySR vs Theoretical Equations', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.view_init(elev=elev_angle, azim=azim_angle)

    # Subplot 2: Theoretical vs EQL Complete
    ax2 = fig.add_subplot(132, projection='3d')
    if trajectory_true is not None:
        ax2.plot(trajectory_true[:, 0], trajectory_true[:, 1], trajectory_true[:, 2],
                 'gray', linewidth=2, alpha=0.7, label='Theoretical (σ=10, ρ=28, β=8/3)')
    if trajectory_eql_complete is not None:
        ax2.plot(trajectory_eql_complete[:, 0], trajectory_eql_complete[:, 1], trajectory_eql_complete[:, 2],
                 'blue', linewidth=1.5, alpha=0.8, label='EQL Complete')
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_zlabel('Z', fontsize=10)
    ax2.set_title('EQL Complete vs Theoretical Equations', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.view_init(elev=elev_angle, azim=azim_angle)

    # Subplot 3: Theoretical vs EQL Simplified
    ax3 = fig.add_subplot(133, projection='3d')
    if trajectory_true is not None:
        ax3.plot(trajectory_true[:, 0], trajectory_true[:, 1], trajectory_true[:, 2],
                 'gray', linewidth=2, alpha=0.7, label='Theoretical (σ=10, ρ=28, β=8/3)')
    if trajectory_eql_simplified is not None:
        ax3.plot(trajectory_eql_simplified[:, 0], trajectory_eql_simplified[:, 1], trajectory_eql_simplified[:, 2],
                 'green', linewidth=1.5, alpha=0.8, label='EQL Simplified')
    ax3.set_xlabel('X', fontsize=10)
    ax3.set_ylabel('Y', fontsize=10)
    ax3.set_zlabel('Z', fontsize=10)
    ax3.set_title('EQL Simplified vs Theoretical Equations', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.view_init(elev=elev_angle, azim=azim_angle)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = 'outputs/lorenz_all_models_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")

    # Calculate MSE for each model against ground truth
    print("\n" + "="*60)
    print("Error Analysis (vs Ground Truth):")
    print("="*60)

    n_points = min(len(trajectory_true), 10000)

    def calc_mse(traj1, traj2, name):
        if traj1 is None or traj2 is None:
            print(f"{name}: N/A (trajectory generation failed)")
            return
        n = min(len(traj1), len(traj2))
        mse_x = np.mean((traj1[:n, 0] - traj2[:n, 0])**2)
        mse_y = np.mean((traj1[:n, 1] - traj2[:n, 1])**2)
        mse_z = np.mean((traj1[:n, 2] - traj2[:n, 2])**2)
        mse_overall = (mse_x + mse_y + mse_z) / 3
        print(f"\n{name}:")
        print(f"  MSE (X): {mse_x:.6e}")
        print(f"  MSE (Y): {mse_y:.6e}")
        print(f"  MSE (Z): {mse_z:.6e}")
        print(f"  Overall MSE: {mse_overall:.6e}")

    calc_mse(trajectory_pysr, trajectory_true, "PySR")
    calc_mse(trajectory_eql_complete, trajectory_true, "EQL Complete")
    calc_mse(trajectory_eql_simplified, trajectory_true, "EQL Simplified")

    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("Lorenz Attractor - Comprehensive Model Comparison")
    print("="*60)
    visualize_all_comparison()
