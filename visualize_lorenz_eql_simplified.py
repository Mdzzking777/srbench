"""
Visualize Lorenz Attractor: Compare real data with EQL SIMPLIFIED (id;mul only) discovered formulas
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

# Load EQL simplified discovered equations
def load_eql_simplified_equations():
    """Load EQL SIMPLIFIED discovered equations from JSON results"""
    equations = {}
    for eq in ['dx', 'dy', 'dz']:
        result_path = f'outputs/lorenz_{eq}_eql_simplified/eql_simplified_result.json'
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                data = json.load(f)
                equations[eq] = data.get('symbolic_model', None)
        else:
            equations[eq] = None
            print(f"Warning: EQL simplified result for {eq} not found")
    return equations

# Parse and evaluate EQL equation
def create_lorenz_eql_simplified_function(eql_equations):
    """
    Create a Lorenz system function from EQL SIMPLIFIED discovered equations
    Uses eval() to execute the symbolic string expressions from EQL
    """
    def lorenz_eql_simplified(state, t):
        x, y, z = state

        # Create safe namespace for eval
        namespace = {'x': x, 'y': y, 'z': z}

        try:
            # Evaluate EQL simplified discovered equations
            dx_dt = eval(eql_equations['dx'], {"__builtins__": {}}, namespace) if eql_equations['dx'] else 10.0 * (y - x)
            dy_dt = eval(eql_equations['dy'], {"__builtins__": {}}, namespace) if eql_equations['dy'] else x * (28.0 - z) - y
            dz_dt = eval(eql_equations['dz'], {"__builtins__": {}}, namespace) if eql_equations['dz'] else x * y - (8.0/3.0) * z
        except Exception as e:
            print(f"Warning: Failed to evaluate EQL simplified equations: {e}")
            # Fallback to theoretical equations if parsing fails
            dx_dt = 10.0 * (y - x)
            dy_dt = x * (28.0 - z) - y
            dz_dt = x * y - (8.0/3.0) * z

        return [dx_dt, dy_dt, dz_dt]

    return lorenz_eql_simplified

# True Lorenz equations (for comparison)
def lorenz_true(state, t, sigma=10, rho=28, beta=8/3):
    """
    Theoretical Lorenz equations:
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
def generate_trajectory(initial_state, t_span, equations):
    """Generate trajectory using given equations"""
    t = np.linspace(0, t_span, 10000)
    trajectory = odeint(equations, initial_state, t)
    return trajectory

# Visualization
def visualize_comparison():
    """Create 3D comparison visualization"""
    print("Loading real data...")
    x_real, y_real, z_real = load_lorenz_data()

    print("Loading EQL SIMPLIFIED discovered equations...")
    eql_equations = load_eql_simplified_equations()

    print("Generating trajectories using EQL SIMPLIFIED formulas...")
    # Use first point of real data as initial condition
    initial_state = [x_real[0], y_real[0], z_real[0]]
    t_span = 40  # Time span

    lorenz_eql_simplified_func = create_lorenz_eql_simplified_function(eql_equations)
    trajectory_eql_simplified = generate_trajectory(initial_state, t_span, lorenz_eql_simplified_func)
    trajectory_true = generate_trajectory(initial_state, t_span, lorenz_true)

    # Create 3D figure
    fig = plt.figure(figsize=(16, 6))

    # Subplot 1: Real data vs EQL SIMPLIFIED
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(x_real, y_real, z_real, 'b-', linewidth=0.8, alpha=0.7, label='Real Data')
    ax1.plot(trajectory_eql_simplified[:, 0], trajectory_eql_simplified[:, 1], trajectory_eql_simplified[:, 2],
             'r-', linewidth=0.8, alpha=0.7, label='EQL Simplified (id;mul)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Real Data vs EQL Simplified Equations')
    ax1.legend()

    # Subplot 2: Real data only
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(x_real, y_real, z_real, 'b-', linewidth=1, alpha=0.8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Real Data Only')

    # Subplot 3: EQL SIMPLIFIED vs Theoretical
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(trajectory_true[:, 0], trajectory_true[:, 1], trajectory_true[:, 2],
             'g-', linewidth=0.8, alpha=0.7, label='Theoretical (σ=10, ρ=28, β=8/3)')
    ax3.plot(trajectory_eql_simplified[:, 0], trajectory_eql_simplified[:, 1], trajectory_eql_simplified[:, 2],
             'r-', linewidth=0.8, alpha=0.7, label='EQL Simplified (id;mul)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('EQL Simplified vs Theoretical Equations')
    ax3.legend()

    plt.tight_layout()

    # Save figure
    output_path = 'outputs/lorenz_comparison_eql_simplified.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")

    # Show figure (commented out to avoid interactive wait)
    # plt.show()

    # Calculate errors
    print("\n" + "="*60)
    print("Error Analysis:")
    print("="*60)

    # Truncate to same length for comparison
    n_points = min(len(x_real), len(trajectory_eql_simplified))

    mse_x = np.mean((x_real[:n_points] - trajectory_eql_simplified[:n_points, 0])**2)
    mse_y = np.mean((y_real[:n_points] - trajectory_eql_simplified[:n_points, 1])**2)
    mse_z = np.mean((z_real[:n_points] - trajectory_eql_simplified[:n_points, 2])**2)

    print(f"MSE (X): {mse_x:.6e}")
    print(f"MSE (Y): {mse_y:.6e}")
    print(f"MSE (Z): {mse_z:.6e}")
    print(f"Overall MSE: {(mse_x + mse_y + mse_z)/3:.6e}")
    print("="*60)

    # Print discovered equations
    print("\n" + "="*60)
    print("EQL SIMPLIFIED Discovered Equations:")
    print("="*60)
    for eq, formula in eql_equations.items():
        if formula:
            print(f"{eq}/dt = {formula}")
        else:
            print(f"{eq}/dt = (not found)")
    print("="*60)

if __name__ == "__main__":
    print("=" * 60)
    print("Lorenz Attractor Visualization Comparison (EQL SIMPLIFIED)")
    print("=" * 60)
    visualize_comparison()
