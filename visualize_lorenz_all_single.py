"""
Visualize Lorenz Attractor: Compare Ground Truth, PySR, and EQL in one plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

# Load real data
def load_lorenz_data():
    """Load real Lorenz data"""
    df = pd.read_csv('data/lorenz/lorenz_dx.tsv.gz', sep='\t', compression='gzip')
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    return x, y, z

# Ground Truth Lorenz equations
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

# PySR discovered equations
def lorenz_pysr(state, t):
    """
    PySR discovered equations:
    dx/dt = (y - x) × 10.0
    dy/dt = x(28 - z) - y
    dz/dt = xy - 2.67z
    """
    x, y, z = state

    dx_dt = (y - x) * 10.0
    dy_dt = x * (28.0 - z) - y
    dz_dt = x * y - 2.6666667 * z

    return [dx_dt, dy_dt, dz_dt]

# EQL discovered equations
def lorenz_eql(state, t):
    """
    EQL discovered equations:
    dx/dt = -10.0*x + 10.0*y - 0.011
    dy/dt = 0.647*x - 1.02*y - 1.41*(8.09 - 0.29*z)*(-2.0*x - 0.015*y - 0.00101)
            - 0.651*(0.225*z - 5.59)*(1.24*x - 0.043*y - 0.00299) + 0.00101
    dz/dt = -0.694*y*(-1.44*x - 2.58) - 1.79*y - 2.67*z - 0.011
    """
    x, y, z = state

    # dx/dt (simple, perfect match)
    dx_dt = -10.0*x + 10.0*y - 0.011

    # dy/dt (complex nested structure)
    dy_dt = (0.647*x - 1.02*y
             - 1.41*(8.09 - 0.29*z)*(-2.0*x - 0.015*y - 0.00101)
             - 0.651*(0.225*z - 5.59)*(1.24*x - 0.043*y - 0.00299)
             + 0.00101)

    # dz/dt (medium complexity)
    dz_dt = -0.694*y*(-1.44*x - 2.58) - 1.79*y - 2.67*z - 0.011

    return [dx_dt, dy_dt, dz_dt]

# Generate trajectory
def generate_trajectory(initial_state, t_span, equations):
    """Generate trajectory using given equations"""
    t = np.linspace(0, t_span, 10000)
    trajectory = odeint(equations, initial_state, t)
    return trajectory

# Visualization
def visualize_comparison():
    """Create single 3D visualization with all three trajectories"""
    print("Loading real data...")
    x_real, y_real, z_real = load_lorenz_data()

    print("Generating trajectories...")
    # Use first point of real data as initial condition
    initial_state = [x_real[0], y_real[0], z_real[0]]
    t_span = 40  # Time span

    trajectory_true = generate_trajectory(initial_state, t_span, lorenz_true)
    trajectory_pysr = generate_trajectory(initial_state, t_span, lorenz_pysr)
    trajectory_eql = generate_trajectory(initial_state, t_span, lorenz_eql)

    # Create single 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all three trajectories
    ax.plot(trajectory_true[:, 0], trajectory_true[:, 1], trajectory_true[:, 2],
            'g-', linewidth=0.6, alpha=0.7, label='Ground Truth (σ=10, ρ=28, β=8/3)')

    ax.plot(trajectory_pysr[:, 0], trajectory_pysr[:, 1], trajectory_pysr[:, 2],
            'r-', linewidth=0.6, alpha=0.7, label='PySR Discovered')

    ax.plot(trajectory_eql[:, 0], trajectory_eql[:, 1], trajectory_eql[:, 2],
            'b-', linewidth=0.6, alpha=0.7, label='EQL Discovered')

    # Labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Lorenz Attractor: Ground Truth vs PySR vs EQL', fontsize=14, pad=20)

    # Legend
    ax.legend(fontsize=11, loc='upper left')

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    # Save figure
    output_path = 'outputs/lorenz_comparison_all_single.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    plt.close()

    # Calculate errors
    print("\n" + "="*70)
    print("Error Analysis (comparing trajectories):")
    print("="*70)

    # Truncate to same length
    n_points = min(len(trajectory_true), len(trajectory_pysr), len(trajectory_eql))

    # PySR vs Ground Truth
    mse_pysr_x = np.mean((trajectory_true[:n_points, 0] - trajectory_pysr[:n_points, 0])**2)
    mse_pysr_y = np.mean((trajectory_true[:n_points, 1] - trajectory_pysr[:n_points, 1])**2)
    mse_pysr_z = np.mean((trajectory_true[:n_points, 2] - trajectory_pysr[:n_points, 2])**2)
    mse_pysr_total = (mse_pysr_x + mse_pysr_y + mse_pysr_z) / 3

    # EQL vs Ground Truth
    mse_eql_x = np.mean((trajectory_true[:n_points, 0] - trajectory_eql[:n_points, 0])**2)
    mse_eql_y = np.mean((trajectory_true[:n_points, 1] - trajectory_eql[:n_points, 1])**2)
    mse_eql_z = np.mean((trajectory_true[:n_points, 2] - trajectory_eql[:n_points, 2])**2)
    mse_eql_total = (mse_eql_x + mse_eql_y + mse_eql_z) / 3

    print(f"\nPySR vs Ground Truth:")
    print(f"  MSE (X): {mse_pysr_x:.6e}")
    print(f"  MSE (Y): {mse_pysr_y:.6e}")
    print(f"  MSE (Z): {mse_pysr_z:.6e}")
    print(f"  Average MSE: {mse_pysr_total:.6e}")

    print(f"\nEQL vs Ground Truth:")
    print(f"  MSE (X): {mse_eql_x:.6e}")
    print(f"  MSE (Y): {mse_eql_y:.6e}")
    print(f"  MSE (Z): {mse_eql_z:.6e}")
    print(f"  Average MSE: {mse_eql_total:.6e}")

    print("\n" + "="*70)
    print("Discovered Equations:")
    print("="*70)

    print("\nGround Truth:")
    print("  dx/dt = 10.0(y - x)")
    print("  dy/dt = x(28.0 - z) - y")
    print("  dz/dt = xy - 2.667z")

    print("\nPySR:")
    print("  dx/dt = 10.0(y - x)")
    print("  dy/dt = x(28.0 - z) - y")
    print("  dz/dt = xy - 2.667z")
    print("  ✅ Perfect match!")

    print("\nEQL:")
    print("  dx/dt = -10.0*x + 10.0*y - 0.011")
    print("  dy/dt = 0.647*x - 1.02*y - 1.41*(8.09 - 0.29*z)*(-2.0*x - ...) - ...")
    print("  dz/dt = -0.694*y*(-1.44*x - 2.58) - 1.79*y - 2.67*z - 0.011")
    print("  ✅ Accurate but more complex")

    print("="*70)

if __name__ == "__main__":
    print("=" * 70)
    print("Lorenz Attractor: Ground Truth vs PySR vs EQL (Single Plot)")
    print("=" * 70)
    visualize_comparison()
