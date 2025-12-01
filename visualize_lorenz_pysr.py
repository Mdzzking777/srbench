"""
Visualize Lorenz Attractor: Compare real data with PySR discovered formulas
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

# PySR discovered Lorenz equations
def lorenz_pysr(state, t):
    """
    PySR discovered Lorenz equations:
    dx/dt = (y - x) × 10.0
    dy/dt = x(28 - z) - y
    dz/dt = xy - 2.67z
    """
    x, y, z = state

    dx_dt = (y - x) * 10.0
    dy_dt = x * (28.0 - z) - y
    dz_dt = x * y - 2.6666667 * z

    return [dx_dt, dy_dt, dz_dt]

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

    print("Generating trajectories using PySR formulas...")
    # Use first point of real data as initial condition
    initial_state = [x_real[0], y_real[0], z_real[0]]
    t_span = 40  # Time span

    trajectory_pysr = generate_trajectory(initial_state, t_span, lorenz_pysr)
    trajectory_true = generate_trajectory(initial_state, t_span, lorenz_true)

    # Create 3D figure
    fig = plt.figure(figsize=(16, 6))

    # Subplot 1: Real data vs PySR
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(x_real, y_real, z_real, 'b-', linewidth=0.8, alpha=0.7, label='Real Data')
    ax1.plot(trajectory_pysr[:, 0], trajectory_pysr[:, 1], trajectory_pysr[:, 2],
             'r-', linewidth=0.8, alpha=0.7, label='PySR Discovered')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Real Data vs PySR Discovered Equations')
    ax1.legend()

    # Subplot 2: Real data only
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(x_real, y_real, z_real, 'b-', linewidth=1, alpha=0.8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Real Data Only')

    # Subplot 3: PySR vs Theoretical
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(trajectory_true[:, 0], trajectory_true[:, 1], trajectory_true[:, 2],
             'g-', linewidth=0.8, alpha=0.7, label='Theoretical (sigma=10, rho=28, beta=8/3)')
    ax3.plot(trajectory_pysr[:, 0], trajectory_pysr[:, 1], trajectory_pysr[:, 2],
             'r-', linewidth=0.8, alpha=0.7, label='PySR Discovered')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('PySR vs Theoretical Equations')
    ax3.legend()

    plt.tight_layout()

    # Save figure
    output_path = 'outputs/lorenz_comparison_pysr.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")

    # Show figure
    plt.show()

    # Calculate errors
    print("\n" + "="*60)
    print("Error Analysis:")
    print("="*60)

    # Truncate to same length for comparison
    n_points = min(len(x_real), len(trajectory_pysr))

    mse_x = np.mean((x_real[:n_points] - trajectory_pysr[:n_points, 0])**2)
    mse_y = np.mean((y_real[:n_points] - trajectory_pysr[:n_points, 1])**2)
    mse_z = np.mean((z_real[:n_points] - trajectory_pysr[:n_points, 2])**2)

    print(f"MSE (X): {mse_x:.6e}")
    print(f"MSE (Y): {mse_y:.6e}")
    print(f"MSE (Z): {mse_z:.6e}")
    print(f"Overall MSE: {(mse_x + mse_y + mse_z)/3:.6e}")
    print("="*60)

if __name__ == "__main__":
    print("=" * 60)
    print("Lorenz Attractor Visualization Comparison")
    print("=" * 60)
    visualize_comparison()
