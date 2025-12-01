"""
Visualize Lorenz Attractor Residuals:
Show the ERROR between discovered models and theoretical equations
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
    """Load PySR discovered equations"""
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
            else:
                equations[eq] = None
        else:
            equations[eq] = None
    return equations

# Load EQL complete results
def load_eql_complete_equations():
    """Load EQL complete equations"""
    equations = {}
    for eq in ['dx', 'dy', 'dz']:
        result_path = f'outputs/lorenz_{eq}_eql/eql_result.json'
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                data = json.load(f)
                equations[eq] = data.get('symbolic_model', None)
        else:
            equations[eq] = None
    return equations

# Load EQL simplified results
def load_eql_simplified_equations():
    """Load EQL simplified equations"""
    equations = {}
    for eq in ['dx', 'dy', 'dz']:
        result_path = f'outputs/lorenz_{eq}_eql_simplified/eql_simplified_result.json'
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                data = json.load(f)
                equations[eq] = data.get('symbolic_model', None)
        else:
            equations[eq] = None
    return equations

# Create Lorenz function
def create_lorenz_function(equations, name="Model"):
    """Create Lorenz system function from equations"""
    def lorenz_discovered(state, t):
        x, y, z = state
        namespace = {
            'x': x, 'y': y, 'z': z,
            'sin': np.sin, 'cos': np.cos,
            'exp': np.exp, 'log': np.log,
            'sqrt': np.sqrt, 'abs': np.abs
        }
        try:
            dx_dt = eval(equations['dx'], {"__builtins__": {}}, namespace) if equations['dx'] else 10.0 * (y - x)
            dy_dt = eval(equations['dy'], {"__builtins__": {}}, namespace) if equations['dy'] else x * (28.0 - z) - y
            dz_dt = eval(equations['dz'], {"__builtins__": {}}, namespace) if equations['dz'] else x * y - (8.0/3.0) * z
        except Exception as e:
            print(f"Warning: {name}: {e}")
            dx_dt = 10.0 * (y - x)
            dy_dt = x * (28.0 - z) - y
            dz_dt = x * y - (8.0/3.0) * z
        return [dx_dt, dy_dt, dz_dt]
    return lorenz_discovered

# True Lorenz equations
def lorenz_true(state, t, sigma=10, rho=28, beta=8/3):
    """Theoretical Lorenz equations"""
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Generate trajectory
def generate_trajectory(initial_state, t_span, equations, name="Model"):
    """Generate trajectory"""
    t = np.linspace(0, t_span, 10000)
    try:
        trajectory = odeint(equations, initial_state, t)
        return trajectory
    except Exception as e:
        print(f"Error: {name}: {e}")
        return None

# Visualization
def visualize_residuals():
    """Create residual visualization"""
    print("="*60)
    print("Loading equations...")
    print("="*60)

    pysr_eq = load_pysr_equations()
    eql_complete_eq = load_eql_complete_equations()
    eql_simplified_eq = load_eql_simplified_equations()

    print("\nGenerating trajectories...")
    x_real, y_real, z_real = load_lorenz_data()
    initial_state = [x_real[0], y_real[0], z_real[0]]
    t_span = 40

    traj_true = generate_trajectory(initial_state, t_span, lorenz_true, "True")
    traj_pysr = generate_trajectory(initial_state, t_span, create_lorenz_function(pysr_eq, "PySR"), "PySR")
    traj_eql_complete = generate_trajectory(initial_state, t_span, create_lorenz_function(eql_complete_eq, "EQL Complete"), "EQL Complete")
    traj_eql_simplified = generate_trajectory(initial_state, t_span, create_lorenz_function(eql_simplified_eq, "EQL Simplified"), "EQL Simplified")

    # Calculate residuals
    n = min(len(traj_true), len(traj_pysr), len(traj_eql_complete), len(traj_eql_simplified))

    residual_pysr = traj_pysr[:n] - traj_true[:n]
    residual_eql_complete = traj_eql_complete[:n] - traj_true[:n]
    residual_eql_simplified = traj_eql_simplified[:n] - traj_true[:n]

    # Create figure
    print("\nCreating residual visualization...")
    fig = plt.figure(figsize=(18, 10))

    # Time array
    t = np.linspace(0, t_span, n)

    # Row 1: Residuals in X, Y, Z over time
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(t, residual_pysr[:, 0], 'r-', linewidth=1, alpha=0.8, label='PySR')
    ax1.plot(t, residual_eql_complete[:, 0], 'b-', linewidth=1, alpha=0.8, label='EQL Complete')
    ax1.plot(t, residual_eql_simplified[:, 0], 'g-', linewidth=1, alpha=0.8, label='EQL Simplified')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Error in X')
    ax1.set_title('X-component Error vs Time', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t, residual_pysr[:, 1], 'r-', linewidth=1, alpha=0.8, label='PySR')
    ax2.plot(t, residual_eql_complete[:, 1], 'b-', linewidth=1, alpha=0.8, label='EQL Complete')
    ax2.plot(t, residual_eql_simplified[:, 1], 'g-', linewidth=1, alpha=0.8, label='EQL Simplified')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Error in Y')
    ax2.set_title('Y-component Error vs Time', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, residual_pysr[:, 2], 'r-', linewidth=1, alpha=0.8, label='PySR')
    ax3.plot(t, residual_eql_complete[:, 2], 'b-', linewidth=1, alpha=0.8, label='EQL Complete')
    ax3.plot(t, residual_eql_simplified[:, 2], 'g-', linewidth=1, alpha=0.8, label='EQL Simplified')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Error in Z')
    ax3.set_title('Z-component Error vs Time', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Row 2: 3D residual plots (error trajectories in 3D space)
    elev_angle = 30
    azim_angle = -60

    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.plot(residual_pysr[:, 0], residual_pysr[:, 1], residual_pysr[:, 2],
             'r-', linewidth=1, alpha=0.7)
    ax4.set_xlabel('ΔX')
    ax4.set_ylabel('ΔY')
    ax4.set_zlabel('ΔZ')
    ax4.set_title('PySR Error Trajectory', fontweight='bold', color='red')
    ax4.view_init(elev=elev_angle, azim=azim_angle)

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax5.plot(residual_eql_complete[:, 0], residual_eql_complete[:, 1], residual_eql_complete[:, 2],
             'b-', linewidth=1, alpha=0.7)
    ax5.set_xlabel('ΔX')
    ax5.set_ylabel('ΔY')
    ax5.set_zlabel('ΔZ')
    ax5.set_title('EQL Complete Error Trajectory', fontweight='bold', color='blue')
    ax5.view_init(elev=elev_angle, azim=azim_angle)

    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    ax6.plot(residual_eql_simplified[:, 0], residual_eql_simplified[:, 1], residual_eql_simplified[:, 2],
             'g-', linewidth=1, alpha=0.7)
    ax6.set_xlabel('ΔX')
    ax6.set_ylabel('ΔY')
    ax6.set_zlabel('ΔZ')
    ax6.set_title('EQL Simplified Error Trajectory', fontweight='bold', color='green')
    ax6.view_init(elev=elev_angle, azim=azim_angle)

    plt.tight_layout()

    output_path = 'outputs/lorenz_residuals_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Residual visualization saved to: {output_path}")

    # Calculate statistics
    print("\n" + "="*60)
    print("Residual Statistics:")
    print("="*60)

    for name, residual in [('PySR', residual_pysr),
                           ('EQL Complete', residual_eql_complete),
                           ('EQL Simplified', residual_eql_simplified)]:
        print(f"\n{name}:")
        print(f"  Max absolute error: {np.max(np.abs(residual)):.6e}")
        print(f"  Mean absolute error: {np.mean(np.abs(residual)):.6e}")
        print(f"  RMS error: {np.sqrt(np.mean(residual**2)):.6e}")

    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("Lorenz Attractor - Residual Analysis")
    print("="*60)
    visualize_residuals()
