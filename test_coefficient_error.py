"""
Test: How much error does the coefficient difference (8/3 vs 2.6666667) cause?
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def lorenz_exact(state, t):
    """Theoretical Lorenz with EXACT 8/3"""
    x, y, z = state
    dx_dt = 10.0 * (y - x)
    dy_dt = x * (28.0 - z) - y
    dz_dt = x * y - (8.0/3.0) * z  # EXACT
    return [dx_dt, dy_dt, dz_dt]

def lorenz_pysr(state, t):
    """PySR discovered equations"""
    x, y, z = state
    dx_dt = (y - x) * 10.0
    dy_dt = (x - y) - (x * (z + -27.0))
    dz_dt = (y * x) - (2.6666667 * z)  # Slightly different
    return [dx_dt, dy_dt, dz_dt]

# Initial condition
initial_state = [1.0, 1.0, 1.0]
t_span = 40
t = np.linspace(0, t_span, 10000)

print("="*80)
print("Testing: Impact of Coefficient Precision on Trajectory")
print("="*80)

print(f"\nCoefficient difference in dz/dt:")
print(f"  Exact 8/3:     {8/3:.15f}")
print(f"  PySR:          {2.6666667:.15f}")
print(f"  Difference:    {8/3 - 2.6666667:.15e}")

# Integrate both
print("\nIntegrating with exact 8/3...")
traj_exact = odeint(lorenz_exact, initial_state, t)

print("Integrating with PySR's 2.6666667...")
traj_pysr = odeint(lorenz_pysr, initial_state, t)

# Calculate error
error = np.sqrt(np.sum((traj_pysr - traj_exact)**2, axis=1))

print("\n" + "="*80)
print("Trajectory Divergence Due to Coefficient Error Alone:")
print("="*80)

print(f"\nError at t=10s:  {error[np.argmin(np.abs(t - 10))]:.6f}")
print(f"Error at t=20s:  {error[np.argmin(np.abs(t - 20))]:.6f}")
print(f"Error at t=40s:  {error[-1]:.6f}")
print(f"Max error:       {np.max(error):.6f}")
print(f"Mean error:      {np.mean(error):.6f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Error over time
ax1 = axes[0, 0]
ax1.plot(t, error, 'r-', linewidth=2)
ax1.set_xlabel('Time (seconds)', fontsize=11)
ax1.set_ylabel('Position Error', fontsize=11)
ax1.set_title('Trajectory Error from Coefficient Precision\n(8/3 vs 2.6666667)', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Error over time (log scale)
ax2 = axes[0, 1]
ax2.semilogy(t, error + 1e-15, 'r-', linewidth=2)
ax2.set_xlabel('Time (seconds)', fontsize=11)
ax2.set_ylabel('Position Error (log scale)', fontsize=11)
ax2.set_title('Error Growth (Log Scale)', fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Component-wise errors
ax3 = axes[1, 0]
error_x = np.abs(traj_pysr[:, 0] - traj_exact[:, 0])
error_y = np.abs(traj_pysr[:, 1] - traj_exact[:, 1])
error_z = np.abs(traj_pysr[:, 2] - traj_exact[:, 2])
ax3.plot(t, error_x, 'r-', linewidth=1.5, alpha=0.7, label='X error')
ax3.plot(t, error_y, 'g-', linewidth=1.5, alpha=0.7, label='Y error')
ax3.plot(t, error_z, 'b-', linewidth=1.5, alpha=0.7, label='Z error')
ax3.set_xlabel('Time (seconds)', fontsize=11)
ax3.set_ylabel('Component Error', fontsize=11)
ax3.set_title('Error in Each Component', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
COEFFICIENT PRECISION IMPACT ANALYSIS
{"="*50}

DZ/DT COEFFICIENT:
• Exact (8/3):      {8/3:.15f}
• PySR discovered:  {2.6666667:.15f}
• Difference:       {8/3 - 2.6666667:.3e}

TRAJECTORY ERROR OVER 40 SECONDS:
• Error at t=10s:   {error[np.argmin(np.abs(t - 10))]:.6f}
• Error at t=20s:   {error[np.argmin(np.abs(t - 20))]:.6f}
• Error at t=40s:   {error[-1]:.6f}
• Maximum error:    {np.max(error):.6f}

KEY INSIGHT:
A coefficient error of only 3.3×10⁻⁸
(0.0000033%) leads to trajectory error
of ~{error[-1]:.1f} after 40 seconds!

This is {error[-1] / (8/3 - 2.6666667):.2e}× amplification!

In chaotic systems, even "perfect" symbolic
regression (R²=1.0, MSE=10⁻²⁹) cannot
guarantee perfect long-term predictions.

The fundamental limitation is:
1. Floating-point representation
2. Lyapunov exponent amplification
3. Numerical integration errors
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=8.5, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('outputs/coefficient_precision_impact.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved to: outputs/coefficient_precision_impact.png")
print("="*80)
