"""
Test: Even with PERFECT equations, does numerical integration produce errors?
Compare two integrations with IDENTICAL equations but different numerical methods
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

def lorenz_true(state, t, sigma=10, rho=28, beta=8/3):
    """Theoretical Lorenz equations"""
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def lorenz_true_ivp(t, state, sigma=10, rho=28, beta=8/3):
    """Same equations for solve_ivp (different argument order)"""
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Initial condition
initial_state = [1.0, 1.0, 1.0]
t_span = 40
t = np.linspace(0, t_span, 10000)

print("="*80)
print("Testing: Do IDENTICAL equations produce IDENTICAL trajectories?")
print("="*80)

# Method 1: odeint (LSODA)
print("\nIntegrating with odeint (LSODA method)...")
traj1 = odeint(lorenz_true, initial_state, t)

# Method 2: odeint again (should be identical if deterministic)
print("Integrating with odeint again (same method)...")
traj2 = odeint(lorenz_true, initial_state, t)

# Method 3: solve_ivp with RK45
print("Integrating with solve_ivp (RK45 method)...")
sol = solve_ivp(lorenz_true_ivp, [0, t_span], initial_state,
                t_eval=t, method='RK45', rtol=1e-9, atol=1e-12)
traj3 = sol.y.T

# Calculate differences
diff_12 = np.sqrt(np.sum((traj1 - traj2)**2, axis=1))
diff_13 = np.sqrt(np.sum((traj1 - traj3)**2, axis=1))

print("\n" + "="*80)
print("Results:")
print("="*80)

print(f"\nDifference between two odeint runs (SAME method):")
print(f"  Max difference: {np.max(diff_12):.6e}")
print(f"  Mean difference: {np.mean(diff_12):.6e}")
print(f"  Final difference (t=40s): {diff_12[-1]:.6e}")

print(f"\nDifference between odeint and solve_ivp (DIFFERENT methods):")
print(f"  Max difference: {np.max(diff_13):.6e}")
print(f"  Mean difference: {np.mean(diff_13):.6e}")
print(f"  Final difference (t=40s): {diff_13[-1]:.6e}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Difference over time
ax1 = axes[0]
ax1.semilogy(t, diff_12 + 1e-20, 'b-', linewidth=2, label='odeint vs odeint (same)')
ax1.semilogy(t, diff_13 + 1e-20, 'r-', linewidth=2, label='odeint vs solve_ivp (different)')
ax1.set_xlabel('Time (seconds)', fontsize=11)
ax1.set_ylabel('Position Error (log scale)', fontsize=11)
ax1.set_title('Numerical Integration Error\n(Same Equation, Different Runs/Methods)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, which='both')

# Plot 2: Summary
ax2 = axes[1]
ax2.axis('off')

summary_text = f"""
NUMERICAL INTEGRATION ERROR TEST
{"="*50}

SETUP:
• Same initial condition: [1.0, 1.0, 1.0]
• Same equation: Lorenz (σ=10, ρ=28, β=8/3)
• Integration time: 40 seconds
• Number of points: 10,000

RESULTS:

1. Two odeint runs (LSODA, same method):
   Final error: {diff_12[-1]:.6e}

2. odeint vs solve_ivp (different methods):
   Final error: {diff_13[-1]:.6e}

KEY INSIGHT:
Even with IDENTICAL equations, different
numerical integration methods produce
DIFFERENT trajectories in chaotic systems!

This is due to:
• Floating-point rounding errors
• Different step size adaptation
• Different order approximations
• Chaotic sensitivity (Lyapunov exponent)

Even "perfect" equations ≠ perfect trajectory!
"""

ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('outputs/numerical_integration_error.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved to: outputs/numerical_integration_error.png")
print("="*80)
