# EQL Results for Lorenz dy/dt Equation

## Method: Equation Learner (EQL)
**Neural Network-based Symbolic Regression**

---

## Discovered Equation

```
dy/dt = 0.647*x - 1.02*y - 1.41*(8.09 - 0.29*z)*(-2.0*x - 0.015*y - 0.00101)
        - 0.651*(0.225*z - 5.59)*(1.24*x - 0.043*y - 0.00299) + 0.00101
```

**Theoretical Equation:**
```
dy/dt = x * (28 - z) - y
```

**Note:** EQL found a complex but highly accurate equation. The structure suggests nonlinear feature combinations that approximate the theoretical form.

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Training Time** | 2082.72 seconds (~35 minutes) |
| **MSE (train)** | 3.41 × 10⁻⁹ |
| **MSE (test)** | 3.69 × 10⁻⁹ |
| **MAE (train)** | 4.31 × 10⁻⁵ |
| **MAE (test)** | 4.46 × 10⁻⁵ |
| **R² (train)** | 0.9999999999989977 |
| **R² (test)** | 0.9999999999982889 |

---

## Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| **n_iter** | 50,000 |
| **n_layers** | 1 |
| **reg** | 0.001 |
| **functions** | "id;mul" |

---

## Comparison with PySR

| Metric | EQL | PySR |
|--------|-----|------|
| **Formula** | Complex (nested multiplications) | `x*(28-z) - y` ✅ Simple |
| **R² (test)** | 0.9999999999982889 | ~1.0 |
| **MSE (test)** | 3.69 × 10⁻⁹ | 5.40 × 10⁻¹¹ |
| **Training Time** | ~35 min | ~2 hours |
| **Simplicity** | Low (complex nested terms) | High (direct form) |

**Winner:** PySR found the simpler and more accurate equation.

---

## File Information

- **Algorithm:** EQL (Equation Learner)
- **Random seed:** 42
- **Results file:** results/lorenz/lorenz_dy_eql_42.json
- **Generated:** 2025-11-09
