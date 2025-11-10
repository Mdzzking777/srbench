# EQL Results for Lorenz dz/dt Equation

## Method: Equation Learner (EQL)
**Neural Network-based Symbolic Regression**

---

## Discovered Equation

```
dz/dt = -0.694*y*(-1.44*x - 2.58) - 1.79*y - 2.67*z - 0.011
```

**Simplified Analysis:**
```
= 0.694*y*(1.44*x + 2.58) - 1.79*y - 2.67*z - 0.011
= 0.999*x*y + 1.79*y - 1.79*y - 2.67*z - 0.011
≈ x*y - 2.67*z - 0.011
```

**Theoretical Equation:**
```
dz/dt = x * y - 2.67 * z
```

**Note:** EQL found a more complex representation but it simplifies close to the theoretical form!

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Training Time** | 2104.56 seconds (~35 minutes) |
| **MSE (train)** | 2.26 × 10⁻¹⁰ |
| **MSE (test)** | 2.59 × 10⁻¹⁰ |
| **MAE (train)** | 1.08 × 10⁻⁵ |
| **MAE (test)** | 1.07 × 10⁻⁵ |
| **R² (train)** | 0.999999999999948 |
| **R² (test)** | 0.9999999999999458 |

---

## Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| **n_iter** | 50,000 |
| **n_layers** | 1 |
| **reg** | 0.001 |
| **functions** | "id;mul" |

---

## Coefficient Analysis

| Term | EQL | Theoretical | Match |
|------|-----|-------------|-------|
| **x*y** | ~0.999 | 1.0 | ✅ Very close |
| **z** | -2.67 | -2.67 | ✅ Perfect |
| **constant** | -0.011 | 0.0 | ~0 (negligible) |

---

## Comparison with PySR

| Metric | EQL | PySR |
|--------|-----|------|
| **Formula** | `-0.694*y*(-1.44*x - 2.58) - 1.79*y - 2.67*z` | `x*y - 2.67*z` ✅ |
| **R² (test)** | 0.9999999999999458 | ~1.0 |
| **MSE (test)** | 2.59 × 10⁻¹⁰ | 2.75 × 10⁻¹¹ |
| **Training Time** | ~35 min | ~2 hours |
| **Simplicity** | Medium (but simplifies well) | High (direct form) |

**Winner:** PySR found the simpler form, but EQL achieved comparable accuracy faster!

---

## File Information

- **Algorithm:** EQL (Equation Learner)
- **Random seed:** 42
- **Results file:** results/lorenz/lorenz_dz_eql_42.json
- **Generated:** 2025-11-09
