# EQL (Equation Learner) Explained

## What is EQL?

**EQL (Equation Learner)** is a neural network architecture specifically designed to discover **symbolic mathematical equations** from data, rather than just making numerical predictions.

**Paper:** Sahoo et al. (2018) - "Learning Equations for Extrapolation and Control" (ICML 2018)

---

## Key Idea: Symbolic Activation Functions

### Normal Neural Network
```
Input: x, y, z
    ↓
Layer 1: σ(W₁·X + b₁)  ← tanh, ReLU, sigmoid
    ↓
Layer 2: σ(W₂·X + b₂)
    ↓
Output: numerical prediction
```
**Problem:** You get a black box that makes predictions, but you can't extract a formula!

### EQL Network
```
Input: x, y, z
    ↓
Layer 1: [x, y, z, x², sin(x), x*y, ...]  ← Symbolic functions
         ↑
    Weights select which functions matter
    ↓
Layer 2: [combine previous terms]
    ↓
Output: SYMBOLIC EQUATION that can be read!
```

---

## Core Difference from Normal NN

| Aspect | Normal NN | EQL |
|--------|-----------|-----|
| **Activation Functions** | tanh, ReLU, sigmoid | **Symbolic: id, mul, sin, exp, div** |
| **Output** | Numerical prediction | **Symbolic equation** |
| **Interpretability** | Black box | **Fully interpretable** |
| **Weights** | Hidden parameters | **Coefficients in equation** |
| **Purpose** | Predict | **Discover equations** |

---

## Detailed Architecture

### 1. Symbolic Layer Structure

Each "neuron" applies a symbolic mathematical function:

```python
# Normal neuron:
output = tanh(w₁*x₁ + w₂*x₂ + w₃*x₃ + b)

# EQL neuron with function 'mul':
output = w * (input₁ * input₂)  # Multiplication function

# EQL neuron with function 'sin':
output = w * sin(input)  # Sine function
```

### 2. Available Symbolic Functions

From the code we saw (`utils.py`):
```python
f_dict_jax = {
    "sin": (jnp.sin, 1),        # Unary: sin(x)
    "cos": (jnp.cos, 1),        # Unary: cos(x)
    "id": (identity, 1),         # Unary: x (pass through)
    "mul": (jnp.multiply, 2),    # Binary: x * y
    "div": (div, 2),             # Binary: x / y
    "sqrt": (sqrt_jax, 1),       # Unary: sqrt(x)
    "exp": (jnp.exp, 1),         # Unary: exp(x)
    "log": (log_jax, 1),         # Unary: log(x)
    "square": (square, 1),       # Unary: x²
    "cube": (cube, 1),           # Unary: x³
}
```

Each function has an "arity" (number of inputs):
- **Unary (1):** Takes one input (sin, cos, id, sqrt, exp, log, square, cube)
- **Binary (2):** Takes two inputs (mul, div)

### 3. Layer Construction

**Example with functions = "id;mul;id;mul":**

```
Layer:
├── Neuron 1: id(x)      ← w₁ * x
├── Neuron 2: mul(x,y)   ← w₂ * (x * y)
├── Neuron 3: id(z)      ← w₃ * z
└── Neuron 4: mul(x,z)   ← w₄ * (x * z)

Output = w₁*x + w₂*(x*y) + w₃*z + w₄*(x*z) + bias
```

---

## How Training Works

### Step 1: Forward Pass
```python
# Input: x=1, y=2, z=3
# Functions: id, mul, id

neuron1 = w₁ * id(x) = w₁ * 1
neuron2 = w₂ * mul(x,y) = w₂ * (1*2) = w₂ * 2
neuron3 = w₃ * id(z) = w₃ * 3

output = neuron1 + neuron2 + neuron3 + bias
       = w₁*1 + w₂*2 + w₃*3 + b
```

### Step 2: Gradient Descent
```python
loss = MSE(output, target)
# Update weights using backpropagation (just like normal NN)
w₁ ← w₁ - lr * ∂loss/∂w₁
w₂ ← w₂ - lr * ∂loss/∂w₂
w₃ ← w₃ - lr * ∂loss/∂w₃
```

### Step 3: L0 Regularization (Sparsity)
This is **key** to getting simple equations!

```python
# Encourage many weights to become exactly zero
regularization = λ * Σ(|wᵢ| > threshold)
```

This forces the network to use **only the necessary terms**.

### Step 4: Extract Equation
After training, read the weights:
```python
if |w₁| > 0.01: keep "x"
if |w₂| > 0.01: keep "x*y"
if |w₃| < 0.01: DROP "z"

Final equation: w₁*x + w₂*(x*y) + bias
```

---

## Example: Learning dx/dt = 10(y - x)

### Training Process:

**Iteration 1:** (random initialization)
```
equation = 0.5*x + 0.3*y - 0.1*z + 0.2*(x*y) + 0.4*(y*z)
loss = 1000
```

**Iteration 1000:**
```
equation = -8.2*x + 8.5*y - 0.05*z + 0.01*(x*y)
loss = 0.5
```

**Iteration 10000:**
```
equation = -9.9*x + 9.9*y - 0.001*z + 0.0*(x*y)
loss = 0.001
```

**Iteration 50000:** (converged)
```
equation = -10.0*x + 10.0*y - 0.011
loss = 2.7e-11
```

L0 regularization zeroed out unnecessary terms!

---

## Why "Equation Learner"?

Unlike normal neural networks that are black boxes, EQL:

1. **Learns in symbolic space** - Uses mathematical functions as building blocks
2. **Produces interpretable output** - You can read the discovered equation
3. **Enforces sparsity** - L0 regularization keeps equations simple
4. **Gradient-based** - Fast convergence compared to evolutionary methods

---

## Comparison: EQL vs Normal NN vs PySR

### Normal Neural Network
```python
model = Sequential([
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
# After training:
prediction = model.predict([x, y, z])  # Get a number
# What's the equation? → Can't extract! Black box!
```

### EQL (Equation Learner)
```python
model = EQL(functions='id;mul', n_iter=50000)
model.fit(X, y)
# After training:
equation = model.get_eqn()  # → "-10.0*x + 10.0*y - 0.011"
# ✅ Interpretable equation!
```

### PySR (Genetic Programming)
```python
model = PySRRegressor(niterations=100)
model.fit(X, y)
# After training:
equation = model.sympy()  # → "10.0*(y - x)"
# ✅ Even simpler form through evolution!
```

---

## Advantages of EQL

### ✅ Strengths

1. **Fast convergence** (gradient descent, not evolution)
   - ~35 minutes for Lorenz equations
   - vs. PySR's ~2 hours

2. **Deterministic** (with fixed random seed)
   - Same result every time
   - vs. PySR's stochastic nature

3. **Neural network infrastructure**
   - GPU acceleration via JAX
   - Well-established optimization techniques
   - Easy to integrate with deep learning pipelines

4. **Automatic hyperparameter tuning**
   - GridSearchCV built-in
   - Tests multiple configurations

5. **Extrapolation focus**
   - Designed for learning dynamical systems
   - Good at temporal extrapolation

### ⚠️ Limitations

1. **Less interpretable than PySR**
   - May find complex but equivalent forms
   - Example: `-0.694*y*(-1.44*x - 2.58)` vs. `x*y`

2. **Fixed function set**
   - Must pre-specify symbolic functions
   - Can't discover novel functional forms

3. **Local optima**
   - Gradient descent may get stuck
   - vs. GP's global exploration

4. **No natural complexity-accuracy tradeoff**
   - Doesn't automatically generate multiple candidates
   - vs. PySR's Hall of Fame (Pareto front)

---

## Technical Implementation Details

### L0 Regularization Layer

```python
class L0Dense(nn.Module):
    """Dense layer with L0 regularization for sparsity"""

    def __call__(self, x, deterministic):
        # Standard linear transformation
        kernel = self.param('kernel', nn.initializers.normal(), shape)

        # L0 gate: learns which connections to keep
        log_alpha = self.param('log_alpha', nn.initializers.zeros(), shape)

        if deterministic:
            # At inference: hard threshold
            mask = (log_alpha > threshold)
        else:
            # During training: stochastic gates
            mask = sample_bernoulli(log_alpha)

        # Apply mask to enforce sparsity
        y = jnp.dot(x, kernel * mask) + bias

        return y
```

This creates **sparse** networks where most weights are exactly zero!

### Symbolic Layer

```python
class SymbolicLayer:
    def __init__(self, functions):
        # Parse function string: "id;mul;sin" → [id, mul, sin]
        self.functions = [f_dict[f] for f in functions.split(';')]

    def __call__(self, x):
        outputs = []
        idx = 0

        for func, arity in self.functions:
            if arity == 1:  # Unary function
                outputs.append(func(x[idx]))
                idx += 1
            elif arity == 2:  # Binary function
                outputs.append(func(x[idx], x[idx+1]))
                idx += 2

        return jnp.concatenate(outputs)
```

---

## Real-World Use Cases

### When EQL Excels:
- 🔬 **Discovering physical laws** from sensor data
- 🤖 **Learning control equations** for robotics
- 📈 **Time series extrapolation** (dynamical systems)
- 🧪 **Scientific discovery** where speed matters
- 💻 **GPU acceleration** is available

### When to Use PySR Instead:
- 📖 Need **maximally simple** equations
- 🎯 **Interpretability** is critical
- 🔍 Want **multiple candidate equations**
- ⏰ Time is not a constraint
- 🔬 Scientific publication requires exact forms

---

## Our Lorenz Results: What Happened?

### dx/dt: Perfect! ✅
```
EQL found: -10.0*x + 10.0*y - 0.011
Theory:     10.0*(y - x)
```
**Why perfect?** Linear equation, simple structure, EQL excels at these!

### dy/dt: Complex ⚠️
```
EQL found: 0.647*x - 1.02*y - 1.41*(8.09 - 0.29*z)*(-2.0*x - 0.015*y...)
Theory:     x*(28 - z) - y
```
**Why complex?** Nonlinear term `x*z` requires multiple multiplications to construct

### dz/dt: Good Approximation ✅
```
EQL found: -0.694*y*(-1.44*x - 2.58) - 1.79*y - 2.67*z - 0.011
Simplifies: ≈ x*y - 2.67*z
Theory:     x*y - 2.67*z
```
**Why good?** Product term `x*y` naturally fits EQL's multiplicative functions!

---

## Conclusion

**EQL is a neural network that learns equations, not just predictions.**

It bridges the gap between:
- **Neural networks** (fast gradient-based optimization)
- **Symbolic regression** (interpretable mathematical formulas)

Think of it as: **"Symbolic Regression with Neural Network Speed"**

---

## Further Reading

- **Original Paper:** Sahoo, S., Lampert, C., & Martius, G. (2018). "Learning Equations for Extrapolation and Control." ICML 2018
- **Code:** https://github.com/martius-lab/EQL
- **Related:** Physics-Informed Neural Networks (PINNs), SINDy

---

**Generated:** 2025-11-09
**Framework:** SRBENCH with EQL 0.2
