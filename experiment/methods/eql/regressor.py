"""
EQL (Equation Learner) configuration for Lorenz system
Neural network-based symbolic regression using JAX
"""
import sys
import os

# Add algorithms/eql to path to import eql module
algorithms_eql_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms', 'eql')
sys.path.insert(0, os.path.abspath(algorithms_eql_path))

from eql.est import EQL
import pandas as pd
from sklearn.model_selection import GridSearchCV


# Base EQL estimator
base = EQL(n_iter=50000)  # Increased iterations for better convergence

# Hyperparameter grid optimized for Lorenz
# Note: EQL implements addition/subtraction through neural network weights
# Available functions: sin, cos, id, mul, div, sqrt, exp, log, square, cube
hp = {
    "reg": (1e-5, 1e-4, 1e-3),  # Regularization strength
    "n_layers": (1, 2),          # Network depth
    "functions": (
        # Simple multiplicative structure for Lorenz
        "id;mul;id;mul",
        # With more layers for complex interactions
        "id;mul;id;mul;id;mul",
    ),
}

# GridSearchCV with reduced CV folds for faster training
est = GridSearchCV(
    estimator=base,
    param_grid=hp,
    cv=2,          # 2-fold cross-validation
    refit=True,
    n_jobs=1       # Single thread for Windows compatibility
)


def model(est, X=None):
    """
    Extract symbolic equation from trained EQL model

    Args:
        est: Trained GridSearchCV estimator
        X: Optional DataFrame with feature names

    Returns:
        String representation of discovered equation
    """
    model_str = str(est.best_estimator_.get_eqn())

    # Replace x0, x1, x2 with actual feature names
    if isinstance(X, pd.DataFrame):
        for i, f in reversed(list(enumerate(X.columns))):
            model_str = model_str.replace(f'x{i}', f)

    return model_str


def complexity(est):
    """
    Calculate equation complexity

    Args:
        est: Trained estimator

    Returns:
        Integer representing equation complexity
    """
    # Get the equation string
    eqn_str = str(est.best_estimator_.get_eqn())
    # Simple complexity: count operators and variables
    return len(eqn_str.split())


# Evaluation configuration - disable normalization (consistent with PySR)
eval_kwargs = {
    'test_params': {'param_grid': {'n_iter': [1000], **hp}},  # Quick test params
    'scale_x': False,  # No feature normalization
    'scale_y': False,  # No target normalization
    'use_dataframe': False
}
