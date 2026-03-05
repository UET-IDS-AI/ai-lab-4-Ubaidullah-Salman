"""
AI_stats_lab.py

Autograded lab: Gradient Descent + Linear Regression (Diabetes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


# =========================
# Helpers
# =========================

def add_bias_column(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return np.hstack([np.ones((X.shape[0], 1)), X])


def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)

    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


@dataclass
class GDResult:
    theta: np.ndarray
    losses: np.ndarray
    thetas: np.ndarray


# =========================
# Q1 Gradient Descent
# =========================

def gradient_descent_linreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    theta0: Optional[np.ndarray] = None,
) -> GDResult:

    n, d = X.shape

    if theta0 is None:
        theta = np.zeros(d)
    else:
        theta = theta0.copy()

    losses = []
    theta_path = []

    for _ in range(epochs):

        y_pred = X @ theta

        loss = mse(y, y_pred)
        losses.append(loss)

        theta_path.append(theta.copy())

        gradient = (2 / n) * (X.T @ (y_pred - y))

        theta = theta - lr * gradient

    return GDResult(
        theta=theta,
        losses=np.array(losses),
        thetas=np.array(theta_path),
    )


# =========================
# Visualization data
# =========================

def visualize_gradient_descent(
    lr: float = 0.1,
    epochs: int = 60,
    seed: int = 0,
) -> Dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)

    n = 50

    X = rng.normal(size=(n, 1))
    X = add_bias_column(X)

    true_theta = np.array([2.0, 3.0])

    noise = rng.normal(scale=0.5, size=n)

    y = X @ true_theta + noise

    gd = gradient_descent_linreg(X, y, lr=lr, epochs=epochs)

    return {
        "theta_path": gd.thetas,
        "losses": gd.losses,
        "X": X,
        "y": y,
    }


# =========================
# Q2 Diabetes using GD
# =========================

def diabetes_linear_gd(
    lr: float = 0.05,
    epochs: int = 2000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:

    data = load_diabetes()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)

    gd = gradient_descent_linreg(X_train, y_train, lr=lr, epochs=epochs)

    theta = gd.theta

    train_pred = X_train @ theta
    test_pred = X_test @ theta

    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q3 Analytical solution
# =========================

def diabetes_linear_analytical(
    ridge_lambda: float = 1e-8,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:

    data = load_diabetes()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)

    d = X_train.shape[1]

    I = np.eye(d)

    theta = np.linalg.inv(
        X_train.T @ X_train + ridge_lambda * I
    ) @ (X_train.T @ y_train)

    train_pred = X_train @ theta
    test_pred = X_test @ theta

    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q4 Comparison
# =========================

def diabetes_compare_gd_vs_analytical(
    lr: float = 0.05,
    epochs: int = 4000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Dict[str, float]:

    train_mse_gd, test_mse_gd, train_r2_gd, test_r2_gd, theta_gd = diabetes_linear_gd(
        lr=lr, epochs=epochs, test_size=test_size, seed=seed
    )

    train_mse_an, test_mse_an, train_r2_an, test_r2_an, theta_an = diabetes_linear_analytical(
        test_size=test_size, seed=seed
    )

    theta_l2_diff = np.linalg.norm(theta_gd - theta_an)

    train_mse_diff = abs(train_mse_gd - train_mse_an)
    test_mse_diff = abs(test_mse_gd - test_mse_an)

    train_r2_diff = abs(train_r2_gd - train_r2_an)
    test_r2_diff = abs(test_r2_gd - test_r2_an)

    theta_cosine_sim = np.dot(theta_gd, theta_an) / (
        np.linalg.norm(theta_gd) * np.linalg.norm(theta_an)
    )

    return {
        "theta_l2_diff": float(theta_l2_diff),
        "train_mse_diff": float(train_mse_diff),
        "test_mse_diff": float(test_mse_diff),
        "train_r2_diff": float(train_r2_diff),
        "test_r2_diff": float(test_r2_diff),
        "theta_cosine_sim": float(theta_cosine_sim),
    }
