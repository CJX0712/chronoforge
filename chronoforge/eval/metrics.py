"""预测误差指标。接口语义统一：分数越大越差（sMAPE/RMSE/MAE/MASE 越小越好）。"""

from __future__ import annotations

import numpy as np


def smape(forecast: np.ndarray, actual: np.ndarray) -> float:
    """对称 MAPE（百分比）。0 表示完美。"""
    f = np.asarray(forecast, dtype=float).ravel()
    a = np.asarray(actual, dtype=float).ravel()
    denom = np.abs(a) + np.abs(f)
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(200.0 * np.abs(f[mask] - a[mask]) / denom[mask]))


def mase(forecast: np.ndarray, actual: np.ndarray, train_scale: float) -> float:
    """MASE：相对 in-sample 朴素预测（1 步）的 MAE 缩放。"""
    f = np.asarray(forecast, dtype=float).ravel()
    a = np.asarray(actual, dtype=float).ravel()
    if train_scale <= 0:
        return float("nan")
    mae = float(np.mean(np.abs(f - a)))
    return mae / train_scale


def rmse(forecast: np.ndarray, actual: np.ndarray) -> float:
    f = np.asarray(forecast, dtype=float).ravel()
    a = np.asarray(actual, dtype=float).ravel()
    return float(np.sqrt(np.mean((f - a) ** 2)))


def mae(forecast: np.ndarray, actual: np.ndarray) -> float:
    f = np.asarray(forecast, dtype=float).ravel()
    a = np.asarray(actual, dtype=float).ravel()
    return float(np.mean(np.abs(f - a)))


def naive_scale(train: np.ndarray) -> float:
    """in-sample 1 步朴素预测 MAE，用作 MASE 缩放分母。"""
    t = np.asarray(train, dtype=float).ravel()
    if len(t) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(t))))
