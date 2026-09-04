"""Reusable training functions without experiment orchestration."""

from __future__ import annotations
import copy
import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from .architectures import CorrectnessMLP, MaskedFusionRegressor
from .features import MODALITIES, SUBSETS


def stable_transform(scaler: StandardScaler, values) -> np.ndarray:
    transformed = scaler.transform(values).astype(np.float32)
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=20.0, neginf=-20.0)
    return np.clip(transformed, -20.0, 20.0)


def _linear_predict(model: Ridge, values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    return np.einsum("ij,j->i", values, coefficients, optimize=False) + float(model.intercept_)


@dataclass
class RidgeArtifact:
    subset: str
    scaler: StandardScaler
    model: Ridge
    label_mean: float


def fit_ridge_backbones(
    train_pooled: Mapping[str, np.ndarray],
    evaluation_pooled: Mapping[str, np.ndarray],
    labels,
    groups,
    alpha: float = 10.0,
    folds: int = 5,
    solver: str = "lsqr",
):
    labels = np.asarray(labels, dtype=np.float32)
    oof, evaluation, artifacts = {}, {}, {}
    for subset in SUBSETS:
        x_train = np.concatenate([train_pooled[m] for m in MODALITIES if m in subset], axis=1)
        x_evaluation = np.concatenate([evaluation_pooled[m] for m in MODALITIES if m in subset], axis=1)
        subset_oof = np.empty(len(labels), dtype=np.float32)
        splitter = GroupKFold(n_splits=folds)
        for fit_index, holdout_index in splitter.split(x_train, labels, groups):
            scaler = StandardScaler().fit(x_train[fit_index])
            mean = float(np.mean(labels[fit_index], dtype=np.float64))
            model = Ridge(alpha=alpha, solver=solver, fit_intercept=False).fit(
                stable_transform(scaler, x_train[fit_index]).astype(np.float64),
                labels[fit_index].astype(np.float64) - mean,
            )
            subset_oof[holdout_index] = (
                _linear_predict(model, stable_transform(scaler, x_train[holdout_index])) + mean
            ).astype(np.float32)
        scaler = StandardScaler().fit(x_train)
        mean = float(np.mean(labels, dtype=np.float64))
        model = Ridge(alpha=alpha, solver=solver, fit_intercept=False).fit(
            stable_transform(scaler, x_train).astype(np.float64),
            labels.astype(np.float64) - mean,
        )
        oof[subset] = subset_oof
        evaluation[subset] = (
            _linear_predict(model, stable_transform(scaler, x_evaluation)) + mean
        ).astype(np.float32)
        artifacts[subset] = RidgeArtifact(subset, scaler, model, mean)
    return oof, evaluation, artifacts


def train_correctness_model(
    features,
    targets,
    metadata,
    seed: int,
    hidden: Sequence[int] = (64, 32),
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 80,
    batch_size: int = 256,
    patience: int = 10,
):
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    calibration_groups = {
        item[-1]
        for item in metadata
        if int(hashlib.sha256(str(item[-1]).encode()).hexdigest()[:8], 16) % 5 == 0
    }
    train_mask = np.asarray([item[-1] not in calibration_groups for item in metadata])
    calibration_mask = ~train_mask
    if not train_mask.any() or not calibration_mask.any():
        raise ValueError("correctness training and calibration groups must be non-empty")
    scaler = StandardScaler().fit(features[train_mask])
    x_train = torch.from_numpy(stable_transform(scaler, features[train_mask]))
    y_train = torch.from_numpy(targets[train_mask])
    x_calibration = torch.from_numpy(stable_transform(scaler, features[calibration_mask]))
    y_calibration = torch.from_numpy(targets[calibration_mask])
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = CorrectnessMLP(features.shape[1], hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    best_state, best_loss, stale, best_epoch = None, float("inf"), 0, -1
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(len(x_train))
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            calibration_loss = float(loss_function(model(x_calibration), y_calibration).item())
        if calibration_loss < best_loss - 1e-6:
            best_loss = calibration_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("correctness model did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, {"best_epoch": best_epoch, "calibration_loss": best_loss}


def fit_modality_scalers(pooled: Mapping[str, np.ndarray], indices):
    result = {}
    for modality, values in pooled.items():
        sample = values[indices].astype(np.float64)
        mean, scale = sample.mean(axis=0), sample.std(axis=0)
        scale[scale < 1e-8] = 1.0
        result[modality] = (mean.astype(np.float32), scale.astype(np.float32))
    return result


def train_masked_fusion(
    pooled: Mapping[str, np.ndarray],
    labels,
    indices,
    config: Mapping[str, int | float],
    seed: int = 20260903,
    subsets: Sequence[str] = SUBSETS,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    scalers = fit_modality_scalers(pooled, indices)
    modalities = []
    for modality in MODALITIES:
        mean, scale = scalers[modality]
        values = (pooled[modality][indices] - mean) / scale
        modalities.append(torch.from_numpy(
            np.clip(np.nan_to_num(values), -20.0, 20.0).astype(np.float32)
        ))
    labels_tensor = torch.from_numpy(np.asarray(labels)[indices].astype(np.float32))
    model = MaskedFusionRegressor([value.shape[1] for value in modalities], config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 5e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    masks = torch.tensor(
        [[float(m in subset) for m in MODALITIES] for subset in subsets],
        dtype=torch.float32,
    )
    rng = np.random.default_rng(seed)
    batch_size = int(config.get("batch_size", 64))
    final_loss = float("nan")
    model.train()
    for _ in range(int(config.get("epochs", 30))):
        order = rng.permutation(len(indices))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            size = len(batch)
            expanded = [
                value[batch].unsqueeze(1).expand(size, len(subsets), value.shape[1]).reshape(
                    size * len(subsets), value.shape[1]
                )
                for value in modalities
            ]
            batch_masks = masks.unsqueeze(0).expand(size, len(subsets), 3).reshape(-1, 3)
            targets = labels_tensor[batch].unsqueeze(1).expand(size, len(subsets)).reshape(-1)
            optimizer.zero_grad()
            loss = torch.nn.functional.smooth_l1_loss(model(expanded, batch_masks), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        final_loss = float(np.mean(losses))
    model.eval()
    return model, scalers, final_loss
