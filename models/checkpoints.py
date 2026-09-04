"""Explicit loaders for the released PyTorch checkpoints."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import torch
from .architectures import CorrectnessMLP, MaskedFusionRegressor

MODALITIES = ("T", "A", "V")


def _load(path: str | Path):
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


@dataclass
class CorrectnessMember:
    seed: int
    model: CorrectnessMLP
    mean: np.ndarray
    scale: np.ndarray


@dataclass
class CorrectnessEnsemble:
    members: list[CorrectnessMember]
    used_output: int = 0


@dataclass
class MaskedFusionCheckpoint:
    model: MaskedFusionRegressor
    scalers: Mapping[str, tuple[np.ndarray, np.ndarray]]
    correctness: CorrectnessEnsemble


def _correctness_from_mapping(mapping, architecture=None) -> CorrectnessEnsemble:
    architecture = architecture or {}
    hidden = tuple(architecture.get("hidden", (64, 32)))
    dropout = float(architecture.get("dropout", 0.1))
    used_output = int(architecture.get("used_output", 0))
    members = []
    for seed_text, item in sorted(mapping.items(), key=lambda pair: int(pair[0])):
        mean = np.asarray(item["scaler_mean"], dtype=np.float32)
        scale = np.asarray(item["scaler_scale"], dtype=np.float32)
        model = CorrectnessMLP(len(mean), hidden, dropout)
        model.load_state_dict(item["state_dict"])
        model.eval()
        members.append(CorrectnessMember(int(seed_text), model, mean, scale))
    return CorrectnessEnsemble(members, used_output)


def load_correctness_ensemble(path: str | Path) -> CorrectnessEnsemble:
    payload = _load(path)
    mapping = payload.get("models") or payload.get("risk_models")
    if not mapping:
        raise ValueError("checkpoint does not contain correctness models")
    return _correctness_from_mapping(mapping, payload.get("architecture"))


@torch.inference_mode()
def predict_correctness(ensemble: CorrectnessEnsemble, features, aggregate: bool = True) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    predictions = []
    for member in ensemble.members:
        transformed = (values - member.mean) / member.scale
        transformed = np.clip(
            np.nan_to_num(transformed, nan=0.0, posinf=20.0, neginf=-20.0),
            -20.0,
            20.0,
        )
        logits = member.model(torch.from_numpy(transformed.astype(np.float32)))
        predictions.append(torch.sigmoid(logits)[:, ensemble.used_output].numpy())
    stacked = np.stack(predictions, axis=0)
    return stacked.mean(axis=0) if aggregate else stacked


def _infer_masked_config(state_dict):
    dims = [int(state_dict[f"projections.{index}.0.weight"].shape[1]) for index in range(3)]
    width = int(state_dict["modality_embedding"].shape[1])
    layers = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("transformer.layers.")
    }
    config = {
        "token_dim": width,
        "attention_heads": 4,
        "feedforward_dim": int(state_dict["transformer.layers.0.linear1.weight"].shape[0]),
        "dropout": 0.1,
        "transformer_layers": max(layers) + 1,
        "fusion_hidden": int(state_dict["head.0.weight"].shape[0]),
    }
    return dims, config


def load_masked_fusion_checkpoint(path: str | Path) -> MaskedFusionCheckpoint:
    payload = _load(path)
    backbone = payload["backbone"]
    state_dict = backbone["full_state_dict"]
    dims, config = _infer_masked_config(state_dict)
    model = MaskedFusionRegressor(dims, config)
    model.load_state_dict(state_dict)
    model.eval()
    scalers = {
        modality: (
            np.asarray(backbone["full_scalers"][modality][0], dtype=np.float32),
            np.asarray(backbone["full_scalers"][modality][1], dtype=np.float32),
        )
        for modality in MODALITIES
    }
    return MaskedFusionCheckpoint(
        model, scalers, _correctness_from_mapping(payload["risk_models"])
    )


@torch.inference_mode()
def predict_masked_fusion(
    checkpoint: MaskedFusionCheckpoint,
    pooled: Mapping[str, Sequence[float]],
    visible: Sequence[str],
) -> np.ndarray:
    mask = torch.tensor(
        [[float(modality in visible) for modality in MODALITIES]], dtype=torch.float32
    )
    tensors = []
    for modality in MODALITIES:
        mean, scale = checkpoint.scalers[modality]
        values = np.asarray(pooled.get(modality, np.zeros_like(mean)), dtype=np.float32)
        transformed = (values - mean) / scale
        transformed = np.clip(
            np.nan_to_num(transformed, nan=0.0, posinf=20.0, neginf=-20.0),
            -20.0,
            20.0,
        )
        tensors.append(torch.from_numpy(transformed[None, :].astype(np.float32)))
    return checkpoint.model(tensors, mask).numpy()
