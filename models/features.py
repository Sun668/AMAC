"""Feature construction used by the released AMAC checkpoints."""

from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np

MODALITIES = ("T", "A", "V")
SUBSETS = ("T", "A", "V", "TA", "TV", "AV", "TAV")


def canonical_subset(chars: str) -> str:
    present = set(chars)
    return "".join(modality for modality in MODALITIES if modality in present)


def affect_state(values, boundaries=(-0.1, 0.1)) -> np.ndarray:
    values = np.asarray(values)
    return np.where(
        values < boundaries[0], -1, np.where(values > boundaries[1], 1, 0)
    ).astype(np.int8)


def pool_modality(array) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(values, -10000.0, 10000.0, out=values)
    return np.concatenate((values.mean(axis=1), values.std(axis=1)), axis=1).astype(np.float32)


def modality_quality(pooled: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for modality, values in pooled.items():
        half = values.shape[1] // 2
        result[modality] = np.column_stack((
            np.linalg.norm(values[:, :half], axis=1) / max(1.0, np.sqrt(half)),
            np.mean(np.abs(values[:, half:]), axis=1),
        )).astype(np.float32)
    return result


def _one_hot_state(state: int) -> list[float]:
    return [float(state == -1), float(state == 0), float(state == 1)]


def chsimsv2_event_features(
    index: int,
    order: str,
    stage: int,
    predictions: Mapping[str, np.ndarray],
    qualities: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Build one 25-dimensional CH-SIMS v2 prefix feature vector."""

    subsets = [canonical_subset(order[:position]) for position in (1, 2, 3)]
    history = [float(predictions[subset][index]) for subset in subsets[:stage + 1]]
    current = history[-1]
    previous = history[-2] if len(history) > 1 else 0.0
    current_state = int(affect_state([current])[0])
    previous_state = int(affect_state([previous])[0]) if len(history) > 1 else 99
    visible = set(subsets[stage])
    vector = [current, abs(current), previous, current - previous if len(history) > 1 else 0.0]
    vector.extend([float(np.mean(history)), float(np.std(history)), min(history), max(history)])
    vector.append(stage / 2.0)
    vector.extend(float(modality in visible) for modality in MODALITIES)
    vector.extend(_one_hot_state(current_state))
    vector.extend(_one_hot_state(previous_state) + [float(previous_state == 99)])
    for modality in MODALITIES:
        vector.extend(qualities[modality][index].tolist() if modality in visible else [0.0, 0.0])
    return np.asarray(vector, dtype=np.float32)


def apply_frozen_feature_ablation(
    features,
    history_feature_indices: Sequence[int],
    quality_feature_start: int = 19,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32).copy()
    values[..., quality_feature_start:] = 0.0
    values[..., list(history_feature_indices)] = 0.0
    return values


def emotiontalk_prefix_features(
    sensors: Mapping[str, Mapping[str, object]],
    visible: Sequence[str],
    classes: Sequence[str],
) -> tuple[int, float, np.ndarray]:
    votes = np.zeros(len(classes), dtype=np.float32)
    certainty: list[float] = []
    labels: list[str] = []
    for modality in visible:
        sensor = sensors[modality]
        label = str(sensor["label"])
        weight = float(sensor["agreement"]) * float(sensor["confidence"])
        votes[classes.index(label)] += weight
        certainty.append(weight)
        labels.append(label)
    order = np.argsort(votes)
    top = int(order[-1])
    second = float(votes[order[-2]])
    confidence = float(votes[top] / len(visible))
    margin = float((votes[top] - second) / len(visible))
    consensus = max(labels.count(value) for value in set(labels)) / len(labels)
    features = np.asarray(
        [confidence, margin, float(np.mean(certainty)), consensus, len(visible) / 3.0]
        + [float(modality in visible) for modality in MODALITIES]
        + [float(index == top) for index in range(len(classes))],
        dtype=np.float32,
    )
    return top, confidence, features
