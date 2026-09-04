"""Classifier and flow training used by the core experiment."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def class_stratified_validation_split(
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint fit/validation indices with every exact class represented."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    labels = np.asarray(labels, dtype=np.int64)
    generator = np.random.default_rng(seed)
    fit_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < 2:
            raise ValueError(f"Class {class_id} needs at least two training examples.")
        indices = generator.permutation(indices)
        validation_count = min(
            max(int(round(len(indices) * validation_fraction)), 1),
            len(indices) - 1,
        )
        validation_parts.append(indices[:validation_count])
        fit_parts.append(indices[validation_count:])
    fit = np.sort(np.concatenate(fit_parts)).astype(np.int64, copy=False)
    validation = np.sort(np.concatenate(validation_parts)).astype(np.int64, copy=False)
    return fit, validation


def train_linear_head(
    *,
    backbone,
    head,
    dataset,
    output_classes: list[int],
    loss_start: int,
    training: dict[str, Any],
    runtime: dict[str, Any],
    stage_index: int,
    joint: bool,
    device,
    seed: int,
) -> list[dict[str, Any]]:
    """Train either the sequential classifier or offline joint reference head."""

    import torch
    from torch.nn import functional
    from torch.utils.data import DataLoader

    initial = stage_index == 0 or joint
    epochs = int(training["epochs_first"] if initial else training["epochs_later"])
    learning_rate = float(
        training["learning_rate_first"] if initial else training["learning_rate_later"]
    )
    weight_decay = float(
        training["weight_decay_first"] if initial else training["weight_decay_later"]
    )
    milestones = list(
        training["milestones_first"] if initial else training["milestones_later"]
    )
    decay = float(training["lr_decay_first"] if initial else training["lr_decay_later"])
    generator = torch.Generator().manual_seed(
        seed * 10_000 + stage_index + int(joint) * 1000
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(runtime.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    optimizer = torch.optim.SGD(
        head.parameters(),
        lr=learning_rate,
        momentum=float(training.get("momentum", 0.9)),
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=decay,
    )
    class_to_output = {class_id: index for index, class_id in enumerate(output_classes)}
    rows: list[dict[str, Any]] = []
    backbone.eval()
    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for _, images, original_targets in loader:
            images = images.to(device, non_blocking=True)
            targets = torch.tensor(
                [class_to_output[int(value)] for value in original_targets],
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad():
                features = backbone(images)
            logits = head(features)
            local_targets = targets - loss_start
            loss = functional.cross_entropy(logits[:, loss_start:], local_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            total += len(targets)
            correct += int((logits.argmax(1) == targets).sum())
        scheduler.step()
        rows.append(
            {
                "stage_index": stage_index,
                "method": "offline_joint_linear_head"
                if joint
                else "sequential_linear_probe",
                "epoch": epoch + 1,
                "loss": total_loss / total,
                "global_train_accuracy": correct / total,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "weight_decay": weight_decay,
            }
        )
    return rows


def classifier_logits(
    head,
    features: np.ndarray,
    device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Evaluate a linear head over cached features in deterministic batches."""

    import torch

    outputs = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            values = torch.from_numpy(features[start : start + batch_size]).to(device)
            outputs.append(head(values).float().cpu().numpy())
    return np.concatenate(outputs)


def train_flow(
    flow,
    fit_values: np.ndarray,
    validation_values: np.ndarray,
    config: dict[str, Any],
    device,
    seed: int,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Fit one task flow, restore its best validation checkpoint, and report NLLs."""

    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset

    flow.to(device)
    fit_tensor = torch.from_numpy(np.asarray(fit_values, dtype=np.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(fit_tensor),
        batch_size=int(config.get("batch_size", 512)),
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    optimizer = AdamW(
        flow.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    epoch_limit = int(max_epochs or config.get("max_epochs", 100))
    patience = int(config.get("patience", 10))
    min_delta = float(config.get("min_delta", 1e-3))
    gradient_clip = float(config.get("gradient_clip_norm", 5.0))
    if epoch_limit < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive.")

    best_validation_nll = math.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epoch_limit + 1):
        flow.train()
        loss_sum = 0.0
        example_count = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            loss = -flow.log_prob(batch).mean()
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite flow loss at epoch {epoch}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(flow.parameters(), gradient_clip)
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            example_count += len(batch)
        online_train_nll = loss_sum / max(example_count, 1)
        validation_log_probs = flow_log_probabilities(
            flow,
            validation_values,
            int(config.get("eval_batch_size", 2048)),
            device,
        )
        validation_nll = float(-validation_log_probs.mean())
        improved = validation_nll < best_validation_nll - min_delta
        if improved:
            best_validation_nll = validation_nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in flow.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "online_train_nll": online_train_nll,
                "validation_nll": validation_nll,
                "generalization_gap": validation_nll - online_train_nll,
                "is_best": improved,
            }
        )
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            "Flow training never produced a finite validation checkpoint."
        )
    flow.load_state_dict(best_state)
    restored_train_log_probs = flow_log_probabilities(
        flow,
        fit_values,
        int(config.get("eval_batch_size", 2048)),
        device,
    )
    restored_validation_log_probs = flow_log_probabilities(
        flow,
        validation_values,
        int(config.get("eval_batch_size", 2048)),
        device,
    )
    restored_train_nll = float(-restored_train_log_probs.mean())
    restored_validation_nll = float(-restored_validation_log_probs.mean())
    return {
        "history": history,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_validation_nll": best_validation_nll,
        "restored_train_nll": restored_train_nll,
        "restored_validation_nll": restored_validation_nll,
        "generalization_gap": restored_validation_nll - restored_train_nll,
        "validation_log_prob_mean": float(restored_validation_log_probs.mean()),
        "validation_log_prob_std": float(restored_validation_log_probs.std()),
    }


def flow_log_probabilities(
    flow,
    values: np.ndarray,
    batch_size: int,
    device,
) -> np.ndarray:
    """Evaluate flow log densities in batches."""

    import torch

    flow.to(device)
    flow.eval()
    output = np.empty(len(values), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            batch = torch.from_numpy(
                np.asarray(values[start:stop], dtype=np.float32)
            ).to(device)
            output[start:stop] = flow.log_prob(batch).float().cpu().numpy()
    return output
