from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(array, add_channel_dim=False):
    tensor = torch.tensor(np.asarray(array, dtype=np.float32))
    if add_channel_dim:
        tensor = tensor.unsqueeze(1)
    return tensor


def build_dataloader(X, y, batch_size, shuffle=False):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(model, data_loader, loss_fn, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            losses.append(loss_fn(predictions, y_batch).item())

    return float(np.mean(losses))


def predict(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            batch_predictions = model(X_batch).cpu().numpy()
            predictions.append(batch_predictions)

    return np.vstack(predictions)


def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    device,
    epochs,
    patience=None,
    min_delta=0.0,
):
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    history = []
    epochs_without_improvement = 0
    stopped_early = False
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            "Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}".format(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )
        )

        if patience is not None and epochs_without_improvement >= patience:
            stopped_early = True
            stopped_epoch = epoch
            print(
                "Early stopping: ingen forbedring i val_loss de siste "
                f"{patience} epochene."
            )
            break

    model.load_state_dict(best_state)
    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "stopped_epoch": stopped_epoch,
    }
