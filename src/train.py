from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import preprocessing
from baselines import inventory_simulation, mean_absolute_error, moving_window_baseline
from models.tcn import SimpleTCN

BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3
SEED = 42


def set_seed(seed=SEED):
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


def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs=EPOCHS):
    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    history = []

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:>2}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

    model.load_state_dict(best_state)
    return history, best_val_loss


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = to_tensor(preprocessing.X_train, add_channel_dim=True)
    y_train = to_tensor(preprocessing.y_train)
    X_val = to_tensor(preprocessing.X_val, add_channel_dim=True)
    y_val = to_tensor(preprocessing.y_val)
    X_test = to_tensor(preprocessing.X_test, add_channel_dim=True)
    y_test = to_tensor(preprocessing.y_test)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("Device:", device)

    train_loader = build_dataloader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(X_val, y_val, batch_size=BATCH_SIZE)
    test_loader = build_dataloader(X_test, y_test, batch_size=BATCH_SIZE)

    model = SimpleTCN(
        input_length=preprocessing.INPUT_WINDOW,
        output_size=preprocessing.TARGET_WINDOW,
        hidden_channels=32,
    ).to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    _, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    test_predictions = predict(model, test_loader, device)
    test_targets = preprocessing.y_test

    baseline_predictions = moving_window_baseline(
        preprocessing.X_test, forecast_horizon=preprocessing.TARGET_WINDOW
    )

    raw_test_mae = mean_absolute_error(test_predictions, test_targets)
    rounded_test_predictions = np.clip(np.rint(test_predictions), 0, None)
    rounded_test_mae = mean_absolute_error(rounded_test_predictions, test_targets)
    baseline_mae = mean_absolute_error(baseline_predictions, test_targets)

    baseline_inventory = inventory_simulation(baseline_predictions, test_targets)
    model_inventory = inventory_simulation(rounded_test_predictions, test_targets)

    print("\nBeste val-loss:", f"{best_val_loss:.4f}")
    print("TCN test MAE (rå):", f"{raw_test_mae:.4f}")
    print("TCN test MAE (avrundet):", f"{rounded_test_mae:.4f}")
    print("Baseline test MAE:", f"{baseline_mae:.4f}")

    print("\nFørste TCN-prediksjon (rå):", np.round(test_predictions[0], 3))
    print("Første TCN-prediksjon (avrundet):", rounded_test_predictions[0].astype(int))
    print("Første baseline-prediksjon:", baseline_predictions[0].astype(int))
    print("Første faktisk target:", test_targets[0].astype(int))

    print("\nInventory metrics på testsett:")
    print(
        "TCN total cost: {:.2f}, stockout rate: {:.3f}, fill rate: {:.3f}".format(
            *model_inventory
        )
    )
    print(
        "Baseline total cost: {:.2f}, stockout rate: {:.3f}, fill rate: {:.3f}".format(
            *baseline_inventory
        )
    )

if __name__ == "__main__":
    main()
