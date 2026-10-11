import copy

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import ConfusionMatrixDisplay

from preprocessing import PLOTS_DIR, prepare_xy, split_data, scale


def load_tensors():
    # rovnaka priprava dat ako pri jednoduchom modeli, rozdelenie 70 / 15 / 15
    X, y = prepare_xy()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, 0.15, 0.15, seed=42)
    X_train, X_val, X_test = scale(X_train, X_val, X_test)
    data = {}
    for name, X_part, y_part in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        data[name] = (torch.tensor(X_part, dtype=torch.float32),
                      torch.tensor(y_part.values, dtype=torch.float32))
    return data


def build_model(input_size, hidden_layers, dropout=0.0):
    # hidden_layers = pocty neuronov v skrytych vrstvach, napr. (64, 32)
    layers = []
    size = input_size
    for neurons in hidden_layers:
        layers.append(nn.Linear(size, neurons))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        size = neurons
    layers.append(nn.Linear(size, 1))   # jeden vystup, sigmoid je sucastou loss funkcie
    return nn.Sequential(*layers)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return (model(X).squeeze(1) > 0).float()   # vystup > 0 znamena pravdepodobnost > 0.5


def evaluate(model, X, y, loss_fn):
    model.eval()
    with torch.no_grad():
        loss = loss_fn(model(X).squeeze(1), y).item()
    accuracy = (predict(model, X) == y).float().mean().item() * 100
    return loss, accuracy


def train_network(data, hidden_layers, lr=0.001, epochs=200, batch_size=64, dropout=0.0, patience=None):
    # patience = kolko epoch cakam na zlepsenie validacnej chyby, None = bez EarlyStopping
    torch.manual_seed(0)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    model = build_model(X_train.shape[1], hidden_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    history = []
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = None
    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch).squeeze(1), y_batch)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, X_train, y_train, loss_fn)
        val_loss, val_acc = evaluate(model, X_val, y_val, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_acc": train_acc, "val_acc": val_acc})

        if patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            elif epoch - best_epoch >= patience:
                break   # validacna chyba sa uz dlho nezlepsila

    if patience is not None:
        model.load_state_dict(best_weights)   # vratim sa k vaham z najlepsej epochy
    return model, pd.DataFrame(history), best_epoch


def accuracies(model, data):
    result = {}
    for name in ["train", "test"]:
        X, y = data[name]
        result[name] = (predict(model, X) == y).float().mean().item() * 100
    return result


def plot_history(history, filename, title, best_epoch=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="trénovacia")
    axes[0].plot(history["epoch"], history["val_loss"], label="validačná")
    axes[0].set_ylabel("chyba (loss)")
    axes[1].plot(history["epoch"], history["train_acc"], label="trénovacia")
    axes[1].plot(history["epoch"], history["val_acc"], label="validačná")
    axes[1].set_ylabel("úspešnosť (%)")
    for ax in axes:
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", linestyle="--", label="najlepšia epocha")
        ax.set_xlabel("epocha")
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def plot_confusion(model, data, filename, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name, label in [(axes[0], "train", "trénovacia"), (axes[1], "test", "testovacia")]:
        X, y = data[name]
        predictions = predict(model, X)
        ConfusionMatrixDisplay.from_predictions(
            y.numpy(), predictions.numpy(), display_labels=["bez nákupu", "nákup"],
            cmap="Blues", colorbar=False, ax=ax)
        accuracy = (predictions == y).float().mean().item() * 100
        ax.set_title(f"{label} množina, úspešnosť {accuracy:.2f} %")
        ax.set_xlabel("predikcia")
        ax.set_ylabel("skutočnosť")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


if __name__ == "__main__":
    data = load_tensors()
    print("train:", data["train"][0].shape, "val:", data["val"][0].shape, "test:", data["test"][0].shape)

    print("\n--- pretrenovanie ---")
    big_network = (512, 512, 256)
    model, history, _ = train_network(data, big_network, lr=0.001, epochs=200)
    print(accuracies(model, data))
    print(history.iloc[[0, 4, 9, 19, 49, 99, 199]].round(3).to_string(index=False))
    plot_history(history, "nn_overfit_history.png", "Pretrénovaná sieť, 200 epoch bez EarlyStopping")
    plot_confusion(model, data, "nn_overfit_confusion.png", "Pretrénovaná sieť")

    print("\n--- ta ista siet s EarlyStopping ---")
    model, history, best_epoch = train_network(data, big_network, lr=0.001, epochs=200, patience=10)
    print(accuracies(model, data), "najlepsia epocha:", best_epoch, "zastavene po:", len(history))
    plot_history(history, "nn_early_history.png", "Tá istá sieť s EarlyStopping (patience 10)", best_epoch)
    plot_confusion(model, data, "nn_early_confusion.png", "Sieť s EarlyStopping")

    print("\n--- experimenty ---")
    # zaklad: skryte vrstvy (64, 32), lr 0.001, bez dropoutu, vsetko s EarlyStopping
    experiments = [
        {"hidden_layers": (64, 32), "lr": 0.001, "dropout": 0.0},
        {"hidden_layers": (64, 32), "lr": 0.01, "dropout": 0.0},
        {"hidden_layers": (64, 32), "lr": 0.1, "dropout": 0.0},
        {"hidden_layers": (64, 32), "lr": 0.0001, "dropout": 0.0},
        {"hidden_layers": (16,), "lr": 0.001, "dropout": 0.0},
        {"hidden_layers": (256, 128, 64), "lr": 0.001, "dropout": 0.0},
        {"hidden_layers": (64, 32), "lr": 0.001, "dropout": 0.3},
        {"hidden_layers": (256, 128, 64), "lr": 0.001, "dropout": 0.3},
    ]
    runs = []
    for number, settings in enumerate(experiments, start=1):
        model, history, best_epoch = train_network(data, patience=10, **settings)
        result = accuracies(model, data)
        runs.append({"model": model, "history": history, "best_epoch": best_epoch, "number": number})
        settings.update({"experiment": number, "epocha": best_epoch,
                         "train (%)": result["train"], "test (%)": result["test"]})
    table = pd.DataFrame(experiments).set_index("experiment")
    print(table.to_string(float_format=lambda value: f"{value:g}" if value < 1 else f"{value:.2f}"))

    for kind, label, index in [("best", "Najlepší", table["test (%)"].idxmax()),
                               ("worst", "Najhorší", table["test (%)"].idxmin())]:
        run = runs[index - 1]
        title = f"{label} experiment (č. {index})"
        plot_history(run["history"], f"nn_{kind}_history.png", title, run["best_epoch"])
        plot_confusion(run["model"], data, f"nn_{kind}_confusion.png", title)
        print(kind, "experiment:", index)
