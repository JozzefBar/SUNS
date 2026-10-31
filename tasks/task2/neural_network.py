import copy

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import PLOTS_DIR, add_title, prepare_data, split_data
from models import evaluate, plot_residuals

# (skryte vrstvy, rychlost ucenia), prve je zakladne nastavenie, ostatne menia jednu vec
EXPERIMENTS = [((64, 32), 0.001), ((32,), 0.001), ((128, 64), 0.001), ((64, 32), 0.01)]
PATIENCE = 20


class NetworkModel:
    # obal okolo siete, aby sa dala vyhodnotit rovnakymi funkciami ako modely zo sklearn
    def __init__(self, network):
        self.network = network

    def predict(self, X):
        self.network.eval()
        with torch.no_grad():
            return self.network(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()


def build_network(input_size, hidden_layers):
    layers = []
    size = input_size
    for neurons in hidden_layers:
        layers += [nn.Linear(size, neurons), nn.ReLU()]
        size = neurons
    layers.append(nn.Linear(size, 1))   # linearny vystup, siet predpoveda priamo koncentraciu
    return nn.Sequential(*layers)


def train_network(X_train, y_train, X_val, y_val, hidden_layers, lr, epochs=500, batch_size=64):
    torch.manual_seed(0)
    X_train, X_val = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32)
    network = build_network(X_train.shape[1], hidden_layers)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()   # regresia: stredna kvadraticka chyba namiesto cross-entropy
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    history = []
    best_val_loss, best_weights, best_epoch = float("inf"), None, None
    for epoch in range(1, epochs + 1):
        network.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(network(X_batch).squeeze(1), y_batch)
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            train_loss = loss_fn(network(X_train).squeeze(1), y_train).item()
            val_loss = loss_fn(network(X_val).squeeze(1), y_val).item()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss, best_weights, best_epoch = val_loss, copy.deepcopy(network.state_dict()), epoch
        elif epoch - best_epoch >= PATIENCE:
            break   # EarlyStopping: validacna chyba sa 20 epoch nezlepsila

    network.load_state_dict(best_weights)   # vratim sa k vaham z najlepsej epochy
    return NetworkModel(network), pd.DataFrame(history), best_epoch


def describe(hidden_layers, lr):
    return f"skryté vrstvy {', '.join(str(neurons) for neurons in hidden_layers)}, rýchlosť učenia {lr:g}"


def plot_history(history, best_epoch, settings):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(history["epoch"], history["train_loss"], label="trénovacia")
    ax.plot(history["epoch"], history["val_loss"], label="validačná")
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"najlepšia epocha ({best_epoch})")
    ax.set_xlabel("epocha")
    ax.set_ylabel("MSE (mg/m³)²")
    ax.set_ylim(0, history["val_loss"].iloc[2:].max() * 1.1)
    ax.legend()
    add_title(fig, "Priebeh trénovania najlepšej neurónovej siete",
              f"{settings}, Adam, EarlyStopping (patience {PATIENCE})")
    plt.savefig(PLOTS_DIR / "nn_history.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    X_train_raw, X_test_raw, y_train, y_test = split_data(prepare_data())
    # validacnu mnozinu pre EarlyStopping odrezem z trenovacej (15 %), testovacia ostava rovnaka ako pri ostatnych modeloch
    X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(X_train_raw, y_train, test_size=0.15, random_state=42)
    scaler = StandardScaler().fit(X_fit_raw)
    X_fit, X_val, X_test = (scaler.transform(X) for X in (X_fit_raw, X_val_raw, X_test_raw))
    print("train:", X_fit.shape, "val:", X_val.shape, "test:", X_test.shape)

    rows, runs = [], []
    for hidden_layers, lr in EXPERIMENTS:
        model, history, best_epoch = train_network(X_fit, y_fit, X_val, y_val, hidden_layers, lr)
        settings = describe(hidden_layers, lr)
        rows.append({"nastavenie": settings, "epoch": len(history), "najlepsia epocha": best_epoch,
                     "MSE val": history["val_loss"].min(), **evaluate(model, X_fit, X_test, y_fit, y_test)})
        runs.append((model, history, best_epoch, settings))
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))

    # najlepsia siet podla validacnej chyby, testovacia mnozina sa do vyberu nemiesa
    model, history, best_epoch, settings = runs[table["MSE val"].idxmin()]
    print("najlepsia:", settings)
    residuals = y_test - model.predict(X_test)
    print(f"test rezidualov v rozsahu +-0.5: {(residuals.abs() <= 0.5).mean():.1%}")
    plot_history(history, best_epoch, settings)
    plot_residuals(model, X_fit, X_test, y_fit, y_test, "residuals_nn.png", "Reziduály: neurónová sieť",
                   f"{settings}, najlepšia epocha {best_epoch}")
