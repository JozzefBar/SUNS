from itertools import product

import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import PLOTS_DIR, add_title
from neural_network import load_tensors, train_network, predict

# vsetky kombinacie tychto hodnot: 3 * 3 * 2 = 18 sieti
GRID = {
    "hidden_layers": [(16,), (64, 32), (256, 128)],
    "lr": [0.1, 0.01, 0.001],
    "dropout": [0.0, 0.3],
}

if __name__ == "__main__":
    data = load_tensors()
    results = []
    for hidden_layers, lr, dropout in product(*GRID.values()):
        model, history, best_epoch = train_network(data, hidden_layers, lr=lr, dropout=dropout, patience=10)
        row = {"skryté vrstvy": ", ".join(str(n) for n in hidden_layers), "lr": lr, "dropout": dropout, "epocha": best_epoch}
        for name in ["train", "val", "test"]:
            X, y = data[name]
            row[f"{name} (%)"] = (predict(model, X) == y).float().mean().item() * 100
        row["rozdiel train-test"] = row["train (%)"] - row["test (%)"]
        results.append(row)

    # najlepsiu kombinaciu vyberam podla validacnej mnoziny, testovacia sluzi len na konecne vyhodnotenie
    table = pd.DataFrame(results).sort_values("val (%)", ascending=False).reset_index(drop=True)
    print(table.to_string(float_format=lambda value: f"{value:.2f}",
                          formatters={"lr": "{:g}".format, "dropout": "{:g}".format}))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, dropout in zip(axes, GRID["dropout"]):
        part = table[table["dropout"] == dropout]
        pivot = part.pivot(index="skryté vrstvy", columns="lr", values="test (%)")
        pivot = pivot.loc[[", ".join(str(n) for n in layers) for layers in GRID["hidden_layers"]]]
        image = ax.imshow(pivot.values, cmap="Blues", vmin=table["test (%)"].min(), vmax=table["test (%)"].max())
        ax.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_xlabel("rýchlosť učenia")
        ax.set_ylabel("skryté vrstvy")
        ax.set_title(f"dropout {dropout:g}")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        color="white" if value > table["test (%)"].mean() else "black")
    add_title(fig, "Grid search: úspešnosť na testovacej množine (%)", "18 kombinácií, všetky s EarlyStopping (patience 10)")
    plt.savefig(PLOTS_DIR / "grid_search.png", dpi=150)
    plt.close()
