import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

from preprocessing import PLOTS_DIR, add_title, prepare_xy, split_data, scale

SEEDS = [0, 1, 2, 3, 4]
# (trenovacia, validacna, testovacia)
SPLITS = [(0.90, 0.05, 0.05), (0.80, 0.10, 0.10), (0.70, 0.15, 0.15),
          (0.60, 0.20, 0.20), (0.40, 0.30, 0.30), (0.10, 0.45, 0.45)]


def train_model(X, y, val_size, test_size, seed, make_model=None):
    # make_model = funkcia, ktora vrati novy nenatrenovany model, bez nej sa pouzije logisticka regresia
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, val_size, test_size, seed)
    X_train, X_val, X_test = scale(X_train, X_val, X_test)
    model = make_model() if make_model is not None else LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def run_experiment(X, y, val_size, test_size, make_model=None):
    # to iste rozdelenie s roznym seedom, aby vysledok nezavisel od jedneho nahodneho vyberu
    rows = []
    for seed in SEEDS:
        model, X_train, X_test, y_train, y_test = train_model(X, y, val_size, test_size, seed, make_model)
        rows.append({
            "train": accuracy_score(y_train, model.predict(X_train)),
            "test": accuracy_score(y_test, model.predict(X_test)),
        })
    runs = pd.DataFrame(rows) * 100
    return {
        "train (%)": runs["train"].mean(),
        "test (%)": runs["test"].mean(),
        "test odchylka": runs["test"].std(),
        "rozdiel train-test": runs["train"].mean() - runs["test"].mean(),
    }


if __name__ == "__main__":
    X, y = prepare_xy()
    print("X:", X.shape, "y:", y.shape)
    print(f"model, ktory vzdy povie 'bez nakupu': {(1 - y.mean()) * 100:.2f} %")

    print("\n--- experimenty s rozdelenim ---")
    results = []
    for train_size, val_size, test_size in SPLITS:
        result = run_experiment(X, y, val_size, test_size)
        X_train, X_val, X_test, *_ = split_data(X, y, val_size, test_size)   # len kvoli poctom riadkov
        result = {"rozdelenie": f"{train_size:.0%} / {val_size:.0%} / {test_size:.0%}",
                  "train riadkov": len(X_train), "test riadkov": len(X_test), **result}
        results.append(result)
    table = pd.DataFrame(results).round(2)
    print(table.to_string(index=False))

    # najlepsie rozdelenie podla uspesnosti na testovacej mnozine,
    # pri zhode ma prednost vacsia testovacia mnozina
    best = SPLITS[table.sort_values(["test (%)", "test riadkov"]).index[-1]]
    print("\nnajlepsie rozdelenie:", best)
    model, X_train, X_test, y_train, y_test = train_model(X, y, best[1], best[2], seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name, X_part, y_part in [(axes[0], "trénovacia", X_train, y_train),
                                     (axes[1], "testovacia", X_test, y_test)]:
        ConfusionMatrixDisplay.from_estimator(
            model, X_part, y_part, display_labels=["bez nákupu", "nákup"],
            cmap="Blues", colorbar=False, ax=ax)
        accuracy = accuracy_score(y_part, model.predict(X_part)) * 100
        ax.set_title(f"{name} množina, úspešnosť {accuracy:.2f} %")
        ax.set_xlabel("predikcia")
        ax.set_ylabel("skutočnosť")
    add_title(fig, "Konfúzne matice logistickej regresie",
              f"rozdelenie {best[0]:.0%} / {best[1]:.0%} / {best[2]:.0%} (trénovacia / validačná / testovacia), seed 0")
    plt.savefig(PLOTS_DIR / "confusion_simple_model.png", dpi=150)

    print("\n--- experiment: co s Informational_Duration ---")
    results = []
    for variant in ["drop", "zeros_drop", "zeros_median"]:
        X_variant, y_variant = prepare_xy(variant)
        result = run_experiment(X_variant, y_variant, 0.15, 0.15)
        results.append({"variant": variant, "riadkov": len(X_variant),
                        "podiel nakupov (%)": y_variant.mean() * 100, **result})
    print(pd.DataFrame(results).round(2).to_string(index=False))
