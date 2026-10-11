import time

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from preprocessing import PLOTS_DIR, add_title, prepare_xy
from simple_model import run_experiment

# nazov -> funkcia, ktora vytvori novy model
MODELS = {
    "logistická regresia": lambda: LogisticRegression(max_iter=1000),
    "k najbližších susedov (k = 15)": lambda: KNeighborsClassifier(n_neighbors=15),
    "SVM (RBF jadro)": lambda: SVC(),
    "náhodný les (200 stromov)": lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
    "gradient boosting": lambda: GradientBoostingClassifier(random_state=0),
}

if __name__ == "__main__":
    X, y = prepare_xy()
    results = []
    for name, make_model in MODELS.items():
        start = time.time()
        # rovnake rozdelenie 70 / 15 / 15 a rovnakych 5 seedov ako pri logistickej regresii
        result = run_experiment(X, y, 0.15, 0.15, make_model)
        results.append({"model": name, **result, "cas (s)": time.time() - start})
    table = pd.DataFrame(results).sort_values("test (%)", ascending=False).round(2)
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    table = table.iloc[::-1]   # najlepsi model hore
    ax.barh(table["model"], table["train (%)"], color="lightsteelblue", label="trénovacia")
    test_bars = ax.barh(table["model"], table["test (%)"], height=0.4, color="darkred", label="testovacia")
    ax.bar_label(test_bars, fmt="%.2f", padding=3)
    ax.axvline((1 - y.mean()) * 100, color="gray", linestyle="--", label="vždy „bez nákupu“")
    ax.set_xlim(75, 103)
    ax.set_xlabel("úspešnosť (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    add_title(fig, "Porovnanie modelov zo Sklearn", "rozdelenie 70 / 15 / 15, priemer z 5 behov")
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
    plt.close()
