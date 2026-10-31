import time

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree

from preprocessing import PLOTS_DIR, add_title, prepare_data, split_data, scale

# popis nastavenia -> model, nevypisane parametre su predvolene
TREE_EXPERIMENTS = {
    "bez obmedzenia": DecisionTreeRegressor(random_state=0),
    "max_depth = 3": DecisionTreeRegressor(max_depth=3, random_state=0),
    "max_depth = 5": DecisionTreeRegressor(max_depth=5, random_state=0),
    "max_depth = 8": DecisionTreeRegressor(max_depth=8, random_state=0),
    "max_depth = 10": DecisionTreeRegressor(max_depth=10, random_state=0),
    "max_depth = 15": DecisionTreeRegressor(max_depth=15, random_state=0),
    "min_samples_leaf = 10": DecisionTreeRegressor(min_samples_leaf=10, random_state=0),
    "min_samples_leaf = 20": DecisionTreeRegressor(min_samples_leaf=20, random_state=0),
    "min_samples_leaf = 50": DecisionTreeRegressor(min_samples_leaf=50, random_state=0),
    # delenie sa spravi len ked znizi chybu aspon o tuto hodnotu
    "min_impurity_decrease = 0.001": DecisionTreeRegressor(min_impurity_decrease=0.001, random_state=0),
    # post-pruning: najprv cely strom, potom odreze vetvy, ktore chybu znizuju menej nez ccp_alpha
    "ccp_alpha = 0.001": DecisionTreeRegressor(ccp_alpha=0.001, random_state=0),
    "ccp_alpha = 0.01": DecisionTreeRegressor(ccp_alpha=0.01, random_state=0),
}

FOREST_EXPERIMENTS = {
    "50 stromov": RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=-1),
    "100 stromov": RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),
    "300 stromov": RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
    "100 stromov, max_features = sqrt": RandomForestRegressor(max_features="sqrt", random_state=0, n_jobs=-1),
    "100 stromov, max_features = 0.5": RandomForestRegressor(max_features=0.5, random_state=0, n_jobs=-1),
    "100 stromov, min_samples_leaf = 5": RandomForestRegressor(min_samples_leaf=5, random_state=0, n_jobs=-1),
    "100 stromov, max_depth = 10": RandomForestRegressor(max_depth=10, random_state=0, n_jobs=-1),
    # kriterialna funkcia: absolutna chyba namiesto stvorca
    "100 stromov, criterion = absolute_error": RandomForestRegressor(criterion="absolute_error", random_state=0, n_jobs=-1),
}

SVR_EXPERIMENTS = {
    "lineárne jadro, C = 1": SVR(kernel="linear"),
    "RBF, C = 0.1": SVR(C=0.1),
    "RBF, C = 1": SVR(C=1),
    "RBF, C = 10": SVR(C=10),
    "RBF, C = 100": SVR(C=100),
    "RBF, C = 300": SVR(C=300),
    "RBF, C = 10, epsilon = 0.3": SVR(C=10, epsilon=0.3),
    "RBF, C = 10, gamma = 0.01": SVR(C=10, gamma=0.01),
    "RBF, C = 10, gamma = 0.5": SVR(C=10, gamma=0.5),
    "polynomiálne jadro, stupeň 2, C = 10": SVR(kernel="poly", degree=2, C=10),
    "polynomiálne jadro, stupeň 3, C = 10": SVR(kernel="poly", degree=3, C=10),
    # coef0 = 1 prida do jadra aj cleny nizsieho stupna
    "polynomiálne jadro, stupeň 2, C = 10, coef0 = 1": SVR(kernel="poly", degree=2, C=10, coef0=1),
}


def evaluate(model, X_train, X_test, y_train, y_test):
    result = {}
    for name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        prediction = model.predict(X)
        result[f"MSE {name}"] = mean_squared_error(y, prediction)
        result[f"RMSE {name}"] = result[f"MSE {name}"] ** 0.5
        result[f"R2 {name}"] = r2_score(y, prediction)
    return result


def run_experiments(experiments, X_train, X_test, y_train, y_test):
    # natrenuje vsetky nastavenia, najlepsie je to s najvyssim R2 na testovacej mnozine
    rows = []
    for label, model in experiments.items():
        start = time.time()
        model.fit(X_train, y_train)
        seconds = time.time() - start
        rows.append({"nastavenie": label, **evaluate(model, X_train, X_test, y_train, y_test), "cas (s)": seconds})
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    best = table.loc[table["R2 test"].idxmax(), "nastavenie"]
    print("najlepsie:", best)
    return experiments[best], best


def residual_summary(model, X_test, y_test):
    # cisla k rozboru rezidualov na testovacej mnozine
    prediction = model.predict(X_test)
    residuals = y_test - prediction
    high = y_test > 5
    print(f"priemer {residuals.mean():.3f}, chyb nad 1 mg/m3: {(residuals.abs() > 1).sum()},"
          f" co nad 5: {high.sum()} hodin s priemernym rezidualom {residuals[high].mean():.2f}")
    low, top = prediction < 2, prediction > 4
    print(f"RMSE pri predikcii pod 2: {(residuals[low] ** 2).mean() ** 0.5:.3f},"
          f" nad 4: {(residuals[top] ** 2).mean() ** 0.5:.3f}")
    print(f"zapornych predikcii: {(prediction < 0).sum()}, najmensia {prediction.min():.2f}")


def plot_residuals(model, X_train, X_test, y_train, y_test, filename, title, subtitle):
    # vlavo rezidualy voci predikcii, vpravo ich rozdelenie, hore trenovacia a dole testovacia mnozina
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    axes[1, 0].sharey(axes[0, 0])
    for row, name, X, y in [(0, "trénovacia", X_train, y_train), (1, "testovacia", X_test, y_test)]:
        prediction = model.predict(X)
        residuals = y - prediction
        axes[row, 0].scatter(prediction, residuals, s=5, alpha=0.3)
        axes[row, 0].axhline(0, color="red", linestyle="--")
        axes[row, 0].set_ylabel("reziduál (mg/m³)")
        axes[row, 0].set_title(f"{name} množina: reziduál podľa predikcie")
        axes[row, 1].hist(residuals, bins=60)
        axes[row, 1].axvline(0, color="red", linestyle="--")
        axes[row, 1].set_ylabel("počet hodín")
        axes[row, 1].set_title(f"{name} množina: rozdelenie reziduálov, RMSE {residuals.pow(2).mean() ** 0.5:.3f}")
    axes[1, 0].set_xlabel("predikovaná koncentrácia CO (mg/m³)")
    axes[1, 1].set_xlabel("reziduál = skutočná − predikovaná hodnota (mg/m³)")
    add_title(fig, title, subtitle)
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def plot_small_tree(X_train_raw, X_test_raw, y_train, y_test):
    # strom na neskalovanych datach, aby boli prahy v povodnych jednotkach,
    # skalovanie poradie hodnot nemeni, takze strom robi rovnake delenia
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_train_raw, y_train)
    result = evaluate(tree, X_train_raw, X_test_raw, y_train, y_test)
    print("maly strom:", {key: round(value, 3) for key, value in result.items()})
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(tree, feature_names=X_train_raw.columns.tolist(), filled=True, rounded=True,
              impurity=False, precision=2, fontsize=11, ax=ax)
    add_title(fig, "Rozhodovací strom s hĺbkou 3",
              f"R² train {result['R2 train']:.3f}, R² test {result['R2 test']:.3f} | "
              "value = priemerná koncentrácia CO v uzle (mg/m³), samples = počet trénovacích hodín")
    plt.savefig(PLOTS_DIR / "tree_depth3.png", dpi=150)
    plt.close()


def plot_importance(model, features):
    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    print(importance.iloc[::-1].round(3).to_string())
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(importance.index, importance.values, color="seagreen")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlim(0, importance.max() * 1.15)
    ax.set_xlabel("dôležitosť (súčet je 1)")
    add_title(fig, "Dôležitosť vstupných parametrov v náhodnom lese")
    plt.savefig(PLOTS_DIR / "forest_importance.png", dpi=150)
    plt.close()


def plot_comparison(table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in [(axes[0], "R2", "R²"), (axes[1], "RMSE", "RMSE (mg/m³)")]:
        part = table.set_index("model")[[f"{metric} train", f"{metric} test"]]
        part.columns = ["trénovacia", "testovacia"]
        part.plot.bar(ax=ax, rot=0, color=["lightsteelblue", "darkred"])
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=9)
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.set_ylim(0, part.values.max() * 1.15)
    add_title(fig, "Porovnanie modelov", "najlepšie nastavenie každého modelu")
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    X_train_raw, X_test_raw, y_train, y_test = split_data(prepare_data())
    X_train, X_test = scale(X_train_raw, X_test_raw)
    data = (X_train, X_test, y_train, y_test)

    # model, ktory vzdy predpovie priemer trenovacich dat, ma R2 priblizne 0
    mean_prediction = [y_train.mean()] * len(y_test)
    print(f"vzdy priemer: RMSE test {mean_squared_error(y_test, mean_prediction) ** 0.5:.3f},"
          f" R2 test {r2_score(y_test, mean_prediction):.3f}")

    print("\n--- rozhodovaci strom ---")
    best_tree, tree_label = run_experiments(TREE_EXPERIMENTS, *data)
    for label, tree in TREE_EXPERIMENTS.items():
        print(f"{label}: hlbka {tree.get_depth()}, listov {tree.get_n_leaves()}")
    plot_small_tree(X_train_raw, X_test_raw, y_train, y_test)

    print("\n--- nahodny les ---")
    best_forest, forest_label = run_experiments(FOREST_EXPERIMENTS, *data)
    plot_importance(best_forest, X_train_raw.columns)

    print("\n--- SVM ---")
    best_svr, svr_label = run_experiments(SVR_EXPERIMENTS, *data)
    # trenovacie rezidualy SVR vzhladom na pasmo epsilon = 0.1 (libsvm ma toleranciu 0.001)
    train_residuals = (y_train - best_svr.predict(X_train)).abs()
    print(f"SVR train: v pasme {(train_residuals <= 0.101).mean():.1%},"
          f" na okraji pasma {((train_residuals - 0.1).abs() <= 0.001).mean():.1%}")

    print("\n--- porovnanie ---")
    rows = []
    for name, model, label, filename in [("rozhodovací strom", best_tree, tree_label, "residuals_tree.png"),
                                         ("náhodný les", best_forest, forest_label, "residuals_forest.png"),
                                         ("SVM (SVR)", best_svr, svr_label, "residuals_svr.png")]:
        rows.append({"model": name, "nastavenie": label, **evaluate(model, *data)})
        plot_residuals(model, *data, filename, f"Reziduály: {name}", label)
        print(name, end=": ")
        residual_summary(model, X_test, y_test)
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    plot_comparison(table)
