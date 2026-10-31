import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from preprocessing import PLOTS_DIR, add_title, prepare_data, split_data, scale
from models import evaluate

# najlepsi model z casti 2
BEST_MODEL = "SVR, RBF, C = 100"
MIN_TARGET_CORR = 0.3    # priznak musi mat s co aspon takuto koreláciu (v absolutnej hodnote)
MAX_MUTUAL_CORR = 0.85   # dva priznaky s vyssou vzajomnou koreláciou nesu skoro to iste
IMPORTANCE_SHARE = 0.95  # podiel dolezitosti z lesa, ktory musia vybrane priznaky pokryt
PCA_VARIANCE = 0.95      # podiel variancie, ktory musia pokryt komponenty PCA


def select_by_correlation(X_train_raw, y_train):
    # korelacie len z trenovacich dat
    corr = X_train_raw.assign(co=y_train).corr()
    target = corr["co"].drop("co").abs().sort_values(ascending=False)
    print(target.round(2).to_string())
    selected = target[target >= MIN_TARGET_CORR].index.tolist()
    # zo silno korelovanej dvojice necham ten priznak, ktory ma vyssiu koreláciu s co
    without_pairs = []
    for feature in selected:
        if all(abs(corr.loc[feature, kept]) <= MAX_MUTUAL_CORR for kept in without_pairs):
            without_pairs.append(feature)
    plot_correlation(corr)
    return selected, without_pairs


def plot_correlation(corr):
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
                annot_kws={"size": 8}, ax=ax)
    add_title(fig, "Korelačná matica (trénovacia množina)")
    plt.savefig(PLOTS_DIR / "correlation_matrix.png", dpi=150)
    plt.close()


def select_by_importance(X_train, y_train, features):
    # rovnaky les ako najlepsi v casti 2
    forest = RandomForestRegressor(max_features="sqrt", random_state=0, n_jobs=-1).fit(X_train, y_train)
    importance = pd.Series(forest.feature_importances_, index=features).sort_values(ascending=False)
    cumulative = importance.cumsum()
    print(pd.DataFrame({"dolezitost": importance, "kumulativne": cumulative}).round(3).to_string())
    # najdolezitejsie priznaky, kym spolu nedaju aspon 95 % dolezitosti
    count = (cumulative < IMPORTANCE_SHARE).sum() + 1
    return importance.index[:count].tolist()


def plot_pca_variance(X_train):
    cumulative = PCA().fit(X_train).explained_variance_ratio_.cumsum()
    count = (cumulative < PCA_VARIANCE).sum() + 1
    print("kumulativna variancia:", cumulative.round(3))
    fig, ax = plt.subplots(figsize=(9, 5))
    components = range(1, len(cumulative) + 1)
    ax.plot(components, cumulative, "o-")
    ax.axhline(PCA_VARIANCE, color="red", linestyle="--", label=f"{PCA_VARIANCE:.0%} variancie")
    ax.axvline(count, color="gray", linestyle=":", label=f"{count} komponentov")
    ax.set_xticks(list(components))
    ax.set_xlabel("počet komponentov")
    ax.set_ylabel("kumulatívna vysvetlená variancia")
    ax.legend(loc="lower right")
    add_title(fig, "PCA: vysvetlená variancia podľa počtu komponentov")
    plt.savefig(PLOTS_DIR / "pca_variance.png", dpi=150)
    plt.close()


def plot_residuals_comparison(results):
    # hore trenovacia, dole testovacia mnozina, kazdy stlpec jedna mnozina priznakov
    fig, axes = plt.subplots(2, len(results), figsize=(4.2 * len(results), 7.5), sharex=True, sharey=True)
    for col, (label, model, data) in enumerate(results):
        X_train, X_test, y_train, y_test = data
        for row, name, X, y in [(0, "train", X_train, y_train), (1, "test", X_test, y_test)]:
            prediction = model.predict(X)
            residuals = y - prediction
            ax = axes[row, col]
            ax.scatter(prediction, residuals, s=4, alpha=0.3)
            ax.axhline(0, color="red", linestyle="--")
            ax.set_title(f"{label}\n{name}, RMSE {residuals.pow(2).mean() ** 0.5:.3f}", fontsize=10)
        axes[1, col].set_xlabel("predikovaná koncentrácia CO (mg/m³)")
    axes[0, 0].set_ylabel("reziduál (mg/m³)")
    axes[1, 0].set_ylabel("reziduál (mg/m³)")
    add_title(fig, "Reziduály SVR pri rôznych množinách príznakov", BEST_MODEL)
    plt.savefig(PLOTS_DIR / "residuals_selection.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    X_train_raw, X_test_raw, y_train, y_test = split_data(prepare_data())
    X_train, X_test = scale(X_train_raw, X_test_raw)
    features = X_train_raw.columns

    print("--- (a) korelacna matica ---")
    corr_selected, corr_without_pairs = select_by_correlation(X_train_raw, y_train)
    print("|r| s co >= 0.3:", corr_selected)
    print("bez silno korelovanych dvojic:", corr_without_pairs)

    print("\n--- (b) dolezitost z nahodneho lesa ---")
    importance_selected = select_by_importance(X_train, y_train, features)
    print("vybrane:", importance_selected)

    print("\n--- (c) PCA podla variancie ---")
    plot_pca_variance(X_train)
    pca = PCA(n_components=PCA_VARIANCE).fit(X_train)
    print("pocet komponentov:", pca.n_components_)

    # kazda mnozina sa skaluje znova, scaler aj PCA su nastavene len na trenovacich datach
    variants = [
        ("všetky príznaky", scale(X_train_raw, X_test_raw)),
        ("korelácia s co", scale(X_train_raw[corr_selected], X_test_raw[corr_selected])),
        ("korelácia s co, bez dvojíc",
         scale(X_train_raw[corr_without_pairs], X_test_raw[corr_without_pairs])),
        ("dôležitosť z lesa", scale(X_train_raw[importance_selected], X_test_raw[importance_selected])),
        ("PCA 95 % variancie", (pca.transform(X_train), pca.transform(X_test))),
    ]

    print(f"\n--- {BEST_MODEL} na zmensenych mnozinach ---")
    rows, results = [], []
    for label, (X_tr, X_te) in variants:
        model = SVR(C=100).fit(X_tr, y_train)
        data = (X_tr, X_te, y_train, y_test)
        rows.append({"mnozina": label, "vstupov": X_tr.shape[1], **evaluate(model, *data),
                     "test rezidual < -1.5": (y_test - model.predict(X_te) < -1.5).sum()})
        if label != "korelácia s co, bez dvojíc":
            results.append((label, model, data))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    plot_residuals_comparison(results)
