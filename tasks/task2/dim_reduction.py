import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from preprocessing import PLOTS_DIR, add_title, prepare_data, split_data, scale

# senzor s najvyssou kladnou koreláciou s co (0.92), senzor so zapornou (-0.70) a hodina dna (0.36)
RAW_FEATURES = ["sensor_nmhc", "sensor_nox", "hour"]


def plot_3d(points, co, labels, filename, title, subtitle=None, view=(30, -60)):
    # body s vysokym co kreslim az nakoniec, aby ich neprekryli tie s nizkym
    order = np.argsort(co)
    points, co = points[order], co[order]
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(projection="3d")
    dots = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=co, cmap="viridis", s=3, alpha=0.6)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.view_init(*view)   # vyska a otocenie pohladu v stupnoch
    fig.colorbar(dots, ax=ax, shrink=0.6, pad=0.12, label="koncentrácia CO (mg/m³)")
    add_title(fig, title, subtitle)
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def run_pca(X_train, features):
    # PCA sa uci len na trenovacich datach, rovnako ako scaler
    pca = PCA(n_components=3).fit(X_train)
    ratio = pca.explained_variance_ratio_
    print("vysvetlena variancia:", ratio.round(3), "spolu:", ratio.sum().round(3))
    # loadingy: kolko kazdy povodny priznak prispieva do komponentu
    loadings = pd.DataFrame(pca.components_.T, index=features, columns=["PC1", "PC2", "PC3"])
    print(loadings.round(2).to_string())
    return pca


if __name__ == "__main__":
    df = prepare_data()
    print("--- 3 povodne priznaky pred normalizaciou ---")
    plot_3d(df[RAW_FEATURES].values, df["co"].values, ["sensor_nmhc", "sensor_nox", "hodina"],
            "scatter3d_raw.png", "Pôvodné príznaky pred normalizáciou", "farba = koncentrácia CO",
            view=(10, -100))   # zboku, aby bola os hodin zvisla a sensor_nmhc vodorovne
    # cisla k rozboru grafu
    print(df.groupby(pd.cut(df["sensor_nmhc"], [0, 750, 1000, 1250, 1500, 1750, 2500]), observed=True)["co"]
          .agg(["mean", "count"]).round(2).to_string())
    print(f"co pri sensor_nox nad 1500: {df.loc[df['sensor_nox'] > 1500, 'co'].mean():.2f},"
          f" pod 800: {df.loc[df['sensor_nox'] < 800, 'co'].mean():.2f}")
    print(df.groupby("hour")["co"].mean().round(2).to_string())

    print("\n--- PCA na 3 komponenty ---")
    X_train_raw, X_test_raw, y_train, y_test = split_data(df)
    X_train, X_test = scale(X_train_raw, X_test_raw)
    pca = run_pca(X_train, X_train_raw.columns)
    train_components = pca.transform(X_train)
    season = X_train_raw.filter(like="season").idxmax(axis=1).values
    print(pd.DataFrame(train_components, columns=["PC1", "PC2", "PC3"]).groupby(season).mean().round(2).to_string())
    print("korelacia komponentov s co (train):",
          [round(np.corrcoef(train_components[:, i], y_train)[0, 1], 2) for i in range(3)])
    # do grafu idu vsetky hodiny, trenovacie aj testovacie
    components = pca.transform(np.vstack([X_train, X_test]))
    co = pd.concat([y_train, y_test]).values
    ratio = pca.explained_variance_ratio_
    plot_3d(components, co, [f"PC{i + 1} ({ratio[i]:.1%})" for i in range(3)],
            "scatter3d_pca.png", "PCA: normalizované vstupy bez CO zmenšené na 3 komponenty",
            f"komponenty spolu vysvetľujú {ratio.sum():.1%} variancie, farba = koncentrácia CO", view=(20, -60))
