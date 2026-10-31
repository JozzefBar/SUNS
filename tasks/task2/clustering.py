import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.svm import SVR

from preprocessing import PLOTS_DIR, add_title, prepare_data, split_data, scale


def choose_k(X_train):
    # zadanie chce aspon 3 zhluky, z k = 3 az 6 beriem to s najvyssim silhouette skore
    scores = {}
    for k in range(3, 7):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X_train)
        scores[k] = silhouette_score(X_train, labels)
        print(f"k = {k}: silhouette {scores[k]:.3f}")
    return max(scores, key=scores.get)


def plot_clusters(X_train, labels):
    # zhlukujem v 14 rozmeroch, do grafu idu 3 komponenty PCA
    points = PCA(n_components=3).fit_transform(X_train)
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(projection="3d")
    for cluster in np.unique(labels):
        part = points[labels == cluster]
        ax.scatter(part[:, 0], part[:, 1], part[:, 2], s=3, alpha=0.5, label=f"zhluk {cluster}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(markerscale=4)
    add_title(fig, "Zhluky K-means v priestore prvých troch komponentov PCA",
              "zhlukovanie na 14 normalizovaných vstupoch bez CO, trénovacia množina")
    plt.savefig(PLOTS_DIR / "clusters_3d.png", dpi=150)
    plt.close()


def scores(y, prediction):
    return {"RMSE": mean_squared_error(y, prediction) ** 0.5, "R2": r2_score(y, prediction)}


if __name__ == "__main__":
    X_train_raw, X_test_raw, y_train, y_test = split_data(prepare_data())
    X_train, X_test = scale(X_train_raw, X_test_raw)

    k = choose_k(X_train)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_train)
    train_clusters = kmeans.labels_
    test_clusters = kmeans.predict(X_test)   # testovacie hodiny priradim k najblizsiemu stredu
    plot_clusters(X_train, train_clusters)

    # profil zhlukov: priemery povodnych hodnot, co sa do zhlukovania nepouzilo, je len na porovnanie
    profile = X_train_raw.assign(co=y_train).groupby(train_clusters).mean()
    profile.insert(0, "hodin", np.bincount(train_clusters))
    print(profile.round(2).T.to_string())

    # najlepsi model z casti 2 na vsetkych datach a zvlast pre kazdy zhluk
    global_model = SVR(C=100).fit(X_train, y_train)
    global_test = global_model.predict(X_test)
    cluster_train, cluster_test = np.zeros(len(y_train)), np.zeros(len(y_test))
    rows = []
    for cluster in range(k):
        tr, te = train_clusters == cluster, test_clusters == cluster
        model = SVR(C=100).fit(X_train[tr], y_train[tr])
        cluster_train[tr] = model.predict(X_train[tr])
        cluster_test[te] = model.predict(X_test[te])
        rows.append({"zhluk": cluster, "train": tr.sum(), "test": te.sum(),
                     **{f"jeden model {key}": value for key, value in scores(y_test[te], global_test[te]).items()},
                     **{f"model zhluku {key}": value for key, value in scores(y_test[te], cluster_test[te]).items()}})
    rows.append({"zhluk": "spolu", "train": len(y_train), "test": len(y_test),
                 **{f"jeden model {key}": value for key, value in scores(y_test, global_test).items()},
                 **{f"model zhluku {key}": value for key, value in scores(y_test, cluster_test).items()}})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("modely zhlukov na trenovacej mnozine:", {key: round(value, 3) for key, value in scores(y_train, cluster_train).items()})
