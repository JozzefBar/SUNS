import copy
import time
from itertools import product
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC



# 1. Predspracovanie dat


TASK_DIR = Path(__file__).parent
PLOTS_DIR = TASK_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# stlpce, ktore su kategorie (aj ked su niektore zapisane cislom)
CATEGORICAL = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
# horne hranice pre outliery, zvolene podla skoku v zoradenych hodnotach
UPPER_LIMITS = {"Administrative_Duration": 10000, "ProductRelated": 2000}


def add_title(fig, title, subtitle=None):
    # tucny nadpis grafu, pod nim normalnym pismom podnadpis
    fig.suptitle(title, fontweight="bold", y=0.985)
    if subtitle is not None:
        fig.text(0.5, 0.9, subtitle, ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle is not None else 0.96))


def load_data():
    return pd.read_csv(TASK_DIR / "online_shoppers.csv")


def remove_outliers(df):
    # hodnoty, ktore podla popisu stlpcov nemozu existovat
    df = df[~(df["Administrative_Duration"] < 0)]   # 5 riadkov
    df = df[~(df["ProductRelated_Duration"] < 0)]   # 5 riadkov, ~(x < 0) necha aj NaN, tie riesim zvlast
    df = df[df["BounceRates"].between(0, 1)]        # 8 riadkov
    df = df[df["VisitorType"] != "Admin"]           # 1 riadok
    # prilis velke hodnoty
    df = df[df["Administrative_Duration"] < UPPER_LIMITS["Administrative_Duration"]]  # 15 riadkov
    df = df[df["ProductRelated"] < UPPER_LIMITS["ProductRelated"]]                    # 10 riadkov
    # rovnake SessionID = ten isty riadok dvakrat
    df = df.drop_duplicates(subset="SessionID")     # 3 riadky
    return df   # spolu prec 47 riadkov, z 12 333 zostane 12 286


def handle_missing(df, info_duration="drop"):
    # info_duration: "drop" = zmazat stlpec, "zeros_drop" = doplnit nuly a zvysok zmazat,
    #                "zeros_median" = doplnit nuly a zvysok medianom
    df = df.drop(columns="SessionID")   # len identifikator, ziadna informacia pre model

    if info_duration == "drop":
        df = df.drop(columns="Informational_Duration")
    else:
        df = df.copy()
        # ziadna informacna stranka = nulovy cas
        no_info_pages = df["Informational"] == 0
        df.loc[no_info_pages, "Informational_Duration"] = df.loc[no_info_pages, "Informational_Duration"].fillna(0)
        if info_duration == "zeros_median":
            # zjednodusenie: median z celeho datasetu, nie len z trenovacich dat
            median = df.loc[~no_info_pages, "Informational_Duration"].median()
            df["Informational_Duration"] = df["Informational_Duration"].fillna(median)

    return df.dropna()   # zvysne riadky s chybajucou hodnotou


def encode(df):
    # one-hot: kazda hodnota kategorie dostane vlastny stlpec 0/1
    df = pd.get_dummies(df, columns=CATEGORICAL, dtype=int)
    df["Weekend"] = df["Weekend"].astype(int)
    df["Revenue"] = df["Revenue"].astype(int)
    return df


def prepare_xy(info_duration="drop"):
    df = encode(handle_missing(remove_outliers(load_data()), info_duration))
    X = df.drop(columns="Revenue")
    y = df["Revenue"]
    return X, y


def split_data(X, y, val_size=0.15, test_size=0.15, seed=42):
    # najprv oddelim trenovacie data, zvysok rozdelim na validacne a testovacie
    # stratify drzi rovnaky podiel nakupov vo vsetkych troch mnozinach
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=val_size + test_size, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=test_size / (val_size + test_size), stratify=y_rest, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale(X_train, X_val, X_test):
    # priemer a odchylka sa pocitaju len z trenovacich dat
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


def explore_data():
    # kontrolne vypisy a graf, podla ktorych som nastavil cistenie dat
    df = load_data()
    print(df.shape)
    df.info()
    print(df.describe().T.to_string())

    print("\n--- outliery ---")
    print("zaporny Administrative_Duration:", (df["Administrative_Duration"] < 0).sum())
    print("zaporny ProductRelated_Duration:", (df["ProductRelated_Duration"] < 0).sum())
    print("BounceRates mimo [0, 1]:", (~df["BounceRates"].between(0, 1)).sum())
    print("ExitRates mimo [0, 1]:", (~df["ExitRates"].between(0, 1)).sum())
    print(df["VisitorType"].value_counts(dropna=False))
    print("opakovane SessionID:", df["SessionID"].duplicated().sum())

    # zoradene hodnoty, hladam skok
    columns = ["Administrative_Duration", "ProductRelated", "ProductRelated_Duration", "PageValues"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, columns):
        values = df[col].dropna().sort_values().values[-200:]
        ax.plot(values, ".")
        if col in UPPER_LIMITS:
            # cervena ciara = hranica, cervene body = odstranene hodnoty
            limit = UPPER_LIMITS[col]
            removed = values >= limit
            ax.plot(removed.nonzero()[0], values[removed], ".", color="red")
            ax.axhline(limit, color="red", linestyle="--", label=f"hranica {limit}, odstránených {removed.sum()}")
            ax.legend(loc="upper left")
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("poradie (200 najväčších hodnôt)")
        print(col, df[col].nlargest(17).round(0).tolist())
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "outliers_sorted.png", dpi=150)
    plt.close()

    rows_before = len(df)
    df = remove_outliers(df)
    print("odstranenych riadkov:", rows_before - len(df), "zostava:", len(df))

    print("\n--- chybajuce hodnoty ---")
    missing = df.isna().sum()
    print(pd.DataFrame({"pocet": missing, "percent": (missing / len(df) * 100).round(2)})[missing > 0])

    # da sa Informational_Duration doplnit? ak Informational == 0, cas musi byt 0
    missing_info = df[df["Informational_Duration"].isna()]
    print("Informational_Duration chyba:", len(missing_info))
    print("  z toho Informational == 0:", (missing_info["Informational"] == 0).sum())
    print("  z toho Informational > 0:", (missing_info["Informational"] > 0).sum())

    print("riadky bez ProductRelated_Duration:", df["ProductRelated_Duration"].isna().sum())
    print("riadky bez VisitorType:", df["VisitorType"].isna().sum())
    print("riadky, kde chyba oboje:", (df["ProductRelated_Duration"].isna() & df["VisitorType"].isna()).sum())

    for variant in ["drop", "zeros_drop", "zeros_median"]:
        cleaned = handle_missing(df, variant)
        print(f"{variant}: {cleaned.shape[0]} riadkov, {cleaned.shape[1]} stlpcov,"
              f" {cleaned.shape[0] / len(load_data()) * 100:.1f} % povodnych dat,"
              f" nakup {cleaned['Revenue'].mean() * 100:.1f} %")


# 1. Jednoduchy model (logisticka regresia)


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


def simple_model_experiments():
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
    plt.close()

    print("\n--- experiment: co s Informational_Duration ---")
    results = []
    for variant in ["drop", "zeros_drop", "zeros_median"]:
        X_variant, y_variant = prepare_xy(variant)
        result = run_experiment(X_variant, y_variant, 0.15, 0.15)
        results.append({"variant": variant, "riadkov": len(X_variant),
                        "podiel nakupov (%)": y_variant.mean() * 100, **result})
    print(pd.DataFrame(results).round(2).to_string(index=False))


# 2. Analyza dat (EDA)


MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NUMERIC = ["Administrative", "Administrative_Duration", "Informational", "ProductRelated",
           "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]


def eda():
    raw = load_data()
    # data po odstraneni outlierov a chybajucich hodnot, este pred kodovanim
    df = handle_missing(remove_outliers(raw))
    df["Nákup"] = df["Revenue"].map({False: "nie", True: "áno"})


    # 1. korelacie medzi ciselnymi stlpcami
    corr = df[NUMERIC + ["Revenue"]].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Korelácie medzi číselnými stĺpcami", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_correlation.png", dpi=150)
    plt.close()
    print(corr.round(2).to_string())


    # 2. BounceRates a ExitRates
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df, x="BounceRates", y="ExitRates", hue="Nákup", alpha=0.4, s=15)
    plt.plot([0, 0.2], [0, 0.2], color="gray", linestyle="--", label="ExitRates = BounceRates")
    plt.title("BounceRates a ExitRates", fontweight="bold")
    plt.legend(title="Nákup")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_bounce_exit.png", dpi=150)
    plt.close()
    print("\nExitRates < BounceRates:", (df["ExitRates"] < df["BounceRates"]).sum(), "riadkov")
    print("nakupy s BounceRates > 0.05:", (df[df["Revenue"]]["BounceRates"] > 0.05).sum(), "z", df["Revenue"].sum())


    # 3. pocet navstev a podiel nakupov podla mesiaca
    by_month = df.groupby("Month")["Revenue"].agg(["size", "mean"]).loc[MONTHS]
    fig, ax_visits = plt.subplots(figsize=(9, 5))
    ax_visits.bar(MONTHS, by_month["size"], color="lightsteelblue")
    ax_visits.set_ylabel("počet návštev")
    ax_rate = ax_visits.twinx()   # druha os y pre percenta
    ax_rate.plot(MONTHS, by_month["mean"] * 100, color="darkred", marker="o")
    ax_rate.set_ylabel("podiel návštev s nákupom (%)", color="darkred")
    ax_rate.set_ylim(0, 30)
    plt.title("Návštevy a nákupy podľa mesiaca", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_months.png", dpi=150)
    plt.close()
    print("\n", by_month.round(3))


    # 4. PageValues a typ navstevnika
    two_types = df[df["VisitorType"] != "Other"].copy()   # Other ma len 78 riadkov
    two_types["PageValues"] = two_types["PageValues"].gt(0).map({False: "PageValues = 0", True: "PageValues > 0"})
    rates = two_types.groupby(["PageValues", "VisitorType"])["Revenue"].agg(["size", "mean"])
    print("\n", rates.round(3))
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=two_types, x="PageValues", y="Revenue", hue="VisitorType",
                     estimator=lambda values: values.mean() * 100, errorbar=None)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f %%")
    plt.xlabel("")
    plt.ylabel("podiel návštev s nákupom (%)")
    plt.title("Nákupy podľa PageValues a typu návštevníka", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_pagevalues_visitor.png", dpi=150)
    plt.close()


    # 5. pocet produktovych stranok podla typu navstevnika a nakupu
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=two_types, x="VisitorType", y="ProductRelated", hue="Nákup", showfliers=False)
    plt.xlabel("")
    plt.ylabel("počet produktových stránok (ProductRelated)")
    plt.title("Produktové stránky podľa typu návštevníka", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_pages_visitor.png", dpi=150)
    plt.close()
    print("\n", two_types.groupby(["VisitorType", "Nákup"])["ProductRelated"].median())
    print(two_types.groupby("VisitorType")["Revenue"].mean().round(3))


    # 6. chybajuce hodnoty v povodnom datasete a nakup
    missing_cols = ["ProductRelated_Duration", "VisitorType", "Informational_Duration"]
    rows = []
    for col in missing_cols:
        for label, part in [("chýba", raw[raw[col].isna()]), ("vyplnená", raw[raw[col].notna()])]:
            rows.append({"skupina": f"{col}\n{label} ({len(part)} riadkov)",
                         "bez nákupu": (~part["Revenue"]).mean() * 100,
                         "nákup": part["Revenue"].mean() * 100})
            print(col, label, len(part), "riadkov, z toho nakup:", part["Revenue"].sum())
    shares = pd.DataFrame(rows).set_index("skupina")
    print(shares.round(1))

    # kazdy pruh ma spolu 100 %, siva cast su navstevy bez nakupu
    ax = shares.plot.barh(stacked=True, figsize=(9, 5), color=["lightgray", "seagreen"], width=0.75)
    for i, value in enumerate(shares["nákup"]):
        ax.text(101, i, f"nákup {value:.1f} %", va="center")
    ax.set_xlabel("podiel návštev (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 125)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=2)
    plt.title("Chýbajúce hodnoty v pôvodných dátach a nákup", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_missing.png", dpi=150)
    plt.close()


# 3. Neuronova siet


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


def describe(hidden_layers, lr, dropout=0.0):
    # popis nastavenia do podnadpisu grafu
    layers = ", ".join(str(neurons) for neurons in hidden_layers)
    return f"skryté vrstvy: {layers} | rýchlosť učenia: {lr:g} | dropout: {dropout:g}"


def plot_history(history, filename, title, subtitle, best_epoch=None):
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
    add_title(fig, title, subtitle)
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def plot_confusion(model, data, filename, title, subtitle):
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
    add_title(fig, title, subtitle)
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def neural_network_experiments():
    data = load_tensors()
    print("train:", data["train"][0].shape, "val:", data["val"][0].shape, "test:", data["test"][0].shape)

    print("\n--- pretrenovanie ---")
    big_network = (512, 512, 256)
    model, history, _ = train_network(data, big_network, lr=0.001, epochs=200)
    print(accuracies(model, data))
    print(history.iloc[[0, 4, 9, 19, 49, 99, 199]].round(3).to_string(index=False))
    settings_text = describe(big_network, 0.001)
    plot_history(history, "nn_overfit_history.png", "Pretrénovaná sieť, 200 epoch bez EarlyStopping", settings_text)
    plot_confusion(model, data, "nn_overfit_confusion.png", "Pretrénovaná sieť", settings_text)

    print("\n--- ta ista siet s EarlyStopping ---")
    model, history, best_epoch = train_network(data, big_network, lr=0.001, epochs=200, patience=10)
    print(accuracies(model, data), "najlepsia epocha:", best_epoch, "zastavene po:", len(history))
    plot_history(history, "nn_early_history.png",
                 "Tá istá sieť s EarlyStopping (patience 10)", settings_text, best_epoch)
    plot_confusion(model, data, "nn_early_confusion.png", "Sieť s EarlyStopping (patience 10)", settings_text)

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
        runs.append({"model": model, "history": history, "best_epoch": best_epoch,
                     "settings_text": describe(**settings)})
        settings.update({"experiment": number, "epocha": best_epoch,
                         "train (%)": result["train"], "test (%)": result["test"]})
    table = pd.DataFrame(experiments).set_index("experiment")
    print(table.to_string(float_format=lambda value: f"{value:g}" if value < 1 else f"{value:.2f}"))

    for kind, label, index in [("best", "Najlepší", table["test (%)"].idxmax()),
                               ("worst", "Najhorší", table["test (%)"].idxmin())]:
        run = runs[index - 1]
        title = f"{label} experiment (č. {index})"
        plot_history(run["history"], f"nn_{kind}_history.png", title, run["settings_text"], run["best_epoch"])
        plot_confusion(run["model"], data, f"nn_{kind}_confusion.png", title, run["settings_text"])
        print(kind, "experiment:", index)


# 4. Nepovinne ulohy: grid search a porovnanie modelov


# vsetky kombinacie tychto hodnot: 3 * 3 * 2 = 18 sieti
GRID = {
    "hidden_layers": [(16,), (64, 32), (256, 128)],
    "lr": [0.1, 0.01, 0.001],
    "dropout": [0.0, 0.3],
}


def grid_search():
    data = load_tensors()
    results = []
    for hidden_layers, lr, dropout in product(*GRID.values()):
        model, history, best_epoch = train_network(data, hidden_layers, lr=lr, dropout=dropout, patience=10)
        row = {"skryté vrstvy": ", ".join(str(n) for n in hidden_layers), "lr": lr, "dropout": dropout, "epocha": best_epoch}
        for name in ["train", "test"]:
            X, y = data[name]
            row[f"{name} (%)"] = (predict(model, X) == y).float().mean().item() * 100
        row["rozdiel train-test"] = row["train (%)"] - row["test (%)"]
        results.append(row)

    table = pd.DataFrame(results).sort_values("test (%)", ascending=False, kind="stable").reset_index(drop=True)
    print(table.to_string(float_format=lambda value: f"{value:.2f}",
                          formatters={"lr": "{:g}".format, "dropout": "{:g}".format}))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, dropout in zip(axes, GRID["dropout"]):
        part = table[table["dropout"] == dropout]
        pivot = part.pivot(index="skryté vrstvy", columns="lr", values="test (%)")
        pivot = pivot.loc[[", ".join(str(n) for n in layers) for layers in GRID["hidden_layers"]]]
        ax.imshow(pivot.values, cmap="Blues", vmin=table["test (%)"].min(), vmax=table["test (%)"].max())
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


# nazov -> funkcia, ktora vytvori novy model
MODELS = {
    "logistická regresia": lambda: LogisticRegression(max_iter=1000),
    "k najbližších susedov (k = 15)": lambda: KNeighborsClassifier(n_neighbors=15),
    "SVM (RBF jadro)": lambda: SVC(),
    "náhodný les (200 stromov)": lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
    "gradient boosting": lambda: GradientBoostingClassifier(random_state=0),
}


def compare_models():
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


if __name__ == "__main__":
    steps = [("1. Predspracovanie dat", explore_data),
             ("1. Jednoduchy model", simple_model_experiments),
             ("2. Analyza dat (EDA)", eda),
             ("3. Neuronova siet", neural_network_experiments),
             ("4. Grid search", grid_search),
             ("4. Porovnanie modelov", compare_models)]
    for title, step in steps:
        print(f"\n\n########## {title} ##########")
        step()
