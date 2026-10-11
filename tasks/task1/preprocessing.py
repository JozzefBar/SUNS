from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


if __name__ == "__main__":
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
