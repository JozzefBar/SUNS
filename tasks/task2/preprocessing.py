from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TASK_DIR = Path(__file__).parent
PLOTS_DIR = TASK_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# hodnoty senzorov a pocasia, ked chybaju, chybaju vsetky naraz
MEASUREMENTS = ["sensor_co", "sensor_nmhc", "sensor_nox", "sensor_no2", "sensor_o3", "temperature", "humidity"]
# popis stlpca v zadani: teplota -5 az 40 °C, pod -5 nic nie je (minimum -1.9), preto len horna hranica
MAX_TEMPERATURE = 40


def add_title(fig, title, subtitle=None):
    # tucny nadpis grafu, pod nim normalnym pismom podnadpis
    fig.suptitle(title, fontweight="bold", y=0.985)
    if subtitle is not None:
        fig.text(0.5, 0.9, subtitle, ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle is not None else 0.96))


def load_data():
    return pd.read_csv(TASK_DIR / "air_quality.csv")


def remove_outliers(df):
    df = df.drop_duplicates()   # 150 riadkov, rovnaky instant aj vsetky hodnoty
    # ~(x > 40) necha aj NaN, tie riesim zvlast
    df = df[~(df["temperature"] > MAX_TEMPERATURE)]   # 59 riadkov
    return df


def handle_missing(df):
    # instant je len identifikator, datum ma 391 roznych hodnot a mesiac aj den v tyzdni su z neho uz vytiahnute
    df = df.drop(columns=["instant", "date"])
    # riadky bez merani senzorov a riadky bez co (cielovu hodnotu nema zmysel vymyslat)
    return df.dropna()


def encode(df):
    # one-hot: 4 rocne obdobia = 4 stlpce 0/1
    return pd.get_dummies(df, columns=["season"], dtype=int)


def prepare_data():
    return encode(handle_missing(remove_outliers(load_data())))


def split_data(df, test_size=0.2, seed=42):
    # najprv rozdelim riadky na trenovacie a testovacie, potom kazdu cast na vstupy X a vystup y
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    X_train, y_train = train.drop(columns="co"), train["co"]
    X_test, y_test = test.drop(columns="co"), test["co"]
    return X_train, X_test, y_train, y_test


def scale(X_train, X_test):
    # priemer a odchylka sa pocitaju len z trenovacich dat, vystup co sa neskaluje
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def explore_data():
    # kontrolne vypisy a graf, podla ktorych som nastavil cistenie dat
    df = load_data()
    print(df.shape)
    df.info()
    print(df.describe().T.to_string())

    print("\n--- duplikaty ---")
    print("rovnake riadky:", df.duplicated().sum())
    print("opakovany instant:", df["instant"].duplicated().sum())
    df = df.drop_duplicates()

    print("\n--- kontrola stlpcov ---")
    dates = pd.to_datetime(df["date"])
    print("roznych datumov:", df["date"].nunique(), "od", dates.min().date(), "do", dates.max().date())
    print("mesiac nesedi s datumom:", (dates.dt.month != df["month"]).sum())
    print("den v tyzdni nesedi s datumom:", ((dates.dt.dayofweek + 1) % 7 != df["weekday"]).sum())   # 0 = nedela
    print(pd.crosstab(df["month"], df["season"]))

    print("\n--- outliery ---")
    for col in MEASUREMENTS + ["co"]:
        print(f"{col}: {df[col].min()} az {df[col].max()}")
    print("teplota nad 40 °C:", (df["temperature"] > MAX_TEMPERATURE).sum())
    print("mesiace s teplotou nad 40 °C:", df[df["temperature"] > MAX_TEMPERATURE]["month"].value_counts().to_dict())

    # zoradene hodnoty, hladam skok
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axes, ["co", "temperature"]):
        values = df[col].dropna().sort_values().values[-200:]
        ax.plot(values, ".")
        if col == "temperature":
            # cervena ciara = hranica z popisu, cervene body = odstranene hodnoty
            removed = values > MAX_TEMPERATURE
            ax.plot(removed.nonzero()[0], values[removed], ".", color="red")
            ax.axhline(MAX_TEMPERATURE, color="red", linestyle="--",
                       label=f"hranica {MAX_TEMPERATURE} °C, odstránených {removed.sum()}")
            ax.legend(loc="upper left")
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("poradie (200 najväčších hodnôt)")
        print(col, df[col].nlargest(15).tolist())
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "outliers_sorted.png", dpi=150)
    plt.close()

    df = remove_outliers(load_data())
    print("po odstraneni duplikatov a outlierov:", len(df))

    print("\n--- chybajuce hodnoty ---")
    missing = df.isna().sum()
    print(pd.DataFrame({"pocet": missing, "percent": (missing / len(df) * 100).round(2)})[missing > 0])
    no_measurements = df[MEASUREMENTS].isna().all(axis=1)
    print("riadky bez vsetkych merani:", no_measurements.sum(),
          "| chyba len cast merani:", (df[MEASUREMENTS].isna().any(axis=1) & ~no_measurements).sum())
    print("riadky bez co:", df["co"].isna().sum(), "| z toho aj bez merani:", (df["co"].isna() & no_measurements).sum())

    df = handle_missing(df)
    print("po odstraneni chybajucich hodnot:", df.shape, f"{len(df) / len(load_data()) * 100:.1f} % povodnych dat")
    print("stlpce po kodovani:", encode(df).columns.tolist())


if __name__ == "__main__":
    explore_data()

    print("\n--- rozdelenie a skalovanie ---")
    X_train, X_test, y_train, y_test = split_data(prepare_data())
    print("train:", X_train.shape, "test:", X_test.shape)
    X_train_scaled, X_test_scaled = scale(X_train, X_test)
    print("priemer train po skalovani:", X_train_scaled.mean(axis=0).round(2))
    print("odchylka train po skalovani:", X_train_scaled.std(axis=0).round(2))
    print("priemer test po skalovani:", X_test_scaled.mean(axis=0).round(2))
    print(f"co train: priemer {y_train.mean():.2f}, odchylka {y_train.std():.2f} | test: priemer {y_test.mean():.2f}")
