import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import PLOTS_DIR, add_title, load_data, remove_outliers

SEASONS = ["winter", "spring", "summer", "autumn"]


if __name__ == "__main__":
    # data bez duplikatov a outlierov, este s chybajucimi hodnotami a so slovnym rocnym obdobim
    df = remove_outliers(load_data())
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # 1. co podla hodiny, pracovny den a vikend (0 = nedela, 6 = sobota)
    weekend = df["weekday"].isin([0, 6]).map({False: "pracovný deň", True: "víkend"})
    by_hour = df.groupby([df["hour"], weekend])["co"].mean().unstack()
    print(by_hour.round(2).to_string())
    by_hour.plot(ax=axes[0, 0], marker="o", markersize=3)
    axes[0, 0].set_title("1. Priemerné CO podľa hodiny dňa")
    axes[0, 0].set_xlabel("hodina")
    axes[0, 0].set_ylabel("CO (mg/m³)")
    axes[0, 0].legend(title="")

    # 2. rozdelenie co podla mesiaca
    print(df.groupby("month")["co"].median().round(2).to_string())
    sns.boxplot(data=df, x="month", y="co", color="lightsteelblue", fliersize=1.5, ax=axes[0, 1])
    axes[0, 1].set_title("2. Rozdelenie CO podľa mesiaca")
    axes[0, 1].set_xlabel("mesiac")
    axes[0, 1].set_ylabel("CO (mg/m³)")

    # 3. teplota a vlhkost, farba = rocne obdobie
    print("korelacia teplota a vlhkost:", round(df["temperature"].corr(df["humidity"]), 2))
    sns.scatterplot(data=df, x="temperature", y="humidity", hue="season", hue_order=SEASONS,
                    s=5, alpha=0.4, linewidth=0, ax=axes[0, 2])
    axes[0, 2].set_title("3. Teplota a vlhkosť podľa ročného obdobia")
    axes[0, 2].set_xlabel("teplota (°C)")
    axes[0, 2].set_ylabel("relatívna vlhkosť (%)")
    axes[0, 2].legend(title="obdobie", markerscale=3)

    # 4. podiel hodin bez nameraneho co v kazdom mesiaci
    missing = df["co"].isna().groupby(df["month"]).mean() * 100
    print(missing.round(1).to_string())
    axes[1, 0].bar(missing.index, missing.values, color="indianred")
    axes[1, 0].set_xticks(missing.index)
    axes[1, 0].set_title("4. Hodiny bez nameranej hodnoty CO")
    axes[1, 0].set_xlabel("mesiac")
    axes[1, 0].set_ylabel("podiel hodín (%)")

    # 5. odozva senzora no2 a teplota
    print("korelacia sensor_no2 a teplota:", round(df["sensor_no2"].corr(df["temperature"]), 2))
    axes[1, 1].hexbin(df["temperature"], df["sensor_no2"], gridsize=40, cmap="Blues", mincnt=1)
    axes[1, 1].set_title("5. Odozva senzora NO2 a teplota")
    axes[1, 1].set_xlabel("teplota (°C)")
    axes[1, 1].set_ylabel("sensor_no2")

    axes[1, 2].axis("off")
    add_title(fig, "EDA: päť vzťahov v dátach")
    plt.savefig(PLOTS_DIR / "eda.png", dpi=150)
    plt.close()
