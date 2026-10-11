import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import PLOTS_DIR, load_data, remove_outliers, handle_missing

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NUMERIC = ["Administrative", "Administrative_Duration", "Informational", "ProductRelated",
           "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]

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

# kazdy pruh ma spolu 100 %, sivá cast su navstevy bez nakupu
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
