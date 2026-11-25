import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors  # artimiausių kaimynų paieškai (kelio metodui)
from sklearn.cluster import DBSCAN

os.makedirs("python/results/dbscan/figs/", exist_ok=True)  # vieta rezultatams ir paveikslėliams

# tiriamos min_samples reikšmės: taisyklė ~2 * požymių skaičius (8 požymiai -> 16), plius mažesnės palyginimui
MIN_SAMPLES = [4, 8, 16]

# 1 ŽINGSNIS: duomenų nuskaitymas ir paruošimas

# nuskaitome išvalytus duomenis (tie patys, kaip 2 laboratoriniame)
df = pd.read_csv("python/results/A21_cleaned.csv")

# klasė NEnaudojama klasterizavimui, ją pasidedame atskirai - vėliau palyginsime klasterius su tikromis klasėmis
klases = df["class"].to_numpy()

# paliekame tik skaitinius požymius
pozymiai = df.drop(columns=["class", "change_log"])

# šaliname stipriai koreliuotus požymius (|r| > 0.9), tas pats ciklas kaip umap_analize.py
koreliacija = pozymiai.corr(method="spearman").abs()
stulpeliai = list(pozymiai.columns)

koreliuoti = []
for i, pirmas in enumerate(stulpeliai):
    if pirmas in koreliuoti:  # jau pašalintas požymis nieko nebešalina
        continue
    for antras in stulpeliai[i + 1:]:
        if koreliacija.loc[pirmas, antras] > 0.9 and antras not in koreliuoti:
            koreliuoti.append(antras)

pozymiai = pozymiai.drop(columns=koreliuoti)

# standartizavimas: (reikšmė - vidurkis) / standartinis nuokrypis, ta pati formulė kaip umap_analize.py
# .to_numpy paverčia lentelę paprastu skaičių masyvu
X = ((pozymiai - pozymiai.mean()) / pozymiai.std()).to_numpy()

print("Objektų skaičius:", X.shape[0])
print("Pašalinti požymiai:", koreliuoti)
print("Likę požymiai:", list(pozymiai.columns))
print("Klasės:", pd.Series(klases).value_counts().to_dict())


# 2 ŽINGSNIS: eps parinkimas kelio (knee) metodu
def EksperimentasA1(X):
    # Eksperimento tikslas - kiekvienai min_samples reikšmei rasti tinkamą eps
    # Idėja: kiekvienam taškui randame atstumą iki min_samples-ojo kaimyno ir surūšiuojame.
    # Tankiose vietose atstumai maži (kreivė plokščia), atskirtuose taškuose - staigiai išauga.
    # Vieta, kur kreivė "lūžta" (kelias), yra geras eps: taškai arčiau už jį laikomi kaimynais.

    rezultatai = []
    fig, asys = plt.subplots(1, len(MIN_SAMPLES), figsize=(15, 4))

    for ax, min_samples in zip(asys, MIN_SAMPLES):
        # n_neighbors=min_samples, nes pirmasis rastas "kaimynas" yra pats taškas,
        # o DBSCAN min_samples taip pat įskaičiuoja patį tašką
        atstumai, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
        atstumai = np.sort(atstumai[:, -1])  # paskutinis stulpelis - atstumas iki tolimiausio iš min_samples kaimynų

        # kelio taškas: abi ašis sunormuojame į [0;1], tada tiesė tarp kreivės galų yra y = x.
        # Kelias - taškas, kuris labiausiai nutolęs žemyn nuo tos tiesės, t.y. didžiausias (x - y)
        x = np.linspace(0, 1, len(atstumai))
        y = (atstumai - atstumai[0]) / (atstumai[-1] - atstumai[0])
        kelias = np.argmax(x - y)
        eps = atstumai[kelias]

        rezultatai.append({"min_samples": min_samples, "eps": eps})
        print(f"min_samples={min_samples}: eps={eps:.3f}")

        ax.plot(atstumai)
        ax.axhline(eps, linestyle="--", color="grey")
        ax.scatter(kelias, eps, color="red", zorder=3)  # zorder - kad taškas būtų virš linijos
        ax.set_title(f"min_samples={min_samples}, eps={eps:.3f}")
        ax.set_xlabel("Taškai, surūšiuoti pagal atstumą")
        ax.set_ylabel(f"Atstumas iki {min_samples}-ojo kaimyno")

    fig.suptitle("Kelio metodas: eps parinkimas (originali erdvė)")
    fig.tight_layout()
    fig.savefig("python/results/dbscan/figs/A1_kelio_metodas.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(rezultatai).to_csv("python/results/dbscan/A1_eps.csv", index=False)
    return rezultatai


# 3 ŽINGSNIS: DBSCAN paleidimas originalioje erdvėje
def EksperimentasA2(X, klases, parametrai):
    # Eksperimento tikslas - su A1 rastais (min_samples, eps) paleisti DBSCAN ir pažiūrėti:
    # kiek klasterių susidarė, kiek taškų pažymėta triukšmu ir kurios klasės pateko į kuriuos klasterius
    # DBSCAN kiekvienam taškui grąžina klasterio numerį (0, 1, 2, ...) arba -1, jei taškas yra triukšmas

    suvestine = []
    for p in parametrai:
        zymes = DBSCAN(eps=p["eps"], min_samples=p["min_samples"]).fit_predict(X)

        klasteriu_sk = len(set(zymes) - {-1})  # skirtingi numeriai, išskyrus triukšmą
        triuksmas = np.mean(zymes == -1)  # triukšmo taškų dalis (True=1, False=0, vidurkis = dalis)

        print(f"\nmin_samples={p['min_samples']}, eps={p['eps']:.3f}: "
              f"klasterių {klasteriu_sk}, triukšmas {triuksmas:.1%}")

        # kryžminė lentelė: eilutės - DBSCAN klasteriai, stulpeliai - tikros klasės
        lentele = pd.crosstab(zymes, klases, rownames=["klasteris"], colnames=["klasė"])
        print(lentele)
        lentele.to_csv(f"python/results/dbscan/A2_klasteriai_ms{p['min_samples']}.csv")

        suvestine.append({"min_samples": p["min_samples"], "eps": p["eps"],
                          "klasteriu_sk": klasteriu_sk, "triuksmas": triuksmas})

    pd.DataFrame(suvestine).to_csv("python/results/dbscan/A2_suvestine.csv", index=False)
    return suvestine


parametrai = EksperimentasA1(X)
EksperimentasA2(X, klases, parametrai)