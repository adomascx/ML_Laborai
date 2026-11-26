import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors  # artimiausių kaimynų paieškai (kelio metodui)
from sklearn.cluster import DBSCAN
from sklearn.metrics import (silhouette_score, silhouette_samples, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score)
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.optimize import linear_sum_assignment  # klasterių numerių suderinimui tarp dviejų rezultatų
import umap
from itertools import combinations  # visos imčių poros stabilumo palyginimui
from datetime import date

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


# 4 ŽINGSNIS: parametrų jautrumas
EPS_DAUGIKLIAI = [0.5, 0.75, 1.0, 1.25, 1.5]  # eps = daugiklis * kelio metodu rastas eps


def EksperimentasA3(X, klases, parametrai):
    # Eksperimento tikslas - ištirti, kaip keičiasi rezultatai keičiant eps ir min_samples (užduoties 8 skaidrė)
    # Kiekvienai min_samples reikšmei eps imame aplink kelio tašką: nuo 0.5x iki 1.5x
    # Kiekvienam deriniui fiksuojame:
    #   klasterių skaičių, triukšmo dalį,
    #   silhouette (skaičiuojamas BE triukšmo taškų, nes triukšmas nėra klasteris; reikia bent 2 klasterių),
    #   ARI su tikromis klasėmis (1 - sutampa visiškai, 0 - ne geriau nei atsitiktinai)

    eilutes = []
    for p in parametrai:
        for daugiklis in EPS_DAUGIKLIAI:
            eps = p["eps"] * daugiklis
            zymes = DBSCAN(eps=eps, min_samples=p["min_samples"]).fit_predict(X)

            klasteriu_sk = len(set(zymes) - {-1})
            triuksmas = np.mean(zymes == -1)

            be_triuksmo = zymes != -1  # True tiems taškams, kurie priklauso klasteriui
            if klasteriu_sk >= 2:
                silhouette = silhouette_score(X[be_triuksmo], zymes[be_triuksmo])
            else:
                silhouette = np.nan  # su vienu klasteriu silhouette neapibrėžtas

            ari = adjusted_rand_score(klases, zymes)  # triukšmas (-1) čia laikomas atskira grupe

            eilutes.append({"min_samples": p["min_samples"], "daugiklis": daugiklis, "eps": eps,
                            "klasteriu_sk": klasteriu_sk, "triuksmas": triuksmas,
                            "silhouette": silhouette, "ARI_klases": ari})
            print(f"min_samples={p['min_samples']:>2}, eps={eps:.3f} (x{daugiklis}): "
                  f"klasterių {klasteriu_sk:>2}, triukšmas {triuksmas:6.1%}, "
                  f"silhouette {silhouette:.3f}, ARI {ari:.3f}")

    lentele = pd.DataFrame(eilutes)
    lentele.to_csv("python/results/dbscan/A3_jautrumas.csv", index=False)

    # heatmap'ai: eilutės - min_samples, stulpeliai - eps daugiklis, langelyje - mato reikšmė
    matai = {"klasteriu_sk": "Klasterių skaičius", "triuksmas": "Triukšmo dalis",
             "silhouette": "Silhouette (be triukšmo)", "ARI_klases": "ARI su klasėmis"}
    fig, asys = plt.subplots(1, len(matai), figsize=(20, 4))
    for ax, (stulpelis, pavadinimas) in zip(asys, matai.items()):
        tinklelis = lentele.pivot(index="min_samples", columns="daugiklis", values=stulpelis)
        ax.imshow(tinklelis.to_numpy(dtype=float), cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(EPS_DAUGIKLIAI)), [f"x{d}" for d in EPS_DAUGIKLIAI])
        ax.set_yticks(range(len(tinklelis.index)), tinklelis.index)
        ax.set_xlabel("eps daugiklis (x kelio taškas)")
        ax.set_ylabel("min_samples")
        ax.set_title(pavadinimas)
        # skaičiai langeliuose
        for i in range(tinklelis.shape[0]):
            for j in range(tinklelis.shape[1]):
                reiksme = tinklelis.iat[i, j]
                if pd.isna(reiksme):
                    tekstas = "-"
                elif stulpelis == "klasteriu_sk":
                    tekstas = f"{reiksme:.0f}"
                else:
                    tekstas = f"{reiksme:.2f}"
                ax.text(j, i, tekstas, ha="center", va="center", color="white")
    fig.suptitle("DBSCAN parametrų jautrumas (originali erdvė)")
    fig.tight_layout()
    fig.savefig("python/results/dbscan/figs/A3_jautrumas.png", dpi=150)
    plt.close(fig)

    return lentele


# 5 ŽINGSNIS: galutinio rezultato kokybė (vidiniai matai)
GALUTINIS_MIN_SAMPLES = 8  # pasirinkta pagal EksperimentasA3 (eps = kelio taškas, daugiklis x1.0)


def EksperimentasA4(X, klases, parametrai):
    # Eksperimento tikslas - įvertinti galutinį originalios erdvės rezultatą. Jis bus ATSKAITA,
    # su kuria vėliau lyginsime klasterizavimą sumažintose erdvėse.
    # Vidiniai matai (skaičiuojami BE triukšmo):
    #   Silhouette ↑ - kaip gerai taškas tinka savo klasteriui, palyginus su artimiausiu kitu (-1..1)
    #   Calinski-Harabasz ↑ - sklaida TARP klasterių / sklaida klasterių VIDUJE
    #   Davies-Bouldin ↓ - kiek klasteriai panašūs į artimiausią kaimyną (0 = visiškai atskirti)
    # Papildomai (turime tikras klases): ARI ir NMI su klasėmis

    eps = next(p["eps"] for p in parametrai if p["min_samples"] == GALUTINIS_MIN_SAMPLES)  # A1 kelio taškas
    zymes = DBSCAN(eps=eps, min_samples=GALUTINIS_MIN_SAMPLES).fit_predict(X)

    be_triuksmo = zymes != -1
    Xb, zb = X[be_triuksmo], zymes[be_triuksmo]

    eilute = {"erdve": "Originali", "algoritmas": "DBSCAN",
              "parametrai": f"eps={eps:.3f}, min_samples={GALUTINIS_MIN_SAMPLES}",
              "k": len(set(zb)), "triuksmas": np.mean(~be_triuksmo),
              "silhouette": silhouette_score(Xb, zb),
              "CH": calinski_harabasz_score(Xb, zb),
              "DB": davies_bouldin_score(Xb, zb),
              "ARI_klases": adjusted_rand_score(klases, zymes),
              "NMI_klases": normalized_mutual_info_score(klases, zymes),
              "data": date.today().isoformat()}

    print("\nGalutinis rezultatas (originali erdvė):")
    for raktas, reiksme in eilute.items():
        print(f"  {raktas}: {reiksme:.3f}" if isinstance(reiksme, float) else f"  {raktas}: {reiksme}")

    # silhouette kiekvienam klasteriui atskirai: parodo, kurie klasteriai gerai susiformavę, o kurie ne
    s = silhouette_samples(Xb, zb)
    print("\nSilhouette pagal klasterius:")
    for k in sorted(set(zb)):
        print(f"  klasteris {k}: {s[zb == k].mean():.3f} (taškų {np.sum(zb == k)})")

    pd.DataFrame([eilute]).to_csv("python/results/dbscan/A4_atskaita.csv", index=False)
    return zymes


# 9 ŽINGSNIS (originali erdvė): klasterių profiliai ir triukšmo analizė
def EksperimentasA5(df, pozymiai, klases, zymes):
    # Eksperimento tikslas - apibūdinti, kuo skiriasi klasteriai (užduoties 8 punktas),
    # ir patikrinti, ar triukšmas / mažas klasteris yra README minimos iškraipytos eilutės.
    # Iškraipyta eilutė - kai nesutampa formulės (README "Žinomos valymo kodo klaidos" 1 ir 2 punktai):
    #   Area turi būti = pi/4 * EquivDiameter^2
    #   MajorAxisLength turi būti = AspectRation * MinorAxisLength
    # Leidžiame 5 % paklaidą (su 1 % ir 5 % randama tiek pat - 104 eilutės, t.y. riba aiški)

    blogas_area = np.abs(df["Area"] / (np.pi / 4 * df["EquivDiameter"] ** 2) - 1) > 0.05
    blogas_major = np.abs(df["MajorAxisLength"] / (df["AspectRation"] * df["MinorAxisLength"]) - 1) > 0.05
    iskraipyta = (blogas_area | blogas_major).to_numpy()
    print(f"\nIškraipytų eilučių: {iskraipyta.sum()} (Area {blogas_area.sum()}, MajorAxisLength {blogas_major.sum()})")

    # kur pateko iškraipytos eilutės?
    lentele = pd.crosstab(zymes, iskraipyta, rownames=["klasteris"], colnames=["iškraipyta"])
    print(lentele)
    lentele.to_csv("python/results/dbscan/A5_iskraipytos.csv")

    # klasterių profiliai: medianos ORIGINALIAIS vienetais (ne standartizuotos), kad būtų galima interpretuoti
    profiliai = pozymiai.groupby(zymes).median()
    profiliai.insert(0, "tasku_sk", pd.Series(zymes).value_counts().sort_index().to_numpy())
    profiliai.index.name = "klasteris"
    print("\nKlasterių medianos (originalūs vienetai):")
    print(profiliai.round(3).T)
    profiliai.to_csv("python/results/dbscan/A5_profiliai.csv")
    return iskraipyta


# 6 ŽINGSNIS: dimensijos mažinimas
# kiekvienam metodui: 2D (vizualizacijai) + geriausia 2 laboratorinio dimensija
#   PCA 4D, MDS 4D (classical_mds pradžia) - pagal PCA_analize.py ir MDS_analize.py
#   UMAP 5D (n_neighbors=15, min_dist=0.0) - pagal umap_analize.py
# t-SNE bus pridėtas, kai grupė jį užbaigs
SEKLA = 0  # fiksuota sėkla, kad rezultatai būtų atkartojami
ERDVIU_FAILAS = "python/results/dbscan/erdves.npz"


def EksperimentasB1(X):
    # Visos sumažintos erdvės skaičiuojamos iš TOS PAČIOS paruoštos matricos X (užduoties 5 skaidrė).
    # MDS užtrunka ~1 min, todėl erdvės išsaugomos faile ir kitą kartą tiesiog nuskaitomos.
    # (jei pakeisite paruošimą ar parametrus - ištrinkite erdves.npz!)
    if os.path.exists(ERDVIU_FAILAS):
        print("\nErdvės nuskaitomos iš", ERDVIU_FAILAS)
        return dict(np.load(ERDVIU_FAILAS))

    erdves = {"Originali": X}
    for d in [2, 4]:
        erdves[f"PCA {d}D"] = PCA(n_components=d).fit_transform(X)
    for d in [2, 4]:
        print(f"Skaičiuojamas MDS {d}D...")
        erdves[f"MDS {d}D"] = MDS(n_components=d, metric_mds=True, metric="euclidean",
                                  init="classical_mds", n_init=1, random_state=SEKLA).fit_transform(X)
    for d in [2, 5]:
        erdves[f"UMAP {d}D"] = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=d,
                                         random_state=SEKLA, n_jobs=1).fit_transform(X)

    np.savez(ERDVIU_FAILAS, **erdves)
    return erdves


def poru_jaccard(a, b):
    # porų Jaccard indeksas: imame visas taškų poras.
    # J = poros, kurios kartu ABIEJUOSE rezultatuose / poros, kurios kartu BENT VIENAME
    # (sklearn jaccard_score netinka - jis lygina žymių numerius, o klasterių numeriai tarp rezultatų skiriasi)
    n = pd.crosstab(a, b).to_numpy().astype(np.int64)
    kartu_abiejuose = (n * (n - 1) // 2).sum()
    kartu_a = (n.sum(axis=1) * (n.sum(axis=1) - 1) // 2).sum()
    kartu_b = (n.sum(axis=0) * (n.sum(axis=0) - 1) // 2).sum()
    return kartu_abiejuose / (kartu_a + kartu_b - kartu_abiejuose)


def sutampantys_taskai(a, b):
    # kiek taškų yra "tame pačiame" klasteryje abiejuose rezultatuose (užduoties 10 skaidrė)
    # klasterių numeriai skiriasi (0 viename gali būti 2 kitame), todėl pirma suderiname numerius taip,
    # kad sutampančių taškų būtų kuo daugiau (linear_sum_assignment), tada sudedame
    lentele = pd.crosstab(a, b).to_numpy()
    eilutes, stulpeliai = linear_sum_assignment(-lentele)  # minuso ženklas - nes funkcija ieško minimumo
    return lentele[eilutes, stulpeliai].sum()


# 7 ŽINGSNIS: DBSCAN kiekvienoje erdvėje
def EksperimentasB2(erdves, X, klases, atskaita):
    # Eksperimento tikslas - suklasterizuoti kiekvieną erdvę ir palyginti su originalios erdvės rezultatu.
    # Parametrai kiekvienoje erdvėje parenkami IŠ NAUJO (atstumų skalė kiekvienoje erdvėje skirtinga):
    #   tas pats tinklelis kaip A3 (min_samples x eps daugiklis aplink kelio tašką),
    #   ir ta pati taisyklė pasirinkimui (BE tikrų klasių): triukšmas <= 10 %, bent 2 klasteriai,
    #   iš jų - didžiausias silhouette. Originalioje erdvėje ši taisyklė parenka tą patį (8, x1.0).

    rezultatai = []
    galutines_zymes = {}
    for pavadinimas, Y in erdves.items():
        # parametrų paieška šioje erdvėje
        paieska = []
        for min_samples in MIN_SAMPLES:
            atstumai, _ = NearestNeighbors(n_neighbors=min_samples).fit(Y).kneighbors(Y)
            atstumai = np.sort(atstumai[:, -1])
            x = np.linspace(0, 1, len(atstumai))
            y = (atstumai - atstumai[0]) / (atstumai[-1] - atstumai[0])
            eps_kelio = atstumai[np.argmax(x - y)]

            for daugiklis in EPS_DAUGIKLIAI:
                zymes = DBSCAN(eps=eps_kelio * daugiklis, min_samples=min_samples).fit_predict(Y)
                klasteriu_sk = len(set(zymes) - {-1})
                be_triuksmo = zymes != -1
                sil = silhouette_score(Y[be_triuksmo], zymes[be_triuksmo]) if klasteriu_sk >= 2 else np.nan
                paieska.append({"min_samples": min_samples, "daugiklis": daugiklis, "eps": eps_kelio * daugiklis,
                                "klasteriu_sk": klasteriu_sk, "triuksmas": np.mean(~be_triuksmo),
                                "silhouette": sil, "zymes": zymes})

        paieska_df = pd.DataFrame(paieska)
        paieska_df.drop(columns="zymes").to_csv(f"python/results/dbscan/B2_paieska_{pavadinimas}.csv", index=False)

        # pasirinkimo taisyklė
        tinkami = paieska_df[(paieska_df["triuksmas"] <= 0.10) & (paieska_df["klasteriu_sk"] >= 2)]
        geriausias = tinkami.loc[tinkami["silhouette"].idxmax()]
        zymes = geriausias["zymes"]
        galutines_zymes[pavadinimas] = zymes

        # matai
        be_triuksmo = zymes != -1
        Yb, zb = Y[be_triuksmo], zymes[be_triuksmo]
        eilute = {"erdve": pavadinimas, "algoritmas": "DBSCAN",
                  "parametrai": f"eps={geriausias['eps']:.3f}, min_samples={geriausias['min_samples']}",
                  # atskirai - kad stabilumo eksperimentas (C1) galėtų pakartoti tą patį pasirinkimą
                  "min_samples": int(geriausias["min_samples"]), "daugiklis": geriausias["daugiklis"],
                  "k": geriausias["klasteriu_sk"], "triuksmas": geriausias["triuksmas"],
                  # vidiniai matai SAVO erdvėje
                  "silhouette": silhouette_score(Yb, zb),
                  "CH": calinski_harabasz_score(Yb, zb),
                  "DB": davies_bouldin_score(Yb, zb),
                  # silhouette tų pačių žymių, bet ORIGINALIOJE erdvėje - sąžiningas palyginimas (UMAP dirbtinai jį padidina)
                  "silhouette_orig": silhouette_score(X[be_triuksmo], zb),
                  # išoriniai matai lyginant su atskaita (DBSCAN originalioje erdvėje)
                  "ARI": adjusted_rand_score(atskaita, zymes),
                  "NMI": normalized_mutual_info_score(atskaita, zymes),
                  "Jaccard": poru_jaccard(atskaita, zymes),
                  "sutampa_tasku": sutampantys_taskai(atskaita, zymes),
                  # papildomai - su tikromis klasėmis
                  "ARI_klases": adjusted_rand_score(klases, zymes),
                  "NMI_klases": normalized_mutual_info_score(klases, zymes),
                  "data": date.today().isoformat()}
        rezultatai.append(eilute)

        print(f"\n{pavadinimas}: {eilute['parametrai']} -> k={eilute['k']}, triukšmas {eilute['triuksmas']:.1%}, "
              f"silhouette {eilute['silhouette']:.3f} (orig. erdvėje {eilute['silhouette_orig']:.3f}), "
              f"ARI su atskaita {eilute['ARI']:.3f}, sutampa {eilute['sutampa_tasku']}/{len(zymes)}")
        lentele = pd.crosstab(zymes, klases, rownames=["klasteris"], colnames=["klasė"])
        print(lentele)
        lentele.to_csv(f"python/results/dbscan/B2_klasteriai_{pavadinimas}.csv")

    lentele = pd.DataFrame(rezultatai)
    lentele.to_csv("python/results/dbscan/B2_rezultatai.csv", index=False)
    return lentele, galutines_zymes


def klasteriu_grafikai(erdves, galutines_zymes):
    # vienas paveikslas su visomis erdvėmis (kaip užduoties 18 skaidrė), triukšmas - pilki taškai
    # aukštesnės dimensijos erdvės piešiamos savo 2D "sesers" koordinatėse:
    #   Originali ir PCA 4D -> PCA 2D, MDS 4D -> MDS 2D, UMAP 5D -> UMAP 2D
    piesimo_erdve = {"Originali": "PCA 2D", "PCA 2D": "PCA 2D", "PCA 4D": "PCA 2D",
                     "MDS 2D": "MDS 2D", "MDS 4D": "MDS 2D", "UMAP 2D": "UMAP 2D", "UMAP 5D": "UMAP 2D"}
    fig, asys = plt.subplots(2, 4, figsize=(20, 9))
    asys = asys.flatten()
    for ax, (pavadinimas, zymes) in zip(asys, galutines_zymes.items()):
        Y2 = erdves[piesimo_erdve[pavadinimas]]
        triuksmas = zymes == -1
        ax.scatter(Y2[triuksmas, 0], Y2[triuksmas, 1], s=4, c="lightgrey", label="triukšmas")
        ax.scatter(Y2[~triuksmas, 0], Y2[~triuksmas, 1], s=4, c=zymes[~triuksmas], cmap="tab10", vmin=0, vmax=9)
        ax.set_title(f"{pavadinimas} (piešta {piesimo_erdve[pavadinimas]})\n"
                     f"k={len(set(zymes) - {-1})}, triukšmas {triuksmas.mean():.1%}")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in asys[len(galutines_zymes):]:
        ax.axis("off")  # tušti langeliai
    fig.suptitle("DBSCAN rezultatai skirtingose erdvėse (pilki taškai - triukšmas)")
    fig.tight_layout()
    fig.savefig("python/results/dbscan/figs/B2_klasteriai.png", dpi=150)
    plt.close(fig)


# 8 ŽINGSNIS: stabilumas (užduoties 13-15 skaidrės)
IMCIU_SK = 20  # N = 20 imčių, random_state = 0..19
STABILUMO_ERDVES = ["Originali", "PCA 2D", "PCA 4D", "UMAP 2D", "UMAP 5D"]  # MDS praleistas - per lėtas (~1 min/kartą)


def sumazinti(Xs, pavadinimas, seed):
    # dimensijos mažinimas vienai imčiai - tie patys nustatymai kaip EksperimentasB1
    if pavadinimas == "Originali":
        return Xs
    metodas, dimensija = pavadinimas.split()  # "PCA 4D" -> "PCA", "4D"
    d = int(dimensija[:-1])  # "4D" -> 4
    if metodas == "PCA":
        return PCA(n_components=d).fit_transform(Xs)
    return umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=d, random_state=seed, n_jobs=1).fit_transform(Xs)


def kelio_eps(Y, min_samples):
    # tas pats kelio metodas kaip EksperimentasA1
    atstumai, _ = NearestNeighbors(n_neighbors=min_samples).fit(Y).kneighbors(Y)
    atstumai = np.sort(atstumai[:, -1])
    x = np.linspace(0, 1, len(atstumai))
    y = (atstumai - atstumai[0]) / (atstumai[-1] - atstumai[0])
    return atstumai[np.argmax(x - y)]


def EksperimentasC1(pozymiai, rezultatai):
    # Eksperimento tikslas - patikrinti, ar klasteriai išlieka tie patys, kai duomenys šiek tiek keičiasi.
    # Kiekvienai imčiai (seed = 0..N-1) VISKAS daroma iš naujo (15 skaidrė):
    #   80 % eilučių -> standartizavimas pagal imtį -> dimensijos mažinimas -> eps kelio metodu -> DBSCAN
    # (jei dimensiją sumažintume vieną kartą visiems duomenims, stabilumas būtų pervertintas)
    # min_samples ir eps daugiklis - tie patys, kuriuos B2 parinko kiekvienai erdvei.
    # Požymių rinkinys (8 požymiai) fiksuotas - koreliuotų požymių šalinimas nekartojamas.
    # Palyginimas: visos imčių poros (N*(N-1)/2 = 190), ARI/NMI/Jaccard TIK bendriems objektams (14 skaidrė)

    parinkta = rezultatai.set_index("erdve")
    poros, vidiniai = [], []

    for pavadinimas in STABILUMO_ERDVES:
        min_samples = int(parinkta.loc[pavadinimas, "min_samples"])
        daugiklis = parinkta.loc[pavadinimas, "daugiklis"]
        print(f"\nStabilumas: {pavadinimas} (min_samples={min_samples}, eps = kelio taškas x{daugiklis})")

        imciu_zymes = []  # kiekvienai imčiai: pd.Series, indeksas - originalus eilutės numeris df
        for seed in range(IMCIU_SK):
            imtis = pozymiai.sample(frac=0.8, random_state=seed)
            Xs = ((imtis - imtis.mean()) / imtis.std()).to_numpy()  # standartizuojama pagal IMTĮ
            Y = sumazinti(Xs, pavadinimas, seed)
            eps = kelio_eps(Y, min_samples) * daugiklis
            zymes = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Y)
            imciu_zymes.append(pd.Series(zymes, index=imtis.index))

            # vidiniai matai kiekvienai imčiai (be triukšmo)
            be_triuksmo = zymes != -1
            klasteriu_sk = len(set(zymes) - {-1})
            if klasteriu_sk >= 2:
                Yb, zb = Y[be_triuksmo], zymes[be_triuksmo]
                sil, ch, db = silhouette_score(Yb, zb), calinski_harabasz_score(Yb, zb), davies_bouldin_score(Yb, zb)
            else:
                sil = ch = db = np.nan
            vidiniai.append({"erdve": pavadinimas, "seed": seed, "eps": eps, "k": klasteriu_sk,
                             "triuksmas": np.mean(~be_triuksmo), "silhouette": sil, "CH": ch, "DB": db})

        # visų porų palyginimas tik bendriems objektams
        for i, j in combinations(range(IMCIU_SK), 2):
            bendri = imciu_zymes[i].index.intersection(imciu_zymes[j].index)  # objektai, esantys abiejose imtyse
            a, b = imciu_zymes[i].loc[bendri], imciu_zymes[j].loc[bendri]  # derinama pagal indeksą, ne pagal eilės nr.
            poros.append({"erdve": pavadinimas, "i": i, "j": j, "bendru": len(bendri),
                          "ARI": adjusted_rand_score(a, b), "NMI": normalized_mutual_info_score(a, b),
                          "Jaccard": poru_jaccard(a, b)})

        p = pd.DataFrame([x for x in poros if x["erdve"] == pavadinimas])
        print(f"  ARI {p['ARI'].mean():.3f} ± {p['ARI'].std():.3f} (min {p['ARI'].min():.3f}), "
              f"bendrų objektų vidutiniškai {p['bendru'].mean():.0f}")

    poros = pd.DataFrame(poros)
    vidiniai = pd.DataFrame(vidiniai)
    poros.to_csv("python/results/dbscan/C1_poros.csv", index=False)
    vidiniai.to_csv("python/results/dbscan/C1_imtys.csv", index=False)

    # suvestinė: vidurkis ± st. nuokrypis kiekvienai erdvei (15 skaidrė)
    suvestine = pd.concat([poros.groupby("erdve")[["ARI", "NMI", "Jaccard"]].agg(["mean", "std", "min"]),
                           vidiniai.groupby("erdve")[["k", "triuksmas", "silhouette", "CH", "DB"]].agg(["mean", "std"])],
                          axis=1).loc[STABILUMO_ERDVES]
    suvestine.to_csv("python/results/dbscan/C1_suvestine.csv")
    print("\n", suvestine.round(3).T)

    # 1 grafikas: ARI pasiskirstymas (boxplot) kiekvienai erdvei
    # 2 grafikas: ar imčių pakanka - ARI vidurkis naudojant pirmas n imčių (n = 5, 10, 15, 20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    ax1.boxplot([poros.loc[poros["erdve"] == e, "ARI"] for e in STABILUMO_ERDVES], tick_labels=STABILUMO_ERDVES)
    ax1.set_ylabel("ARI tarp imčių porų")
    ax1.set_title(f"Stabilumas: ARI per visas {IMCIU_SK * (IMCIU_SK - 1) // 2} imčių poras")
    for e in STABILUMO_ERDVES:
        pe = poros[poros["erdve"] == e]
        n_reiksmes = [5, 10, 15, 20]
        vidurkiai = [pe[(pe["i"] < n) & (pe["j"] < n)]["ARI"].mean() for n in n_reiksmes]
        ax2.plot(n_reiksmes, vidurkiai, marker="o", label=e)
    ax2.set_xlabel("Imčių skaičius N")
    ax2.set_ylabel("ARI vidurkis")
    ax2.set_title("Ar imčių pakanka? (vidurkis turi nusistovėti)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig("python/results/dbscan/figs/C1_stabilumas.png", dpi=150)
    plt.close(fig)

    return suvestine


parametrai = EksperimentasA1(X)
EksperimentasA2(X, klases, parametrai)
EksperimentasA3(X, klases, parametrai)
atskaita = EksperimentasA4(X, klases, parametrai)
EksperimentasA5(df, pozymiai, klases, atskaita)

erdves = EksperimentasB1(X)
rezultatai, galutines_zymes = EksperimentasB2(erdves, X, klases, atskaita)
klasteriu_grafikai(erdves, galutines_zymes)

EksperimentasC1(pozymiai, rezultatai)