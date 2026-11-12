import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, trustworthiness #algoritmų išvesčių įvertinimui
from sklearn.neighbors import NearestNeighbors #artimiausių kaimynų paieškai
from scipy.spatial.distance import pdist #atstumai tarp visų taškų porų
from scipy.stats import spearmanr #Spearmano koreliacija

#tiriami parametrai
SEKLOS = [627774, 123456, 654321, 8888, 149] #random_state, tik atsitiktinei pradžiai
DIMENSIJOS = [2, 3, 4, 5, 10] #n_components

#bendri kintamieji
SPALVOS = {"BARBUNYA": "#1f77b4", "BOMBAY": "#d62728", "CALI": "#2ca02c"} #tokios pačios, kaip umap_analize.py

#geriausi parametrai
GERIAUSIA_PRADZIA = "classical_mds" #pagal eksperimentasA1()
GERIAUSIA_DIMENSIJA = 4 #pagal eksperimentasA2(): mažiausia dimensija, kurioje visiems k T > 0.95 ir |T - C| < 0.05
TIRIAMOS_DIMENSIJOS = [2, GERIAUSIA_DIMENSIJA] #2 - vizualizacijai

#eksperimentui A1: pavadinimas -> init. Numatytasis sklearn 1.9 init - "random"
NUSTATYMAI = {"numatytieji": "random",
              "geriausi": GERIAUSIA_PRADZIA}


#PAGALBINĖS FUNKCIJOS
def projekcija(X, dimensija, pradzia=GERIAUSIA_PRADZIA, seed=0):
    #vienas metrinio MDS (SMACOF) paleidimas. Sėkla įtakoja tik atsitiktinę pradžią
    modelis = MDS(n_components=dimensija, metric_mds=True, metric="euclidean", init=pradzia, n_init=1, random_state=seed)
    return modelis.fit_transform(X)

def kaimynai_lentele(Y, k):
    #kiekvieno taško k artimiausių kaimynų eilučių numeriai (k+1, nes artimiausias yra pats taškas)
    return NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]

def neighborhood_hit(Y, pavadinimai, k):
    #kaimynystės poveikis (NH): kokia dalis kaimynų projekcijoje yra tos pačios klasės
    return (pavadinimai[kaimynai_lentele(Y, k)] == pavadinimai[:, None]).mean()

def kaimynu_sutapimas(Y1, Y2, k):
    #kokia dalis kaimynų sutampa dviejose erdvėse (1 - visi tie patys, 0 - nė vieno bendro)
    #(X, Y) - KNN kaimynų išsaugojimas, (Y_a, Y_b) - dviejų projekcijų palyginimas
    n1, n2 = kaimynai_lentele(Y1, k), kaimynai_lentele(Y2, k)
    return sum(len(set(a) & set(b)) for a, b in zip(n1, n2)) / (len(n1) * k)

def atstumu_koreliacija(Y1, Y2):
    #Spearmano koreliacija tarp visų porų atstumų: ar išliko atstumų tvarka
    return spearmanr(pdist(Y1), pdist(Y2)).statistic

def normuotas_stress(X, Y):
    #normuotoji įtempimo funkcija pagal skaidrę: sum((dX - dY)^2) / sum(dX^2), 0 - tobula
    dX, dY = pdist(X), pdist(Y)
    return ((dX - dY) ** 2).sum() / (dX ** 2).sum()

def ivertinti(X, Y, pavadinimai):
    #visos projekcijos kokybės metrikos. Klasės naudojamos tik NH
    eilute = {"spearman": atstumu_koreliacija(X, Y), "stress": normuotas_stress(X, Y)}
    for k in [5, 15, 50]:
        eilute[f"T{k}"] = trustworthiness(X, Y, n_neighbors=k)
        eilute[f"C{k}"] = trustworthiness(Y, X, n_neighbors=k)
        eilute[f"KNN{k}"] = kaimynu_sutapimas(X, Y, k)
        eilute[f"NH{k}"] = neighborhood_hit(Y, pavadinimai, k)
    return eilute

def paveiksliukogeneravimas(rinkinys, pavadinimai, titulas, failopavadinimas):
    plt.figure(figsize=(8, 6))
    for klase, spalva in SPALVOS.items():
        eilutes = pavadinimai == klase
        plt.scatter(rinkinys[eilutes, 0], rinkinys[eilutes, 1], s=5, c=spalva, label=klase)
    plt.gca().set_aspect("equal", adjustable="datalim") #ašių santykis 1:1
    plt.legend()
    plt.title(titulas)
    plt.savefig(f"python/results/mds/figs/{failopavadinimas}.png", dpi=150)
    plt.close()


def eksperimentasA1(X, pavadinimai):
    #Eksperimento tikslas - palyginti numatytuosius (atsitiktinė pradžia) ir pagrįstus (klasikinio MDS pradžia) parametrus 2D
    #atsitiktinė pradžia stochastinė, todėl paleidžiama su 5 sėklomis ir pateikiamas vidurkis ± sd
    print("\nPradedamas eksperimentas A1 - numatytieji vs geriausi parametrai")
    rezultatai = []
    for nustatymas, pradzia in NUSTATYMAI.items():
        seklos = SEKLOS if pradzia == "random" else [0]
        for seed in seklos:
            Y = projekcija(X, 2, pradzia, seed)
            rezultatai.append({"nustatymas": nustatymas, "seed": seed, **ivertinti(X, Y, pavadinimai)})
            print(nustatymas, seed)
        paveiksliukogeneravimas(Y, pavadinimai, f"MDS 2D, init: {pradzia}", f"A1_{nustatymas}")

    lentele = pd.DataFrame(rezultatai)
    lentele.to_csv("python/results/mds/mds_A1.csv", index=False)
    suvestine = lentele.drop(columns="seed").groupby("nustatymas").agg(["mean", "std"])
    suvestine.round(4).to_csv("python/results/mds/mds_A1_suvestine.csv")
    print(suvestine.round(4).T)
    print("\nEksperimentas A1 baigtas.\n")

def eksperimentasA2(X, pavadinimai):
    #Eksperimento tikslas - nustatyti tinkamiausią dimensiją
    print("\nPradedamas eksperimentas A2 - nustatoma geriausia dimensija")
    rezultatai = []
    for dimensija in DIMENSIJOS:
        Y = projekcija(X, dimensija)
        rezultatai.append({"n_components": dimensija, **ivertinti(X, Y, pavadinimai)})
        print(dimensija)

    lentele = pd.DataFrame(rezultatai)
    lentele["TC15"] = (lentele["T15"] + lentele["C15"]) / 2
    for k in [5, 15, 50]:
        lentele[f"skirtumas{k}"] = (lentele[f"T{k}"] - lentele[f"C{k}"]).abs() #gera, kai |T - C| < 0.05
    lentele.round(4).to_csv("python/results/mds/mds_dimensijos.csv", index=False)
    print(lentele.round(4))
    print("\nEksperimentas A2 baigtas.\n")

def eksperimentasBirC(X_a, X_b, X_c, pavadinimai, pavadinimai_c, liko):
    #Eksperimento tikslas - ištirti projekcijos jautrumą duomenų paruošimui
    #B: be koreliuotų požymių, objektai tie patys -> lyginami tie patys objektai prieš ir po
    #C: be išskirčių, objektų mažiau -> lyginami tik bendri objektai (Y_a[liko] su Y_c)
    print("\nPradedamas eksperimentas B ir C - projekcijos jautrumas")
    rezultatai = []
    for dimensija in TIRIAMOS_DIMENSIJOS:
        Y_a, Y_b, Y_c = projekcija(X_a, dimensija), projekcija(X_b, dimensija), projekcija(X_c, dimensija)

        eilute = {"n_components": dimensija,
                  "rho_AB": atstumu_koreliacija(Y_a, Y_b),
                  "rho_AC": atstumu_koreliacija(Y_a[liko], Y_c)}
        for k in [5, 15, 50]:
            eilute[f"sutapimas{k}_AB"] = kaimynu_sutapimas(Y_a, Y_b, k)
            eilute[f"sutapimas{k}_AC"] = kaimynu_sutapimas(Y_a[liko], Y_c, k)
        for vardas, X, Y, pav in [("A", X_a, Y_a, pavadinimai), ("B", X_b, Y_b, pavadinimai), ("C", X_c, Y_c, pavadinimai_c)]:
            for metrika, reiksme in ivertinti(X, Y, pav).items():
                eilute[f"{metrika}_{vardas}"] = reiksme
        rezultatai.append(eilute)
        print(dimensija)

        if dimensija == 2:
            paveiksliukogeneravimas(Y_b, pavadinimai, "MDS B (be koreliuotų požymių)", "B_2d")
            paveiksliukogeneravimas(Y_c, pavadinimai_c, "MDS C (be išskirčių)", "C_2d")

    lentele = pd.DataFrame(rezultatai).set_index("n_components").T #požymiai eilutėmis, kad tilptų ekrane
    lentele.round(4).to_csv("python/results/mds/mds_BC.csv")
    print(lentele.round(4))
    print("\nEksperimentas B ir C baigtas.\n")


def main():
    os.makedirs("python/results/mds/figs/", exist_ok=True)

    #nuskaitymas, tas pats kaip umap_analize.py
    df = pd.read_csv("python/results/A21_cleaned.csv")
    pavadinimai = df["class"].to_numpy()
    pozymiai = df.drop(columns=["class", "change_log"])
    X = ((pozymiai - pozymiai.mean()) / pozymiai.std()).to_numpy()

    #EKSPERIMENTAS A1 - numatytieji vs geriausi parametrai
    eksperimentasA1(X, pavadinimai)

    #EKSPERIMENTAS A2 - tinkamiausia dimensija
    eksperimentasA2(X, pavadinimai)

    #C rinkinys be išskirčių (ta pati riba kaip umap_analize.py)
    q1 = pozymiai.quantile(0.25)
    q3 = pozymiai.quantile(0.75)
    isskirtis = (pozymiai > q3 + 7.5 * (q3 - q1)).any(axis=1)
    liko = (~isskirtis).to_numpy()
    pozymiai_c = pozymiai[liko]
    pavadinimai_c = pavadinimai[liko]
    X_c = ((pozymiai_c - pozymiai_c.mean()) / pozymiai_c.std()).to_numpy()

    #B rinkinys be koreliuotų požymių (tas pats algoritmas kaip umap_analize.py)
    koreliacija = pozymiai.corr(method="spearman").abs()
    stulpeliai = list(pozymiai.columns)
    KORELIUOTI = []
    for i, pirmas in enumerate(stulpeliai):
        if pirmas in KORELIUOTI:
            continue
        for antras in stulpeliai[i + 1:]:
            if koreliacija.loc[pirmas, antras] > 0.9 and antras not in KORELIUOTI:
                KORELIUOTI.append(antras)
    pozymiai_b = pozymiai.drop(columns=KORELIUOTI)
    X_b = ((pozymiai_b - pozymiai_b.mean()) / pozymiai_b.std()).to_numpy()

    #EKSPERIMENTAS B ir C
    eksperimentasBirC(X, X_b, X_c, pavadinimai, pavadinimai_c, liko)

main()
