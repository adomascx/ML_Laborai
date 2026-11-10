import umap
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness #algoritmų išvesčių įvertinimui
from itertools import combinations #visos sėklų poros stabilumo palyginimui
from sklearn.neighbors import NearestNeighbors #artimiausių kaimynų paieškai
from scipy.spatial.distance import pdist #atstumai tarp visų taškų porų
from scipy.spatial import procrustes #dviejų projekcijų formos palyginimui
from scipy.stats import spearmanr #Spearmano koreliacija

#tiriami parametrai
SEKLOS = [627774, 123456, 654321, 8888, 149] #random_state parametras
KAIMYNAI = [5,15,30,50,100] #n_neighbors default 15
MIN_ATSTUMAI = [0.0, 0.1, 0.5] #min_dist default 0.1
DIMENSIJOS = [2,3,5,10] #n_components

#bendri kintamieji
SPALVOS = {"BARBUNYA": "#1f77b4", "BOMBAY": "#d62728","CALI": "#2ca02c"} # tokios pačios, kaip pirmame laboratoriniame

#geriausias parametrų derinys
GERIAUSI_KAIMYNAI = 15 #pagal eksperimentasA1()
GERIAUSIAS_MIN_ATSTUMAS = 0.0 #pagal eksperimentasA1()
GERIAUSIA_DIMENSIJA = 5 #pagal eksperimentasA2(). Kuo daugiau dimensiju -> tuo geresni rezultatai, tiesa, 10 buvo geresnis už 5, tačiau tik per 0.0001 TC15, kas pasirinkus reikštų, kad pridedama per daug kompleksiškumo.
#tačiau 2 dimensija, su kuria galima piešti grafikus, buvo tik per 0.0099 nuo 5 dimensijos. Todėl naudosime ją, kad galėtume piešti vienodus grafikus su kitais algoritmais.
#todel eksperimentai B ir C bus atlikti ir su 5 ir su 2
TIRIAMOS_DIMENSIJOS = [2, GERIAUSIA_DIMENSIJA]

#eksperimentui A3: pavadinimas -> (n_neighbors, min_dist)
NUSTATYMAI = {"numatytieji": (15, 0.1),
              "geriausi": (GERIAUSI_KAIMYNAI, GERIAUSIAS_MIN_ATSTUMAS)}


#PAGALBINĖS FUNKCIJOS
def projekcija(X, seed, dimensija, kaimynai=GERIAUSI_KAIMYNAI, min_atstumas=GERIAUSIAS_MIN_ATSTUMAS):
    #vienas UMAP paleidimas. Jei kaimynai ir min_atstumas nenurodyti, naudojami geriausi parametrai
    modelis = umap.UMAP(random_state=seed, n_neighbors=kaimynai, min_dist=min_atstumas,
                        n_components=dimensija, n_jobs=1)
    return modelis.fit_transform(X)

def kaimynai_lentele(Y, k):
    #kiekvieno taško k artimiausių kaimynų eilučių numeriai
    #prašome k+1, nes artimiausias kiekvienam taškui yra jis pats, o [:, 1:] tą pirmą stulpelį išmeta
    return NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]

def neighborhood_hit(Y, pavadinimai, k):
    #kaimynystės poveikis (NH): kokia dalis kaimynų projekcijoje yra tos pačios klasės, vidurkis per visus taškus
    #pavadinimai[...] kaimynų numerius pakeičia jų klasėmis, pavadinimai[:, None] - kiekvieno taško klasė stulpeliu
    return (pavadinimai[kaimynai_lentele(Y, k)] == pavadinimai[:, None]).mean()

def kaimynu_sutapimas(Y1, Y2, k):
    #kokia dalis kaimynų sutampa dviejose projekcijose (1 - visi tie patys, 0 - nė vieno bendro)
    n1, n2 = kaimynai_lentele(Y1, k), kaimynai_lentele(Y2, k)
    #set(a) & set(b) - kaimynai, esantys abiejuose sąrašuose
    return sum(len(set(a) & set(b)) for a, b in zip(n1, n2)) / (len(n1) * k)

def atstumu_koreliacija(Y1, Y2):
    #Spearmano koreliacija tarp visų porų atstumų: ar išliko atstumų tvarka
    #(X, Y) - originali erdvė prieš projekciją, (Y_a, Y_b) - dvi projekcijos tarpusavyje
    return spearmanr(pdist(Y1), pdist(Y2)).statistic

def normuotas_stress(X, Y):
    #normuotoji įtempimo funkcija pagal skaidrę: sum((dX - dY)^2) / sum(dX^2), 0 - tobula
    #UMAP koordinatės neturi mastelio, todėl projekcija pirma padauginama iš a, kuris geriausiai atitinka originalius atstumus
    dX, dY = pdist(X), pdist(Y)
    a = (dX @ dY) / (dY @ dY)
    return ((dX - a * dY) ** 2).sum() / (dX ** 2).sum()

def procrustes_skirtumas(Y1, Y2):
    #dviejų projekcijų formos skirtumas: viena pasukama, apverčiama ir keičiamas mastelis, kad atitiktų kitą
    #lieka tik tikras formos skirtumas (0 - ta pati forma). [2] - trečia grąžinama reikšmė, skirtumas
    return procrustes(Y1, Y2)[2]

def paveiksliukogeneravimas(rinkinys, pavadinimai, titulas, failopavadinimas):
    #paruošiame ir piešiame grafikus
    plt.figure(figsize=(8, 6)) #matuojama coliais
    for klase, spalva in SPALVOS.items(): #kiekvienai kombinacijai / 3 ciklai
        eilutes = pavadinimai == klase #palygina visą masyvą su viena klase iš karto, kad pritaikytų spalvą
        plt.scatter(rinkinys[eilutes, 0], rinkinys[eilutes, 1], s=5, c=spalva, label=klase) #nupiešiamas taškelis ir pritaikoma spalva
    plt.legend()
    plt.title(titulas)
    plt.savefig(f"python/results/umap/figs/{failopavadinimas}.png", dpi=150)
    plt.close()
    

def eksperimentasA1(rinkinys,pavadinimai):
    rezultatai = []
    
    #apskaičiuojamos pirmos 75 kombinacijos, iš jų padaromi piešinukai, nustatoma, kokie parametrai patikimiausi ir tęstiniausi.
    print("\nGeneruojamos A1 eksperimento kombinacijos...:")
    for seed in SEKLOS: #5
        for kaimynai in KAIMYNAI: #5x5 =25
            for min_atstumas in MIN_ATSTUMAI: # 25x3 = 75 kombinacijos
                modelis = umap.UMAP(random_state=seed, n_neighbors=kaimynai, min_dist=min_atstumas, n_components=2, n_jobs=1) #n_jobs - kad naudotų tik vieną procesoriaus branduolį (nemestų įspėjimų.)
                Y = modelis.fit_transform(rinkinys) #perdaromas grafas iš pvz 3474x16 į 3474x2. Fit - "mokykis iš duomenų", "transform" - užpildyk grafą/lentelę. Grąžinamos (x,y) koordinačių sekos.
                # išlieka eiliškumas, t.y. X[0]=Y[0]=pavadinimai[0].
                
                #įvertinimas
                eilute = {"seed": seed, "n_neighbors": kaimynai, "min_dist": min_atstumas}
                for k in [5,15,50]: #įvertiname tikrindami 5 kaimynus, tada 15 kaimynų ir t.t.
                    #f"T{k}" yra paprastas string formatas, kad skaičius užsirašytų: "T5" arba "T15"
                    eilute[f"T{k}"] = trustworthiness(rinkinys,Y,n_neighbors=k) #T - patikimumas (trustworthiness), matuojamas nuo 0-1, 1 - tobula 0 - blogiausia. Nurodo, ar galima pasitikėti grafiku, t.y. ar kaimynai 2d erdvėje yra ir kaimynai 16d erdvėje.
                    eilute[f"C{k}"] = trustworthiness(Y, rinkinys, n_neighbors=k) #C - tęstinumas (continuity), matuojamas nuo 0-1, 1 - tobula. Nurodo ar tikri kaimynai iš tikro ir liko kaimynais, ar ne.
                rezultatai.append(eilute)
                print(seed, kaimynai, min_atstumas) #labiau progreso stebėjimui, neįtakoja jokio proceso

                paveiksliukogeneravimas(Y,pavadinimai, f"UMAP Kaimynai: {kaimynai}, min_dist: {min_atstumas}, sekla: {seed}", f"A1_S{seed}_K{kaimynai}_M{min_atstumas}")

    lentele = pd.DataFrame(rezultatai) #lentelėje gauname visą įvertinimo log'ą
    lentele.to_csv("python/results/umap/umap_parametrai.csv", index=False)
    print("generavimas baigtas \n")
    print("Atliekama 1-ojo eksperimento analizė (išvedami apdoroti rezultatai)\n")
    lentele["TC15"] = (lentele["T15"] + lentele["C15"]) / 2
    for k in [5, 15, 50]:
        lentele[f"skirtumas{k}"] = (lentele[f"T{k}"] - lentele[f"C{k}"]).abs()

    #sugrupuoju visus rezultatus pagal sėklą ir išvedu vidurkius, taip randu geriausią kombinaciją.
    suvestine = lentele.drop(columns="seed").groupby(["n_neighbors", "min_dist"]).agg(["mean", "std"])
    suvestine = suvestine.sort_values(("TC15", "mean"), ascending=False)

    suvestine.round(4).to_csv("python/results/umap/umap_suvestine.csv")
    print(suvestine[["TC15"]].round(4))
    
    #gaunama, kad tiksliausi rezultatai prie min_dist 0.0, kas turbūt reiškia, kad pupelės nesiskyrė tiek daug kiek atrodė.
    #tačiau pastebime, kad min_dist pakelia C, bet nuleidžia T. Tuo tarpu n_neighbors keliant taip pat nuleidžia T, o pakelia C.
    #sėklos labai mažai daro įtakos: standartinis nuokrypis vidutiniškai mažesnis už 0,002 visose kombinacijose.
    #min_dist daro daugiau įtakos nei n_neighbors
    
    #rastas geriausias derinys: n_neighbors=15 ir min_dist=0.0, prie kurio (T15+C15)/2 = 0.9774 su 0.0005 sd (paklaida)
    #todėl šis derinys nustatytas projekto viršuje kaip konstantos
    print("Analize baigta.\n")

def eksperimentasA2(rinkinys,pavadinimai):
    print("\nPradedamas eksperimentas A2 - nustatoma geriausia dimensija")
    rezultatai = []
    #logika ta pati, kaip pirmame eksperimente
    for seed in SEKLOS:
        for dimension in DIMENSIJOS:
            modelis = umap.UMAP(random_state=seed, n_neighbors=GERIAUSI_KAIMYNAI, min_dist=GERIAUSIAS_MIN_ATSTUMAS, n_components=dimension, n_jobs=1)
            Y = modelis.fit_transform(rinkinys)

            eilute = {"seed": seed, "n_components": dimension}
            for k in [5,15,50]:
                eilute[f"T{k}"] = trustworthiness(rinkinys,Y,n_neighbors=k) 
                eilute[f"C{k}"] = trustworthiness(Y, rinkinys, n_neighbors=k)
            rezultatai.append(eilute)
            print(seed, )
            
            if(dimension == 2): paveiksliukogeneravimas(Y,pavadinimai, f"UMAP dimensija: {dimension} sekla: {seed}", f"A2_S{seed}_D{dimension}") #daugiau nei 2d negali nupiešti.
    
    lentele = pd.DataFrame(rezultatai) #lentelėje gauname visą įvertinimo log'ą
    lentele.to_csv("python/results/umap/umap_dimensijos.csv", index=False)
    print("Generavimas baigtas.\nPradedama analize.\n")

    lentele["TC15"] = (lentele["T15"]+lentele["C15"])/2
    for k in [5,15,50]:
        lentele[f"skirtumas{k}"] =(lentele[f"T{k}"] - lentele[f"C{k}"]).abs() 

    suvestine = lentele.drop(columns="seed").groupby("n_components").agg(["mean","std"])
    suvestine.round(4).to_csv("python/results/umap/umap_dimensijos_suvestine.csv")
    print(suvestine[["TC15"]].round(4))
    print("\nAnalize baigta.\nEksperimentas A2 baigtas.\n")

def eksperimentasA3(X, pavadinimai):
    #Eksperimento tikslas: 1) palyginti numatytuosius ir geriausius parametrus visomis metrikomis (T, C, NH, Spearman, stress)
    #2) įvertinti stabilumą - kiek panašios skirtingų sėklų projekcijos (Procrustes, kaimynų sutapimas). Tai ir A vs A atskaitos taškas B ir C eksperimentams
    #2 nustatymai x 2 dimensijos x 5 sėklos = 20 paleidimų
    print("\nPradedamas eksperimentas A3 - numatytieji vs geriausi parametrai ir stabilumas")
    rezultatai = []
    stabilumas = []

    for nustatymas, (kaimynai, min_atstumas) in NUSTATYMAI.items(): #išpakuojamas pavadinimas ir abu skaičiai
        for dimensija in TIRIAMOS_DIMENSIJOS:
            projekcijos = {} #kiekvienos sėklos projekcija, kad vėliau galėtume jas palyginti tarpusavyje

            for seed in SEKLOS:
                Y = projekcija(X, seed, dimensija, kaimynai, min_atstumas)
                projekcijos[seed] = Y

                eilute = {"nustatymas": nustatymas, "n_components": dimensija, "seed": seed,
                          "spearman": atstumu_koreliacija(X, Y), #ar išliko atstumų tvarka tarp 16d ir projekcijos
                          "stress": normuotas_stress(X, Y)}
                for k in [5, 15, 50]:
                    eilute[f"T{k}"] = trustworthiness(X, Y, n_neighbors=k)
                    eilute[f"C{k}"] = trustworthiness(Y, X, n_neighbors=k)
                    eilute[f"NH{k}"] = neighborhood_hit(Y, pavadinimai, k)
                rezultatai.append(eilute)
                print(nustatymas, dimensija, seed)

            #stabilumas: lyginamos visos sėklų poros (10 porų)
            for s1, s2 in combinations(SEKLOS, 2):
                eilute = {"nustatymas": nustatymas, "n_components": dimensija, "seed1": s1, "seed2": s2,
                          "procrustes": procrustes_skirtumas(projekcijos[s1], projekcijos[s2])}
                for k in [5, 15, 50]:
                    eilute[f"sutapimas{k}"] = kaimynu_sutapimas(projekcijos[s1], projekcijos[s2], k)
                stabilumas.append(eilute)

    metrikos = pd.DataFrame(rezultatai)
    metrikos.to_csv("python/results/umap/umap_A3_metrikos.csv", index=False)
    stab = pd.DataFrame(stabilumas)
    stab.to_csv("python/results/umap/umap_A3_stabilumas.csv", index=False)

    print("Generavimas baigtas.\nPradedama analize.\n")
    grupes = ["nustatymas", "n_components"]
    suvestine = metrikos.drop(columns="seed").groupby(grupes).agg(["mean", "std"])
    suvestine.round(4).to_csv("python/results/umap/umap_A3_metrikos_suvestine.csv")
    stab_suvestine = stab.drop(columns=["seed1", "seed2"]).groupby(grupes).agg(["mean", "std"])
    stab_suvestine.round(4).to_csv("python/results/umap/umap_A3_stabilumas_suvestine.csv")
    print(suvestine.round(4))
    print(stab_suvestine.round(4))
    print("\nAnalize baigta.\nEksperimentas A3 baigtas.\n")

def eksperimentasBirC(X_a, X_b, X_c, pavadinimai, pavadinimai_c, liko):
    #Eksperimento tikslas - ištirti projekcijos jautrumą duomenų paruošimui: ar struktūrą lemia koreliuoti požymiai (B) ir ekstremalios išskirtys (C)
    #B: be koreliuotų požymių, objektai tie patys -> lyginami tie patys objektai prieš ir po
    #C: be išskirčių, objektų mažiau -> lyginami tik bendri objektai (Y_a[liko] su Y_c)
    #visur naudojami geriausi parametrai, 2 dimensijos x 5 sėklos x 3 rinkiniai = 30 paleidimų
    print("\nPradedamas eksperimentas B ir C - projekcijos jautrumas")
    rezultatai = []
    baze = [] #A vs A tarp sėklų - kas yra "normalu" (atskaitos taškas B ir C palyginimui)

    for dimensija in TIRIAMOS_DIMENSIJOS:
        projekcijos_a = {} #A projekcijos atskaitos taškui

        for seed in SEKLOS:
            Y_a = projekcija(X_a, seed, dimensija)
            Y_b = projekcija(X_b, seed, dimensija)
            Y_c = projekcija(X_c, seed, dimensija)
            projekcijos_a[seed] = Y_a

            eilute = {"n_components": dimensija, "seed": seed,
                      "rho_AB": atstumu_koreliacija(Y_a, Y_b), #ar išliko atstumų tvarka po požymių pašalinimo
                      "rho_AC": atstumu_koreliacija(Y_a[liko], Y_c)} #ar išliko bendrų objektų atstumų tvarka po išskirčių pašalinimo

            for k in [5, 15, 50]:
                #kiekvieno rinkinio projekcijos kokybė lyginant su jo paties duomenimis
                for vardas, X, Y, pav in [("A", X_a, Y_a, pavadinimai), ("B", X_b, Y_b, pavadinimai), ("C", X_c, Y_c, pavadinimai_c)]:
                    eilute[f"T{k}_{vardas}"] = trustworthiness(X, Y, n_neighbors=k)
                    eilute[f"C{k}_{vardas}"] = trustworthiness(Y, X, n_neighbors=k)
                    eilute[f"NH{k}_{vardas}"] = neighborhood_hit(Y, pav, k)

                eilute[f"sutapimas{k}_AB"] = kaimynu_sutapimas(Y_a, Y_b, k) #kiek tų pačių kaimynų liko po požymių pašalinimo
                eilute[f"sutapimas{k}_AC"] = kaimynu_sutapimas(Y_a[liko], Y_c, k) #kiek tų pačių kaimynų liko bendriems objektams
            rezultatai.append(eilute)
            print(dimensija, seed)

            if dimensija == 2 and seed == SEKLOS[0]: #po vieną paveikslėlį, kitos sėklos skiriasi tik pasukimu
                paveiksliukogeneravimas(Y_b, pavadinimai, f"UMAP B (be koreliuotų požymių), sėkla: {seed}", "B_2d")
                paveiksliukogeneravimas(Y_c, pavadinimai_c, f"UMAP C (be išskirčių), sėkla: {seed}", "C_2d")

        #atskaitos taškas: A vs A tarp skirtingų sėklų, visiems objektams (B) ir tik bendriems objektams (C)
        for s1, s2 in combinations(SEKLOS, 2):
            Y1, Y2 = projekcijos_a[s1], projekcijos_a[s2]
            eilute = {"n_components": dimensija, "seed1": s1, "seed2": s2,
                      "rho_AA": atstumu_koreliacija(Y1, Y2),
                      "rho_AA_bendri": atstumu_koreliacija(Y1[liko], Y2[liko])}
            for k in [5, 15, 50]:
                eilute[f"sutapimas{k}_AA"] = kaimynu_sutapimas(Y1, Y2, k)
                eilute[f"sutapimas{k}_AA_bendri"] = kaimynu_sutapimas(Y1[liko], Y2[liko], k)
            baze.append(eilute)
        print(dimensija, "atskaitos taskas apskaiciuotas")

    lentele = pd.DataFrame(rezultatai)
    lentele.to_csv("python/results/umap/umap_BC.csv", index=False)
    baze_lentele = pd.DataFrame(baze)
    baze_lentele.to_csv("python/results/umap/umap_BC_atskaita.csv", index=False)

    print("Generavimas baigtas.\nPradedama analize.\n")
    suvestine = lentele.drop(columns="seed").groupby("n_components").agg(["mean", "std"])
    suvestine.round(4).to_csv("python/results/umap/umap_BC_suvestine.csv")
    baze_suvestine = baze_lentele.drop(columns=["seed1", "seed2"]).groupby("n_components").agg(["mean", "std"])
    baze_suvestine.round(4).to_csv("python/results/umap/umap_BC_atskaita_suvestine.csv")

    #pagrindiniai palyginimai: B ir C prieš A vs A atskaitos tašką
    palyginimas = pd.DataFrame({
        "rho_AB": lentele.groupby("n_components")["rho_AB"].mean(),
        "rho_AA": baze_lentele.groupby("n_components")["rho_AA"].mean(),
        "sutapimas15_AB": lentele.groupby("n_components")["sutapimas15_AB"].mean(),
        "sutapimas15_AA": baze_lentele.groupby("n_components")["sutapimas15_AA"].mean(),
        "rho_AC": lentele.groupby("n_components")["rho_AC"].mean(),
        "rho_AA_bendri": baze_lentele.groupby("n_components")["rho_AA_bendri"].mean(),
        "sutapimas15_AC": lentele.groupby("n_components")["sutapimas15_AC"].mean(),
        "sutapimas15_AA_bendri": baze_lentele.groupby("n_components")["sutapimas15_AA_bendri"].mean()})
    print(palyginimas.round(4))
    print("\nAnalize baigta.\nEksperimentas B ir C baigtas.\n")



def main():
    os.makedirs("python/results/umap/figs/", exist_ok=True) #pasidarome vietą, kurioje generuojami paveikslėliai

    #nuskaitymas
    df = pd.read_csv("python/results/A21_cleaned.csv")

    pavadinimai = df["class"].to_numpy() #nuskaitoma klasė, to_numpy paverčia į masyvą ["CALI","BOMBAY"...]. Praktiškai nuskaito kiekvienos eilutės klasę
    pozymiai = df.drop(columns=["class", "change_log"]) #atmetami neaktualūs stulpeliai

    #EKSPERIMENTĄ ATLIKTI SU: originaliu duomenų rinkiniu, pašalintom išskirtim, pašalintais koreliuotais požymiais
    #Pašalinus duomenis normuoti iš naujo!
    #EKSPERIMENTAS A1 su originaliu duomenų rinkiniu (atliekamas normavimas kiekvienam rinkiniui)
    #Eksperimento tikslas - atrinkti parametrus: n_neighbors, min_dist ir išbandyti kelias sėklas
    #Mano eksperimento metu tikrinami 5 n_neighbors, 3 min_dist ir 5 sėklos - 75 kombinacijos su originaliu rinkiniu
    X = ((pozymiai - pozymiai.mean()) / pozymiai.std()).to_numpy() #formulė iš adomas.py + .to_numpy padaro, kad lentelė būtų paprastas skaičių masyvas
    #eksperimentasA1(X,pavadinimai)
    #EKSPERIMENTAS A2 su originaliu duomenų rinkiniu
    #Eksperimento tikslas - ištirti, prie kurios dimensijos geriausiai veikia pasirinkti parametrai
    #Tiriamas parametras - n_components, tiriamos 4 vertės: 2,3,5,10 ir 5 sėklos, praktiškai 20 kombinacijų
    #eksperimentasA2(X,pavadinimai)  #jau atlikta, rezultatai: python/results/umap/umap_dimensijos.csv

    #EKSPERIMENTAS A3 su originaliu duomenų rinkiniu
    #Eksperimento tikslas - palyginti numatytuosius ir geriausius parametrus visomis metrikomis ir įvertinti stabilumą tarp sėklų
    #eksperimentasA3(X, pavadinimai)  #jau atlikta, rezultatai: python/results/umap/umap_A3_*.csv

    #EKSPERIMENTAS C su rinkiniu be išskirčių
    #Eksperimento tikslas - ištirti pasirinktų parametrų jautrumą (kartu su eksperimentu B), įrodyti, kad parametrai teisingi
    #Tiriami tie patys parametrai su 5 sėklom
    #pasalinamos isskirtys
    q1=pozymiai.quantile(0.25)
    q3=pozymiai.quantile(0.75)
    virsutine_riba=q3+7.5*(q3-q1) #7.5 iqr: mažiausias praradimas, kai dar sugaunamos visos 104 ekstremalios eilutės (prarandama 21 reali BOMBAY). Su 8 iqr išlieka daugiau BOMBAY, bet praleidžiamos 43 ekstremalios eilutės
    
    isskirtis = (pozymiai > virsutine_riba).any(axis=1) #tikrinama kiekviena eilutė, any metodas pažymi EILUTĘ(axis 1) kaip išskirtį (priskiria prie isskirtis), jei tenkina salygą
    liko = (~isskirtis).to_numpy() # ~ apverčia True į False, todėl pasirenkamos eilutės, kurios nėra išskirtys. to_numpy - kad tiktų indeksuoti projekcijas Y_a[liko]

    pozymiai_c=pozymiai[liko] #pasirenku eilutes tik tas, kurios ne išskirtys
    pavadinimai_c=pavadinimai[liko]

    X_c = ((pozymiai_c - pozymiai_c.mean()) / pozymiai_c.std()).to_numpy()

    #EKSPERIMENTAS B su rinkiniu be koreliuotų požymių (eilutės tos pačios)
    #randu koreliuotus požymius
    koreliacija = pozymiai.corr(method="spearman").abs() #aptinkame Spyrmano koreleaciją, teigiamą ar neigiamą.
    stulpeliai = list(pozymiai.columns) #apsirašau stulpelius

    KORELIUOTI = []
    for i, pirmas in enumerate(stulpeliai):
        if pirmas in KORELIUOTI: #praleidžiame jau pašalintą požymį
            continue
        for antras in stulpeliai[i + 1:]: #antras - visi stulpeliai po pirmo ([i + 1:] reiškia nuo i+1 iki galo)
            if koreliacija.loc[pirmas,antras] > 0.9 and antras not in KORELIUOTI: #.loc paima vieną lentelės langelį: pirmo ir antro požymio koreliaciją
                KORELIUOTI.append(antras)

    print(KORELIUOTI)
    #pašalinu koreliuotus požymius
    pozymiai_b = pozymiai.drop(columns=KORELIUOTI)

    X_b = ((pozymiai_b - pozymiai_b.mean()) / pozymiai_b.std()).to_numpy()

    #EKSPERIMENTAS B ir C: visi trys rinkiniai paleidžiami kartu, kad kiekvienai sėklai būtų galima palyginti A su B ir A su C
    #B naudoja tuos pačius pavadinimus (šalinami stulpeliai, o ne eilutės), C - atfiltruotus pavadinimai_c
    eksperimentasBirC(X, X_b, X_c, pavadinimai, pavadinimai_c, liko)

main()
