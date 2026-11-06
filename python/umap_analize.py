import umap
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness #algoritmų išvesčių įvertinimui

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
GERIAUSIA_DIMENSIJA = 0 #pagal eksperimentasA2()


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

def analizuotiA1(lentele):
    print("Atliekama 1-ojo eksperimento analizė (išvedami apdoroti rezultatai)\n")
    lentele = pd.read_csv("python/results/umap/umap_parametrai.csv")
    #sugrupuoju visus rezultatus pagal sėklą ir išvedu vidurkius, taip randu geriausią kombinaciją.

    lentele["TC15"] = (lentele["T15"] + lentele["C15"]) / 2
    for k in [5, 15, 50]:
        lentele[f"skirtumas{k}"] = (lentele[f"T{k}"] - lentele[f"C{k}"]).abs()

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
    analizuotiA1(lentele)

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
        lentele[f"skirtumas[k]"] = (lentele[f"T{k}"] - lentele[f"C{k}"]).abs() 

    suvestine = lentele.drop(columns="seed").groupby("n_components").agg(["mean","std"])
    suvestine.round(4).to_csv("python/results/umap/umap_dimensijos_suvestine.csv")
    print(suvestine[["TC15"]].round(4))
    print("\nAnalize baigta.\nEksperimentas A2 baigtas.")

def eksperimentasBirC(rinkinys,pavadinimai):
    print("hi")



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
    eksperimentasA2(X,pavadinimai)

    #EKSPERIMENTAS B su rinkiniu be išskirčių
    #Eksperimento tikslas - ištirti pasirinktų parametrų jautrumą (kartu su eksperimentu C), įrodyti, kad parametrai teisingi
    #Tiriami tie patys parametrai su 5 sėklom
    #pasalinamos isskirtys
    q1=pozymiai.quantile(0.25)
    q3=pozymiai.quantile(0.75)
    virsutine_riba=q3+7.5*(q3-q1) #naudojamas 7.5 iqr daugiklis, nes taip išlieka daugiausia Bombay realių duomenų, nepaliekant jokių realių išskirčių iš kitų klasių. (8 jau palieka, nors tikslesnis su Bombay)
    
    isskirtis = (pozymiai > virsutine_riba).any(axis=1) #tikrinama kiekviena eilutė, any metodas pažymi EILUTĘ(axis 1) kaip išskirtį (priskiria prie isskirtis), jei tenkina salygą
    liko = ~isskirtis # ~ apverčia True į False, todėl pasirenkamos eilutės, kurios nėra išskirtys

    pozymiai_b=pozymiai[liko] #pasirenku eilutes tik tas, kurios ne išskirtys
    pavadinimai_b=pavadinimai[liko] 

    X = ((pozymiai_b - pozymiai_b.mean()) / pozymiai_b.std()).to_numpy()
    eksperimentasBirC(X,pavadinimai_b)

    #EKSPERIMENTAS C su rinkiniu be išskirčių ir be koreliuotų požymių
    #randu koreliuotus požymius
    koreliacija = pozymiai.corr(method="spearman").abs() #aptinkame Spyrmano koreleaciją, teigiamą ar neigiamą.
    stulpeliai = list(pozymiai.columns) #apsirašau stulpelius

    KORELIUOTI = []
    for i, pirmas in enumerate(stulpeliai):
        if pirmas in KORELIUOTI: #praleidžiame jau pašalintą požymį
            continue
        for antras in stulpeliai[i + 1:]: #antras visad = pirmas + 1 (: padaro, kad imtų visą stulpelį)
            if koreliacija.loc[pirmas,antras] > 0.9 and antras not in KORELIUOTI: #.loc pasirenka to pačio stulpelio tas pačias eilutes palyginimui
                KORELIUOTI.append(antras)

    print(KORELIUOTI)
    #pašalinu koreliuotus požymius
    pozymiai_c = pozymiai.drop(columns=KORELIUOTI)

    X = ((pozymiai_c - pozymiai_c.mean()) / pozymiai_c.std()).to_numpy()
    eksperimentasBirC(X, pavadinimai) #pavadinimai tie patys, nes jų skaičius nesikeičia (šalinami stulpeliai, o ne eilutės)

main()
