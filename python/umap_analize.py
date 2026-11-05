import umap
import pandas as pd
import os
import matplotlib.pyplot as plt

def eksperimentas(rinkinys,pavadinimai):
    #EKSPERIMENTĄ ATLIKTI SU: originaliu duomenų rinkiniu, pašalintom išskirtim, pašalintais koreliuotais požymiais
    #Pašalinus duomenis normuoti iš naujo!
    modelis = umap.UMAP(n_components=2,random_state=42) # 2 dimensijos ir sėkla = 42, naudojami numatytieji n_neighbors = 15 ir min_dist = 0.1
    Y = modelis.fit_transform(rinkinys) #perdaromas grafas iš pvz 3474x16 į 3474x2. Fit - "mokykis iš duomenų", "transform" - užpildyk grafą/lentelę. Grąžinamos (x,y) koordinačių sekos.
    print(Y.shape) #patikriname, ar gaunasi 3474x2 (grąžina dydį: eilutės, stilpeliai)
    # išlieka eiliškumas, t.y. X[0]=Y[0]=pavadinimai[0].

    #paruošiame ir piešiame grafikus
    SPALVOS = {"BARBUNYA": "#1f77b4", "BOMBAY": "#d62728","CALI": "#2ca02c"} # tokios pačios, kaip pirmame laboratoriniame
    os.makedirs("python/results/figs", exist_ok=True) #pasidarome vietą, kurioje generuojami paveikslėliai

    plt.figure(figsize=(8, 6)) #matuojama coliais
    for klase, spalva in SPALVOS.items(): #kiekvienai kombinacijai / 3 ciklai
        eilutes = pavadinimai == klase #palygina visą masyvą su viena klase iš karto, kad pritaikytų spalvą
        plt.scatter(Y[eilutes, 0], Y[eilutes, 1], s=5, c=spalva, label=klase) #nupiešiamas taškelis ir pritaikoma spalva
    plt.legend()
    plt.title("UMAP, numatytieji parametrai")
    plt.savefig("python/results/figs/umap_default.png", dpi=150)

def main():
    #nuskaitymas
    df = pd.read_csv("python/results/A21_cleaned.csv")

    pavadinimai = df["class"].to_numpy() #nuskaitoma klasė, to_numpy paverčia į masyvą ["CALI","BOMBAY"...]. Praktiškai nuskaito kiekvienos eilutės klasę
    pozymiai = df.drop(columns=["class", "change_log"]) #atmetami neaktualūs stulpeliai

    #EKSPERIMENTAS A su originaliu duomenų rinkiniu (atliekamas normavimas kiekvienam rinkiniui)
    X = ((pozymiai - pozymiai.mean()) / pozymiai.std()).to_numpy() #formulė iš adomas.py + .to_numpy padaro, kad lentelė būtų paprastas skaičių masyvas
    eksperimentas(X,pavadinimai)

    #EKSPERIMENTAS B su rinkiniu be išskirčių
    #pasalinamos isskirtys
    q1=pozymiai.quantile(0.25)
    q3=pozymiai.quantile(0.75)
    virsutine_riba=q3+7.5*(q3-q1) #naudojamas 7.5 iqr daugiklis, nes taip išlieka daugiausia Bombay realių duomenų, nepaliekant jokių realių išskirčių iš kitų klasių. (8 jau palieka, nors tikslesnis su Bombay)
    
    isskirtis = (pozymiai > virsutine_riba).any(axis=1) #tikrinama kiekviena eilutė, any metodas pažymi EILUTĘ(axis 1) kaip išskirtį (priskiria prie isskirtis), jei tenkina salygą
    liko = ~isskirtis # ~ apverčia True į False, todėl pasirenkamos eilutės, kurios nėra išskirtys

    pozymiai_b=pozymiai[liko] #pasirenku eilutes tik tas, kurios ne išskirtys
    pavadinimai_b=pavadinimai[liko] 

    X = ((pozymiai_b - pozymiai_b.mean()) / pozymiai_b.std()).to_numpy()
    eksperimentas(X,pavadinimai_b)

    #EKSPERIMENTAS C su rinkiniu be išskirčių ir be koreliuotų požymių
    #randu koreliuotus požymius
    koreliacija = pozymiai.corr(method="spearman").abs() #aptinkame Spyrmano koreleaciją, teigiamą ar neigiamą.
    stulpeliai = list(pozymiai.columns) #apsirašau stulpelius

    KORELIUOTI = []
    for i, pirmas in enumerate(stulpeliai):
        if pirmas in KORELIUOTI:
            continue
        for antras in stulpeliai[i + 1]: #antras visad = pirmas + 1
            if koreliacija.loc[pirmas,antras] > 0.9 and antras not in KORELIUOTI: #.loc pasirenka to pačio stulpelio tas pačias eilutes palyginimui
                KORELIUOTI.append(antras)

    print(KORELIUOTI)
    #pašalinu koreliuotus požymius
    pozymiai_c = pozymiai.drop(columns=KORELIUOTI)

    X = ((pozymiai_c - pozymiai_c.mean()) / pozymiai_c.std()).to_numpy()
    eksperimentas(X, pavadinimai) #pavadinimai tie patys, nes jų skaičius nesikeičia (šalinami stulpeliai, o ne eilutės)



main()
