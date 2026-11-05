import umap
import pandas as pd
import os
import matplotlib.pyplot as plt

def eksperimentas(rinkinys):
    #EKSPERIMENTĄ ATLIKTI SU: originaliu duomenų rinkiniu, pašalintom išskirtim, pašalintais koreliuotais požymiais
    #Pašalinus duomenis normuoti iš naujo!
    modelis = umap.UMAP(n_components=2,random_state=42) # 2 dimensijos ir sėkla = 42, naudojami numatytieji n_neighbors = 15 ir min_dist = 0.1
    Y = modelis.fit_transform(X) #perdaromas grafas iš 3474x16 į 3474x2. Fit - "mokykis iš duomenų", "transform" - užpildyk grafą. Grąžinamos (x,y) koordinačių sekos.
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

    pavadinimai = df["class"].to_numpy() #nuskaitoma klasė, to_numpy paverčia į masyvą ["CALI","BOMBAY"...]
    pozymiai = df.drop(columns=["class", "change_log"]) #atmetami neaktualūs stulpeliai

    #eksperimentas su originaliu duomenų rinkiniu (atliekamas normavimas kiekvienam rinkiniui)
    X = ((pozymiai - pozymiai.mean()) / pozymiai.std()).to_numpy() #formulė iš adomas.py + .to_numpy padaro, kad lentelė būtų paprastas skaičių masyvas
    eksperimentas(X)

    #eksperimentas su rinkiniu be išskirčių
    #pasalinamos isskirtys
    q1=pozymiai.quantile(0.25)
    q3=pozymiai.quantile(0.75)
    virsutine_riba=q3+7.5*(q3-q1) #naudojamas 7.5 iqr daugiklis, nes taip išlieka daugiausia Bombay realių duomenų, o kitos išskirtys tarpe nuo 3-8 nedaug skiriasi
    
    isskirtis = (pozymiai > virsutine_riba).any(axis=1) #tikrinama kiekviena eilutė, any metodas pažymi EILUTĘ(axis 1) kaip išskirtį (priskiria prie isskirtis), jei tenkina salygą
    liko = ~isskirtis # ~ apverčia True į False, todėl pasirenkamos eilutės, kurios nėra išskirtys

    pozymiai=pozymiai[liko] #pasirenku eilutes tik tas, kurios ne išskirtys
    pavadinimai=pavadinimai[liko] 

    X = ((pozymiai - pozymiai.mean()) / pozymiai.std()).to_numpy()
    eksperimentas(X)

    #eksperimentas su rinkiniu be išskirčių ir be koreliuotų požymių
    X = ((pozymiai - pozymiai.mean()) / pozymiai.std()).to_numpy()
    eksperimentas(X)



main()
