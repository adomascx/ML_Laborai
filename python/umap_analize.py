import umap
import pandas as pd
import os
import matplotlib.pyplot as plt

df = pd.read_csv("python/results/A21_cleaned.csv")

pavadinimai = df["class"].to_numpy() #nuskaitoma klasė, to_numpy paverčia į masyvą ["CALI","BOMBAY"...]
pozymiai = df.drop(columns=["class", "change_log"]) #atmetami neaktualūs stulpeliai

#standartizuoju visus likusius stulpelius
X = ((pozymiai - pozymiai.mean()) / pozymiai.std()).to_numpy() #formulė iš adomas.py + .to_numpy padaro, kad lentelė būtų paprastas skaičių masyvas

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
    #eksperimentas su originaliu duomenų rinkiniu (atliekamas normavimas kiekvienam rinkiniui)
    X1 = ((pozymiai - pozymiai.mean()) / pozymiai.std()).to_numpy()
    eksperimentas(X1)
    #eksperimentas su rinkiniu be išskirčių
    X2 = ((pozymiai - pozymiai.mean()) / pozymiai.std()).to_numpy()
    eksperimentas(X2)
    #eksperimentas su rinkiniu be išskirčių ir be koreliuotų požymių
    X3 = ((pozymiai - pozymiai.mean()) / pozymiai.std()).to_numpy()
    eksperimentas(X3)

def eksperimentas(rinkinys):
    #EKSPERIMENTĄ ATLIKTI SU: originaliu duomenų rinkiniu, pašalintom išskirtim, pašalintais koreliuotais požymiais
    #Pašalinus duomenis normuoti iš naujo!
    
