import umap
import pandas as pd
import numpy

df = pd.read_csv("python/results/A21_cleaned.csv")

pavadinimai = df["class"].to_numpy() #nuskaitoma klasė, to_numpy paverčia į masyvą ["CALI","BOMBAY"...]
pozymiai = df.drop(columns=["class", "change_log"]) #atmetami neaktualūs stulpeliai

#standartizuoju visus likusius stulpelius
X = ((pozymiai - pozymiai.mean()) / pozymiai.std()).to_numpy() #formulė iš adomas.py + .to_numpy padaro, kad lentelė būtų paprastas skaičių masyvas

modelis = umap.UMAP(n_components=2,random_state=42) # 2 dimensijos ir sėkla = 42, naudojami numatytieji n_neighbors = 15 ir min_dist = 0.1
Y = modelis.fit_transform(X)
print(Y.shape)    