import pandas as pd

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
