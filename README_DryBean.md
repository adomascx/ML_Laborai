# DRY BEAN A20–A23 STUDENTAMS

Duomenys parengti iš vartotojo pateiktos originalios UCI Dry Bean aibės.

Svarbu:

1. Prieš EDA patikrinti duomenų tipus ir reikšmių formatą.
2. Patikrinti missing, dublikatus, logiškai galimas ribas ir išskirtis.
3. Nešalinti statistinių išskirčių automatiškai.
4. Visus valymo sprendimus realizuoti atkuriamu programiniu kodu.

## Dry Bean – bendras duomenų aibės aprašas

Duomenys gauti iš realių sausų pupelių vaizdų.
 Viena eilutė atitinka vieną pupelės grūdą.
 Vaizdai buvo segmentuoti, o iš pupelės formos ir dydžio apskaičiuota 16 skaitinių morfologinių požymių.

**Originali UCI aibė:** <https://archive.ics.uci.edu/dataset/602/dry>
**DOI:** <https://doi.org/10.24432/C50S4B>

Originalioje aibėje yra 13 611 objektų ir 7 pupelių veislės. Originali UCI aibė neturi praleistų reikšmių;
Duomenys yra parengti duomenų kokybės ir pirminio apdorojimo analizei.

### Požymiai

- **`Area`** – Pupelės srities plotas; pikselių skaičius jos ribose. (pikseliai)
- **`Perimeter`** – Pupelės kontūro ilgis. (pikselių atstumo vienetai)
- **`MajorAxisLength`** – Ilgiausios pagrindinės ašies ilgis. (vaizdo ilgio vienetai)
- **`MinorAxisLength`** – Pagrindinei ašiai statmenos mažosios ašies ilgis. (vaizdo ilgio vienetai)
- **`AspectRation`** – Pagrindinės ir mažosios ašies ilgių santykis (UCI apraše – AspectRatio). (be mato)
- **`Eccentricity`** – Elipsės, turinčios tuos pačius momentus kaip pupelės sritis, ekscentricitetas. (0–1)
- **`ConvexArea`** – Mažiausio pupelę apgaubiančio iškilo daugiakampio plotas. (pikseliai)
- **`EquivDiameter`** – Tokio pat ploto apskritimo ekvivalentinis skersmuo. (vaizdo ilgio vienetai)
- **`Extent`** – Pupelės srities ir ją apgaubiančio stačiakampio ploto santykis. (0–1)
- **`Solidity`** – Pupelės ploto ir iškilo apvalkalo ploto santykis. (0–1)
- **`roundness`** – Apvalumo rodiklis 4πA/P². (paprastai 0–1)
- **`Compactness`** – Kompaktiškumo rodiklis, paremtas ekvivalentiniu skersmeniu ir pagrindine ašimi. (santykinis)
- **`ShapeFactor1`** – Formos deskriptorius 1. (be mato)
- **`ShapeFactor2`** – Formos deskriptorius 2. (be mato)
- **`ShapeFactor3`** – Formos deskriptorius 3. (be mato)
- **`ShapeFactor4`** – Formos deskriptorius 4. (be mato)
- **`class`** – pupelių veislė.

Prieš analizę būtina patikrinti požymių tipus, trūkstamas reikšmes, dublikatus, logines reikšmių ribas, išskirtis ir požymių tarpusavio ryšius.
Aptikta statistinė išskirtis nėra automatiškai šalinama.
