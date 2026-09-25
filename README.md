# 1 Laboratorinis Darbas

Duomenų tyrybos ir mašininio mokymosi 1-as laboratorinis darbas, naudojant "UCI Dry Bean" duomenų aibę

## Komandos nariai

- Marijus Kuprys
- Dominykas Pronskus
- Adomas Lukoševičius

## Projekto Struktūra

- [1 Laboratorinis Darbas](#1-laboratorinis-darbas)
  - [Komandos nariai](#komandos-nariai)
  - [Projekto Struktūra](#projekto-struktūra)
  - [Pradiniai duomenys - DRY BEAN A20–A23 STUDENTAMS](#pradiniai-duomenys---dry-bean-a20a23-studentams)
    - [Bendras duomenų aibės aprašas](#bendras-duomenų-aibės-aprašas)
    - [Požymiai](#požymiai)
  - [Analizės metodai](#analizės-metodai)
    - [Prieš analizę](#prieš-analizę)
    - [Loginių ribų patikrinimas](#loginių-ribų-patikrinimas)
    - [Išskirčių analizė ir pašalinimas](#išskirčių-analizė-ir-pašalinimas)
    - [Požymių tarpusavio ryšių analizė](#požymių-tarpusavio-ryšių-analizė)

## Pradiniai duomenys - DRY BEAN A20–A23 STUDENTAMS

<!-- TBD ar cia dalis dokumentacijos turi but, ar tsg mums info -->

Duomenys parengti iš vartotojo pateiktos originalios UCI Dry Bean aibės.

Svarbu:

1. Prieš EDA patikrinti duomenų tipus ir reikšmių formatą.
2. Patikrinti missing, dublikatus, logiškai galimas ribas ir išskirtis.
3. Nešalinti statistinių išskirčių automatiškai.
4. Visus valymo sprendimus realizuoti atkuriamu programiniu kodu.

### Bendras duomenų aibės aprašas

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

## Analizės metodai

### Prieš analizę

Duomenų rinkinys pirmiausia buvo paruoštas analizės veiksmams: skaitiniai požymiai konvertuoti į skaitinį formatą, o iš `Perimeter` reikšmių pašalintas matavimo vienetas `px`.

### Loginių ribų patikrinimas

Patikrintos loginės požymių reikšmių ribos: visi skaitiniai matavimai turėjo būti teigiami, `Eccentricity`, `Extent`, `Solidity`, `roundness` ir `Compactness` reikšmės turėjo būti intervale `(0; 1]`, `AspectRation` reikšmė turėjo būti ne mažesnė už 1, `ConvexArea` negalėjo būti mažesnis už `Area`, o `MajorAxisLength` negalėjo būti mažesnis už `MinorAxisLength`; šias sąlygas pažeidusios eilutės buvo pašalintos.  

Trūkstamos reikšmės ir pasikartojančios eilutės šiame etape nebuvo šalinamos, nes jų tvarkymas darytas anksčiau minėtuose etapuose.

### Išskirčių analizė ir pašalinimas

Statistiniai išskirtiniai stebėjimai nustatyti naudojant Tukey 1,5 × IQR taisyklę: išskirtinėmis laikytos reikšmės, mažesnės už `Q1 - 1,5 × IQR` arba didesnės už `Q3 + 1,5 × IQR`. Eilutė buvo pašalinta, jei bent viename skaitiniame požymyje reikšmė pateko už nustatytų ribų. Pradiniame duomenų rinkinyje buvo 3509 eilutės. Dėl loginių ribų pažeidimų pašalintos 87 eilutės, o kaip statistinės išskirtys identifikuotos ir pašalintos 965 eilutės. Po valymo liko 2544 eilutės. Kadangi kai kurios loginės ribos pažeidžiančios reikšmės taip pat pateko už IQR ribų, bendras pašalintų eilučių skaičius buvo 965, o ne 87 + 965.

### Požymių tarpusavio ryšių analizė

Požymių tarpusavio ryšiai įvertinti naudojant Pirsono koreliacijos koeficientą. Nustatyti stiprūs teigiami ryšiai tarp dydį apibūdinančių požymių, pavyzdžiui, tarp `Perimeter` ir `EquivDiameter` (`r = 0,990`), bei stiprūs neigiami ryšiai tarp kai kurių formos požymių, pavyzdžiui, tarp `AspectRation` ir `ShapeFactor3` (`r = -0,992`). Tai rodo, kad dalis požymių teikia panašią arba priešingai susijusią informaciją, todėl vėlesniame modeliavimo etape gali būti svarstoma požymių atranka arba dimensijos mažinimas.
