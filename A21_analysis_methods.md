
### Prieš analizę:
Duomenų rinkinys pirmiausia buvo paruoštas analizės veiksmams: skaitiniai požymiai konvertuoti į skaitinį formatą, o iš `Perimeter` reikšmių pašalintas matavimo vienetas `px`. 

### Loginių ribų patikrinimas
Patikrintos loginės požymių reikšmių ribos: visi skaitiniai matavimai turėjo būti teigiami, `Eccentricity`, `Extent`, `Solidity`, `roundness` ir `Compactness` reikšmės turėjo būti intervale `(0; 1]`, `AspectRation` reikšmė turėjo būti ne mažesnė už 1, `ConvexArea` negalėjo būti mažesnis už `Area`, o `MajorAxisLength` negalėjo būti mažesnis už `MinorAxisLength`; šias sąlygas pažeidusios eilutės buvo pašalintos.  

Trūkstamos reikšmės ir pasikartojančios eilutės šiame etape nebuvo šalinamos, nes jų tvarkymas darytas anksčiau minėtuose etapuose. 

### Išskirčių analizė ir pašalinimas
Statistiniai išskirtiniai stebėjimai nustatyti naudojant Tukey 1,5 × IQR taisyklę: išskirtinėmis laikytos reikšmės, mažesnės už `Q1 - 1,5 × IQR` arba didesnės už `Q3 + 1,5 × IQR`. Eilutė buvo pašalinta, jei bent viename skaitiniame požymyje reikšmė pateko už nustatytų ribų. Pradiniame duomenų rinkinyje buvo 3509 eilutės. Dėl loginių ribų pažeidimų pašalintos 87 eilutės, o kaip statistinės išskirtys identifikuotos ir pašalintos 965 eilutės. Po valymo liko 2544 eilutės. Kadangi kai kurios loginės ribos pažeidžiančios reikšmės taip pat pateko už IQR ribų, bendras pašalintų eilučių skaičius buvo 965, o ne 87 + 965. 

### Požymių tarpusavio ryšių analizė
Požymių tarpusavio ryšiai įvertinti naudojant Pirsono koreliacijos koeficientą. Nustatyti stiprūs teigiami ryšiai tarp dydį apibūdinančių požymių, pavyzdžiui, tarp `Perimeter` ir `EquivDiameter` (`r = 0,990`), bei stiprūs neigiami ryšiai tarp kai kurių formos požymių, pavyzdžiui, tarp `AspectRation` ir `ShapeFactor3` (`r = -0,992`). Tai rodo, kad dalis požymių teikia panašią arba priešingai susijusią informaciją, todėl vėlesniame modeliavimo etape gali būti svarstoma požymių atranka arba dimensijos mažinimas.