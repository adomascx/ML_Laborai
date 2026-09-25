# Ka dar reik mum padaryt

Claude parase tai per daug nesigilinkit i frazavima

## Kodo funkcionalumas

### Missing values

Destytojos duotas example:

- fills them in through several methods:
  - fixed values from city
  - the column mean
  - the per-industry median
  - derived values such as Profit = Revenue − Expenses

Musu:

- leaves missing values untouched and only reports how many there are.

### Outliers

Destytojos duotas example:

- applies the 1.5×IQR rule to Revenue only. It caps them with winsorization, replacing values beyond the fences with the 5th or 95th percentile, so no rows are lost.

Musu:

- applies 1.5×IQR to every numeric column. The quartiles come only from rows that passed the logical checks. It drops any row with an outlier in any column. Winsorization is not used.

### Grafinis pavaizdavimas

Destytojos duotas example:

- plots (scatter, histogram, boxplots)
- summary/head console output

Musu:

- no plots
- It writes four CSV files:
  - cleaned data,
  - per-column fence report,
  - summary counts,
  - correlation matrix.

### Scaling and normalization

Destytojos duotas example:

- z-score,
- min-max
- robust (median/IQR) scaling

Musu:

- no scaling

## Whole ass ataskaita

yep cia bus darbelio
