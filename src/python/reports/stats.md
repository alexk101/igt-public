# Statistics

## Summary

### ANOVA 1

Final Score as a function of winners vs losers did not prove significant

The main effect, winners vs losers, was insignificant, F(1, 107) = 1.54, p = 0.22.

### RM-ANOVA 1

Score within trial proved significant

The within factor, trial, was significant, F(99, 10692) = 2.34, p = 1.62e-12.

### MIXED ANOVA 1

Score as a function of winners vs losers within trial, the interaction proved significant

The main effect, winners vs losers, was insignificant, F(1, 107) = 1.08, p = 0.3.

The within factor, trial, was significant, F(99, 10593) = 2.35, p = 1.21e-12.

The interaction was significant, F(99, 10593) = 1.47, p = 1.76e-3.

### RM-ANOVA 2

Score within quintile proved insignificant

The main effect was insignificant, F(4, 432) = 1.81, p = 0.12.

### MIXED ANOVA 2

Score as a function of winners vs losers within quintile, nothing proved significant

The main effect, winners vs losers, was insignificant, F(1, 107) = 1.08, p = 0.3.

The within factor, trial, was insignificant, F(4, 428) = 1.82, p = 0.12.

The interaction was insignificant, F(4, 428) = 1.71, p = 0.15.

## Model Results

### ANOVA 1

dv=final score

iv=win/loss_2

|    | Source     |          SS |   DF |          MS |         F |          p |         np2 |
|---:|:-----------|------------:|-----:|------------:|----------:|-----------:|------------:|
|  0 | win/loss_2 | 4.27468e+06 |    1 | 4.27468e+06 |   1.54781 |   0.216177 |   0.0142593 |
|  1 | Within     | 2.95507e+08 |  107 | 2.76175e+06 | nan       | nan        | nan         |

### RM-ANOVA 1

dv=score

within=trial

|    | Source   |          SS |    DF |               MS |         F |             p |   p-GG-corr |         ng2 |
|---:|:---------|------------:|------:|-----------------:|----------:|--------------:|------------:|------------:|
|  0 | trial    | 1.23185e+08 |    99 |      1.24429e+06 |   2.34462 |   1.62207e-12 |    0.076931 |   0.0084271 |
|  1 | Error    | 5.67428e+09 | 10692 | 530703           | nan       | nan           |  nan        | nan         |

### MIXED ANOVA 1

dv=score

within=trial

between=win/loss_2

|    | Source      |          SS |   DF1 |   DF2 |               MS |       F |          p |   p-GG-corr |       np2 |
|---:|:------------|------------:|------:|------:|-----------------:|--------:|-----------:|------------:|----------:|
|  0 | win/loss_2  | 8.83519e+07 |     1 |   107 |      8.83519e+07 | 1.08265 | 0.30045    |  nan        | 0.0100169 |
|  1 | trial       | 1.23185e+08 |    99 | 10593 |      1.24429e+06 | 2.35479 | 1.218e-12  |    0.076931 | 0.0215335 |
|  2 | Interaction | 7.684e+07   |    99 | 10593 | 776161           | 1.46886 | 0.00176271 |  nan        | 0.0135418 |

### RM-ANOVA 2

dv=score

within=quintile

|    | Source   |          SS |   DF |     MS |         F |          p |   p-GG-corr |          ng2 |
|---:|:---------|------------:|-----:|-------:|----------:|-----------:|------------:|-------------:|
|  0 | quintile | 3.61704e+06 |    4 | 904259 |   1.81454 |   0.124979 |     0.16988 |   0.00548107 |
|  1 | Error    | 2.15284e+08 |  432 | 498342 | nan       | nan        |   nan       | nan          |

### MIXED ANOVA 2

dv=score

within=quintile

between=win/loss_2

|    | Source      |          SS |   DF1 |   DF2 |              MS |       F |        p |   p-GG-corr |       np2 |
|---:|:------------|------------:|------:|------:|----------------:|--------:|---------:|------------:|----------:|
|  0 | win/loss_2  | 4.4176e+06  |     1 |   107 |      4.4176e+06 | 1.08265 | 0.30045  |   nan       | 0.0100169 |
|  1 | quintile    | 3.61704e+06 |     4 |   428 | 904259          | 1.82647 | 0.122717 |     0.16988 | 0.0167833 |
|  2 | Interaction | 3.38638e+06 |     4 |   428 | 846595          | 1.70999 | 0.146746 |   nan       | 0.0157299 |