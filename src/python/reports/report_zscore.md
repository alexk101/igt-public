# Learners/Non-learners Statistics (z-scored within subject)
dv='pupil'

between=['learners/non-learners']

within=['decile, 'bin']

## ANOVA Pupil Change Scores
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/ad_pupil/anova_bin_baseline.png)  |  ![](./r_stats/zscore/ad_pupil/anova_decile_baseline.png)

Significant Effects/Interactions

- bin

| Effect                             |   num Df |   den Df |     MSE |          F |        ges |           P |
|:-----------------------------------|---------:|---------:|--------:|-----------:|-----------:|------------:|
| `learners/non-learners`            |  1       |   89     | 3.87039 |   1.60767  | 0.0018106  | 0.208127    |
| decile                             |  6.89363 |  613.533 | 2.54052 |   0.933637 | 0.00474398 | 0.478747    |
| `learners/non-learners`:decile     |  6.89363 |  613.533 | 2.54052 |   0.984592 | 0.00500159 | 0.440668    |
| bin                                |  3.73411 |  332.336 | 1.71216 | 121.932    | 0.185172   | 2.97019e-61 |
| `learners/non-learners`:bin        |  3.73411 |  332.336 | 1.71216 |   0.638675 | 0.00118893 | 0.624506    |
| decile:bin                         | 99       | 8811     | 0.10875 |   1.08275  | 0.00338673 | 0.270984    |
| `learners/non-learners`:decile:bin | 99       | 8811     | 0.10875 |   1.08283  | 0.00338696 | 0.270827    |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/ad_pupil/anova_bin_planning.png)  |  ![](./r_stats/zscore/ad_pupil/anova_decile_planning.png)

Significant Effects/Interactions

- bin

| Effect                             |   num Df |   den Df |       MSE |           F |         ges |           P |
|:-----------------------------------|---------:|---------:|----------:|------------:|------------:|------------:|
| `learners/non-learners`            |  1       |   94     | 2.54226   |   0.0129578 | 4.70176e-05 | 0.909613    |
| decile                             |  7.54693 |  709.412 | 0.142086  |   1.71131   | 0.00261243  | 0.0970631   |
| `learners/non-learners`:decile     |  7.54693 |  709.412 | 0.142086  |   1.37644   | 0.00210229  | 0.207224    |
| bin                                |  1.50506 |  141.476 | 1.72952   | 327.531     | 0.548925    | 2.40538e-47 |
| `learners/non-learners`:bin        |  1.50506 |  141.476 | 1.72952   |   0.250287  | 0.000929068 | 0.714619    |
| decile:bin                         | 16.3734  | 1539.1   | 0.0754625 |   0.694245  | 0.00122287  | 0.805603    |
| `learners/non-learners`:decile:bin | 16.3734  | 1539.1   | 0.0754625 |   1.13384   | 0.00199564  | 0.316096    |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/ad_pupil/anova_bin_feedback.png)  |  ![](./r_stats/zscore/ad_pupil/anova_decile_feedback.png)

Significant Effects/Interactions

- bin
- decile:bin

| Effect                             |   num Df |   den Df |      MSE |         F |        ges |           P |
|:-----------------------------------|---------:|---------:|---------:|----------:|-----------:|------------:|
| `learners/non-learners`            |  1       |   94     | 2.762    |  0.590798 | 0.00182497 | 0.444038    |
| decile                             |  5.79485 |  544.716 | 0.683701 |  1.52503  | 0.00672423 | 0.170119    |
| `learners/non-learners`:decile     |  5.79485 |  544.716 | 0.683701 |  0.594263 | 0.00263105 | 0.729197    |
| bin                                |  2.77782 |  261.115 | 0.287061 | 41.9286   | 0.036108   | 5.84077e-21 |
| `learners/non-learners`:bin        |  2.77782 |  261.115 | 0.287061 |  2.54718  | 0.00227059 | 0.0609643   |
| decile:bin                         | 16.7513  | 1574.62  | 0.117808 |  1.76917  | 0.00389658 | 0.0276016   |
| `learners/non-learners`:decile:bin | 16.7513  | 1574.62  | 0.117808 |  0.966186 | 0.00213179 | 0.493802    |
## ANOVA Raw Pupil
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/pupil/anova_bin_baseline.png)  |  ![](./r_stats/zscore/pupil/anova_decile_baseline.png)

Significant Effects/Interactions

- decile
- bin
- decile:bin

| Effect                             |   num Df |   den Df |       MSE |          F |         ges |           P |
|:-----------------------------------|---------:|---------:|----------:|-----------:|------------:|------------:|
| `learners/non-learners`            |  1       |   89     | 3.51392   |   0.108202 | 0.000158276 | 0.742973    |
| decile                             |  3.934   |  350.126 | 2.95805   |  30.446    | 0.128549    | 1.73966e-21 |
| `learners/non-learners`:decile     |  3.934   |  350.126 | 2.95805   |   0.836118 | 0.00403466  | 0.501297    |
| bin                                |  2.44686 |  217.77  | 2.25145   | 237.505    | 0.352647    | 9.98506e-62 |
| `learners/non-learners`:bin        |  2.44686 |  217.77  | 2.25145   |   1.9617   | 0.00447929  | 0.132797    |
| decile:bin                         | 99       | 8811     | 0.0639107 |   1.31071  | 0.00344091  | 0.0214046   |
| `learners/non-learners`:decile:bin | 99       | 8811     | 0.0639107 |   1.24018  | 0.00325636  | 0.0540969   |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/pupil/anova_bin_planning.png)  |  ![](./r_stats/zscore/pupil/anova_decile_planning.png)

Significant Effects/Interactions

- decile
- bin

| Effect                             |   num Df |   den Df |      MSE |          F |        ges |           P |
|:-----------------------------------|---------:|---------:|---------:|-----------:|-----------:|------------:|
| `learners/non-learners`            |  1       |   94     | 1.60419  |   2.31795  | 0.00252975 | 0.131246    |
| decile                             |  3.95154 |  371.445 | 1.9139   |  22.5948   | 0.104384   | 1.44507e-16 |
| `learners/non-learners`:decile     |  3.95154 |  371.445 | 1.9139   |   1.45417  | 0.00744513 | 0.21618     |
| bin                                |  1.97361 |  185.519 | 1.90971  | 451.409    | 0.537128   | 1.38383e-71 |
| `learners/non-learners`:bin        |  1.97361 |  185.519 | 1.90971  |   0.890539 | 0.00228405 | 0.410997    |
| decile:bin                         | 18.9151  | 1778.02  | 0.140703 |   1.01404  | 0.00183734 | 0.440554    |
| `learners/non-learners`:decile:bin | 18.9151  | 1778.02  | 0.140703 |   1.28179  | 0.00232135 | 0.185145    |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/pupil/anova_bin_feedback.png)  |  ![](./r_stats/zscore/pupil/anova_decile_feedback.png)

Significant Effects/Interactions

- `learners/non-learners`
- decile
- bin
- decile:bin

| Effect                             |   num Df |   den Df |       MSE |        F |        ges |           P |
|:-----------------------------------|---------:|---------:|----------:|---------:|-----------:|------------:|
| `learners/non-learners`            |  1       |   94     | 1.99796   |  4.76645 | 0.00937795 | 0.0315096   |
| decile                             |  3.93973 |  370.334 | 1.37843   | 44.491   | 0.193666   | 3.72294e-30 |
| `learners/non-learners`:decile     |  3.93973 |  370.334 | 1.37843   |  1.53132 | 0.00819893 | 0.193234    |
| bin                                |  2.37519 |  223.267 | 0.566312  | 45.0001  | 0.0567554  | 2.31481e-19 |
| `learners/non-learners`:bin        |  2.37519 |  223.267 | 0.566312  |  1.46001 | 0.00194841 | 0.232071    |
| decile:bin                         | 21.1399  | 1987.15  | 0.0912062 |  1.84487 | 0.00352352 | 0.0108972   |
| `learners/non-learners`:decile:bin | 21.1399  | 1987.15  | 0.0912062 |  1.34901 | 0.00257891 | 0.132199    |
## ANOVA Respiratory Sinus Arythmia
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/rsa/anova_bin_baseline.png)  |  ![](./r_stats/zscore/rsa/anova_decile_baseline.png)

Significant Effects/Interactions

- decile

| Effect                             |   num Df |   den Df |        MSE |        F |         ges |         P |
|:-----------------------------------|---------:|---------:|-----------:|---------:|------------:|----------:|
| `learners/non-learners`            |  1       |   74     | 0.0702121  | 1.48534  | 5.83956e-05 | 0.226811  |
| decile                             |  5.64411 |  417.664 | 4.21644    | 2.24424  | 0.0290389   | 0.0418272 |
| `learners/non-learners`:decile     |  5.64411 |  417.664 | 4.21644    | 1.48061  | 0.0193493   | 0.187303  |
| bin                                |  1.54551 |  114.367 | 0.0204167  | 1.87762  | 3.31755e-05 | 0.166468  |
| `learners/non-learners`:bin        |  1.54551 |  114.367 | 0.0204167  | 0.717155 | 1.26716e-05 | 0.456384  |
| decile:bin                         | 99       | 7326     | 0.00234954 | 0.871821 | 0.000113544 | 0.813464  |
| `learners/non-learners`:decile:bin | 99       | 7326     | 0.00234954 | 1.18128  | 0.00015384  | 0.107065  |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/rsa/anova_bin_planning.png)  |  ![](./r_stats/zscore/rsa/anova_decile_planning.png)

Significant Effects/Interactions

No statistically significant effects present at P < 0.05

| Effect                             |   num Df |   den Df |        MSE |        F |         ges |         P |
|:-----------------------------------|---------:|---------:|-----------:|---------:|------------:|----------:|
| `learners/non-learners`            |  1       |   74     | 0.0498046  | 0.459197 | 2.56391e-05 | 0.500113  |
| decile                             |  5.57752 |  412.737 | 2.14319    | 2.12215  | 0.0276532   | 0.0545793 |
| `learners/non-learners`:decile     |  5.57752 |  412.737 | 2.14319    | 1.28714  | 0.0169568   | 0.264657  |
| bin                                |  1.67795 |  124.169 | 0.00319365 | 1.75178  | 1.05241e-05 | 0.182827  |
| `learners/non-learners`:bin        |  1.67795 |  124.169 | 0.00319365 | 0.202655 | 1.2175e-06  | 0.77808   |
| decile:bin                         | 12.4339  |  920.109 | 0.00361108 | 1.25992  | 6.34171e-05 | 0.2348    |
| `learners/non-learners`:decile:bin | 12.4339  |  920.109 | 0.00361108 | 1.735    | 8.73278e-05 | 0.0523937 |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/rsa/anova_bin_feedback.png)  |  ![](./r_stats/zscore/rsa/anova_decile_feedback.png)

Significant Effects/Interactions

No statistically significant effects present at P < 0.05

| Effect                             |   num Df |   den Df |        MSE |        F |         ges |         P |
|:-----------------------------------|---------:|---------:|-----------:|---------:|------------:|----------:|
| `learners/non-learners`            |  1       |   74     | 0.04638    | 0.246902 | 1.22232e-05 | 0.620738  |
| decile                             |  5.69933 |  421.75  | 2.20427    | 1.78964  | 0.0234363   | 0.103688  |
| `learners/non-learners`:decile     |  5.69933 |  421.75  | 2.20427    | 1.42106  | 0.0186998   | 0.208036  |
| bin                                |  1.7636  |  130.506 | 0.00404482 | 2.82178  | 2.14856e-05 | 0.0698232 |
| `learners/non-learners`:bin        |  1.7636  |  130.506 | 0.00404482 | 0.803202 | 6.11583e-06 | 0.436184  |
| decile:bin                         | 12.2465  |  906.241 | 0.00356266 | 0.732121 | 3.40949e-05 | 0.723581  |
| `learners/non-learners`:decile:bin | 12.2465  |  906.241 | 0.00356266 | 0.539431 | 2.51215e-05 | 0.892506  |
## ANOVA Skin Conductance Response (Phasic)
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/scr/anova_bin_baseline.png)  |  ![](./r_stats/zscore/scr/anova_decile_baseline.png)

Significant Effects/Interactions

- bin
- decile:bin

| Effect                             |   num Df |   den Df |       MSE |        F |         ges |           P |
|:-----------------------------------|---------:|---------:|----------:|---------:|------------:|------------:|
| `learners/non-learners`            |  1       |   91     | 1.34645   | 0.136713 | 0.000127675 | 0.71243     |
| decile                             |  7.54954 |  687.008 | 0.636214  | 1.69963  | 0.00563104  | 0.0998303   |
| `learners/non-learners`:decile     |  7.54954 |  687.008 | 0.636214  | 0.344152 | 0.00114535  | 0.94226     |
| bin                                |  1.88855 |  171.858 | 1.00618   | 6.39162  | 0.00835473  | 0.00254821  |
| `learners/non-learners`:bin        |  1.88855 |  171.858 | 1.00618   | 0.461239 | 0.000607613 | 0.620142    |
| decile:bin                         | 99       | 9009     | 0.0787043 | 1.70365  | 0.00912421  | 1.85495e-05 |
| `learners/non-learners`:decile:bin | 99       | 9009     | 0.0787043 | 0.712462 | 0.00383608  | 0.986094    |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/scr/anova_bin_planning.png)  |  ![](./r_stats/zscore/scr/anova_decile_planning.png)

Significant Effects/Interactions

- decile
- bin

| Effect                             |   num Df |   den Df |      MSE |         F |         ges |           P |
|:-----------------------------------|---------:|---------:|---------:|----------:|------------:|------------:|
| `learners/non-learners`            |  1       |   91     | 1.03625  |  0.232702 | 0.000405001 | 0.630686    |
| decile                             |  8.0138  |  729.256 | 0.450598 |  3.27882  | 0.0195055   | 0.0010946   |
| `learners/non-learners`:decile     |  8.0138  |  729.256 | 0.450598 |  0.785618 | 0.00474395  | 0.615693    |
| bin                                |  1.35636 |  123.429 | 0.163766 | 11.0846   | 0.00411994  | 0.000324611 |
| `learners/non-learners`:bin        |  1.35636 |  123.429 | 0.163766 |  2.85125  | 0.00106301  | 0.0812438   |
| decile:bin                         | 10.4558  |  951.477 | 0.159801 |  1.06384  | 0.00297771  | 0.387675    |
| `learners/non-learners`:decile:bin | 10.4558  |  951.477 | 0.159801 |  0.529937 | 0.00148553  | 0.876445    |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/scr/anova_bin_feedback.png)  |  ![](./r_stats/zscore/scr/anova_decile_feedback.png)

Significant Effects/Interactions

- decile
- bin
- decile:bin

| Effect                             |   num Df |   den Df |      MSE |          F |         ges |           P |
|:-----------------------------------|---------:|---------:|---------:|-----------:|------------:|------------:|
| `learners/non-learners`            |  1       |   91     | 1.07859  |  0.0705612 | 8.82031e-05 | 0.791122    |
| decile                             |  7.6863  |  699.454 | 0.525624 |  3.95142   | 0.018167    | 0.000176249 |
| `learners/non-learners`:decile     |  7.6863  |  699.454 | 0.525624 |  0.739138  | 0.00344919  | 0.651475    |
| bin                                |  1.1318  |  102.994 | 1.61559  |  8.90169   | 0.0185164   | 0.00249544  |
| `learners/non-learners`:bin        |  1.1318  |  102.994 | 1.61559  |  1.84009   | 0.00388463  | 0.177324    |
| decile:bin                         |  7.73591 |  703.968 | 0.327548 | 10.2586    | 0.0292472   | 2.84486e-13 |
| `learners/non-learners`:decile:bin |  7.73591 |  703.968 | 0.327548 |  1.36403   | 0.00399001  | 0.211166    |
## ANOVA Respiration Rate
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/rsp/anova_bin_baseline.png)  |  ![](./r_stats/zscore/rsp/anova_decile_baseline.png)

Significant Effects/Interactions

- decile
- bin

| Effect                             |   num Df |   den Df |      MSE |         F |         ges |           P |
|:-----------------------------------|---------:|---------:|---------:|----------:|------------:|------------:|
| `learners/non-learners`            |  1       |  104     | 0.577022 |  0.477741 | 0.000124042 | 0.490987    |
| decile                             |  5.06694 |  526.962 | 3.28584  |  3.55291  | 0.0259301   | 0.00345117  |
| `learners/non-learners`:decile     |  5.06694 |  526.962 | 3.28584  |  0.406904 | 0.00303949  | 0.846397    |
| bin                                |  1.48426 |  154.363 | 0.542454 | 44.7469   | 0.0159547   | 3.60484e-13 |
| `learners/non-learners`:bin        |  1.48426 |  154.363 | 0.542454 |  1.22212  | 0.000442619 | 0.288337    |
| decile:bin                         | 13.2345  | 1376.39  | 0.251988 |  1.22764  | 0.00183906  | 0.251735    |
| `learners/non-learners`:decile:bin | 13.2345  | 1376.39  | 0.251988 |  0.885284 | 0.00132688  | 0.569992    |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/rsp/anova_bin_planning.png)  |  ![](./r_stats/zscore/rsp/anova_decile_planning.png)

Significant Effects/Interactions

- decile

| Effect                             |   num Df |   den Df |       MSE |        F |         ges |         P |
|:-----------------------------------|---------:|---------:|----------:|---------:|------------:|----------:|
| `learners/non-learners`            |  1       |  104     | 0.946853  | 0.968723 | 0.000871714 | 0.327283  |
| decile                             |  5.54505 |  576.685 | 1.53032   | 2.30834  | 0.0182911   | 0.0370594 |
| `learners/non-learners`:decile     |  5.54505 |  576.685 | 1.53032   | 0.615226 | 0.00494129  | 0.705268  |
| bin                                |  1.34327 |  139.7   | 0.066041  | 1.51419  | 0.000127753 | 0.224913  |
| `learners/non-learners`:bin        |  1.34327 |  139.7   | 0.066041  | 0.228191 | 1.92547e-05 | 0.705053  |
| decile:bin                         | 10.4231  | 1084     | 0.0563629 | 1.80573  | 0.00100803  | 0.0524343 |
| `learners/non-learners`:decile:bin | 10.4231  | 1084     | 0.0563629 | 0.66002  | 0.000368687 | 0.768699  |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/rsp/anova_bin_feedback.png)  |  ![](./r_stats/zscore/rsp/anova_decile_feedback.png)

Significant Effects/Interactions

- decile
- bin

| Effect                             |   num Df |   den Df |      MSE |         F |         ges |           P |
|:-----------------------------------|---------:|---------:|---------:|----------:|------------:|------------:|
| `learners/non-learners`            |  1       |  104     | 0.86765  |  0.434551 | 0.000329604 | 0.511221    |
| decile                             |  5.28904 |  550.061 | 1.75755  |  3.92781  | 0.030941    | 0.00131157  |
| `learners/non-learners`:decile     |  5.28904 |  550.061 | 1.75755  |  0.858624 | 0.00693132  | 0.513792    |
| bin                                |  1.31124 |  136.369 | 0.125916 | 61.0211   | 0.0087334   | 5.05227e-15 |
| `learners/non-learners`:bin        |  1.31124 |  136.369 | 0.125916 |  2.35875  | 0.000340444 | 0.118276    |
| decile:bin                         | 10.0657  | 1046.83  | 0.066271 |  0.688199 | 0.000401288 | 0.737196    |
| `learners/non-learners`:decile:bin | 10.0657  | 1046.83  | 0.066271 |  0.547369 | 0.000319197 | 0.857892    |
## ANOVA Heart Rate
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/hr/anova_bin_baseline.png)  |  ![](./r_stats/zscore/hr/anova_decile_baseline.png)

Significant Effects/Interactions

- bin
- decile:bin

| Effect                             |   num Df |   den Df |      MSE |         F |         ges |           P |
|:-----------------------------------|---------:|---------:|---------:|----------:|------------:|------------:|
| `learners/non-learners`            |  1       |   98     | 0.926307 |  0.222768 | 9.3565e-05  | 0.637988    |
| decile                             |  4.83937 |  474.258 | 3.28584  |  0.929761 | 0.00665963  | 0.45912     |
| `learners/non-learners`:decile     |  4.83937 |  474.258 | 3.28584  |  2.0754   | 0.0147445   | 0.0695192   |
| bin                                |  2.11539 |  207.308 | 0.656353 | 90.7565   | 0.054053    | 4.01579e-30 |
| `learners/non-learners`:bin        |  2.11539 |  207.308 | 0.656353 |  1.30879  | 0.000823355 | 0.272876    |
| decile:bin                         | 99       | 9702     | 0.043295 |  1.42916  | 0.0027701   | 0.00347695  |
| `learners/non-learners`:decile:bin | 99       | 9702     | 0.043295 |  1.24164  | 0.00240752  | 0.0530237   |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/hr/anova_bin_planning.png)  |  ![](./r_stats/zscore/hr/anova_decile_planning.png)

Significant Effects/Interactions

- bin

| Effect                             |   num Df |   den Df |       MSE |         F |         ges |           P |
|:-----------------------------------|---------:|---------:|----------:|----------:|------------:|------------:|
| `learners/non-learners`            |  1       |   98     | 0.690071  |  1.1485   | 0.000771206 | 0.286496    |
| decile                             |  5.17792 |  507.437 | 1.6421    |  1.19052  | 0.00976144  | 0.312133    |
| `learners/non-learners`:decile     |  5.17792 |  507.437 | 1.6421    |  1.32776  | 0.0108745   | 0.249453    |
| bin                                |  1.50295 |  147.289 | 0.166086  | 48.3329   | 0.0116126   | 5.51897e-14 |
| `learners/non-learners`:bin        |  1.50295 |  147.289 | 0.166086  |  0.121732 | 2.95905e-05 | 0.826877    |
| decile:bin                         | 14.1898  | 1390.6   | 0.0730072 |  0.889233 | 0.000896291 | 0.571572    |
| `learners/non-learners`:decile:bin | 14.1898  | 1390.6   | 0.0730072 |  0.726842 | 0.000732731 | 0.750711    |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/zscore/hr/anova_bin_feedback.png)  |  ![](./r_stats/zscore/hr/anova_decile_feedback.png)

Significant Effects/Interactions

- decile
- bin
- decile:bin

| Effect                             |   num Df |   den Df |       MSE |             F |         ges |           P |
|:-----------------------------------|---------:|---------:|----------:|--------------:|------------:|------------:|
| `learners/non-learners`            |  1       |   98     | 0.844703  |   0.000316219 | 2.64459e-07 | 0.985848    |
| decile                             |  5.11174 |  500.95  | 1.50098   |   2.2325      | 0.0166762   | 0.0485665   |
| `learners/non-learners`:decile     |  5.11174 |  500.95  | 1.50098   |   1.5171      | 0.0113933   | 0.181346    |
| bin                                |  1.64193 |  160.909 | 0.422603  | 105.57        | 0.067622    | 1.28169e-26 |
| `learners/non-learners`:bin        |  1.64193 |  160.909 | 0.422603  |   0.177573    | 0.000121977 | 0.79449     |
| decile:bin                         | 14.1431  | 1386.03  | 0.0774379 |   1.7944      | 0.00194197  | 0.0339254   |
| `learners/non-learners`:decile:bin | 14.1431  | 1386.03  | 0.0774379 |   0.996892    | 0.0010798   | 0.453966    |
