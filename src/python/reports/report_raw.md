# Learners/Non-learners Statistics (raw)
dv='pupil'

between=['learners/non-learners']

within=['decile, 'bin']

## ANOVA Pupil Change Scores
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/ad_pupil/anova_bin_baseline.png)  |  ![](./r_stats/raw/ad_pupil/anova_decile_baseline.png)

Significant Effects/Interactions

- `learners/non-learners`
- bin

| Effect                             |   num Df |   den Df |       MSE |         F |        ges |           P |
|:-----------------------------------|---------:|---------:|----------:|----------:|-----------:|------------:|
| `learners/non-learners`            |  1       |   89     | 4.12195   |  4.27556  | 0.00782276 | 0.0415675   |
| decile                             |  6.46118 |  575.045 | 1.62327   |  0.982829 | 0.00459047 | 0.43922     |
| `learners/non-learners`:decile     |  6.46118 |  575.045 | 1.62327   |  1.16     | 0.00541352 | 0.325279    |
| bin                                |  4.21024 |  374.711 | 0.858064  | 85.0496   | 0.120848   | 3.79206e-53 |
| `learners/non-learners`:bin        |  4.21024 |  374.711 | 0.858064  |  1.93404  | 0.0031161  | 0.100449    |
| decile:bin                         | 99       | 8811     | 0.0696187 |  1.1129   | 0.00341985 | 0.209587    |
| `learners/non-learners`:decile:bin | 99       | 8811     | 0.0696187 |  1.11442  | 0.00342448 | 0.206763    |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/ad_pupil/anova_bin_planning.png)  |  ![](./r_stats/raw/ad_pupil/anova_decile_planning.png)

Significant Effects/Interactions

- `learners/non-learners`
- bin

| Effect                             |   num Df |   den Df |       MSE |          F |        ges |           P |
|:-----------------------------------|---------:|---------:|----------:|-----------:|-----------:|------------:|
| `learners/non-learners`            |  1       |   94     | 0.9313    |   5.3667   | 0.0206857  | 0.0226972   |
| decile                             |  7.22956 |  679.579 | 0.0666511 |   1.94792  | 0.00395114 | 0.0575116   |
| `learners/non-learners`:decile     |  7.22956 |  679.579 | 0.0666511 |   1.47203  | 0.00298875 | 0.171656    |
| bin                                |  2.14301 |  201.443 | 0.286475  | 487.844    | 0.558643   | 2.60022e-80 |
| `learners/non-learners`:bin        |  2.14301 |  201.443 | 0.286475  |   2.07944  | 0.00536629 | 0.12409     |
| decile:bin                         | 17.6984  | 1663.65  | 0.0276939 |   0.781464 | 0.00161613 | 0.721921    |
| `learners/non-learners`:decile:bin | 17.6984  | 1663.65  | 0.0276939 |   1.16992  | 0.00241755 | 0.27908     |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/ad_pupil/anova_bin_feedback.png)  |  ![](./r_stats/raw/ad_pupil/anova_decile_feedback.png)

Significant Effects/Interactions

- bin

| Effect                             |   num Df |   den Df |      MSE |         F |        ges |           P |
|:-----------------------------------|---------:|---------:|---------:|----------:|-----------:|------------:|
| `learners/non-learners`            |  1       |   94     | 1.61853  |  1.45785  | 0.00378742 | 0.230302    |
| decile                             |  4.73534 |  445.122 | 0.649911 |  1.32415  | 0.00652315 | 0.254652    |
| `learners/non-learners`:decile     |  4.73534 |  445.122 | 0.649911 |  0.494255 | 0.00244484 | 0.770804    |
| bin                                |  3.27774 |  308.108 | 0.129857 | 30.4525   | 0.0204571  | 1.92764e-18 |
| `learners/non-learners`:bin        |  3.27774 |  308.108 | 0.129857 |  1.76239  | 0.00120719 | 0.149128    |
| decile:bin                         | 11.408   | 1072.36  | 0.129809 |  1.48021  | 0.00351939 | 0.129658    |
| `learners/non-learners`:decile:bin | 11.408   | 1072.36  | 0.129809 |  0.899053 | 0.00214057 | 0.54353     |
## ANOVA Raw Pupil
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/pupil/anova_bin_baseline.png)  |  ![](./r_stats/raw/pupil/anova_decile_baseline.png)

Significant Effects/Interactions

- decile
- bin
- `learners/non-learners`:decile:bin

| Effect                             |   num Df |   den Df |        MSE |         F |         ges |           P |
|:-----------------------------------|---------:|---------:|-----------:|----------:|------------:|------------:|
| `learners/non-learners`            |  1       |   89     | 29.0738    |  1.37449  | 0.0123883   | 0.244171    |
| decile                             |  4.3043  |  383.082 |  0.59873   | 29.2607   | 0.0231228   | 1.83806e-22 |
| `learners/non-learners`:decile     |  4.3043  |  383.082 |  0.59873   |  0.882692 | 0.000713532 | 0.480338    |
| bin                                |  1.86265 |  165.776 |  1.45614   | 80.6813   | 0.0642742   | 4.38597e-24 |
| `learners/non-learners`:bin        |  1.86265 |  165.776 |  1.45614   |  1.19015  | 0.00101223  | 0.304574    |
| decile:bin                         | 99       | 8811     |  0.0144682 |  0.975113 | 0.000438223 | 0.551276    |
| `learners/non-learners`:decile:bin | 99       | 8811     |  0.0144682 |  1.26786  | 0.00056971  | 0.0381385   |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/pupil/anova_bin_planning.png)  |  ![](./r_stats/raw/pupil/anova_decile_planning.png)

Significant Effects/Interactions

- decile
- bin

| Effect                             |   num Df |   den Df |        MSE |          F |         ges |           P |
|:-----------------------------------|---------:|---------:|-----------:|-----------:|------------:|------------:|
| `learners/non-learners`            |  1       |   94     | 16.5851    |   1.21442  | 0.0109873   | 0.273272    |
| decile                             |  4.27402 |  401.758 |  0.366451  |  20.0883   | 0.0170578   | 7.08702e-16 |
| `learners/non-learners`:decile     |  4.27402 |  401.758 |  0.366451  |   1.49663  | 0.00129123  | 0.198889    |
| bin                                |  2.07345 |  194.904 |  0.297327  | 461.232    | 0.135574    | 8.12049e-76 |
| `learners/non-learners`:bin        |  2.07345 |  194.904 |  0.297327  |   1.78682  | 0.000607217 | 0.168848    |
| decile:bin                         | 17.2375  | 1620.32  |  0.0301401 |   0.966359 | 0.000276845 | 0.494812    |
| `learners/non-learners`:decile:bin | 17.2375  | 1620.32  |  0.0301401 |   1.34139  | 0.000384244 | 0.156305    |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/pupil/anova_bin_feedback.png)  |  ![](./r_stats/raw/pupil/anova_decile_feedback.png)

Significant Effects/Interactions

- decile
- bin

| Effect                             |   num Df |   den Df |        MSE |         F |         ges |           P |
|:-----------------------------------|---------:|---------:|-----------:|----------:|------------:|------------:|
| `learners/non-learners`            |  1       |   94     | 17.483     |  0.347785 | 0.00333266  | 0.556785    |
| decile                             |  4.20006 |  394.805 |  0.278652  | 37.1406   | 0.0233464   | 2.31654e-27 |
| `learners/non-learners`:decile     |  4.20006 |  394.805 |  0.278652  |  1.40317  | 0.000902298 | 0.230174    |
| bin                                |  2.03873 |  191.641 |  0.143469  | 36.69     | 0.00586712  | 2.08638e-14 |
| `learners/non-learners`:bin        |  2.03873 |  191.641 |  0.143469  |  1.51512  | 0.000243654 | 0.222107    |
| decile:bin                         | 17.9652  | 1688.73  |  0.0221909 |  1.30174  | 0.000285313 | 0.17657     |
| `learners/non-learners`:decile:bin | 17.9652  | 1688.73  |  0.0221909 |  1.41237  | 0.000309553 | 0.115692    |
## ANOVA Respiratory Sinus Arythmia
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/rsa/anova_bin_baseline.png)  |  ![](./r_stats/raw/rsa/anova_decile_baseline.png)

Significant Effects/Interactions

- decile
- `learners/non-learners`:decile:bin

| Effect                             |   num Df |   den Df |          MSE |        F |         ges |         P |
|:-----------------------------------|---------:|---------:|-------------:|---------:|------------:|----------:|
| `learners/non-learners`            |  1       |   74     | 126.312      | 0.207673 | 0.00255214  | 0.649932  |
| decile                             |  5.04619 |  373.418 |   2.39677    | 2.48052  | 0.00291779  | 0.0310572 |
| `learners/non-learners`:decile     |  5.04619 |  373.418 |   2.39677    | 1.22448  | 0.00144246  | 0.296716  |
| bin                                |  1.54755 |  114.518 |   0.010284   | 2.1564   | 3.34754e-06 | 0.132111  |
| `learners/non-learners`:bin        |  1.54755 |  114.518 |   0.010284   | 0.392997 | 6.1008e-07  | 0.622583  |
| decile:bin                         | 99       | 7326     |   0.00119558 | 0.731393 | 8.44405e-06 | 0.978956  |
| `learners/non-learners`:decile:bin | 99       | 7326     |   0.00119558 | 1.36231  | 1.57279e-05 | 0.0101827 |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/rsa/anova_bin_planning.png)  |  ![](./r_stats/raw/rsa/anova_decile_planning.png)

Significant Effects/Interactions

- decile

| Effect                             |   num Df |   den Df |         MSE |        F |         ges |         P |
|:-----------------------------------|---------:|---------:|------------:|---------:|------------:|----------:|
| `learners/non-learners`            |  1       |   74     | 63.5325     | 0.219543 | 0.00269547  | 0.640766  |
| decile                             |  4.99783 |  369.84  |  1.23681    | 2.32991  | 0.00278294  | 0.0420786 |
| `learners/non-learners`:decile     |  4.99783 |  369.84  |  1.23681    | 1.04347  | 0.00124828  | 0.391735  |
| bin                                |  1.55859 |  115.335 |  0.00186704 | 1.29076  | 7.27818e-07 | 0.273998  |
| `learners/non-learners`:bin        |  1.55859 |  115.335 |  0.00186704 | 0.142894 | 8.05732e-08 | 0.814382  |
| decile:bin                         | 10.41    |  770.34  |  0.00215834 | 1.15798  | 5.04152e-06 | 0.314773  |
| `learners/non-learners`:decile:bin | 10.41    |  770.34  |  0.00215834 | 1.61853  | 7.04658e-06 | 0.0933799 |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/rsa/anova_bin_feedback.png)  |  ![](./r_stats/raw/rsa/anova_decile_feedback.png)

Significant Effects/Interactions

No statistically significant effects present at P < 0.05

| Effect                             |   num Df |   den Df |         MSE |        F |         ges |         P |
|:-----------------------------------|---------:|---------:|------------:|---------:|------------:|----------:|
| `learners/non-learners`            |  1       |   74     | 62.9567     | 0.240166 | 0.00293203  | 0.625536  |
| decile                             |  5.20054 |  384.84  |  1.2507     | 2.09056  | 0.00263759  | 0.0631104 |
| `learners/non-learners`:decile     |  5.20054 |  384.84  |  1.2507     | 1.15936  | 0.00146444  | 0.32848   |
| bin                                |  1.79955 |  133.167 |  0.0015996  | 2.55404  | 1.42985e-06 | 0.0873018 |
| `learners/non-learners`:bin        |  1.79955 |  133.167 |  0.0015996  | 0.880981 | 4.93208e-07 | 0.407021  |
| decile:bin                         | 12.2386  |  905.656 |  0.00156841 | 0.804841 | 3.00462e-06 | 0.648143  |
| `learners/non-learners`:decile:bin | 12.2386  |  905.656 |  0.00156841 | 0.551373 | 2.05838e-06 | 0.883972  |
## ANOVA Skin Conductance Response (Phasic)
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/scr/anova_bin_baseline.png)  |  ![](./r_stats/raw/scr/anova_decile_baseline.png)

Significant Effects/Interactions

- bin

| Effect                             |   num Df |   den Df |       MSE |        F |         ges |          P |
|:-----------------------------------|---------:|---------:|----------:|---------:|------------:|-----------:|
| `learners/non-learners`            |  1       |   91     | 0.235838  | 0.863593 | 0.000676135 | 0.355194   |
| decile                             |  7.1307  |  648.894 | 0.135089  | 1.30926  | 0.00417222  | 0.242076   |
| `learners/non-learners`:decile     |  7.1307  |  648.894 | 0.135089  | 0.678762 | 0.00216737  | 0.692979   |
| bin                                |  2.06924 |  188.301 | 0.168934  | 6.43263  | 0.00741466  | 0.00175793 |
| `learners/non-learners`:bin        |  2.06924 |  188.301 | 0.168934  | 0.973794 | 0.00112956  | 0.381927   |
| decile:bin                         | 99       | 9009     | 0.0177699 | 1.07435  | 0.00623956  | 0.289763   |
| `learners/non-learners`:decile:bin | 99       | 9009     | 0.0177699 | 1.00269  | 0.00582579  | 0.473924   |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/scr/anova_bin_planning.png)  |  ![](./r_stats/raw/scr/anova_decile_planning.png)

Significant Effects/Interactions

- bin

| Effect                             |   num Df |   den Df |       MSE |        F |         ges |          P |
|:-----------------------------------|---------:|---------:|----------:|---------:|------------:|-----------:|
| `learners/non-learners`            |  1       |   91     | 0.184507  | 0.792084 | 0.00116977  | 0.375818   |
| decile                             |  7.40585 |  673.932 | 0.0959601 | 1.81936  | 0.0102549   | 0.0763751  |
| `learners/non-learners`:decile     |  7.40585 |  673.932 | 0.0959601 | 0.977719 | 0.00553723  | 0.448778   |
| bin                                |  1.3375  |  121.713 | 0.046371  | 7.15767  | 0.00354483  | 0.00425232 |
| `learners/non-learners`:bin        |  1.3375  |  121.713 | 0.046371  | 0.61423  | 0.000305186 | 0.479715   |
| decile:bin                         |  6.18167 |  562.532 | 0.0669904 | 0.889608 | 0.00294347  | 0.504429   |
| `learners/non-learners`:decile:bin |  6.18167 |  562.532 | 0.0669904 | 0.476285 | 0.00157806  | 0.831233   |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/scr/anova_bin_feedback.png)  |  ![](./r_stats/raw/scr/anova_decile_feedback.png)

Significant Effects/Interactions

- decile
- bin
- decile:bin

| Effect                             |   num Df |   den Df |       MSE |        F |         ges |           P |
|:-----------------------------------|---------:|---------:|----------:|---------:|------------:|------------:|
| `learners/non-learners`            |  1       |   91     | 0.282787  | 0.471015 | 0.00080191  | 0.494265    |
| decile                             |  7.02444 |  639.224 | 0.100247  | 3.65736  | 0.0152806   | 0.000687576 |
| `learners/non-learners`:decile     |  7.02444 |  639.224 | 0.100247  | 0.554281 | 0.00234623  | 0.793701    |
| bin                                |  1.16177 |  105.721 | 0.2827    | 8.52373  | 0.0165879   | 0.00283503  |
| `learners/non-learners`:bin        |  1.16177 |  105.721 | 0.2827    | 0.443725 | 0.000877319 | 0.535874    |
| decile:bin                         |  6.8594  |  624.206 | 0.0741192 | 7.47591  | 0.0223886   | 1.57473e-08 |
| `learners/non-learners`:decile:bin |  6.8594  |  624.206 | 0.0741192 | 1.19792  | 0.00365623  | 0.302242    |
## ANOVA Respiration Rate
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/rsp/anova_bin_baseline.png)  |  ![](./r_stats/raw/rsp/anova_decile_baseline.png)

Significant Effects/Interactions

- bin

| Effect                             |   num Df |   den Df |       MSE |         F |         ges |          P |
|:-----------------------------------|---------:|---------:|----------:|----------:|------------:|-----------:|
| `learners/non-learners`            |  1       |  104     | 1046.27   |  1.13496  | 0.00562251  | 0.289187   |
| decile                             |  3.67396 |  382.091 |  206.261  |  1.74272  | 0.00624901  | 0.14549    |
| `learners/non-learners`:decile     |  3.67396 |  382.091 |  206.261  |  0.385664 | 0.00138967  | 0.802922   |
| bin                                |  1.5101  |  157.05  |   16.3221 | 25.1593   | 0.00294412  | 1.6965e-08 |
| `learners/non-learners`:bin        |  1.5101  |  157.05  |   16.3221 |  0.224095 | 2.63002e-05 | 0.735959   |
| decile:bin                         |  2.6415  |  274.716 |   72.1699 |  0.958552 | 0.00086936  | 0.404334   |
| `learners/non-learners`:decile:bin |  2.6415  |  274.716 |   72.1699 |  0.72855  | 0.000660896 | 0.519298   |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/rsp/anova_bin_planning.png)  |  ![](./r_stats/raw/rsp/anova_decile_planning.png)

Significant Effects/Interactions

No statistically significant effects present at P < 0.05

| Effect                             |   num Df |   den Df |       MSE |        F |         ges |        P |
|:-----------------------------------|---------:|---------:|----------:|---------:|------------:|---------:|
| `learners/non-learners`            |  1       |  104     | 580.919   | 0.616725 | 0.0039937   | 0.434052 |
| decile                             |  4.24756 |  441.746 |  60.5337  | 1.76821  | 0.0050626   | 0.129967 |
| `learners/non-learners`:decile     |  4.24756 |  441.746 |  60.5337  | 0.544144 | 0.00156343  | 0.714042 |
| bin                                |  1.36291 |  141.742 |   1.79131 | 1.08998  | 2.97817e-05 | 0.318371 |
| `learners/non-learners`:bin        |  1.36291 |  141.742 |   1.79131 | 1.02698  | 2.80604e-05 | 0.335725 |
| decile:bin                         |  7.92443 |  824.141 |   2.35363 | 1.03404  | 0.000215801 | 0.40821  |
| `learners/non-learners`:decile:bin |  7.92443 |  824.141 |   2.35363 | 0.750973 | 0.000156736 | 0.645104 |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/rsp/anova_bin_feedback.png)  |  ![](./r_stats/raw/rsp/anova_decile_feedback.png)

Significant Effects/Interactions

- decile
- bin

| Effect                             |   num Df |   den Df |       MSE |         F |         ges |           P |
|:-----------------------------------|---------:|---------:|----------:|----------:|------------:|------------:|
| `learners/non-learners`            |  1       |  104     | 719.058   |  0.851924 | 0.00564839  | 0.358145    |
| decile                             |  4.11563 |  428.026 |  71.7767  |  2.45904  | 0.00669098  | 0.0432551   |
| `learners/non-learners`:decile     |  4.11563 |  428.026 |  71.7767  |  1.20285  | 0.00328415  | 0.308769    |
| bin                                |  1.50679 |  156.707 |   2.55338 | 34.9186   | 0.00124424  | 5.47783e-11 |
| `learners/non-learners`:bin        |  1.50679 |  156.707 |   2.55338 |  3.00801  | 0.000107306 | 0.0666814   |
| decile:bin                         |  7.04773 |  732.964 |   2.64082 |  0.500871 | 8.64365e-05 | 0.835335    |
| `learners/non-learners`:decile:bin |  7.04773 |  732.964 |   2.64082 |  0.686264 | 0.000118426 | 0.684829    |
## ANOVA Heart Rate
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/hr/anova_bin_baseline.png)  |  ![](./r_stats/raw/hr/anova_decile_baseline.png)

Significant Effects/Interactions

- bin
- decile:bin
- `learners/non-learners`:decile:bin

| Effect                             |   num Df |   den Df |         MSE |         F |         ges |           P |
|:-----------------------------------|---------:|---------:|------------:|----------:|------------:|------------:|
| `learners/non-learners`            |  1       |   98     | 15073.8     |  0.127398 | 0.00122635  | 0.721913    |
| decile                             |  4.91201 |  481.377 |   125.711   |  0.987707 | 0.00038981  | 0.423821    |
| `learners/non-learners`:decile     |  4.91201 |  481.377 |   125.711   |  1.63708  | 0.000645927 | 0.149861    |
| bin                                |  2.0099  |  196.97  |    32.6812  | 76.8848   | 0.00321866  | 1.72912e-25 |
| `learners/non-learners`:bin        |  2.0099  |  196.97  |    32.6812  |  1.10818  | 4.65399e-05 | 0.332423    |
| decile:bin                         | 99       | 9702     |     2.04252 |  1.29314  | 0.000167163 | 0.0271894   |
| `learners/non-learners`:decile:bin | 99       | 9702     |     2.04252 |  1.32872  | 0.00017176  | 0.0165218   |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/hr/anova_bin_planning.png)  |  ![](./r_stats/raw/hr/anova_decile_planning.png)

Significant Effects/Interactions

- bin

| Effect                             |   num Df |   den Df |        MSE |          F |         ges |           P |
|:-----------------------------------|---------:|---------:|-----------:|-----------:|------------:|------------:|
| `learners/non-learners`            |  1       |   98     | 7707.24    |  0.0701247 | 0.000680858 | 0.791712    |
| decile                             |  4.93736 |  483.861 |   66.7252  |  1.34736   | 0.000559251 | 0.24357     |
| `learners/non-learners`:decile     |  4.93736 |  483.861 |   66.7252  |  1.02503   | 0.00042552  | 0.40169     |
| bin                                |  1.53599 |  150.527 |    6.80527 | 45.3111    | 0.000596706 | 1.49586e-13 |
| `learners/non-learners`:bin        |  1.53599 |  150.527 |    6.80527 |  0.127014  | 1.67365e-06 | 0.826429    |
| decile:bin                         | 12.3192  | 1207.29  |    3.84626 |  0.906163  | 5.41239e-05 | 0.542043    |
| `learners/non-learners`:decile:bin | 12.3192  | 1207.29  |    3.84626 |  0.800405  | 4.78074e-05 | 0.653664    |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/hr/anova_bin_feedback.png)  |  ![](./r_stats/raw/hr/anova_decile_feedback.png)

Significant Effects/Interactions

- decile
- bin

| Effect                             |   num Df |   den Df |        MSE |           F |         ges |          P |
|:-----------------------------------|---------:|---------:|-----------:|------------:|------------:|-----------:|
| `learners/non-learners`            |  1       |   98     | 7714.92    |   0.0766564 | 0.000745023 | 0.782464   |
| decile                             |  5.13604 |  503.332 |   58.8323  |   2.21456   | 0.000842906 | 0.0499656  |
| `learners/non-learners`:decile     |  5.13604 |  503.332 |   58.8323  |   1.45091   | 0.000552404 | 0.202944   |
| bin                                |  1.70742 |  167.327 |   14.0668  | 106.549     | 0.00321587  | 9.5029e-28 |
| `learners/non-learners`:bin        |  1.70742 |  167.327 |   14.0668  |   0.666277  | 2.01741e-05 | 0.492173   |
| decile:bin                         | 10.5722  | 1036.07  |    4.99885 |   1.47696   | 9.83951e-05 | 0.1377     |
| `learners/non-learners`:decile:bin | 10.5722  | 1036.07  |    4.99885 |   1.02367   | 6.8199e-05  | 0.42199    |
## ANOVA Skin Conductance Level (Tonic)
### Baseline
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/scl/anova_bin_baseline.png)  |  ![](./r_stats/raw/scl/anova_decile_baseline.png)

Significant Effects/Interactions

- bin
- `learners/non-learners`:bin
- decile:bin

| Effect                             |   num Df |   den Df |            MSE |         F |         ges |           P |
|:-----------------------------------|---------:|---------:|---------------:|----------:|------------:|------------:|
| `learners/non-learners`            |  1       |   91     | 14121.4        |  0.024884 | 0.000257173 | 0.875006    |
| decile                             |  3.47665 |  316.375 |   255.883      |  0.90953  | 0.000591974 | 0.448004    |
| `learners/non-learners`:decile     |  3.47665 |  316.375 |   255.883      |  2.00352  | 0.00130308  | 0.103592    |
| bin                                | 11       | 1001     |     0.00317192 | 27.2765   | 6.96696e-07 | 3.97918e-50 |
| `learners/non-learners`:bin        | 11       | 1001     |     0.00317192 |  2.98344  | 7.62029e-08 | 0.000649034 |
| decile:bin                         | 99       | 9009     |     0.00286011 |  2.98992  | 6.19753e-07 | 4.80733e-21 |
| `learners/non-learners`:decile:bin | 99       | 9009     |     0.00286011 |  1.14543  | 2.37425e-07 | 0.154426    |
### Planning
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/scl/anova_bin_planning.png)  |  ![](./r_stats/raw/scl/anova_decile_planning.png)

Significant Effects/Interactions

- bin
- decile:bin

| Effect                             |   num Df |   den Df |            MSE |           F |         ges |           P |
|:-----------------------------------|---------:|---------:|---------------:|------------:|------------:|------------:|
| `learners/non-learners`            |  1       |   91     | 7035.31        |   0.0235481 | 0.000243474 | 0.87838     |
| decile                             |  3.47745 |  316.448 |  126.577       |   0.81638   | 0.000527953 | 0.500923    |
| `learners/non-learners`:decile     |  3.47745 |  316.448 |  126.577       |   2.05829   | 0.00133002  | 0.0958163   |
| bin                                |  5       |  455     |    0.000845474 | 110.26      | 6.85179e-07 | 4.00887e-76 |
| `learners/non-learners`:bin        |  5       |  455     |    0.000845474 |   0.0621044 | 3.85932e-10 | 0.997422    |
| decile:bin                         | 45       | 4095     |    0.000771136 |   3.40654   | 1.7377e-07  | 2.03215e-13 |
| `learners/non-learners`:decile:bin | 45       | 4095     |    0.000771136 |   0.620699  | 3.16623e-08 | 0.978073    |
### Feedback
Bin             |  Decile
:-------------------------:|:-------------------------:
![](./r_stats/raw/scl/anova_bin_feedback.png)  |  ![](./r_stats/raw/scl/anova_decile_feedback.png)

Significant Effects/Interactions

- decile:bin

| Effect                             |   num Df |   den Df |            MSE |         F |         ges |           P |
|:-----------------------------------|---------:|---------:|---------------:|----------:|------------:|------------:|
| `learners/non-learners`            |  1       |     91   | 7027.06        | 0.0229771 | 0.000237634 | 0.879852    |
| decile                             |  3.48571 |    317.2 |  125.559       | 0.729139  | 0.000469555 | 0.55466     |
| `learners/non-learners`:decile     |  3.48571 |    317.2 |  125.559       | 2.1123    | 0.00135908  | 0.0885415   |
| bin                                |  5       |    455   |    0.00146579  | 0.965476  | 1.04166e-08 | 0.438525    |
| `learners/non-learners`:bin        |  5       |    455   |    0.00146579  | 0.0171976 | 1.85546e-10 | 0.999887    |
| decile:bin                         | 45       |   4095   |    0.000744432 | 4.76421   | 2.34948e-07 | 3.67988e-23 |
| `learners/non-learners`:decile:bin | 45       |   4095   |    0.000744432 | 0.750984  | 3.70349e-08 | 0.888727    |
