# Analysis
pupil = [24,42,43,54,71,77,82,86,88,107]
# Manual
pupil += [34,21,106]

ecg = [3,6,18,24,43,60,109,103]
eda = [5,16,18,19,21,22,55,67,68,69,93,97,98,99,100]
rsp = [41,56]

FILTER = {
    'pupil': pupil,
    'ad_pupil': pupil,
    'hr': ecg,
    'rsa': ecg,
    'scl': eda,
    'scr': eda,
    'rsp': rsp
}