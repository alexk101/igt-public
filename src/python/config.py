from pathlib import Path
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_path)

DATA_PATH = Path('../../data')
SIGNALS_PATH = DATA_PATH / 'signals'
INDIV_PATH = DATA_PATH / 'indiv'
OUTPUT = Path('./plots')
(OUTPUT / 'indiv').mkdir(exist_ok=True, parents=True)
(OUTPUT / 'group').mkdir(exist_ok=True, parents=True)
# Sample size
N = 100
R_N = 10

# Periods and length in seconds
PERIODS = {
    'baseline': 6,
    'planning': 3,
    'feedback': 3
}
# Signals and Frequencies
SIGNALS = {
    'ad_pupil': 60,
    'hr': 100,
    'pupil': 60,
    'rsa': 5,
    'rsp': 100,
    'scl': 100,
    'scr': 100
}

NAME_MAP = {
    'net_gain': 'Winning Trials vs Losing Trials',
    'win_loss_2': 'Winners vs Losers',
    've_median_split': 'Vagal Efficiency'
}