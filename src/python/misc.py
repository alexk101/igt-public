import polars as pl
import sys
from config import *
import numpy as np
from collections import defaultdict

dir_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_path)

TTP_OUTPUT = SIGNALS_PATH / 'ttp'
TTP_OUTPUT.mkdir(exist_ok=True, parents=True)

def gen_ttp(signal: str, freq: int) -> None:
    """Generates the time to peak (max) and time to trough (min) for a given
    signal at a given frequency.

    Args:
        signal (str): The filename of the signal.
        freq (int): The frequency of previously specified signal.
    """
    
    data = pl.scan_csv(SIGNALS_PATH / f'{signal}.csv')
    id_vars = ['participant', 'trial']

    ttp = []
    for period, time in PERIODS.items():
        data_cols = [f'data_{x}' for x in range(freq*time)]
        print(f'{period}')
        period_data = data.filter(pl.col('period') == period)
        period_data = period_data.select(id_vars+data_cols).collect()

        signal_data = period_data.select(data_cols).to_numpy()
        nan_mask = np.isnan(signal_data)
        all_nans: np.ndarray = np.all(nan_mask, axis=1)
        all_nan_ind = np.where(all_nans)
        print(f'found {all_nans.sum()} all nans out of {nan_mask.shape[0]}.')


        old_glue = period_data.select(id_vars).with_columns(pl.Series('valid', all_nans))
        glue = old_glue.filter(~pl.col('valid'))
        glue = glue.drop('valid')
        invalid = old_glue.filter('valid')
        invalid = invalid.drop('valid')

        signal_data = np.delete(signal_data, all_nan_ind, 0)

        all_data = []

        # valid ttps
        all_data.append(glue.with_columns(
            [
                pl.lit(period).alias('period'),
                pl.Series('min', np.nanargmin(signal_data, axis=1)/freq),
                pl.Series('max', np.nanargmax(signal_data, axis=1)/freq)
            ]
        ))
        # invalid ttps
        all_data.append(invalid.with_columns(
            [
                pl.lit(period).alias('period'),
                pl.lit(np.nan).alias('min'),
                pl.lit(np.nan).alias('max')
            ]
        ))
        all_data = pl.concat(all_data)
        ttp.append(all_data)

    ttp = pl.concat(ttp).sort('participant','trial','period')
    ttp.write_csv(str(TTP_OUTPUT / f'{signal}.csv'))


def add_trial_win_loss():
    data_path = INDIV_PATH / f'igt_trial_info.csv'
    data = pl.read_csv(data_path)
    if 'net_gain' in data.columns:
        print(f'net_gain column already exists')
        return
    net_gain = (data['gain'] - data['loss']).to_numpy()
    net_gain = net_gain > 0
    bruh = np.array(['loss']*len(net_gain))
    bruh[net_gain > 0] = 'gain'

    data.with_columns([pl.Series('net_gain', bruh)]).write_csv(data_path)


def reconstruct_scores():
    trials = pl.read_csv(INDIV_PATH / 'igt_trial_info.csv').drop(['net_gain'])
    trials = trials.with_columns([(trials['gain'] - trials['loss']).alias('net')])
    fixed = []

    for sub in trials['participant'].unique():
        sub_data = trials.filter(pl.col('participant') == sub)
        cur_val = 2000
        net = sub_data.drop_in_place('net').to_numpy()
        reconstructed = []
        for val in net:
            cur_val += val
            reconstructed.append(cur_val)
        reconstructed = np.array(reconstructed)
        sub_data = sub_data.replace('score', pl.Series(reconstructed))
        sub_data = sub_data.with_columns(
            [
                pl.when((pl.col('gain') - pl.col('loss')) > 0)
                .then('gain')
                .otherwise('loss')
                .alias('net_gain')
            ]
        )
        fixed.append(sub_data)
        print(f"sub: {sub} diff: {np.abs(sub_data['score'].to_numpy() - reconstructed).sum()}")
    fixed = pl.concat(fixed)
    fixed.write_csv(INDIV_PATH / 'igt_trial_info_corrected.csv')


def scores():
    data = pl.read_csv(INDIV_PATH / 'igt_trial_info_corrected.csv')
    data = data.select('deck', 'gain', 'loss')
    output = data.groupby('deck', 'gain', 'loss').count()
    sums = output.groupby('deck').sum()

    new_output = []
    for deck in output['deck'].unique():
        count = sums.filter(pl.col('deck') == deck).to_dicts()[0]['count']
        new_output.append(output.filter(pl.col('deck') == deck).with_columns([(pl.col('count') / count).alias('percent')]))
    output = pl.concat(new_output)
    output.write_csv('deck_probs.csv')
    

def main() -> int:
    # for signal, freq in SIGNALS.items():
    #     print(f'Extracting ttp for {signal}')
    #     gen_ttp(signal, freq)
    # add_trial_win_loss()
    # reconstruct_scores()
    scores()
    return 0


if __name__ == '__main__':
    sys.exit(main())
