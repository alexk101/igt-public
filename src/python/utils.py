import polars as pl
import os
from config import *
from typing import Tuple
import numpy as np
from cleaning import FILTER
from typing import List, Dict
from collections import defaultdict
from JaxRPCA import RPCA

dir_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_path)
THRESHOLD = 0.5


def import_factors(group: str, drop_nulls: bool=False) -> Tuple[np.ndarray, pl.DataFrame]:
    # Bad subjects!
    bad_subject = [18,24]
    groups = pl.read_csv(INDIV_PATH / 'groups.csv')
    groups = groups.rename({'exp': 'participant'}).with_columns(pl.col('participant').cast(pl.Int32))
    groups = groups.select(['participant', group])

    factors = pl.read_csv(INDIV_PATH / f'summary.csv')
    factors = factors.rename({'subject': 'participant'}).with_columns(pl.col('participant').cast(pl.Int32))
    factors = factors.join(groups, on='participant')
    factors = factors.filter(~pl.col('participant').is_in(bad_subject))

    # Remove column we no want :)
    not_interesting = [
        'tlx',
        'learned',
        'valid_score'
    ]
    factors = factors.select(list(set(factors.columns).difference(not_interesting)))

    if drop_nulls:
        factors = factors.drop_nulls()

    participants = factors.drop_in_place('participant').to_numpy()
    return participants, factors


def import_signal(name: str, period: str, zscore: bool = False, rpca: bool=False) -> pl.DataFrame:
    groups = pl.read_csv(INDIV_PATH / 'groups.csv')
    groups = groups.rename({'exp': 'participant'}).with_columns(pl.col('participant').cast(pl.Int32))
    
    trial_info = pl.read_csv(INDIV_PATH / 'igt_trial_info_corrected.csv').sort('participant', 'trial')
    sig = pl.read_csv(SIGNALS_PATH / f'{name}.csv')        
    data_cols = [f'data_{x}' for x in range(SIGNALS[name]*PERIODS[period])]

    # Postprocessing
    if zscore:
        print(f'z-scoring within subject...', end='\r')
        id_vars = ['participant', 'trial', 'period']
        
        subset = sig.select(id_vars + data_cols)
        subset = subset.melt(id_vars=id_vars, value_vars=data_cols, variable_name='data_ind', value_name='data')
        new_subset = []
        
        # zscore within subject
        for sub in subset['participant'].unique():
            subject_subset = subset.filter(pl.col('participant') == sub)
            mu = subject_subset['data'].mean()
            sigma = subject_subset['data'].std()
            new_subset.append(subject_subset.with_columns((pl.col('data')-mu)/sigma))
        subset = pl.concat(new_subset)
        sig = subset.pivot(values='data', index=id_vars, columns='data_ind')

    sig = sig.filter(pl.col('period') == period).drop('period')
    sig = sig.with_columns(pl.col('participant').cast(pl.Int32))
    sig = sig.filter(~pl.col('participant').is_in(FILTER[name]))
    subs = sig['participant'].unique()

    if rpca:
        participants = sig['participant']
        trial = sig['trial']
        data = sig.select(data_cols)
        data = data.fill_null(strategy='backward')
        data = data.fill_null(strategy='forward')
        model = RPCA(data.to_numpy())
        d, s = model.fit(max_iter=100000, tol=0.01)
        sig = pl.DataFrame({f'data_{ind}': dat for ind, dat in enumerate(np.array(d.T))})
        sig = sig.with_columns([participants, trial])

    # Filtering signal
    if 'pupil' in name and not rpca:
        trial_err = pl.scan_csv(INDIV_PATH / 'quality_periods.csv')
        trial_err = trial_err.filter(pl.col('period') == period).drop('period')
        id_vars = ['participant']
        trial_err = trial_err.melt(id_vars=id_vars, variable_name='trial', value_name='err')
        trial_err = trial_err.with_columns([pl.col('trial').str.slice(6)]).collect()
        
        trial_err = trial_err.filter(pl.col('participant').is_in(subs))
        sig = sig.with_columns([trial_err.sort('participant','trial')['err']])
        sig = sig.filter(pl.col('err') < 0.5)
    
    sig = sig.join(groups, on=['participant'], how='left')
    trial_info = trial_info.filter(pl.col('participant').is_in(subs))
    trial_info = trial_info.select(['participant', 'trial', 'net_gain'])
    trial_info = trial_info.with_columns(
        pl.col('participant').cast(pl.Int32),
        pl.col('trial').cast(pl.Int32)
    )
    sig = sig.with_columns(
        pl.col('trial').cast(pl.Int32),
        pl.col('participant').cast(pl.Int32)
    )

    sig = sig.sort('participant', 'trial').join(
        trial_info,
        on=['participant', 'trial'], 
        how='left'
    )

    sig = sig.with_columns(
        [
            ((pl.col('trial') // 20)+1).alias('quintile').cast(pl.Int32),
            ((pl.col('trial') // 10)+1).alias('decile').cast(pl.Int32),
        ]
    )

    return sig


def import_trial_data() -> pl.DataFrame:
    groups = pl.read_csv(INDIV_PATH / 'groups.csv')
    groups = groups.rename({'exp': 'participant'}).with_columns(pl.col('participant').cast(pl.Int32))
    trials = pl.read_csv(INDIV_PATH / 'igt_trial_info_corrected.csv')
    
    new_trials = []
    for subject in groups['participant'].unique():
        sub_dat = trials.filter(pl.col('participant') == subject)
        sub_grps = groups.filter(pl.col('participant') == subject).to_dicts()[0]
        sub_grps.pop('participant')
        if sub_grps['ve_median_split'] is None:
            sub_grps['ve_median_split'] = ''
        sub_dat = sub_dat.with_columns([pl.lit(val).alias(key) for key, val in sub_grps.items()])
        new_trials.append(sub_dat)
    trials = pl.concat(new_trials)
    trials = trials.with_columns(
        [
            ((pl.col('trial') // 20)+1).alias('quintile').cast(pl.Int32),
            pl.col('trial').cast(pl.Int32),
            pl.col('participant').cast(pl.Int32)
        ]
    )
    return trials


def import_ttp(name:str, period: str) -> pl.DataFrame:
    trial_data = import_trial_data().sort('participant', 'trial')
    ttp = pl.read_csv(SIGNALS_PATH / 'ttp' / f'{name}.csv').filter(pl.col('period') == period)
    trial_data = trial_data.filter(pl.col('participant').is_in(ttp['participant'].unique()))
    ttp = ttp.sort('participant', 'trial').drop('period').with_columns([trial_data[x] for x in trial_data.columns])
    return ttp


class TransitionMatrix():
    def __init__(self, group_data: pl.DataFrame, domain: List) -> None:
        counts = {card: dict(zip(domain, [0]*len(domain))) for card in domain}
        for sub in group_data['participant'].unique():
            sub_data = group_data.filter(pl.col('participant') == sub).sort('trial')
            deck = sub_data['deck'].to_numpy()
            for card in range(len(deck)-1):
                counts[deck[card]][deck[card+1]] += 1

        self.counts: Dict[str, Dict[str, int]] = counts
        self.probs: Dict[str, Dict[str, float]] = defaultdict(dict)

        for key, vals in self.counts.items():
            total = sum(vals.values())
            for key_2, val_2 in vals.items():
                if total:
                    self.probs[key][key_2] = val_2 / total
                else:
                    self.probs[key][key_2] = total


    def to_numpy(self):
        array = []
        for vals in self.probs.values():
            array.append(np.array(list(vals.values())) / sum(vals.values()))
        return np.nan_to_num(np.array(array))


    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return self.probs
