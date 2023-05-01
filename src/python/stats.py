import numpy as np
import pingouin as pg
import polars as pl
import os
from pathlib import Path
from utils import *
from config import *
import sys
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

dir_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_path)

SRC = Path('../../data')
SIG_SRC = SRC / 'signals'
R_OUTPUT = Path('../R/data')
P = 0.05



########### R Section ###############
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

pandas2ri.activate()

def aov(data: pl.DataFrame, period: str, sig: str, dv: str, between: List[str], within: List[str], id: str='participant', zscore: bool=False) -> pd.DataFrame:
    subdir = 'raw'
    if zscore:
        subdir = 'zscore'

    output = Path(f'./r_stats/{subdir}/{sig}')
    output.mkdir(parents=True, exist_ok=True)

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(data.to_pandas())
    afex = importr("afex")
    ggplt = importr("ggplot2")

    model = afex.aov_ez(
        id=id,
        dv=dv,
        between=between,
        within=within,
        data=r_from_pd_df,
        check_contrasts=True,  # 3-way interaction is unaffected by this, "True" is recommended
        type=3,  # using so-called type 3 sums of squares
        print_formula=True,
    )

    print(model)
    for w in within:
        plot = afex.afex_plot(model, x = w, trace = between[0], error='within')
        ggplt.ggsave(filename=f'anova_{w}_{period}.png', plot=plot, path=str(output))

    model: pd.DataFrame = ro.conversion.rpy2py(model[0])
    model.rename(columns={'Pr(>F)':'P'}, inplace=True)
    return model

##################################################


def physio_anova_uneven(zscore: bool = False) -> Dict[str, Dict[str, pd.DataFrame]]:
    subdir = 'raw'
    if zscore:
        subdir = 'zscore'

    # precision bug for scl https://github.com/jasp-stats/jasp-issues/issues/1924
    signals = ['ad_pupil', 'pupil', 'rsa', 'scr', 'rsp', 'hr']
    if not zscore:
        signals.append('scl')
    groups = ['win/loss_2']
    grouping = ('decile', 10)
    analysis = defaultdict(dict)

    for signal in signals:
        output = Path(f'./r_stats') / subdir / signal
        output.mkdir(parents=True, exist_ok=True)
        for period in PERIODS.keys():
            if (output / f'anova_{period}.csv').exists():
                analysis[signal][period] = pd.read_csv(output / f'anova_{period}.csv')
                continue
            data = import_signal(signal, period, zscore)
            
            for group in groups:
                data = data.sort('participant', grouping[0])
                data = data.fill_null(strategy='backward')
                data = data.fill_null(strategy='forward')
                data = data.groupby('participant', group, grouping[0], maintain_order=True).agg(pl.col(f'data_{x}').mean() for x in range(PERIODS[period]*SIGNALS[signal]))
                exclude = data.select('participant', grouping[0]).groupby('participant').n_unique().filter(~(pl.col(grouping[0]) == grouping[1]))['participant']
                data = data.filter(~pl.col('participant').is_in(exclude))
                half_second = SIGNALS[signal] // 2
                bins = [f'bin_{x+1}' for x in range(PERIODS[period]*2)]

                for bin_ind in range(PERIODS[period]*2):
                    time_bin = data.select([f'data_{x}' for x in range(bin_ind*half_second, (bin_ind+1)*half_second)])
                    data = data.with_columns(time_bin.mean(axis=1).alias(f'bin_{bin_ind+1}'))

                print(f'Running ANOVA for {signal}, {period}, {group}')
                id_vars = ['participant', grouping[0], group]
                temp = data.select(id_vars+bins)
                temp = temp.melt(id_vars, bins, 'bin', signal)

                map_dict = {
                    'winner': 'learner',
                    'loser': 'non-learner'
                }

                temp = temp.with_columns(pl.col(group).map_dict(map_dict))
                temp = temp.rename({'win/loss_2':'learners/non-learners'})
                test = aov(
                    period=period, 
                    sig=signal, 
                    dv=signal, 
                    between=['learners/non-learners'], 
                    within=[grouping[0], 'bin'], 
                    id='participant', data=temp,
                    zscore=zscore
                )
                test.reset_index(inplace=True)
                test.rename(columns = {'index':'Effect'}, inplace=True)
                analysis[signal][period] = test
                test.to_csv(output / f'anova_{period}.csv', index=False)
    return dict(analysis)


def summarize_signals():
    out = Path('./temp')
    out.mkdir(exist_ok=True)
    mixtures = [['ad_pupil', 'hr', 'rsa', 'scr']]
    groups = ['win/loss_2']
    grouping = ('decile', 10)
    id_vars = []

    group_names = {'win/loss_2':'learners_nonlearners'}

    for mixt in mixtures:
        print(f'Running MANOVA for {mixt}')
        extra_exclude = set()
        for temp in mixt:
            extra_exclude = extra_exclude.union(FILTER[temp])
        for group in groups:
            for period in PERIODS.keys():
                period_data = []
                for signal in mixt:
                    print(f'summarizing: {signal} in period {period}...')
                    data = import_signal(signal, period)
                    data = data.sort('participant', grouping[0])
                    data = data.fill_null(strategy='backward')
                    data = data.fill_null(strategy='forward')
                    # Average to Deciles
                    data = data.groupby('participant', group, grouping[0], maintain_order=True).agg(pl.col(f'data_{x}').mean() for x in range(PERIODS[period]*SIGNALS[signal]))
                    exclude = data.select('participant', grouping[0]).groupby('participant').n_unique().filter(~(pl.col(grouping[0]) == grouping[1]))['participant'].to_list()
                    extra_exclude = extra_exclude.union(exclude)
                    half_second = SIGNALS[signal] // 2
                    bins = [f'bin_{x+1}' for x in range(PERIODS[period]*2)]

                    for bin_ind in range(PERIODS[period]*2):
                        time_bin = data.select([f'data_{x}' for x in range(bin_ind*half_second, (bin_ind+1)*half_second)])
                        data = data.with_columns(time_bin.mean(axis=1).alias(f'bin_{bin_ind+1}'))

                    id_vars = ['participant', grouping[0], group]
                    temp = data.select(id_vars+bins)
                    temp = temp.melt(id_vars, bins, 'bin', 'value')

                    map_dict = {
                        'winner': 'learner',
                        'loser': 'non-learner'
                    }

                    temp = temp.with_columns(pl.col(group).map_dict(map_dict))
                    temp = temp.rename(group_names)
                    temp = temp.with_columns([pl.lit(signal).alias('signal')])
                    period_data.append(temp)
                period_data = pl.concat(period_data)
                id_vars.remove(group)
                id_vars.append(group_names[group])
                period_data = period_data.pivot(values='value', index=id_vars+['bin'], columns='signal')
                period_data = period_data.filter(~pl.col('participant').is_in(extra_exclude))
                rsa_oof = period_data.filter(pl.col('rsa').is_null()).groupby('participant').count()

                # Drop rsa subs I guess
                period_data = period_data.filter(~pl.col('participant').is_in(rsa_oof['participant']))
                period_data.write_csv(str(out / f'all_data_{period}.csv'))


def make_report(analysis: Dict[str, Dict[str, pd.DataFrame]], title: str='Learners/Non-learners', zscore: bool = False):
    named_sig = {
        'ad_pupil': 'Pupil Change Scores',
        'pupil': 'Raw Pupil',
        'hr': 'Heart Rate',
        'rsa': 'Respiratory Sinus Arythmia',
        'scr': 'Skin Conductance Response (Phasic)',
        'scl': 'Skin Conductance Level (Tonic)',
        'rsp': 'Respiration Rate'
    }
        
    qual = '(raw)'
    subdir = 'raw'
    if zscore:
        qual = '(z-scored within subject)'
        subdir = 'zscore'

    output = Path(f'./report_{subdir}.md')
    im_table = """Bin             |  Decile
:-------------------------:|:-------------------------:
![]({path1})  |  ![]({path2})\n"""
    with open(output, 'w+') as fp:
        fp.write(f'# {title} Statistics {qual}\n')

        fp.write("dv='pupil'\n\nbetween=['learners/non-learners']\n\nwithin=['decile, 'bin']\n\n")

        for signal in analysis.keys():
            fp.write(f'## ANOVA {named_sig[signal]}\n')
            for period, anova in analysis[signal].items():
                fp.write(f'### {period.capitalize()}\n')
                im1 = './'+str(list(Path(f'./r_stats/{subdir}/{signal}').glob(f'*bin_{period}*'))[0])
                im2 = './'+str(list(Path(f'./r_stats/{subdir}/{signal}').glob(f'*decile_{period}*'))[0])
                fp.write(f'{im_table.format(path1=im1,path2=im2)}\n')

                fp.write(f'Significant Effects/Interactions\n\n')
                meh = pl.from_pandas(anova).filter(pl.col('P') < P)['Effect']
                if not len(meh):
                    fp.write(f'No statistically significant effects present at P < 0.05\n')
                for effect in meh:
                    fp.write(f'- {effect}\n')
                fp.write('\n')

                anova_table = anova.to_markdown(index=False)
                fp.write(f'{anova_table}\n')


def plot_f_dist(rv, dfn: int, dfd: int):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf') # type: ignore
    ax.set_title(f'Plot of F Distribution\ndfn: {dfn}, dfd: {dfd}')
    fig.savefig('f_dist.png')


def basic_anova(data: pd.DataFrame, dv: str='score', between: str='win/loss_2'):
    print(f'ANOVA:\ndv={dv}\niv={between}')
    bruh = pg.anova(data, dv=dv, between=between, detailed=True)
    bruh = pl.from_pandas(bruh)
    bruh = bruh.rename({'p-unc': 'p'})
    bruh = bruh.to_pandas()
    print(f"\n{bruh.to_markdown()}\n")


def rm_anova_stat(data: pd.DataFrame, dv: str='score', within='trial'):
    print(f'RM-ANOVA:\ndv={dv}\nwithin={within}')
    bruh = pg.rm_anova(data, dv=dv, within=within, subject='participant', detailed=True)
    bruh = pl.from_pandas(bruh)
    bruh = bruh.rename({'p-unc': 'p'})
    remove = ['eps', 'sphericity', 'p-spher', 'W-spher']
    for col in remove:
        if col in bruh.columns:
            bruh = bruh.drop(col)
    bruh = bruh.to_pandas()
    print(f"\n{bruh.to_markdown()}\n")


def mixed_anova(data: pd.DataFrame, dv: str='score', within='trial', between: str='win/loss_2'):
    print(f'MIXED ANOVA:\ndv={dv}\nwithin={within}\nbetween={between}')
    bruh = pg.mixed_anova(data, dv=dv, within=within, between=between, subject='participant')
    bruh = pl.from_pandas(bruh)
    bruh = bruh.rename({'p-unc': 'p'})
    remove = ['eps', 'sphericity', 'p-spher', 'W-spher']
    for col in remove:
        if col in bruh.columns:
            bruh = bruh.drop(col)
    bruh = bruh.to_pandas()
    print(f"\n{bruh.to_markdown()}\n")


def plot():
    data = import_trial_data().drop(['valid_score', 'win/loss', 've_median_split'])
    data = data.with_columns([(pl.col('gain')-pl.col('loss')).alias('net')])
    data = data.select(['participant', 'trial', 'score', 'win/loss_2'])
    data = data.filter(pl.col('trial')==99)

    print(data)
    sns.displot(data.to_pandas(), x="score", hue="win/loss_2", multiple="stack")
    plt.show()


def new_stats():
    data = import_trial_data().drop(['valid_score', 'win/loss', 've_median_split'])
    data = data.with_columns([(pl.col('gain')-pl.col('loss')).alias('net')])
    data = data.select(['participant', 'trial', 'score', 'win/loss_2', 'quintile'])

    final_scores = data.filter(pl.col('trial')==99).to_pandas()
    basic_anova(final_scores)

    # Trial
    rm_anova_stat(data.to_pandas())
    mixed_anova(data.to_pandas())

    # Quintile
    rm_anova_stat(data.to_pandas(), within='quintile')
    mixed_anova(data.to_pandas(), within='quintile')


def main() -> int:
    # pl.Config.set_tbl_cols(-1)
    # new_stats()
    data = physio_anova_uneven()
    make_report(data)
    data = physio_anova_uneven(True)
    make_report(data, zscore = True)
    # big_manova()
    # check()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())